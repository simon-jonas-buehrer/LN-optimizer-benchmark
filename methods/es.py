"""es: Evolution Strategies over CONTINUOUS per-LUT weights.

The requested contrast to a GA that evolves binary truth-table bits by crossover: here every 2-input
gate carries four continuous real weights (one per truth-table entry) -- a soft gate -- and an
OpenAI-ES loop (antithetic Gaussian perturbations, rank-shaped update) moves the mean. Wiring is
fixed random (chosen once from the seed); only the continuous weights change. At emit time each
gate's four reals are HARDENED to a boolean truth table (bit k = 1 iff weight_k > 0), and predict()
runs that exact hard net via LutModel -- so the measured circuit is the hard net, not the soft one.
Fitness is the packed simulator. Trains to convergence with early stopping on val loss.

Speed. The search is untouched (same seeds, same draws, same order, same perturbations, same
fitness, same update, same early stop); only the evaluation is. `_Sim` is a bit-identical twin of
methods.lut.lut_sim, specialised for the way ES uses it -- one fixed wiring, `pop` truth tables per
generation, all scored on the same minibatch:

  * the minibatch is thermometer-encoded and bit-packed ONCE per generation instead of once per
    candidate (`pop` times), and the encoder transposes per chunk so the pack stays cache-resident;
  * the 4-mask LUT expression `(m0&~A&~B)|(m1&~A&B)|(m2&A&~B)|(m3&A&B)` is replaced by its
    Reed-Muller form factored over A, `c0 ^ (c1&B) ^ (A & (c2 ^ (c3&B)))`, with c0=t0, c1=t0^t1,
    c2=t0^t2, c3=t0^t1^t2^t3 -- the same bits in 6 ops instead of 13, into preallocated buffers;
  * the readout popcount accumulates in uint8/uint16 blocks instead of the default int64;
  * each antithetic perturbation is drawn from its seed ONCE per generation (cached when it fits in
    memory) instead of three times -- once in the fitness loop and twice in the gradient loop -- and
    the rank-weighted sum is folded over each antithetic pair, `(u+ - u-) * eps`, so the gradient
    costs one float32 pass per PAIR instead of one float64 pass per population member;
  * on CUDA the wiring lives on the device for the whole run and the packed minibatch is uploaded
    once per generation, instead of being rebuilt and re-uploaded for every candidate; a device is
    used only from `_GPU_MIN_GATES` gates upward, because a generation is `pop` small kernels and
    the little tiers are launch-bound.
"""

from __future__ import annotations

import time

import numpy as np

from data import Dataset, DatasetSpec
from hw import even_thresholds
from methods.lut import LutModel, load as lut_load, pack_encoded

TITLE = "es (evolution strategies, continuous per-LUT weights)"

# >=5 points, ~1k -> ~20M gates by pre-opt gate count (= sum of widths). ES is gradient-free and
# per-eval expensive, so the big tiers are heavy (the plateau is itself a result).
_LADDER = {
    "xs": (1, (700, 300)),
    "s": (1, (4000, 2000)),
    "m": (3, (26000, 13000)),
    "l": (3, (160000, 80000, 40000)),
    "xl": (7, (1_000_000, 500_000, 250_000)),
    # xxl (11M, 6M, 3M) = 20M LUT nodes is REMOVED: it cannot be measured inside a 48 h job.
    # Extrapolating the measured l point (280k nodes -> 11.2 MB .sv, 196 s, 11.4 GB) at the fitted
    # exponents gives ~14 h of synthesis and ~2 TB of peak RSS for 20M nodes; the nodes have 515 GB.
}

_SIM_MEM = 1 << 30       # signal-buffer budget per image chunk (bytes)
_MAX_WORDS = 16          # 1024 images/chunk: the sweet spot for the cache-tiled numpy gate loop
_MAX_WORDS_GPU = 64      # 4096 images/chunk: a GPU wants fewer, bigger kernels (measured 1.01-1.24x)
_EPS_MEM = 1 << 30       # how much of a generation's perturbations may be cached (bytes)
_GRAD_BLK = 1 << 21      # rows per block when accumulating the ES gradient (bounds temporaries)
# Measured on an RTX 3090, pop=40 batch=2048 (cuda s/gen vs numpy s/gen, results bit-identical):
#   1k gates 0.59x (GPU LOSES) | 6k 1.40x | 39k 3.40x | 280k 5.00x -- so route by size, not device.
_GPU_MIN_GATES = 3000    # break-even; below it a generation is pop*small kernels and is launch-bound
_L2 = 1 << 19            # gate-loop row block: keep a layer's A/B/tmp tiles roughly L2-resident


def points(spec: DatasetSpec) -> list[dict]:
    return [{"name": n, "bits": b, "widths": w, "gens": 3000} for n, (b, w) in _LADDER.items()]


def _wiring(widths, n_in, rng):
    """Fixed random 2-input wiring: gate reads two strictly-earlier signal ids. Returns per-layer
    (idx_a, idx_b) and the per-layer signal offsets."""
    wires, off = [], n_in
    for w in widths:
        a = rng.integers(0, off, w, dtype=np.int64)
        b = rng.integers(0, off, w, dtype=np.int64)
        wires.append((a, b))
        off += w
    return wires


def _layers(theta, wires, widths):
    """Harden the flat genome theta (G,4) into (idx_a, idx_b, tt) per layer."""
    tt = ((theta[:, 0] > 0) | ((theta[:, 1] > 0) << 1) | ((theta[:, 2] > 0) << 2) |
          ((theta[:, 3] > 0) << 3)).astype(np.int64)
    out, o = [], 0
    for (a, b), w in zip(wires, widths):
        out.append((a, b, tt[o:o + w]))
        o += w
    return out


def _ranks(f):
    """Centered rank utilities in [-0.5, 0.5] (OpenAI-ES fitness shaping)."""
    order = np.argsort(np.argsort(f))
    return order / (len(f) - 1) - 0.5


# ================================================================================================
# Packed simulator (bit-identical to methods.lut.lut_sim, reused across a generation's candidates)
# ================================================================================================

def _pack(pix: np.ndarray, thresholds, spec: DatasetSpec, chunk: int) -> list:
    """[(n, enc)] per image chunk; enc is (n_in, words) int64, 64 images per word, little-endian.

    `methods.lut.pack_encoded` builds exactly the `S[:n_in]` a `lut_sim` chunk starts from (the
    threshold compare writes the transposed layout directly instead of byte-transposing an
    (n, n_in) array afterwards); this only cuts the batch into simulator-sized chunks and views the
    words as int64, whose bits are identical to uint64 and which the torch twin can share.
    """
    out = []
    for i in range(0, len(pix), chunk):
        p = pix[i:i + chunk]
        out.append((len(p), pack_encoded(p, thresholds, spec).view(np.int64)))
    return out


def _popcount_rows(bits: np.ndarray, n: int, n_classes: int, g: int) -> np.ndarray:
    """(n_classes*g, words) int64 readout rows -> (n, n_classes) int64 counts.

    Identical to lut_sim's `unpackbits(...).reshape(n_classes, g, n).sum(1)`, but the group sum is
    accumulated in uint8 blocks of 255 rows (255 zero/one rows cannot overflow a uint8) into a
    uint16/uint32 accumulator instead of numpy's default int64.
    """
    r = np.unpackbits(bits.view(np.uint8), axis=1, bitorder="little")[:, :n]
    r = r.reshape(n_classes, g, n)
    acc = np.zeros((n_classes, n), np.uint16 if g <= 0xFFFF else np.uint32)
    for i in range(0, g, 255):
        blk = r[:, i:i + 255]
        acc += (blk[:, 0] if blk.shape[1] == 1 else blk.sum(1, dtype=np.uint8)).astype(acc.dtype)
    return acc.T.astype(np.int64)


def _rm_coeffs(sgn: np.ndarray):
    """Hardened signs (G,4) -> the four Reed-Muller coefficient bit vectors (G,) each."""
    s0, s1, s2, s3 = sgn[:, 0], sgn[:, 1], sgn[:, 2], sgn[:, 3]
    c1 = s0 ^ s1
    return s0, c1, s0 ^ s2, c1 ^ s2 ^ s3


class _Sim:
    """Fixed-wiring packed LUT simulator; one instance per run, reused by every candidate."""

    def __init__(self, widths, wires, spec: DatasetSpec, n_in: int):
        self.widths = list(widths)
        self.wires = wires
        self.offs = [n_in]
        for w in widths:
            self.offs.append(self.offs[-1] + w)
        self.n_in = n_in
        self.n_sig = self.offs[-1]
        self.n_classes = spec.n_classes
        self.g = widths[-1] // spec.n_classes
        self.words = max(1, min(_MAX_WORDS, int(_SIM_MEM // (max(1, self.n_sig) * 8))))
        self.chunk = self.words * 64
        self._bufs: dict[int, np.ndarray] = {}
        self._tmp: dict[int, list] = {}

    def pack(self, pix, thresholds, spec):
        return _pack(pix, thresholds, spec, self.chunk)

    def _buf(self, words):
        S = self._bufs.get(words)
        if S is None:
            S = self._bufs[words] = np.zeros((self.n_sig, words), np.int64)
            blk = max(256, min(8192, _L2 // (words * 8)))
            self._tmp[words] = [blk] + [np.empty((blk, words), np.int64) for _ in range(3)]
        return S

    def counts(self, sgn: np.ndarray, packed) -> np.ndarray:
        """(G,4) hardened signs + packed images -> (N, n_classes) counts (== lut_sim)."""
        c = _rm_coeffs(sgn)
        masks, o = [], 0
        for w in self.widths:                       # per-layer 0/-1 masks, built once per candidate
            masks.append([-(x[o:o + w].astype(np.int64))[:, None] for x in c])
            o += w
        out = np.empty((sum(n for n, _ in packed), self.n_classes), np.int64)
        pos, offs, last = 0, self.offs, len(self.widths) - 1
        for n, enc in packed:
            words = enc.shape[1]
            S = self._buf(words)
            blk, bA, bB, bT = self._tmp[words]
            S[:self.n_in] = enc
            for l, (ia, ib) in enumerate(self.wires):
                o, e = offs[l], offs[l + 1]
                m0, m1, m2, m3 = masks[l]
                for i in range(0, e - o, blk):       # row blocks: A/B/t tiles stay cache-resident
                    j = min(i + blk, e - o)
                    A = np.take(S, ia[i:j], axis=0, out=bA[:j - i])
                    B = np.take(S, ib[i:j], axis=0, out=bB[:j - i])
                    t, dst = bT[:j - i], S[o + i:o + j]
                    np.bitwise_and(B, m3[i:j], out=t)      # c3&B
                    np.bitwise_xor(t, m2[i:j], out=t)      # ^ c2
                    np.bitwise_and(t, A, out=t)            # &A  == c2&A ^ c3&A&B
                    np.bitwise_xor(t, m0[i:j], out=t)      # ^ c0
                    np.bitwise_and(B, m1[i:j], out=B)      # c1&B (B is a fresh tile, safe in place)
                    np.bitwise_xor(t, B, out=dst)
            out[pos:pos + n] = _popcount_rows(S[offs[last]:offs[last + 1]], n,
                                              self.n_classes, self.g)
            pos += n
        return out


class _TorchSim:
    """CUDA twin of `_Sim`: same 64-images-per-word bit arithmetic, gate loop on the device.

    numpy's int64 words and torch's int64 words share the exact two's-complement layout, so
    `&`/`^`/`>>` produce identical bits (checked against `_Sim` on the torch CPU backend). The
    wiring is uploaded once for the whole run and the packed minibatch once per generation; only
    the 1-byte-per-gate hardened truth table crosses the bus per candidate.
    """

    def __init__(self, widths, wires, spec: DatasetSpec, n_in: int, device):
        import torch

        import batching   # imported here, not at module scope: this file is numpy-only until CUDA

        self.torch = torch
        self.dev = torch.device(device)
        self.widths = list(widths)
        self.offs = [n_in]
        for w in widths:
            self.offs.append(self.offs[-1] + w)
        self.n_in, self.n_sig = n_in, self.offs[-1]
        self.n_classes = spec.n_classes
        self.g = widths[-1] // spec.n_classes
        self.ia = [torch.as_tensor(a, device=self.dev) for a, _ in wires]
        self.ib = [torch.as_tensor(b, device=self.dev) for _, b in wires]
        self.sh = torch.arange(64, dtype=torch.int64, device=self.dev)
        # _SIM_MEM is a ceiling tuned on a 24 GB card; on a smaller one take what is actually free
        # (see `batching.budget`) so a wide net simulates in more, narrower chunks instead of OOMing.
        self.words = max(1, min(_MAX_WORDS_GPU,
                                int(batching.budget(_SIM_MEM, device=self.dev)
                                    // (max(1, self.n_sig) * 8))))
        self.chunk = self.words * 64
        self._bufs, self._tmp = {}, {}
        self._blk = max(1, min(1024, (1 << 24) // max(1, self.words * 64 * 8)))

    def pack(self, pix, thresholds, spec):
        t = self.torch
        return [(n, t.as_tensor(enc, device=self.dev))
                for n, enc in _pack(pix, thresholds, spec, self.chunk)]

    def _buf(self, words):
        S = self._bufs.get(words)
        if S is None:
            t = self.torch
            S = self._bufs[words] = t.zeros((self.n_sig, words), dtype=t.int64, device=self.dev)
            self._tmp[words] = t.empty((max(self.widths), words), dtype=t.int64, device=self.dev)
        return S

    def counts(self, sgn: np.ndarray, packed) -> np.ndarray:
        t = self.torch
        c = _rm_coeffs(sgn)
        # one byte per gate over the bus, unpacked into 0/-1 masks on the device
        code = (c[0].view(np.uint8) | (c[1].view(np.uint8) << 1) | (c[2].view(np.uint8) << 2)
                | (c[3].view(np.uint8) << 3))
        cd = t.as_tensor(code, device=self.dev).to(t.int64)
        cm = [-((cd >> k) & 1) for k in range(4)]
        masks, o = [], 0
        for w in self.widths:
            masks.append([x[o:o + w].view(-1, 1) for x in cm])
            o += w
        out = np.empty((sum(n for n, _ in packed), self.n_classes), np.int64)
        pos, offs, last = 0, self.offs, len(self.widths) - 1
        for n, enc in packed:
            words = enc.shape[1]
            S, tmp = self._buf(words), self._tmp[words]
            S[:self.n_in] = enc
            for l in range(len(self.widths)):
                o, e = offs[l], offs[l + 1]
                m0, m1, m2, m3 = masks[l]
                A, B, dst, tp = S[self.ia[l]], S[self.ib[l]], S[o:e], tmp[:e - o]
                t.bitwise_and(B, m3, out=tp)
                t.bitwise_xor(tp, m2, out=tp)
                t.bitwise_and(tp, A, out=tp)
                t.bitwise_xor(tp, m0, out=tp)
                t.bitwise_and(B, m1, out=B)
                t.bitwise_xor(tp, B, out=dst)
            out[pos:pos + n] = self._readout(S[offs[last]:offs[last + 1]], n)
            pos += n
        return out

    def _readout(self, last, n):
        """Per-image popcount of each class group, on the device."""
        t = self.torch
        words = last.shape[1]
        acc = t.zeros((self.n_classes, words * 64), dtype=t.int32, device=self.dev)
        v = last.view(self.n_classes, self.g, words)
        for i in range(0, self.g, self._blk):
            b = v[:, i:i + self._blk]
            acc += (((b.unsqueeze(-1) >> self.sh) & 1).sum(1, dtype=t.int32)
                    .reshape(self.n_classes, -1))
        return acc[:, :n].t().cpu().numpy().astype(np.int64)


def _softmax_ll(counts, y):
    """mean log-softmax of the true class, with the integer counts as logits (== the original)."""
    z = counts - counts.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(logp[np.arange(len(y)), y].mean())


class ES(LutModel):
    def __init__(self, spec, bits, widths, gens, pop=40, sigma=0.1, lr=0.1, batch=2048,
                 patience=20, eval_every=10):
        super().__init__(spec)
        if widths[-1] % spec.n_classes:
            raise ValueError(f"readout {widths[-1]} not divisible by {spec.n_classes}")
        self.bits, self.widths = bits, tuple(widths)
        self.cfg = dict(gens=gens, pop=pop, sigma=sigma, lr=lr, batch=batch,
                        patience=patience, eval_every=eval_every)

    def train(self, data: Dataset, *, device="cpu", seed=0):
        c = self.cfg
        rng = np.random.default_rng(seed)
        thr = even_thresholds(self.bits)
        n_in = self.spec.n_pixels * self.bits
        wires = _wiring(self.widths, n_in, rng)
        G = sum(self.widths)
        theta = rng.standard_normal((G, 4)).astype(np.float32) * 0.1

        # Route by net size, not by whether a device exists: a generation is `pop` small kernels,
        # so on the little tiers the launch overhead costs more than the gate loop saves. Both
        # backends are bit-identical, so this picks speed only -- it cannot move the search.
        gpu = device is not None and str(device) != "cpu" and G >= _GPU_MIN_GATES
        sim = (_TorchSim(self.widths, wires, self.spec, n_in, device) if gpu
               else _Sim(self.widths, wires, self.spec, n_in))
        val_packed = None                  # the val set is encoded+packed once, not per eval
        pert = np.empty_like(theta)        # scratch for theta +- sigma*eps

        def fit(packed, yb):               # negative CE of the hardened net (higher = better)
            return _softmax_ll(sim.counts(pert > 0, packed), yb)

        best, best_theta, best_gen = float("inf"), theta.copy(), 0
        half = c["pop"] // 2
        keep_eps = half * G * 16 <= _EPS_MEM   # cache a generation's perturbations if they fit
        train_secs, nseen = 0.0, 0  # only the ES search (fitness + update), never the val evals
        for gen in range(c["gens"]):
            t0 = time.perf_counter()
            sel = rng.integers(0, len(data.train_x), c["batch"])
            xb, yb = data.train_x[sel], data.train_y[sel]
            packed = sim.pack(xb, thr, self.spec)   # once per generation, not once per candidate
            seeds = rng.integers(0, 2**31, half)
            fs, cache = [], {}
            for s in seeds:  # antithetic pairs; eps is regenerated from its seed unless cached
                eps = np.random.default_rng(s).standard_normal((G, 4)).astype(np.float32)
                if keep_eps:
                    cache[int(s)] = eps
                np.multiply(eps, c["sigma"], out=pert)
                np.add(theta, pert, out=pert)
                fs.append(fit(packed, yb))
                np.multiply(eps, c["sigma"], out=pert)
                np.subtract(theta, pert, out=pert)
                fs.append(fit(packed, yb))
            u = _ranks(np.array(fs))
            grad = np.zeros_like(theta)
            # sum_j u_j*sign_j*eps_j, folded over each antithetic pair: (u+ - u-) * eps, one pass
            # per PAIR instead of one per member, in float32 blocks instead of float64 temporaries.
            for i in range(0, len(seeds)):
                s = seeds[i]
                eps = cache.get(int(s))
                if eps is None:
                    eps = np.random.default_rng(s).standard_normal((G, 4)).astype(np.float32)
                w = np.float32(u[2 * i] - u[2 * i + 1])
                for b in range(0, G, _GRAD_BLK):
                    grad[b:b + _GRAD_BLK] += w * eps[b:b + _GRAD_BLK]
            theta += np.float32(c["lr"] / (c["pop"] * c["sigma"])) * grad
            train_secs += time.perf_counter() - t0
            nseen += c["pop"] * c["batch"]  # pop candidates each scored on `batch` samples this gen

            if gen % c["eval_every"] == 0 or gen == c["gens"] - 1:
                if val_packed is None:
                    val_packed = sim.pack(data.val_x, thr, self.spec)
                vl = -_softmax_ll(sim.counts(theta > 0, val_packed), data.val_y)
                if vl < best - 1e-4:
                    best, best_gen, best_theta = vl, gen, theta.copy()
                print(f"  gen {gen + 1:5d}/{c['gens']}  val loss {vl:.4f}  (best {best:.4f} @ {best_gen + 1})",
                      flush=True)
                if gen - best_gen >= c["patience"] * c["eval_every"]:
                    print(f"  early stop at gen {gen + 1}", flush=True)
                    break

        self.thresholds = thr
        self.layers = _layers(best_theta, wires, self.widths)
        self.train_seconds = train_secs
        self.train_samples = nseen  # training-example evals until early stop


def build(spec, **point) -> ES:
    return ES(spec, **point)


# The synthesis phase reloads the trained circuit from its .ckpt; (thresholds, layers) is all of it,
# so the shared LUT-net loader covers this method without touching the trainer above.
load = lut_load
