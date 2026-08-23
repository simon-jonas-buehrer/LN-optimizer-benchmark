"""genetic: learn the wiring of a fixed all-NAND net by mutation hill-climbing.

Every gate is a NAND. NAND is functionally complete, so this search space contains every circuit
the LUT net can express; the only free parameters are which two strictly-earlier signals each gate
reads. No gradients.

    for each generation:
        make k-1 mutants of the current wiring (rewire `mut` gate endpoints at random)
        score all k (the incumbent included) on the same large minibatch by MARGIN
        keep the best

Margin (true-class readout count minus the best wrong class) is the fitness rather than accuracy:
minibatch accuracy only moves when a prediction flips, so almost every single-wire mutation scores
the same and the search random-walks; the margin moves whenever any vote moves, turning the plateau
into a slope. The selection batch must be big or one rewired wire drowns in sampling noise.

Fitness and validation come straight from the packed simulator that mirrors the emitted netlist, so
predict()==emit by construction. Each point trains until validation loss stops improving; `gens` is
a ceiling, not a target.

Speed. The search itself is unchanged (same draws, same order, same fitness, same early stop); only
the evaluation is. `_Sim` below is a specialised, bit-identical twin of methods.lut.lut_sim:

  * the minibatch is thermometer-encoded and bit-packed ONCE per generation instead of once per
    candidate (k times), and the encoder transposes per chunk so the pack is cache-resident;
  * every gate is a NAND, so the 4-mask LUT expression collapses to one `~(A & B)` written straight
    into the signal buffer -- no per-layer mask arrays, no 13 temporaries;
  * the readout popcount accumulates in uint8/uint16 blocks instead of the default int64;
  * mutants are evaluated INCREMENTALLY. A mutant differs from the incumbent in `mut` gate
    endpoints, so only those gates and their downstream cone can change value. The incumbent's
    signal buffer is kept, the cone is found by marking changed signals and gathering the marks
    through each later layer, the few changed gates are recomputed in place (and restored
    afterwards), and the class counts are patched by the difference of the changed readout bits.
    Same integers, a small fraction of the work: the incumbent costs a full simulation per
    generation, the other k-1 candidates cost their cone. `_TorchSim` is the same scheme on CUDA,
    wiring resident on the device; it is used only from `_GPU_MIN_GATES` gates up, where the
    incumbent's own full pass has grown big enough to dominate.
"""

from __future__ import annotations

import time

import numpy as np

from data import Dataset, DatasetSpec
from hw import even_thresholds
from methods.lut import LutModel, load as lut_load, pack_encoded

TITLE = "genetic (learned wiring, all gates NAND)"

NAND_TT = 0b0111  # bit (2a+b) of ~(a & b): f(0,0)=f(0,1)=f(1,0)=1, f(1,1)=0

# >=5 size points, ~1k -> ~20M gates by pre-optimisation gate count (~sum of layer widths). xs..l are
# the original record's shapes; xl/xxl extend the curve upward. The hill-climber weakens as the net
# grows (a random rewiring moves the margin less), so the big tiers are expected to plateau -- the
# flattening is the result. Every readout width (widths[-1]) is divisible by 10.
_LADDER = {
    "xs": (1, (256, 256, 160)),
    "s": (1, (1024, 1024, 320)),
    "m": (3, (2048, 2048, 2048, 640)),
    "l": (3, (4096, 4096, 4096, 4096, 1280)),
    # xl and xxl are REMOVED -- they cannot finish, and the limit is early stopping, not the wall
    # clock alone. Stopping needs `eval_every` * `patience` = 5000 * 20 = 100,000 generations with no
    # improvement, so 100,000 generations is a FLOOR on runtime, not an average. Measured on real
    # data: 2.66 s/generation at xl, i.e. a floor of 73.9 h against a 48 h job -- unreachable however
    # training goes -- and xxl is 20x larger again. The jump from l (17,664 nodes) to the old xl
    # (1,025,000 nodes) was also 58x, far past where a mutation-and-select search over 2-input LUTs
    # does anything useful.
}

_EMPTY = np.zeros(0, np.int64)
_SIM_MEM = 1 << 30          # signal-buffer budget per chunk (bytes); caps the images per chunk
_MAX_WORDS = 16             # 1024 images per chunk: the sweet spot for the packed gate loop
_L2 = 1 << 19               # gate-loop row block: keep a layer's A/B tiles roughly L2-resident
_SIM_MEM_GPU = 1 << 31      # device signal-buffer budget per chunk (bytes)
_MAX_WORDS_GPU = 64         # 4096 images per chunk: a GPU wants fewer, bigger kernels
# Measured on an RTX 3090, k=8 batch=16384 (cuda s/gen vs numpy s/gen, wiring identical):
#   155k gates 1.03x (break-even) | 1.03M 12.4x | 4.1M 18.6x. Below break-even the cone evaluator
#   has already removed the work a GPU would parallelise, so the device only adds launch latency.
_GPU_MIN_GATES = 200_000    # just above the measured break-even


# GENERATION CAP -- the backstop for a point that never converges, not the stopping rule. The rule
# is early stopping (eval_every * patience = 100,000 generations with no new best); the cap only
# ever bites on a point that keeps finding marginal improvements forever.
#
# It has to fit the 48 h wall clock, because a point has NO mid-point checkpoint: a run cut off by
# the wall clock restarts at generation 0 in the continuation, so any point that needs more than one
# window never finishes at all -- it just burns a GPU forever. Measured gen/s on an RTX 2080 Ti node
# (2026-08-21): CIFAR10 xs 11, s 11, m 9; MNIST xs 33, s 32, m 24. `l` never got to run there, so its
# ratio to `m` was measured separately on CPU -- every tier here is under `_GPU_MIN_GATES`, so all of
# them use the numpy cone evaluator, whose cost is nearly flat in net size: 4.03 gen/s at m vs 3.63
# at l, i.e. l ~= 0.90 * m, putting CIFAR10 l at ~8 gen/s.
#
# At the old 2,000,000 that made CIFAR10 m 61.7 h and CIFAR10 l 69.4 h -- both unreachable, and both
# observed looping. At 1,000,000 the worst point is CIFAR10 l at ~34.7 h, with CIFAR10 m at 30.9 h
# and every MNIST point under 13 h. Nothing that actually converges is touched: the longest observed
# convergence is CIFAR10 xs at 210,000 generations (MNIST xs 180,000; both s tiers 120,000), so the
# cap still sits ~5x above the point where early stopping takes over.
_GENS = 1_000_000


def points(spec: DatasetSpec) -> list[dict]:
    return [{"name": n, "bits": b, "widths": w, "gens": _GENS} for n, (b, w) in _LADDER.items()]


# ================================================================================================
# Packed simulator (bit-identical to methods.lut.lut_sim for an all-NAND net)
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
    accumulated in uint8 blocks of 255 rows (a 0/1 sum of 255 rows cannot overflow a uint8) into a
    uint16/uint32 accumulator instead of numpy's default int64 -- ~4x less work in the readout.
    """
    r = np.unpackbits(bits.view(np.uint8), axis=1, bitorder="little")[:, :n]
    r = r.reshape(n_classes, g, n)
    acc = np.zeros((n_classes, n), np.uint16 if g <= 0xFFFF else np.uint32)
    for i in range(0, g, 255):
        blk = r[:, i:i + 255]
        acc += (blk[:, 0] if blk.shape[1] == 1 else blk.sum(1, dtype=np.uint8)).astype(acc.dtype)
    return acc.T.astype(np.int64)


class _Sim:
    """All-NAND packed simulator with incremental re-evaluation of single-endpoint mutants."""

    def __init__(self, widths, offs, spec: DatasetSpec):
        self.widths = list(widths)
        self.offs = list(offs)                 # offs[l] = first signal id of layer l; offs[-1]=n_sig
        self.n_in = offs[0]
        self.n_sig = offs[-1]
        self.n_layers = len(widths)
        self.n_classes = spec.n_classes
        self.g = widths[-1] // spec.n_classes
        self.last = self.n_layers - 1
        self.last_off = offs[self.last]
        self.words = max(1, min(_MAX_WORDS, int(_SIM_MEM // (max(1, self.n_sig) * 8))))
        self.chunk = self.words * 64
        self.mark = np.zeros(self.n_sig, np.uint8)   # scratch, always all-zero between calls
        self._bufs: dict[int, np.ndarray] = {}
        self._tiles: dict[int, list] = {}

    def pack(self, pix: np.ndarray, thresholds, spec: DatasetSpec) -> list:
        return _pack(pix, thresholds, spec, self.chunk)

    def _buf(self, words: int) -> np.ndarray:
        S = self._bufs.get(words)
        if S is None:
            S = self._bufs[words] = np.zeros((self.n_sig, words), np.int64)
            blk = max(256, min(8192, _L2 // (words * 8)))
            self._tiles[words] = [blk] + [np.empty((blk, words), np.int64) for _ in range(2)]
        return S

    def _run(self, S: np.ndarray, srcs) -> None:
        """Forward the whole net in place: every gate is `~(a & b)`, in cache-resident row tiles."""
        offs = self.offs
        blk, bA, bB = self._tiles[S.shape[1]]
        for l in range(self.n_layers):
            ia, ib = srcs[l]
            o, w = offs[l], offs[l + 1] - offs[l]
            for i in range(0, w, blk):
                j = min(i + blk, w)
                A = np.take(S, ia[i:j], axis=0, out=bA[:j - i])
                B = np.take(S, ib[i:j], axis=0, out=bB[:j - i])
                np.bitwise_and(A, B, out=A)
                np.invert(A, out=S[o + i:o + j])

    def _readout(self, S: np.ndarray, n: int) -> np.ndarray:
        return _popcount_rows(S[self.last_off:self.offs[-1]], n, self.n_classes, self.g)

    def counts(self, srcs, packed) -> np.ndarray:
        """Full simulation of one wiring -> (N, n_classes) counts (== lut_sim)."""
        out, pos = np.empty((sum(n for n, _ in packed), self.n_classes), np.int64), 0
        for n, enc in packed:
            S = self._buf(enc.shape[1])
            S[:self.n_in] = enc
            self._run(S, srcs)
            out[pos:pos + n] = self._readout(S, n)
            pos += n
        return out

    def generation(self, srcs, packed, patch_list) -> list:
        """Counts for the incumbent and for each mutant patch list, on the same packed batch.

        `srcs` is left exactly as it was found. Each mutant is scored incrementally against the
        incumbent's signal buffer, which is restored bit for bit afterwards.
        """
        N = sum(n for n, _ in packed)
        out = [np.empty((N, self.n_classes), np.int64) for _ in range(1 + len(patch_list))]
        pos = 0
        for n, enc in packed:
            S = self._buf(enc.shape[1])
            S[:self.n_in] = enc
            self._run(S, srcs)
            base = self._readout(S, n)
            out[0][pos:pos + n] = base
            for j, patches in enumerate(patch_list):
                undo = _apply(srcs, patches)
                out[1 + j][pos:pos + n] = self._mutant(S, srcs, base, patches, n)
                _undo(srcs, undo)
            pos += n
        return out

    def _mutant(self, S, srcs, base, patches, n) -> np.ndarray:
        """Counts of the patched wiring: recompute only the changed gates' downstream cone."""
        expl: dict[int, list] = {}
        for l, _, col, _ in patches:
            expl.setdefault(l, []).append(col)
        if not expl:                     # mut=0: the "mutant" is the incumbent
            return base
        mark, offs = self.mark, self.offs
        saved, changed = [], False
        counts = base
        for l in range(min(expl), self.n_layers):
            ia, ib = srcs[l]
            cols = np.flatnonzero(mark[ia] | mark[ib]) if changed else _EMPTY
            if l in expl:
                e = np.unique(np.asarray(expl[l], np.int64))
                cols = np.union1d(cols, e) if cols.size else e
            if cols.size == 0:
                continue
            new = np.bitwise_and(S[ia[cols]], S[ib[cols]])
            np.invert(new, out=new)
            gid = cols + offs[l]
            old = S[gid]
            d = np.flatnonzero((new != old).any(1))
            if d.size == 0:
                continue
            kg, knew, kold = gid[d], new[d], old[d]
            saved.append((kg, kold))
            S[kg] = knew
            mark[kg] = 1
            changed = True
            if l == self.last:
                counts = self._patch_counts(base, kg - self.last_off, knew, kold, n)
        for kg, kold in saved:
            S[kg] = kold
            mark[kg] = 0
        return counts

    def _patch_counts(self, base, rows, new, old, n) -> np.ndarray:
        """base counts + (new bits - old bits) of the changed readout gates, per class."""
        up_new = np.unpackbits(new.view(np.uint8), axis=1, bitorder="little")[:, :n]
        up_old = np.unpackbits(old.view(np.uint8), axis=1, bitorder="little")[:, :n]
        d = up_new.astype(np.int16) - up_old.astype(np.int16)
        counts = base.copy()
        cls = rows // self.g
        for i in range(len(rows)):
            counts[:, cls[i]] += d[i]
        return counts


class _TorchSim:
    """CUDA twin of `_Sim`: the same all-NAND packed simulation AND the same incremental cone
    update, with the wiring resident on the device for the whole run.

    numpy's uint64 words and torch's int64 words share the exact two's-complement layout, so
    `&`/`~`/`>>` produce identical bits; this is checked against `_Sim` and against `lut_sim` (on
    the torch CPU backend and on CUDA) before it is allowed to drive a search. Only the per-chunk
    packed encoding goes up the bus, and only the (n, n_classes) counts come back.
    """

    def __init__(self, widths, offs, spec: DatasetSpec, device):
        import torch

        import batching   # imported here, not at module scope: this file is numpy-only until CUDA

        self.torch = torch
        self.dev = torch.device(device)
        self.widths = list(widths)
        self.offs = list(offs)
        self.n_in, self.n_sig = offs[0], offs[-1]
        self.n_layers = len(widths)
        self.n_classes = spec.n_classes
        self.g = widths[-1] // spec.n_classes
        self.last = self.n_layers - 1
        self.last_off = offs[self.last]
        # _SIM_MEM_GPU is a ceiling tuned on a 24 GB card; on a smaller one take what is actually
        # free (see `batching.budget`), so a wide net uses more, narrower chunks instead of OOMing.
        self.words = max(1, min(_MAX_WORDS_GPU,
                                int(batching.budget(_SIM_MEM_GPU, device=self.dev)
                                    // (max(1, self.n_sig) * 8))))
        self.chunk = self.words * 64
        self.mark = torch.zeros(self.n_sig, dtype=torch.uint8, device=self.dev)
        self.sh = torch.arange(64, dtype=torch.int64, device=self.dev)
        self._empty = torch.zeros(0, dtype=torch.int64, device=self.dev)
        self._bufs: dict[int, object] = {}
        self._w = None          # device wiring: per layer a (2, w) int64 tensor

    def pack(self, pix: np.ndarray, thresholds, spec: DatasetSpec) -> list:
        t = self.torch
        return [(n, t.as_tensor(enc, device=self.dev))
                for n, enc in _pack(pix, thresholds, spec, self.chunk)]

    def wire(self, srcs):
        """Upload the wiring once; `commit` keeps it in step with the numpy copy afterwards."""
        if self._w is None:
            t = self.torch
            self._w = [t.as_tensor(np.ascontiguousarray(s, np.int64), device=self.dev)
                       for s in srcs]
        return self._w

    def commit(self, patches):
        for l, row, col, val in patches:
            self._w[l][row, col] = val

    def _buf(self, words: int):
        S = self._bufs.get(words)
        if S is None:
            t = self.torch
            S = self._bufs[words] = t.zeros((self.n_sig, words), dtype=t.int64, device=self.dev)
        return S

    def _run(self, S, w) -> None:
        t = self.torch
        for l in range(self.n_layers):
            ia, ib = w[l][0], w[l][1]
            dst = S[self.offs[l]:self.offs[l + 1]]
            t.bitwise_and(S[ia], S[ib], out=dst)
            t.bitwise_not(dst, out=dst)

    def _readout(self, S, n):
        """(n, n_classes) int32 counts, on the device (the popcount never leaves the GPU)."""
        t = self.torch
        last = S[self.last_off:self.offs[-1]]
        words = last.shape[1]
        v = last.view(self.n_classes, self.g, words)
        acc = t.zeros((self.n_classes, words * 64), dtype=t.int32, device=self.dev)
        blk = max(1, min(4096, (1 << 24) // max(1, words * 64 * 8)))
        for i in range(0, self.g, blk):
            b = v[:, i:i + blk]
            acc += (((b.unsqueeze(-1) >> self.sh) & 1)
                    .sum(1, dtype=t.int32).reshape(self.n_classes, -1))
        return acc[:, :n].t()

    def counts(self, srcs, packed) -> np.ndarray:
        t = self.torch
        w = self.wire(srcs)
        out, pos = np.empty((sum(n for n, _ in packed), self.n_classes), np.int64), 0
        for n, enc in packed:
            S = self._buf(enc.shape[1])
            S[:self.n_in] = enc
            self._run(S, w)
            out[pos:pos + n] = self._readout(S, n).to(t.int64).cpu().numpy()
            pos += n
        return out

    def generation(self, srcs, packed, patch_list) -> list:
        t = self.torch
        w = self.wire(srcs)
        N = sum(n for n, _ in packed)
        out = [np.empty((N, self.n_classes), np.int64) for _ in range(1 + len(patch_list))]
        pos = 0
        for n, enc in packed:
            S = self._buf(enc.shape[1])
            S[:self.n_in] = enc
            self._run(S, w)
            base = self._readout(S, n)
            out[0][pos:pos + n] = base.to(t.int64).cpu().numpy()
            for j, patches in enumerate(patch_list):
                undo = self._patch_wire(w, patches)
                c = self._mutant(S, w, base, patches, n)
                out[1 + j][pos:pos + n] = c.to(t.int64).cpu().numpy()
                for l, row, col, val in reversed(undo):
                    w[l][row, col] = val
            pos += n
        return out

    def _patch_wire(self, w, patches):
        undo = []
        for l, row, col, val in patches:
            undo.append((l, row, col, int(w[l][row, col])))
            w[l][row, col] = val
        return undo

    def _mutant(self, S, w, base, patches, n):
        t = self.torch
        expl: dict[int, list] = {}
        for l, _, col, _ in patches:
            expl.setdefault(l, []).append(col)
        if not expl:
            return base
        mark, offs = self.mark, self.offs
        saved, changed = [], False
        counts = base
        for l in range(min(expl), self.n_layers):
            ia, ib = w[l][0], w[l][1]
            cols = t.nonzero(mark[ia] | mark[ib]).squeeze(1) if changed else self._empty
            if l in expl:
                e = t.as_tensor(sorted(set(expl[l])), dtype=t.int64, device=self.dev)
                cols = t.unique(t.cat([cols, e])) if cols.numel() else e
            if cols.numel() == 0:
                continue
            new = t.bitwise_and(S[ia[cols]], S[ib[cols]])
            t.bitwise_not(new, out=new)
            gid = cols + offs[l]
            old = S[gid]
            d = t.nonzero((new != old).any(1)).squeeze(1)
            if d.numel() == 0:
                continue
            kg, knew, kold = gid[d], new[d], old[d]
            saved.append((kg, kold))
            S[kg] = knew
            mark[kg] = 1
            changed = True
            if l == self.last:
                counts = self._patch_counts(base, kg - self.last_off, knew, kold, n)
        for kg, kold in saved:
            S[kg] = kold
            mark[kg] = 0
        return counts

    def _patch_counts(self, base, rows, new, old, n):
        t = self.torch
        bn = ((new.unsqueeze(-1) >> self.sh) & 1).reshape(new.shape[0], -1)[:, :n]
        bo = ((old.unsqueeze(-1) >> self.sh) & 1).reshape(old.shape[0], -1)[:, :n]
        d = (bn - bo).to(t.int32).t().contiguous()          # (n, R)
        counts = base.clone()
        counts.index_add_(1, (rows // self.g).to(t.int64), d)
        return counts


def _apply(srcs, patches):
    undo = []
    for l, row, col, val in patches:
        undo.append((l, row, col, int(srcs[l][row, col])))
        srcs[l][row, col] = val
    return undo


def _undo(srcs, undo):
    for l, row, col, val in reversed(undo):
        srcs[l][row, col] = val


# ================================================================================================
# Search
# ================================================================================================

def _mutate(srcs, widths, offs, mut, rng):
    """Rewire `mut` random gate endpoints. Layers chosen with P(l) ~ width.

    Returns the change list [(layer, row, col, new_src)] rather than a whole copy of the wiring
    (which is 2 int64 per gate -- hundreds of MB on the big tiers). The draws are identical to
    `m[l][rng.integers(2), rng.integers(widths[l])] = rng.integers(offs[l])`: python evaluates the
    right-hand side of an assignment first, so the source id is drawn BEFORE the row and the column.
    """
    patches = []
    w = np.asarray(widths, float)
    for l in rng.choice(len(widths), size=mut, p=w / w.sum()):
        val = int(rng.integers(offs[l]))
        row = int(rng.integers(2))
        col = int(rng.integers(widths[l]))
        patches.append((int(l), row, col, val))
    return patches


def _margin(counts, y):
    """Readout count for the true class minus the best other class, averaged. A slope, not a cliff."""
    idx = np.arange(len(y))
    true = counts[idx, y]
    other = counts.copy()
    other[idx, y] = -1  # counts are >= 0, so this excludes the true class from the max
    return float((true - other.max(1)).mean())


def _val_ce(counts, y):
    """Cross-entropy of a softmax over the integer readout counts (used as logits)."""
    z = counts.astype(float)
    z -= z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


class Genetic(LutModel):
    def __init__(self, spec, bits, widths, gens, k=8, mut=1, batch=16384,
                 eval_every=5000, patience=20):
        super().__init__(spec)
        if widths[-1] % spec.n_classes:
            raise ValueError(f"readout {widths[-1]} not divisible by {spec.n_classes}")
        self.cfg = dict(bits=bits, widths=tuple(widths), gens=gens, k=k, mut=mut,
                        batch=batch, eval_every=eval_every, patience=patience)

    def train(self, data: Dataset, *, device="cpu", seed=0):
        c, spec = self.cfg, self.spec
        rng = np.random.default_rng(seed)
        widths = c["widths"]
        thresholds = even_thresholds(c["bits"])
        # signal-id layout: encoder owns 0..n_in-1, then layer 0's gates, then layer 1's, ...
        offs = [spec.n_pixels * c["bits"]]
        for w in widths:
            offs.append(offs[-1] + w)
        # a gate reads any strictly-earlier signal -> the graph is acyclic by construction
        srcs = [rng.integers(0, offs[l], size=(2, w), dtype=np.int64) for l, w in enumerate(widths)]

        tt = [np.full(w, NAND_TT, np.int64) for w in widths]  # constant: only the wiring is learned
        layers = lambda g: [(gi[0], gi[1], ti) for gi, ti in zip(g, tt)]
        # Route by net size. The incremental evaluator costs one cone per mutant instead of a
        # full simulation, which is why the CPU beats a per-candidate GPU re-simulation on the
        # small tiers; the GPU only pays once the incumbent's own full pass dominates. Both
        # backends run the identical bit arithmetic, so this picks speed only.
        gpu = device is not None and str(device) != "cpu" and sum(widths) >= _GPU_MIN_GATES
        sim = (_TorchSim(widths, offs, spec, device) if gpu
               else _Sim(widths, offs, spec))   # exact twins of lut_sim, NAND-specialised
        val_packed = None                       # the val set is encoded+packed once, not per eval

        best_ce, best_srcs, stale = float("inf"), [s.copy() for s in srcs], 0
        t0, train_secs, nseen = time.time(), 0.0, 0  # train_secs: only the search, not the val evals
        for gen in range(c["gens"]):
            ts = time.perf_counter()
            idx = rng.integers(0, len(data.train_x), size=c["batch"])
            bx, by = data.train_x[idx], data.train_y[idx]
            packed = sim.pack(bx, thresholds, spec)
            patch_list = [_mutate(srcs, widths, offs, c["mut"], rng) for _ in range(c["k"] - 1)]
            cnts = sim.generation(srcs, packed, patch_list)
            best_fit = _margin(cnts[0], by)  # incumbent, scored on this batch
            winner = None
            for j in range(c["k"] - 1):
                fit = _margin(cnts[1 + j], by)
                if fit > best_fit:  # strictly better, so the incumbent survives ties
                    best_fit, winner = fit, patch_list[j]
            if winner is not None:
                _apply(srcs, winner)
                if gpu:
                    sim.commit(winner)          # keep the device wiring in step with `srcs`
            train_secs += time.perf_counter() - ts
            nseen += c["k"] * c["batch"]  # k candidates each scored on `batch` samples this gen

            if (gen + 1) % c["eval_every"] == 0 or gen + 1 == c["gens"]:
                if val_packed is None:
                    val_packed = sim.pack(data.val_x, thresholds, spec)
                ce = _val_ce(sim.counts(srcs, val_packed), data.val_y)
                if ce < best_ce - 1e-4:
                    best_ce, best_srcs, stale = ce, [s.copy() for s in srcs], 0
                else:
                    stale += 1
                print(f"  gen {gen + 1:7d}/{c['gens']}  margin {best_fit:+.3f}  val ce {ce:.4f}  "
                      f"(best {best_ce:.4f}, stale {stale})  "
                      f"{(gen + 1) / (time.time() - t0):.0f} gen/s", flush=True)
                if stale >= c["patience"]:  # converged: no new best in patience*eval_every gens
                    print(f"  early stop at gen {gen + 1}: converged (best ce {best_ce:.4f})",
                          flush=True)
                    break

        self.thresholds = thresholds
        self.layers = layers(best_srcs)
        self.train_seconds = train_secs
        self.train_samples = nseen  # training-example evals until early stop


def build(spec, **point) -> Genetic:
    return Genetic(spec, **point)


# The synthesis phase reloads the trained circuit from its .ckpt; (thresholds, layers) is all of it,
# so the shared LUT-net loader covers this method without touching the trainer above.
load = lut_load
