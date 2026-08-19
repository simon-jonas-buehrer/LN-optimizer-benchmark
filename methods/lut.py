"""Shared fan-in-2 LUT-net scaffolding for the logic-net methods (backprop, genetic, dfa, es).

A trained logic net is fully described by `(thresholds, layers)` where each layer is
`(idx_a, idx_b, tt)` (int arrays): gate i reads signals idx_a[i], idx_b[i] and applies the 4-bit
truth table tt[i]. `LutModel` turns that into emit_verilog / predict / scores / save, and `lut_sim`
is a packed numpy simulator that mirrors `hw.emit_lutnet` exactly -- so a method only has to produce
the hard `(thresholds, layers)` and everything else (and bit-exactness vs the synthesized netlist)
comes for free. Optimizers differ only in how they learn those arrays.
"""

from __future__ import annotations

import os

import numpy as np

import hw
from data import Dataset, DatasetSpec, to_bits

FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
_ZERO = np.uint64(0)
_READ_BLOCK = 1024   # gates per unpackbits block in the readout (keeps the working set in cache)
_TILE_BYTES = 1 << 19   # per-array working set of one gate-loop tile (~L2); see _tile
_TORCH_READ_BYTES = 1 << 26   # bit-expansion block of the on-device readout


def _tile(words: int) -> int:
    """Gates per tile of the gate loop: enough that A, B and the destination rows all stay in cache
    on the wide layers, big enough that the per-tile numpy overhead stays negligible."""
    return max(256, min(8192, _TILE_BYTES // (words * 8)))


def _gate_loop(S: np.ndarray, idx, offs, widths, consts, tile: int, bufA, bufB) -> None:
    """The whole gate net, in place in `S`. Row-tiled so the two gathered operands and the
    destination rows stay in cache on the wide layers, and gathered with `np.take(..., out=)` into
    two buffers that are allocated once per call rather than once per tile."""
    for (ia, ib), o, w, (m0, d0, e, gg) in zip(idx, offs, widths, consts):
        for j in range(0, w, tile):
            k = j + tile if j + tile < w else w
            A, B = bufA[:k - j], bufB[:k - j]
            np.take(S, ia[j:k], axis=0, out=A)
            np.take(S, ib[j:k], axis=0, out=B)
            T = S[o + j:o + k]
            np.bitwise_and(B, gg[j:k], out=T)
            np.bitwise_xor(T, e[j:k], out=T)
            np.bitwise_and(T, A, out=T)
            np.bitwise_and(B, d0[j:k], out=B)
            np.bitwise_xor(T, B, out=T)
            np.bitwise_xor(T, m0[j:k], out=T)


def _thr(thresholds):
    """(t_int16, t_uint8_or_None): the uint8 twin exists only when every threshold is a legal
    uint8 comparison bound (0..254), which is what hw.emit_thermometer allows. Comparing uint8
    against uint8 avoids widening the whole image batch to int16."""
    t = np.asarray(thresholds, np.int16)
    fast = t.astype(np.uint8) if t.size and t.min() >= 0 and t.max() <= 254 else None
    return t, fast


def encode_bits(pix: np.ndarray, thresholds, spec: DatasetSpec) -> np.ndarray:
    """(N, n_pixels) uint8 -> (N, n_pixels*k) uint8, byte-major bit p*k+j = pix[p] > thresholds[j].
    Matches hw.emit_thermometer."""
    t, fast = _thr(thresholds)
    b = (pix[:, :, None] > fast) if fast is not None else (pix[:, :, None].astype(np.int16) > t)
    return b.reshape(len(pix), -1).view(np.uint8)


def pack_encoded(pix: np.ndarray, thresholds, spec: DatasetSpec, words: int | None = None
                 ) -> np.ndarray:
    """(n, n_pixels) uint8 -> (n_pixels*k, words) uint64: the thermometer code of `pix`, bit-packed
    64 images per word (bit i of word w is image 64*w+i), zero-padded past image n-1.

    This is exactly the `S[:n_in]` a `lut_sim` chunk starts from. It is a *lot* faster than
    `packbits(encode_bits(...).T)` because the threshold compare writes the transposed layout
    directly instead of byte-transposing an (n, n_in) array afterwards. Callers that evaluate many
    nets on the SAME image batch (es, genetic) can hoist it and feed it to ``lut_sim_packed``.
    """
    n = len(pix)
    t, fast = _thr(thresholds)
    k = t.size
    n_in = spec.n_pixels * k
    pt = np.ascontiguousarray(pix.T)                            # (n_pixels, n)
    if fast is not None:
        b = pt[:, None, :] > fast[None, :, None]                # (n_pixels, k, n) bool
    else:
        b = pt[:, None, :].astype(np.int16) > t[None, :, None]
    b = np.ascontiguousarray(b.reshape(n_in, n)).view(np.uint8)
    packed = np.packbits(b, axis=1, bitorder="little")          # (n_in, ceil(n/8))
    if words is None:
        words = (packed.shape[1] + 7) // 8
    nb = words * 8
    if packed.shape[1] == nb:
        return packed.view(np.uint64)
    out = np.zeros((n_in, nb), np.uint8)
    keep = min(nb, packed.shape[1])
    out[:, :keep] = packed[:, :keep]
    return out.view(np.uint64)


def _masks(tt) -> tuple:
    """The four xor-form gate masks for one layer, as (w, 1) uint64 arrays.

    out = m0 ^ (B & d0) ^ (A & (e ^ (B & gg))) with d0 = m0^m1, e = m0^m2, gg = m0^m1^m2^m3.
    Check the four cases: (0,0)->m0, (0,1)->m1, (1,0)->m2, (1,1)->m3. Six bitwise ops instead of the
    thirteen of the 4-mask sum-of-products form, and every one of them runs with out=.
    """
    tt = np.asarray(tt, np.int64)
    b0, b1, b2, b3 = ((tt >> kk) & 1 for kk in range(4))
    return tuple(np.where(bit, FULL, _ZERO)[:, None]
                 for bit in (b0, b0 ^ b1, b0 ^ b2, b0 ^ b1 ^ b2 ^ b3))


def _plan(thresholds, layers, spec: DatasetSpec):
    """Hoisted, per-net constants: offsets, contiguous int64 wiring, and the 4 gate masks in the
    xor form the kernel uses. Computed once per lut_sim call, not once per image chunk."""
    k = len(thresholds)
    n_in = spec.n_pixels * k
    widths = [len(a) for a, _, _ in layers]
    n_sig = n_in + sum(widths)
    offs, off = [], n_in
    for w in widths:
        offs.append(off)
        off += w
    last_off, last_w = offs[-1], widths[-1]
    if last_w % spec.n_classes:
        raise ValueError(f"readout width {last_w} not divisible by {spec.n_classes}")
    idx = [(np.ascontiguousarray(a, np.int64), np.ascontiguousarray(b, np.int64))
           for a, b, _ in layers]
    consts = [_masks(tt) for _, _, tt in layers]
    return n_in, widths, n_sig, offs, last_off, last_w, idx, consts


def _readout(last: np.ndarray, n_classes: int, g: int, n: int, dst: np.ndarray) -> None:
    """last: (n_classes*g, words) uint64 readout bits -> dst: (n, n_classes) int64 popcounts.

    Same numbers as `unpackbits(last)[:, :n].reshape(n_classes, g, n).sum(1).T`, but it never
    materialises the (n_classes*g, 64*words) uint8 expansion: it unpacks `_READ_BLOCK` gates at a
    time and folds each block into a uint32 accumulator, so the working set stays in cache.
    """
    u8 = last.view(np.uint8)
    cols = u8.shape[1] * 8
    if last.shape[0] <= _READ_BLOCK:      # tiny readout: one shot beats n_classes blocked loops
        read = np.unpackbits(u8, axis=1, bitorder="little")
        dst[:] = read.reshape(n_classes, g, cols).sum(1, dtype=np.uint32)[:, :n].T
        return
    acc = np.empty(cols, np.uint32)
    for c in range(n_classes):
        rows = u8[c * g:(c + 1) * g]
        acc[:] = 0
        for j in range(0, g, _READ_BLOCK):
            u = np.unpackbits(rows[j:j + _READ_BLOCK], axis=1, bitorder="little")
            acc += u.sum(0, dtype=np.uint32)
        dst[:, c] = acc[:n]


def lut_sim_packed(packed_chunks, layers, spec: DatasetSpec, thresholds_len: int,
                   out: np.ndarray | None = None) -> np.ndarray:
    """Low-level entry point: run the net over already-packed image chunks.

    ``packed_chunks`` is an iterable of ``(enc, n)`` where ``enc`` is a ``(n_pixels*k, words)``
    uint64 array from ``pack_encoded`` and ``n`` is how many of its ``64*words`` bit slots are real
    images. Returns the (sum of n, n_classes) int64 readout counts. ``lut_sim`` is a thin wrapper
    that produces the chunks from raw pixels; methods that evaluate MANY nets on the SAME batch can
    pack once and call this repeatedly.
    """
    chunks = list(packed_chunks)
    n_in, widths, n_sig, offs, last_off, last_w, idx, consts = _plan(
        range(thresholds_len), layers, spec)
    g = last_w // spec.n_classes
    if out is None:
        out = np.empty((sum(n for _, n in chunks), spec.n_classes), np.int64)
    words = max((e.shape[1] for e, _ in chunks), default=1)
    tile = _tile(words)
    S = np.empty((n_sig, words), np.uint64)
    bufA = np.empty((tile, words), np.uint64)
    bufB = np.empty((tile, words), np.uint64)
    pos = 0
    for enc, n in chunks:
        w_enc = enc.shape[1]
        S[:n_in, :w_enc] = enc
        if w_enc < words:
            S[:n_in, w_enc:] = 0
        _gate_loop(S, idx, offs, widths, consts, tile, bufA, bufB)
        _readout(S[last_off:last_off + last_w], spec.n_classes, g, n, out[pos:pos + n])
        pos += n
    return out


def candidates(n_src: int, width: int, k: int, rng) -> np.ndarray:
    """(width, k) random source ids in [0, n_src): each gate's k wiring candidates."""
    return rng.integers(0, n_src, size=(width, k), dtype=np.int64)


def lut_sim(thresholds, layers, pix: np.ndarray, spec: DatasetSpec, chunk: int = 512,
            device=None) -> np.ndarray:
    """Exact packed simulation of the emitted LUT net -> (N, n_classes) integer readout counts.

    Same semantics as hw.emit_lutnet + hw.emit_popcount_argmax: argmax of these counts is predict();
    counts / group_size is scores(). Bit-packed (64 images/word) and chunked, so it runs on the big
    nets without materialising an (N, gates) array.

    ``device`` selects the backend: ``None``/``"cpu"`` is the numpy reference below; a CUDA device
    (e.g. ``"cuda"``) runs the identical bit-arithmetic on the GPU via ``_lut_sim_torch``. Both return
    the same int64 (N, n_classes) counts -- the gate ops are exact integer bit-twiddling, so the GPU
    path is bit-for-bit equal to numpy (checked in tests), only faster on the big nets. The gradient-
    free searches (es, genetic) spend almost all their time here, so this is their speed-up.
    """
    if device is not None and str(device) != "cpu":
        return _lut_sim_torch(thresholds, layers, pix, spec, device)
    n_in, widths, n_sig, offs, last_off, last_w, idx, consts = _plan(thresholds, layers, spec)
    g = last_w // spec.n_classes
    N = len(pix)
    chunk = max(64, int(chunk))
    words = max(1, (min(chunk, N) + 63) // 64) if N else 1

    tile = _tile(words)
    out = np.empty((N, spec.n_classes), np.int64)
    S = np.empty((n_sig, words), np.uint64)
    bufA = np.empty((tile, words), np.uint64)
    bufB = np.empty((tile, words), np.uint64)
    for i in range(0, N, chunk):
        p = pix[i:i + chunk]
        n = len(p)
        S[:n_in] = pack_encoded(p, thresholds, spec, words)
        _gate_loop(S, idx, offs, widths, consts, tile, bufA, bufB)
        _readout(S[last_off:last_off + last_w], spec.n_classes, g, n, out[i:i + n])
    return out


def _masks_torch(tt, torch, dev):
    """The four xor-form gate masks for one layer, as (w, 1) int64 tensors on `dev`."""
    t = torch.as_tensor(np.asarray(tt, np.int64), device=dev)
    b0, b1, b2, b3 = ((t >> kk) & 1 for kk in range(4))
    # -1 is 0xFFFF...F as int64 == numpy uint64 FULL, the same bits
    return tuple(torch.where(bit == 1, -1, 0).view(-1, 1)
                 for bit in (b0, b0 ^ b1, b0 ^ b2, b0 ^ b1 ^ b2 ^ b3))


def _gate_loop_torch(S, idx_a, idx_b, masks, offs, widths, torch) -> None:
    """The GPU twin of ``_gate_loop``: same six ops, same order, all with out=."""
    for l, (a, b) in enumerate(zip(idx_a, idx_b)):
        m0, d0, e, gg = masks[l]
        A = S[a]
        B = S[b]
        T = S[offs[l]:offs[l] + widths[l]]
        torch.bitwise_and(B, gg, out=T)
        torch.bitwise_xor(T, e, out=T)
        torch.bitwise_and(T, A, out=T)
        torch.bitwise_and(B, d0, out=B)
        torch.bitwise_xor(T, B, out=T)
        torch.bitwise_xor(T, m0, out=T)


def _readout_torch(last, n_classes: int, g: int, n: int, dst: np.ndarray, torch) -> None:
    """On-device readout popcount: (n_classes*g, words) int64 on the GPU -> (n, n_classes) on the host.

    Bit-for-bit the same numbers as ``_readout``: an int64 word is eight little-endian bytes, so
    viewing the readout rows as uint8 and expanding bit ``i`` of byte ``j`` to flat column ``j*8+i``
    reproduces ``np.unpackbits(..., bitorder="little")`` exactly, and the sum over the g gates of a
    class is the same integer sum. What changes is the traffic: only the (n_classes, 64*words) int32
    counts cross PCIe instead of the whole last layer, which at the 20M-gate tiers is a multi-GB
    device-to-host copy per image chunk.
    """
    dev = last.device
    u8 = last.view(torch.uint8)                       # (n_classes*g, words*8), rows are contiguous
    nb = u8.shape[1]
    cols = nb * 8
    sh = torch.arange(8, device=dev, dtype=torch.uint8)
    acc = torch.zeros((n_classes, cols), dtype=torch.int32, device=dev)
    v = u8.view(n_classes, g, nb)
    blk = max(1, min(g, _TORCH_READ_BYTES // max(1, n_classes * cols)))
    for j in range(0, g, blk):
        b = v[:, j:j + blk]                           # (n_classes, bb, nb) uint8
        bits = b.unsqueeze(-1) >> sh                  # (n_classes, bb, nb, 8) uint8
        bits &= 1
        acc += bits.reshape(n_classes, b.shape[1], cols).sum(1, dtype=torch.int32)
    dst[:] = acc[:, :n].to(torch.int64).t().contiguous().cpu().numpy()


def _lut_sim_torch(thresholds, layers, pix: np.ndarray, spec: DatasetSpec, device) -> np.ndarray:
    """GPU twin of ``lut_sim``: the same 64-images-per-word bit simulation, gate loop on the GPU.

    numpy's uint64 words and torch's int64 words share the exact 64-bit two's-complement layout, so
    ``&``/``^``/``|`` produce identical bits; only the (cheap) encode+pack stays in numpy, and the
    readout popcount runs on the device too, so only the counts come back. Chunked so the
    (n_sig, words) signal buffer fits the GPU even for the ~20M-gate tiers.

    Evaluating MANY nets on the same wiring? Use ``PackedNet``, which keeps the index tensors
    resident instead of re-uploading them on every call.
    """
    import torch

    n_classes = spec.n_classes
    n_in, widths, n_sig, offs, last_off, last_w, idx, _ = _plan(thresholds, layers, spec)
    g = last_w // n_classes

    dev = torch.device(device)
    # wiring + per-gate truth-table masks live on the GPU once (reused across image chunks)
    idx_a = [torch.as_tensor(a, device=dev) for a, _ in idx]
    idx_b = [torch.as_tensor(b, device=dev) for _, b in idx]
    masks = [_masks_torch(tt, torch, dev) for _, _, tt in layers]

    # image chunk sized so the signal buffer (n_sig, words) stays within a memory budget
    words_budget = max(1, min(512, int(1.5e9 // (max(1, n_sig) * 8))))
    chunk = words_budget * 64

    N = len(pix)
    words = max(1, (min(chunk, N) + 63) // 64) if N else 1
    out = np.empty((N, n_classes), np.int64)
    S = torch.empty((n_sig, words), dtype=torch.int64, device=dev)
    for i in range(0, N, chunk):
        p = pix[i:i + chunk]
        n = len(p)
        enc = pack_encoded(p, thresholds, spec, words)
        S[:n_in] = torch.as_tensor(enc.view(np.int64), device=dev)
        _gate_loop_torch(S, idx_a, idx_b, masks, offs, widths, torch)
        _readout_torch(S[last_off:last_off + last_w], n_classes, g, n, out[i:i + n], torch)
    return out


class PackedNet:
    """A net whose WIRING is resident, so only the truth tables move between evaluations.

    Built once from ``layers`` (only the shapes of its ``tt`` entries are read); ``counts(tt, ...)``
    then re-runs that wiring with a new set of per-gate truth tables. This is exactly the ES / GA
    inner loop -- one fixed random wiring, `pop` candidate truth-table vectors per generation, all
    evaluated on the same packed image batch -- where ``lut_sim`` would re-upload the two int64 index
    arrays (2 * sum(widths) words) and re-allocate the signal buffer on every single call.

        net = PackedNet(layers, spec, len(thresholds), device="cuda")
        chunks = [(pack_encoded(x[i:i+c], thr, spec), len(x[i:i+c])) for i in ...]
        for cand in population:
            counts = net.counts(cand_tt, chunks)      # (N, n_classes) int64, == lut_sim

    ``device=None``/``"cpu"`` gives the numpy backend, so the same object works on both.
    """

    def __init__(self, layers, spec: DatasetSpec, thresholds_len: int, device=None):
        (self.n_in, self.widths, self.n_sig, self.offs, self.last_off, self.last_w,
         idx, _) = _plan(range(thresholds_len), layers, spec)
        self.spec = spec
        self.g = self.last_w // spec.n_classes
        self.total = sum(self.widths)
        self._cuda = device is not None and str(device) != "cpu"
        self._S = None
        if self._cuda:
            import torch
            self._torch = torch
            self._dev = torch.device(device)
            self._ia = [torch.as_tensor(a, device=self._dev) for a, _ in idx]
            self._ib = [torch.as_tensor(b, device=self._dev) for _, b in idx]
        else:
            self._ia = [a for a, _ in idx]
            self._ib = [b for _, b in idx]
        self._idx = list(zip(self._ia, self._ib))

    def _split(self, tt):
        """Accept one flat (sum(widths),) array of truth tables, or a per-layer sequence."""
        if isinstance(tt, np.ndarray) and tt.ndim == 1 and tt.size == self.total:
            out, o = [], 0
            for w in self.widths:
                out.append(tt[o:o + w])
                o += w
            return out
        parts = list(tt)
        if len(parts) == len(self.widths) and all(len(p) == w for p, w in zip(parts, self.widths)):
            return parts
        flat = np.asarray(tt, np.int64).reshape(-1)
        if flat.size != self.total:
            raise ValueError(f"expected {self.total} truth tables, got {flat.size}")
        out, o = [], 0
        for w in self.widths:
            out.append(flat[o:o + w])
            o += w
        return out

    def counts(self, tt, packed_chunks, out: np.ndarray | None = None) -> np.ndarray:
        """(N, n_classes) int64 readout counts, identical to ``lut_sim`` on the same net.

        ``packed_chunks`` is an iterable of ``(enc, n)`` (or ``(n, enc)``) where ``enc`` comes from
        ``pack_encoded``; ``tt`` is the per-gate truth tables, flat or per layer.
        """
        chunks = [(c[1], c[0]) if isinstance(c[0], (int, np.integer)) else (c[0], c[1])
                  for c in packed_chunks]
        n_classes = self.spec.n_classes
        if out is None:
            out = np.empty((sum(n for _, n in chunks), n_classes), np.int64)
        words = max((e.shape[1] for e, _ in chunks), default=1)
        tts = self._split(tt)
        pos = 0
        if self._cuda:
            torch = self._torch
            masks = [_masks_torch(t, torch, self._dev) for t in tts]
            if self._S is None or self._S.shape[1] != words:
                self._S = torch.empty((self.n_sig, words), dtype=torch.int64, device=self._dev)
            S = self._S
            for enc, n in chunks:
                w_enc = enc.shape[1]
                S[:self.n_in, :w_enc] = torch.as_tensor(
                    np.ascontiguousarray(enc).view(np.int64), device=self._dev)
                if w_enc < words:
                    S[:self.n_in, w_enc:] = 0
                _gate_loop_torch(S, self._ia, self._ib, masks, self.offs, self.widths, torch)
                _readout_torch(S[self.last_off:self.last_off + self.last_w], n_classes, self.g,
                               n, out[pos:pos + n], torch)
                pos += n
            return out
        consts = [_masks(t) for t in tts]
        tile = _tile(words)
        if self._S is None or self._S.shape[1] != words:
            self._S = np.empty((self.n_sig, words), np.uint64)
            self._bufA = np.empty((tile, words), np.uint64)
            self._bufB = np.empty((tile, words), np.uint64)
        S = self._S
        for enc, n in chunks:
            w_enc = enc.shape[1]
            S[:self.n_in, :w_enc] = enc.view(np.uint64)
            if w_enc < words:
                S[:self.n_in, w_enc:] = 0
            _gate_loop(S, self._idx, self.offs, self.widths, consts, tile, self._bufA, self._bufB)
            _readout(S[self.last_off:self.last_off + self.last_w], n_classes, self.g, n,
                     out[pos:pos + n])
            pos += n
        return out


class LutModel:
    """emit / predict / scores / save for any method that produces hard (thresholds, layers)."""

    # A method sets this during train() (``self.sim_device = "cuda"``) to have EVALUATION run on the
    # same device it trained on. None keeps the numpy simulator, so a CPU run is untouched. dfa's
    # older private ``_sim_device`` is honoured too, so that override can simply be deleted.
    sim_device = None
    # ...but only once the net is big enough to be worth the kernel launches. Measured crossover on
    # an RTX 3090 vs numpy at 16 threads: ~1k gates 0.63x (a LOSS), ~6k gates 1.1-1.3x, 39k 4-5x,
    # 280k 7-11x, 1.75M 8-15x. 4096 sits inside the flat break-even region, so a wrong guess either
    # way costs ~nothing. Override per method, or with MNISTBENCH_LUTSIM_GPU_MIN_GATES.
    gpu_min_gates = int(os.environ.get("MNISTBENCH_LUTSIM_GPU_MIN_GATES", "4096"))

    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self.thresholds: list[int] = []
        self.layers: list = []          # list of (idx_a, idx_b, tt) int arrays
        self._counts_memo = None        # (pix, thresholds, layers, counts) of the last _counts call

    def emit_verilog(self) -> str:
        return hw.emit_lutnet(self.thresholds, self.layers, self.spec)

    def _eval_device(self):
        """The device `_counts` should simulate on, or None for numpy."""
        dev = getattr(self, "sim_device", None) or getattr(self, "_sim_device", None)
        if dev is None or str(dev) == "cpu":
            return None
        if sum(len(a) for a, _, _ in self.layers) < self.gpu_min_gates:
            return None                 # too small to amortise the launches; numpy wins
        try:
            import torch
            if not torch.cuda.is_available():
                return None             # trained on a GPU that is gone now -> numpy
        except Exception:
            return None
        return dev

    def _counts(self, pix: np.ndarray) -> np.ndarray:
        """Readout counts for `pix`, remembering the last call.

        The harness asks for `predict(val_x)` and then `scores(val_x)` -- same images, same net --
        so without this the big nets are simulated twice for one number each. The memo holds strong
        references to the exact objects it was computed from and compares them with `is`, so it can
        only ever hit on a repeat of the identical call; any new batch, or any reassigned/rebuilt
        layer, misses and recomputes.

        Post-training evaluation (the harness's predict/scores on val+test, and the full-test
        circuit cross-check) is the same simulation the trainer just ran, so it goes to the same
        device when `sim_device` says so and the net is big enough -- `lut_sim`'s CUDA backend is
        bit-for-bit the numpy reference, so this cannot move a measured number, only the clock.
        Anything that goes wrong on the device falls back to numpy rather than failing the run.
        """
        memo = self._counts_memo
        if memo is not None:
            p, thr, lay, out = memo
            if (p is pix and thr is self.thresholds and len(lay) == len(self.layers)
                    and all(a is b for a, b in zip(lay, self.layers))):
                return out
        dev = self._eval_device()
        try:
            out = lut_sim(self.thresholds, self.layers, pix, self.spec, device=dev)
        except Exception as e:          # OOM, a vanished device, a stale ordinal -> numpy still works
            if dev is None:
                raise
            print(f"[lutsim] eval on {dev} failed ({type(e).__name__}: {e}); using numpy",
                  flush=True)
            out = lut_sim(self.thresholds, self.layers, pix, self.spec)
        self._counts_memo = (pix, self.thresholds, list(self.layers), out)
        return out

    def predict(self, pix: np.ndarray) -> np.ndarray:
        return self._counts(pix).argmax(1)

    def scores(self, pix: np.ndarray) -> np.ndarray:
        g = len(self.layers[-1][0]) // self.spec.n_classes
        return self._counts(pix) / g

    def save(self, path: str) -> None:
        with open(path, "wb") as f:  # file handle so numpy keeps the .ckpt name (no .npz suffix)
            np.savez(f, thresholds=np.asarray(self.thresholds, np.int64),
                     **{f"a{i}": a for i, (a, _, _) in enumerate(self.layers)},
                     **{f"b{i}": b for i, (_, b, _) in enumerate(self.layers)},
                     **{f"t{i}": t for i, (_, _, t) in enumerate(self.layers)})
