"""Shared fan-in-2 LUT-net scaffolding for the logic-net methods (backprop, genetic, dfa, es).

A trained logic net is fully described by `(thresholds, layers)` where each layer is
`(idx_a, idx_b, tt)` (int arrays): gate i reads signals idx_a[i], idx_b[i] and applies the 4-bit
truth table tt[i]. `LutModel` turns that into emit_verilog / predict / scores / save, and `lut_sim`
is a packed numpy simulator that mirrors `hw.emit_lutnet` exactly -- so a method only has to produce
the hard `(thresholds, layers)` and everything else (and bit-exactness vs the synthesized netlist)
comes for free. Optimizers differ only in how they learn those arrays.
"""

from __future__ import annotations

import numpy as np

import hw
from data import Dataset, DatasetSpec, to_bits

FULL = np.uint64(0xFFFFFFFFFFFFFFFF)


def encode_bits(pix: np.ndarray, thresholds, spec: DatasetSpec) -> np.ndarray:
    """(N, n_pixels) uint8 -> (N, n_pixels*k) uint8, byte-major bit p*k+j = pix[p] > thresholds[j].
    Matches hw.emit_thermometer."""
    t = np.asarray(thresholds, np.int16)
    b = (pix[:, :, None].astype(np.int16) > t)          # (N, n_pixels, k)
    return b.reshape(len(pix), -1).astype(np.uint8)


def candidates(n_src: int, width: int, k: int, rng) -> np.ndarray:
    """(width, k) random source ids in [0, n_src): each gate's k wiring candidates."""
    return rng.integers(0, n_src, size=(width, k), dtype=np.int64)


def lut_sim(thresholds, layers, pix: np.ndarray, spec: DatasetSpec, chunk: int = 512) -> np.ndarray:
    """Exact packed simulation of the emitted LUT net -> (N, n_classes) integer readout counts.

    Same semantics as hw.emit_lutnet + hw.emit_popcount_argmax: argmax of these counts is predict();
    counts / group_size is scores(). Bit-packed (64 images/word) and chunked, so it runs on the big
    nets without materialising an (N, gates) array.
    """
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
    g = last_w // spec.n_classes

    # precompute per-layer tt masks (m_k = 0xFF.. where truth-table bit k is set), k = 2a+b
    masks = []
    for _, _, tt in layers:
        tt = np.asarray(tt, np.int64)
        masks.append([np.where((tt >> kk) & 1, FULL, np.uint64(0))[:, None] for kk in range(4)])

    out = []
    for i in range(0, len(pix), chunk):
        eb = encode_bits(pix[i:i + chunk], thresholds, spec)   # (n, n_in) uint8
        n = len(eb)
        pad = (-n) % 64
        if pad:
            eb = np.concatenate([eb, np.zeros((pad, n_in), np.uint8)])
        enc = np.packbits(np.ascontiguousarray(eb.T), axis=1, bitorder="little").view(np.uint64)
        S = np.zeros((n_sig, enc.shape[1]), np.uint64)
        S[:n_in] = enc
        for (idx_a, idx_b, _), (o, w), m in zip(layers, zip(offs, widths), masks):
            A, B = S[np.asarray(idx_a, np.int64)], S[np.asarray(idx_b, np.int64)]
            S[o:o + w] = ((m[0] & ~A & ~B) | (m[1] & ~A & B) | (m[2] & A & ~B) | (m[3] & A & B))
        read = np.unpackbits(S[last_off:last_off + last_w].view(np.uint8), axis=1, bitorder="little")
        read = read[:, :n].reshape(spec.n_classes, g, n).sum(1)   # (n_classes, n)
        out.append(read.T.astype(np.int64))
    return np.concatenate(out)


class LutModel:
    """emit / predict / scores / save for any method that produces hard (thresholds, layers)."""

    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self.thresholds: list[int] = []
        self.layers: list = []          # list of (idx_a, idx_b, tt) int arrays

    def emit_verilog(self) -> str:
        return hw.emit_lutnet(self.thresholds, self.layers, self.spec)

    def _counts(self, pix: np.ndarray) -> np.ndarray:
        return lut_sim(self.thresholds, self.layers, pix, self.spec)

    def predict(self, pix: np.ndarray) -> np.ndarray:
        return self._counts(pix).argmax(1)

    def scores(self, pix: np.ndarray) -> np.ndarray:
        g = len(self.layers[-1][0]) // self.spec.n_classes
        return self._counts(pix) / g

    def save(self, path: str) -> None:
        np.savez(path, thresholds=np.asarray(self.thresholds, np.int64),
                 **{f"a{i}": a for i, (a, _, _) in enumerate(self.layers)},
                 **{f"b{i}": b for i, (_, b, _) in enumerate(self.layers)},
                 **{f"t{i}": t for i, (_, _, t) in enumerate(self.layers)})
