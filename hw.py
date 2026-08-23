"""Verilog emitters. Every model emits a SystemVerilog `top` module; yosys+ABC maps it to gates.

Fixed interface, derived from the dataset's `DatasetSpec` (not hardcoded MNIST constants):

    module top (input [spec.port_bits-1:0] pix, output [spec.cls_bits-1:0] cls);

`pix[spec.pixel_bits*p +: spec.pixel_bits]` is byte `p`; `cls` is the predicted class. Everything
between the ports is counted.

Two families of emitter:
  * fan-in-2 logic nets  -> `emit_lutnet` (thermometer encoder, 2-input LUT layers, popcount+argmax)
  * quantized MLPs       -> `emit_quant_mlp` (integer MAC / requantize / integer argmax)
Both are just Verilog; yosys lowers them to the same 2-input NAND netlist the harness measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from data import DatasetSpec

# ================================================================================================
# Fan-in-2 logic nets
# ================================================================================================

# The 16 boolean functions of two inputs. Truth table `tt` is a 4-bit int: bit (2a+b) is f(a,b).
_LUT2 = {
    0b0000: "1'b0", 0b0001: "(~{a} & ~{b})", 0b0010: "(~{a} & {b})", 0b0011: "~{a}",
    0b0100: "({a} & ~{b})", 0b0101: "~{b}", 0b0110: "({a} ^ {b})", 0b0111: "~({a} & {b})",
    0b1000: "({a} & {b})", 0b1001: "~({a} ^ {b})", 0b1010: "{b}", 0b1011: "(~{a} | {b})",
    0b1100: "{a}", 0b1101: "({a} | ~{b})", 0b1110: "({a} | {b})", 0b1111: "1'b1",
}


def lut2_expr(tt: int, a: str, b: str) -> str:
    """Verilog for the 2-input boolean function `tt` applied to signals a, b."""
    return _LUT2[int(tt) & 0xF].format(a=a, b=b)


def even_thresholds(bits: int) -> list[int]:
    """`bits` evenly spaced thermometer thresholds on a uint8 byte, on 2^k-1 boundaries: `pix>127`
    is bit 7 of the byte, a free wire; `pix>63` costs two gates."""
    return [round(256 * (j + 1) / (bits + 1)) - 1 for j in range(bits)]


# ================================================================================================
# Multi-operand addition -- the one RTL decision that dominates synthesis cost
#
# Verilog sizes every operand of an expression to the width of its lvalue. So
#
#     wire signed [14:0] acc0_0 = - $signed({1'b0, a0[13]}) - $signed({1'b0, a0[14]}) + ...;
#
# with 784 terms is not "add 784 four-bit numbers". It is a chain of 784 FIFTEEN-bit adders, each
# carrying eleven bits of sign extension that ABC then has to prove redundant -- ~10,500 full-adder
# bits per neuron where the data needs ~3,500. That redundancy is most of what made the quantized
# cells cost hundreds of gigabytes of yosys heap.
#
# `fold_sum` folds the same sum as a 4-ary tree of exact-width partial sums instead: 4-bit leaves,
# widening two bits per level, reaching the accumulator width only at the root. Integer addition is
# associative and nothing here is float, so the result is bit-identical -- this is a pure change of
# structure, not of value, and `circuit == predict()` is unaffected.
#
# Fan-in 4 rather than 2 is deliberate: a binary tree emits ~n wires, a 4-ary tree ~n/3, and the
# three-adder chain inside a 4-term group wastes ~2 bits against a perfect tree. Text size matters
# here -- the widest points emit millions of these.
# ================================================================================================

_CHUNK = 4


def fold_sum(body: list[str], prefix: str, leaves: list[tuple[str, int]]) -> tuple[str, int]:
    """Fold (expr, max_value) leaves into one unsigned sum -> (expr, max_value).

    Appends the intermediate `wire` declarations to `body`, each declared at exactly the width its
    running maximum needs. `prefix` must be unique per call site; returns a bare leaf expression
    unchanged when there is nothing to fold.
    """
    if not leaves:
        return "1'b0", 0
    cur, k = list(leaves), 0
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), _CHUNK):
            grp = cur[i:i + _CHUNK]
            if len(grp) == 1:
                nxt.append(grp[0])
                continue
            mx = sum(m for _, m in grp)
            nm = f"{prefix}{k}"
            k += 1
            body.append(f"  wire [{max(1, mx.bit_length()) - 1}:0] {nm} = "
                        f"{' + '.join(e for e, _ in grp)};")
            nxt.append((nm, mx))
        cur = nxt
    return cur[0]


def scale_sum(body: list[str], prefix: str, term: tuple[str, int], k: int) -> tuple[str, int]:
    """`k` * term for a small positive constant k, as shifts and adds -> (expr, max_value).

    A concatenation is a free shift, so k decomposes into its set bits: 5*T is {T,2'b00} + T. Left
    to yosys as `k * T` this would become a generic $mul macro that ABC has to strip back down.
    """
    e, mx = term
    if k == 1:
        return e, mx
    parts = [(e if b == 0 else f"{{{e}, {b}'b{'0' * b}}}", mx << b)
             for b in range(k.bit_length()) if (k >> b) & 1]
    return fold_sum(body, prefix, parts)


def signed_width(lo: int, hi: int) -> int:
    """Narrowest two's-complement width holding every value in [lo, hi]."""
    w = 2
    while lo < -(1 << (w - 1)) or hi > (1 << (w - 1)) - 1:
        w += 1
    return w


def fold_signed(body: list[str], prefix: str,
                leaves: list[tuple[str, int, int]]) -> tuple[str, int, int]:
    """`fold_sum` for signed (expr, lo, hi) leaves -> (expr, lo, hi)."""
    if not leaves:
        return "0", 0, 0
    cur, k = list(leaves), 0
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), _CHUNK):
            grp = cur[i:i + _CHUNK]
            if len(grp) == 1:
                nxt.append(grp[0])
                continue
            lo = sum(l for _, l, _ in grp)
            hi = sum(h for _, _, h in grp)
            nm = f"{prefix}{k}"
            k += 1
            body.append(f"  wire signed [{signed_width(lo, hi) - 1}:0] {nm} = "
                        f"{' + '.join(e for e, _, _ in grp)};")
            nxt.append((nm, lo, hi))
        cur = nxt
    return cur[0]


def dot_expr(body: list[str], prefix: str, pairs: list[tuple[int, str]],
             amax: int) -> tuple[str, int, int]:
    """sum_k w_k * act_k for small signed constants w_k -> signed (expr, lo, hi).

    Buckets by weight VALUE so each activation is summed exactly once (a bit-plane split would sum
    it once per set weight bit), folds each bucket unsigned and narrow, scales it by shifts, then
    subtracts the negative side from the positive side in one signed line.
    """
    buckets: dict[int, list[str]] = {}
    for w, a in pairs:
        if w:
            buckets.setdefault(int(w), []).append(a)
    side: dict[int, list[tuple[str, int]]] = {1: [], -1: []}
    for v in sorted(buckets):
        tag = f"{prefix}{'m' if v < 0 else 'p'}{abs(v)}_"
        t = fold_sum(body, tag, [(a, amax) for a in buckets[v]])
        side[1 if v > 0 else -1].append(scale_sum(body, f"{tag}k", t, abs(v)))
    parts, lo, hi = [], 0, 0
    for sign in (1, -1):
        if not side[sign]:
            continue
        e, mx = fold_sum(body, f"{prefix}{'P' if sign > 0 else 'N'}", side[sign])
        parts.append(f"{'+' if sign > 0 else '-'} $signed({{1'b0, {e}}})")
        if sign > 0:
            hi = mx
        else:
            lo = -mx
    return (" ".join(parts).lstrip("+ ") or "0"), lo, hi


def emit_thermometer(thresholds: Sequence[int], spec: DatasetSpec, sig: str = "s") -> tuple[str, int]:
    """Encoder: each input byte -> len(thresholds) bits `pix[p] > t`, byte-major (bit p*k+j)."""
    k = len(thresholds)
    pb = spec.pixel_bits
    lines = [f"  // thermometer: {spec.n_pixels} bytes x {k} = {spec.n_pixels * k} bits"]
    for p in range(spec.n_pixels):
        for j, t in enumerate(thresholds):
            t = int(t)
            if not 0 <= t <= (1 << pb) - 2:
                raise ValueError(f"threshold {t} is degenerate for a {pb}-bit byte")
            lines.append(f"  assign {sig}[{p * k + j}] = pix[{p * pb} +: {pb}] > {pb}'d{t};")
    return "\n".join(lines), spec.n_pixels * k


def emit_popcount_argmax(bit_names: Sequence[str], spec: DatasetSpec) -> str:
    """Readout: split bits into n_classes contiguous groups, popcount, argmax (ties: lowest)."""
    n_classes, n = spec.n_classes, len(bit_names)
    if n % n_classes != 0:
        raise ValueError(f"{n} readout bits not divisible by {n_classes} classes")
    g = n // n_classes
    w = max(1, int(g).bit_length())
    cw = spec.cls_bits
    lines = [f"  // readout: {n_classes} groups x {g} bits -> popcount tree -> argmax",
             f"  logic [{w - 1}:0] cnt [0:{n_classes - 1}];"]
    for c in range(n_classes):
        # a popcount tree, not `b0 + b1 + ... + b639` at the full count width (see `fold_sum`)
        e, _ = fold_sum(lines, f"pc{c}_", [(b, 1) for b in bit_names[c * g:(c + 1) * g]])
        lines.append(f"  assign cnt[{c}] = {e};")
    lines += [f"  logic [{w - 1}:0] best;", "  always_comb begin", "    best = cnt[0];",
              f"    cls  = {cw}'d0;"]
    for c in range(1, n_classes):
        lines.append(f"    if (cnt[{c}] > best) begin best = cnt[{c}]; cls = {cw}'d{c}; end")
    lines.append("  end")
    return "\n".join(lines)


def emit_lutnet(thresholds, layers, spec: DatasetSpec, *, top: str = "top") -> str:
    """Whole circuit for a fan-in-2 logic net. `layers` is a list of (idx_a, idx_b, tt); signal ids
    are global (encoder 0..n_in-1, then each layer); a gate may read only strictly-earlier signals.
    The last layer is the readout, width divisible by n_classes."""
    enc, n_in = emit_thermometer(thresholds, spec)
    body, off, offs = [enc], n_in, [n_in]
    for li, (idx_a, idx_b, tt) in enumerate(layers):
        w = len(idx_a)
        if not (len(idx_b) == len(tt) == w):
            raise ValueError(f"layer {li}: idx_a/idx_b/tt length mismatch")
        # one bulk conversion to python ints + one vectorised range check, instead of two int()
        # calls and two comparisons per gate; the emitted text is byte-identical.
        a_l = np.asarray(idx_a, np.int64).tolist()
        b_l = np.asarray(idx_b, np.int64).tolist()
        t_l = (np.asarray(tt, np.int64) & 0xF).tolist()
        if w and not (0 <= min(a_l) and max(a_l) < off and 0 <= min(b_l) and max(b_l) < off):
            for i, (a, b) in enumerate(zip(a_l, b_l)):
                if not (0 <= a < off and 0 <= b < off):
                    raise ValueError(f"layer {li} gate {i} reads {a}/{b}, only 0..{off - 1} exist")
        body.append(f"  // layer {li}: {w} gates, sources < {off}")
        body.extend(f"  assign s[{off + i}] = {_LUT2[t].format(a=f's[{a}]', b=f's[{b}]')};"
                    for i, (a, b, t) in enumerate(zip(a_l, b_l, t_l)))
        off += w
        offs.append(off)
    head = emit_popcount_argmax([f"s[{i}]" for i in range(offs[-2], offs[-1])], spec)
    return f"""// core.hw lutnet -- {spec.name}, {len(layers)} layers, {off - n_in} gates
module {top} (input [{spec.port_bits - 1}:0] pix, output logic [{spec.cls_bits - 1}:0] cls);
  wire [{off - 1}:0] s;

{chr(10).join(body)}

{head}
endmodule
"""


# ================================================================================================
# Quantized MLPs (integer-exact; predict() == synthesized netlist)
#
#   acc_j = sum_i Wq[i,j]*x_i + bias_j                            signed integer MAC
#   y_j   = clamp((acc_j*MUL + 2^(SH-1)) >>> SH, 0, 2^a-1)        requantize + ReLU  (next layer)
#
# w1.58 weights in {-1,0,+1} -> Wq*x is +x/0/-x, an adder tree (no multiplier).
# w4 weights in {-8..7}      -> real signed multiplier + adder tree.
# activations unsigned a-bit; the input layer reads the top `a` bits of each pixel byte (wires).
# final layer -> n_classes signed logits -> integer argmax (ties: lowest class).
# ================================================================================================


@dataclass
class QLayer:
    Wq: np.ndarray      # (in_dim, out_dim) small signed ints
    bias: np.ndarray    # (out_dim,) ints
    mul: int            # requant multiplier (>0); unused if final
    sh: int             # requant arithmetic shift (>=1); unused if final
    out_abits: int      # activation bits out; unused if final
    final: bool = False

    @property
    def in_dim(self) -> int:
        return int(self.Wq.shape[0])

    @property
    def out_dim(self) -> int:
        return int(self.Wq.shape[1])


def input_activations(pix: np.ndarray, spec: DatasetSpec, in_abits: int) -> np.ndarray:
    """(N, n_pixels) uint8 -> top `in_abits` bits of each byte (byte >> (pixel_bits-in_abits))."""
    if not 1 <= in_abits <= spec.pixel_bits:
        raise ValueError(f"in_abits {in_abits} out of range 1..{spec.pixel_bits}")
    return pix.astype(np.int64) >> (spec.pixel_bits - in_abits)


def requantize(acc: np.ndarray, mul: int, sh: int, out_abits: int) -> np.ndarray:
    r = (acc.astype(np.int64) * int(mul) + (1 << (sh - 1))) >> sh
    return np.clip(r, 0, (1 << out_abits) - 1)


_F32_SAFE = 1 << 23     # every partial sum stays an exactly representable float32 integer (< 2^24)
_F64_SAFE = 1 << 52     # ... float64 (< 2^53)
_GEMM_ROWS = 4096       # rows per pass, to bound the float activation buffers


def qmlp_forward_int(x0: np.ndarray, layers: Sequence[QLayer]) -> np.ndarray:
    """Reference integer implementation: (N, in_dim0) int activations -> (N, n_classes) logits."""
    x, logits = x0.astype(np.int64), None
    for L in layers:
        acc = x @ L.Wq.astype(np.int64) + L.bias.astype(np.int64)
        if L.final:
            logits = acc
        else:
            x = requantize(acc, L.mul, L.sh, L.out_abits)
    if logits is None:
        raise ValueError("no final layer")
    return logits


def _gemm_dtype(layers: Sequence[QLayer], in_amax: int):
    """Smallest float dtype in which every layer's MAC is exact, or None if there isn't one.

    Every operand is an integer, so a float GEMM is exact as long as every partial sum is an integer
    strictly inside the mantissa. `sum_i |W_ij| * amax + |b_j|` bounds the full column sum AND every
    partial sum of it, whatever order (or FMA, or blocking) BLAS chooses -- so if that bound is
    representable, the float result equals the int64 result bit for bit. `amax` is the largest
    magnitude any activation entering that layer can have: max|x0| for the first layer, and the
    requantize clip `2^out_abits - 1` afterwards.
    """
    amax = int(in_amax)
    if amax >= _F64_SAFE:
        return None
    dt = np.float64 if amax >= _F32_SAFE else np.float32
    for L in layers:
        col = np.abs(L.Wq).sum(0, dtype=np.int64) * amax + np.abs(L.bias).astype(np.int64)
        bound = int(col.max()) if col.size else 0
        if bound >= _F64_SAFE:
            return None
        if bound >= _F32_SAFE:
            dt = np.float64
        if not L.final:
            amax = (1 << L.out_abits) - 1
    return dt


def qmlp_forward(x0: np.ndarray, layers: Sequence[QLayer], *,
                 cache: dict | None = None) -> np.ndarray:
    """(N, in_dim0) int activations -> (N, n_classes) signed integer logits.

    Bit-identical to `qmlp_forward_int`, but the MACs go through BLAS: numpy has no BLAS path for
    int64 and falls back to a scalar loop, which is orders of magnitude slower on the wide layers.
    The GEMM runs in the smallest float dtype `_gemm_dtype` can prove exact for this net (else the
    int64 reference runs); requantization stays in int64. Pass any dict as `cache` to memoise the
    float weight copies across calls.
    """
    x0 = np.asarray(x0)
    if x0.dtype.kind not in "iub" or not layers or not layers[-1].final:
        return qmlp_forward_int(x0, layers)
    in_amax = int(np.abs(x0).max()) if x0.size else 0
    prep = cache.get("_qmlp_prep") if cache is not None else None
    if prep is None or prep[0] is not layers or prep[1] != in_amax:
        dt = _gemm_dtype(layers, in_amax)
        prep = (layers, in_amax, dt, None if dt is None else
                [(np.ascontiguousarray(L.Wq, dt), L.bias.astype(dt)) for L in layers])
        if cache is not None:
            cache["_qmlp_prep"] = prep
    dt, packed = prep[2], prep[3]
    if dt is None:                       # no provably-exact float dtype -> the int64 reference
        return qmlp_forward_int(x0, layers)
    out = np.empty((len(x0), int(layers[-1].Wq.shape[1])), np.int64)
    for i in range(0, len(x0), _GEMM_ROWS):
        x = np.ascontiguousarray(x0[i:i + _GEMM_ROWS], dt)
        for L, (Wf, bf) in zip(layers, packed):
            acc = x @ Wf
            acc += bf
            if L.final:
                out[i:i + _GEMM_ROWS] = acc.astype(np.int64)
            else:
                x = requantize(acc.astype(np.int64), L.mul, L.sh, L.out_abits).astype(dt)
    return out


def _acc_bound(L: QLayer, in_amax: int) -> int:
    perj = np.abs(L.Wq).sum(0).astype(np.int64) * in_amax + np.abs(L.bias).astype(np.int64)
    return int(perj.max()) if perj.size else 1


def _signed_width(bound: int) -> int:
    return max(2, int(bound).bit_length() + 2)


def emit_argmax_int(logit_names: Sequence[str], width: int, spec: DatasetSpec) -> str:
    n = len(logit_names)
    if n != spec.n_classes:
        raise ValueError(f"{n} logits != {spec.n_classes} classes")
    cw = spec.cls_bits
    lines = [f"  // argmax over {n} signed logits (ties: lowest)",
             f"  logic signed [{width - 1}:0] lg [0:{n - 1}];"]
    for c, nm in enumerate(logit_names):
        lines.append(f"  assign lg[{c}] = {nm};")
    lines += [f"  logic signed [{width - 1}:0] best;", "  always_comb begin", "    best = lg[0];",
              f"    cls  = {cw}'d0;"]
    for c in range(1, n):
        lines.append(f"    if (lg[{c}] > best) begin best = lg[{c}]; cls = {cw}'d{c}; end")
    lines.append("  end")
    return "\n".join(lines)


_GROUPS = (2, 3, 4, 6, 8, 12)


def _group_size(Wq: np.ndarray) -> int:
    """Input-group size that minimises emitted adds for this weight matrix, or 0 for no grouping.

    Neurons in a layer read the SAME activations, so if the input index space is cut into fixed
    groups, neuron j's contribution from group g depends only on (g, its weights on that group).
    Those (group, pattern) pairs repeat across neurons -- ternary weights admit at most 3^K patterns
    per group, against out_dim neurons -- so each distinct one can be emitted once and referenced by
    name by every neuron that shares it. Sharing is what takes the MAC below the per-neuron adder
    floor; nothing else here can, because a carry-save tree costs the same full adders as a ripple
    one, it only shortens them.

    Whether it pays depends entirely on the trained weights, so this COUNTS instead of guessing:
    `build` is the adds needed to emit every distinct group sum once, `combine` the adds each neuron
    still needs over its group sums, and the smallest total wins. A layer whose patterns almost
    never repeat (int4 with a wide output layer) scores worst at every K and gets 0 -- the plain
    per-neuron path, unchanged.
    """
    n_in, n_out = Wq.shape

    def lines(n):                                   # wires a 4-ary fold of n leaves emits
        return np.maximum(n - 1, 0) // (_CHUNK - 1) + (np.maximum(n - 1, 0) % (_CHUNK - 1) > 0)

    # The cost is EMITTED WIRES, not adds. Counting adds alone once picked K=2 for a layer with
    # essentially no reuse: it shaved a few adds while turning every inlined term into its own named
    # wire, and the .sv doubled. Every shared group also costs its own wire on top of its fold.
    best = int(lines(np.count_nonzero(Wq, axis=0)).sum())      # no grouping: one fold per column
    best_k = 0
    for k in _GROUPS:
        if k >= n_in:
            break
        build, live_per_out = 0, np.zeros(n_out, np.int64)
        for lo in range(0, n_in, k):
            uniq, inv = np.unique(np.ascontiguousarray(Wq[lo:lo + k].T), axis=0,
                                  return_inverse=True)
            live = np.count_nonzero(uniq, axis=1)
            build += int((lines(live[live > 0]) + 1).sum())    # +1: the group wire itself
            live_per_out += (live[inv.ravel()] > 0)
        total = build + int(lines(live_per_out).sum())
        if total < best:
            best, best_k = total, k
    return best_k


def emit_quant_mlp(layers: Sequence[QLayer], spec: DatasetSpec, in_abits: int,
                   *, top: str = "top") -> str:
    if not layers or not layers[-1].final:
        raise ValueError("last layer must be final")
    if layers[0].in_dim != spec.n_pixels:
        raise ValueError(f"layer 0 in_dim {layers[0].in_dim} != n_pixels {spec.n_pixels}")
    if layers[-1].out_dim != spec.n_classes:
        raise ValueError(f"final out_dim {layers[-1].out_dim} != n_classes {spec.n_classes}")

    pb = spec.pixel_bits
    body = [f"  // input activations: top {in_abits} of {pb} bits per byte",
            f"  wire [{in_abits - 1}:0] a0 [0:{spec.n_pixels - 1}];"]
    for p in range(spec.n_pixels):
        body.append(f"  assign a0[{p}] = pix[{p * pb + (pb - in_abits)} +: {in_abits}];")

    prev = [f"a0[{i}]" for i in range(spec.n_pixels)]
    in_amax = (1 << in_abits) - 1
    final_names, final_w = [], 2

    for li, L in enumerate(layers):
        if L.in_dim != len(prev):
            raise ValueError(f"layer {li} in_dim {L.in_dim} != {len(prev)} prev activations")
        aw = _signed_width(_acc_bound(L, in_amax))
        body.append(f"  // layer {li}: {L.in_dim} -> {L.out_dim}"
                    f"{' (logits)' if L.final else f', requant >>{L.sh} a{L.out_abits}'}")
        gk = _group_size(L.Wq)
        bounds = [(lo, min(lo + gk, L.in_dim)) for lo in range(0, L.in_dim, gk)] if gk else []
        shared: dict[tuple, tuple[str, int, int]] = {}
        if gk:
            body.append(f"  // shared group sums: {len(bounds)} groups of {gk} inputs")
        acc_names = []
        for j in range(L.out_dim):
            col = L.Wq[:, j].tolist()
            if gk:
                leaves = []
                for g, (lo, hi) in enumerate(bounds):
                    pat = tuple(col[lo:hi])
                    if not any(pat):
                        continue
                    hit = shared.get((g, pat))
                    if hit is None:
                        nm = f"g{li}_{len(shared)}"
                        e, elo, ehi = dot_expr(body, f"{nm}_",
                                               [(w, prev[lo + o]) for o, w in enumerate(pat) if w],
                                               in_amax)
                        body.append(f"  wire signed [{signed_width(elo, ehi) - 1}:0] {nm} = {e};")
                        hit = shared[(g, pat)] = (nm, elo, ehi)
                    leaves.append(hit)
                expr = fold_signed(body, f"s{li}_{j}_", leaves)[0]
            else:
                expr = dot_expr(body, f"c{li}_{j}_",
                                [(w, prev[i]) for i, w in enumerate(col) if w], in_amax)[0]
            if int(L.bias[j]):
                expr = f"{expr} + ({int(L.bias[j])})"
            body.append(f"  wire signed [{aw - 1}:0] acc{li}_{j} = {expr};")
            acc_names.append(f"acc{li}_{j}")
        if L.final:
            final_names, final_w = acc_names, aw
        else:
            amax = (1 << L.out_abits) - 1
            rw = _signed_width(_acc_bound(L, in_amax) * L.mul + (1 << (L.sh - 1)))
            rc = 1 << (L.sh - 1)
            nxt = []
            for j, acc in enumerate(acc_names):
                body.append(f"  wire signed [{rw - 1}:0] rq{li}_{j} = ({acc} * {L.mul} + {rc}) >>> {L.sh};")
                body.append(f"  wire [{L.out_abits - 1}:0] a{li + 1}_{j} = "
                            f"rq{li}_{j} <= 0 ? 0 : (rq{li}_{j} >= {amax} ? {amax} : "
                            f"rq{li}_{j}[{L.out_abits - 1}:0]);")
                nxt.append(f"a{li + 1}_{j}")
            prev, in_amax = nxt, amax

    head = emit_argmax_int(final_names, final_w, spec)
    return f"""// core.hw quant_mlp -- {spec.name}, {len(layers)} quantized layers
module {top} (input [{spec.port_bits - 1}:0] pix, output logic [{spec.cls_bits - 1}:0] cls);

{chr(10).join(body)}

{head}
endmodule
"""
