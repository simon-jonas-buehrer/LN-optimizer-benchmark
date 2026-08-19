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
    lines = [f"  // readout: {n_classes} groups x {g} bits -> popcount -> argmax",
             f"  logic [{w - 1}:0] cnt [0:{n_classes - 1}];"]
    for c in range(n_classes):
        lines.append(f"  assign cnt[{c}] = {' + '.join(bit_names[c * g : (c + 1) * g])};")
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
        body.append(f"  // layer {li}: {w} gates, sources < {off}")
        for i in range(w):
            a, b = int(idx_a[i]), int(idx_b[i])
            if not (0 <= a < off and 0 <= b < off):
                raise ValueError(f"layer {li} gate {i} reads {a}/{b}, only 0..{off - 1} exist")
            body.append(f"  assign s[{off + i}] = {lut2_expr(tt[i], f's[{a}]', f's[{b}]')};")
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


def qmlp_forward(x0: np.ndarray, layers: Sequence[QLayer]) -> np.ndarray:
    """(N, in_dim0) int activations -> (N, n_classes) signed integer logits."""
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


def _acc_bound(L: QLayer, in_amax: int) -> int:
    perj = np.abs(L.Wq).sum(0).astype(np.int64) * in_amax + np.abs(L.bias).astype(np.int64)
    return int(perj.max()) if perj.size else 1


def _signed_width(bound: int) -> int:
    return max(2, int(bound).bit_length() + 2)


def _term(w: int, a: str) -> str:
    sx = f"$signed({{1'b0, {a}}})"
    return f"+ {sx}" if w == 1 else f"- {sx}" if w == -1 else f"+ ({w}) * {sx}"


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
        acc_names = []
        for j in range(L.out_dim):
            col = L.Wq[:, j]
            terms = [_term(int(col[i]), prev[i]) for i in np.nonzero(col)[0]]
            expr = " ".join(terms) if terms else "0"
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
