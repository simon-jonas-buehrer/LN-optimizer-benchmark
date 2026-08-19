"""dfa: fixed butterfly wiring, gate truth tables learned by direct feedback alignment.

Nothing here is learned except the 4 bits of each gate's truth table. The wiring is fixed:

  Forward. A butterfly (FFT) pattern wires every gate to two signals of the layer below: gate j reads
  j and j ^ (1 << k), with the stride k halving every layer. Deterministic, never touched by the
  optimizer; the stride cycle makes the receptive field genuinely double per layer.

  Backward. A fixed random matrix B_l projects the output error DIRECTLY onto layer l -- no backward
  sweep, no chain rule across layers (direct feedback alignment, Nokland 2016). Because the forward
  pass is exact bits, a gate's output is just T[p] with p=2a+b, so its derivative w.r.t. its own
  table is the indicator of the active pattern: each layer's update is a single scatter_add.

    e       = softmax(votes/tau) - onehot(y)     # (B, n_classes) -- the only global signal
    delta_l = e @ B_l                            # (B, w), B_l fixed random (n_classes, w)
    G[i,p]  = sum_b delta_l[b,i] * 1[p_bi == p]  # scatter_add, (w, 4)
    z.grad  = G * 0.5*cos(z)                     # chain THROUGH a gate, never ACROSS one

The readout layer uses its own local gradient (e[:, class(i)] / tau), DFA's standard output-layer
convention. B_l is a training-only object: never synthesized -> zero gates. `.backward()` is never
called; the whole step runs under torch.no_grad(), only the gradient is hand-written. Latents are
sin-binarized (hard = 1[sin(z)>0]); residual init (tt 0b1100, pass input A) at +-pi/4 so cos(z) != 0
and the gradient can move. The forward is exact boolean, so the trained (thresholds, layers) go
straight to LutModel and predict() == the emitted netlist by construction.

Implementation notes. The update rule, the init, the feedback distribution, the schedule and the
early-stopping criterion above are untouched; what changed is how the step is executed. Two rules
divide this file: the TRAINING forward may be reshaped freely (DFA already trains on continuous
latents and evaluates on hardened tables -- that gap is the method), while everything from
`layers_np()` onward is exact and must stay exact, because `predict()` has to be the function
`emit_verilog()` describes.

  * DFA has no backward sweep, so the body layers never interact: their latents, feedback matrices
    and gradient accumulators are stored as ONE stacked tensor each (all body layers share `width`).
    The per-layer feedback matmuls become one, the per-layer scatter_adds become one, Adam sees two
    parameters instead of `layers+1`, and DDP all-reduces two buffers instead of six.
  * the fixed global batch is executed as few, wide forwards instead of many tiny ones -- see
    `_Butterfly.merge`. Same images, same gradient, same optimiser step; only how many kernel
    launches it takes, and the order the micro-batch sums are added in.
  * the active pattern p = 2a+b is built ONCE per micro-batch, as int64, and used twice: as the
    forward's gather index into the (w, 4) table and as the gradient's scatter index. Evaluation has
    no gradient to feed, so it stays in uint8 and reads a packed 4-bit table as `(tt >> p) & 1`.
  * every buffer and every view is built once per batch shape and reused (`_Plane`): the hot loop is
    nothing but `out=` kernels -- no allocation, no zero-fill, no tensor indexing. Which is exactly
    what lets the whole step be captured as a CUDA graph (`_StepGraph`) on a single GPU.
  * a source-index vector that is an arithmetic progression (every body layer's `a` tap) is a
    strided view of the signal plane, not a gather; a layer that gathers both taps gathers them in
    one `index_select`.
  * nothing syncs to the host inside an epoch: the loss is a device tensor, read once per epoch and
    only on the epochs that actually print it.

On CPU this is bit-identical to the straightforward transcription it replaced -- same tables, same
early-stop epoch. On CUDA it is not, and cannot be: `scatter_add_` accumulates the wide batch with
atomics, so the gradient is reproducible only to the last ulp (measured: identical across repeats at
8 images per forward, different on every repeat at 256). That moves which epoch validation loss
bottoms out, hence the extracted net, by the width of the noise. The hardened net is unaffected --
whatever comes out, `predict()`, `scores()` and the emitted netlist agree on it exactly.
"""

from __future__ import annotations

import time

import numpy as np
import torch

import ddp
from data import Dataset, DatasetSpec
from hw import even_thresholds
from methods.lut import LutModel

TITLE = "dfa (fixed butterfly wiring, direct feedback alignment)"

# 5 body layers, bits=1, spend the silicon on the READOUT (the one layer whose delta is the true
# gradient, not a random projection). Small tiers keep the original record's proven shapes; the big
# tiers extend the readout upward, spanning ~2k -> ~20M gates by pre-opt count (width*layers+readout).
# NB: the knob is `readout`/`width`/`layers`, never `depth` -- `depth` collides with a measured field.
_LADDER = {
    # name: (width, layers, readout)          pre-opt gates = width*layers + readout
    "xs":  (256, 5, 640),          #     1,920
    "s":   (512, 5, 1280),         #     3,840
    "m":   (1024, 5, 5120),        #    10,240
    "l":   (2048, 5, 90000),       #   100,240
    "xl":  (4096, 5, 1_000_000),   # 1,020,480
    "xxl": (8192, 5, 20_000_000),  # 20,040,960
}


def points(spec: DatasetSpec) -> list[dict]:
    return [{"name": n, "bits": 1, "width": w, "layers": L, "readout": r, "epochs": 60}
            for n, (w, L, r) in _LADDER.items()]


def _t(a: np.ndarray, device: str) -> torch.Tensor:
    """The harness speaks numpy; torch starts here."""
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


# ---- fixed structure: the butterfly tap --------------------------------------------------------

def _log2(n: int) -> int:
    if n & (n - 1):
        raise ValueError(f"{n} is not a power of two")
    return n.bit_length() - 1


def _butterfly_src(in_dim: int, out_dim: int, stage: int) -> torch.Tensor:
    """(2, out_dim) local source indices into `in_dim`. Fan-in 2, deterministic, non-learnable."""
    j = torch.arange(out_dim)
    if in_dim == out_dim:                       # body: the butterfly proper (stride k halves)
        k = stage % _log2(in_dim)
        return torch.stack([j, j ^ (1 << k)])
    if out_dim > in_dim:                        # encoder -> first layer: cover every input bit
        return torch.stack([(2 * j) % in_dim, (2 * j + 1) % in_dim])
    a = (j * in_dim) // out_dim                 # readout / narrowing: spread the tap over the layer
    return torch.stack([a, (a + in_dim // 2) % in_dim])


def _tap(idx: torch.Tensor):
    """A source vector that is an arithmetic progression is a strided VIEW, not a gather.

    `acts[slice]` costs nothing; `acts[index_tensor]` copies (w, B) bytes. Every body layer's `a`
    tap is `j + base` (step 1), so this drops a third of the butterfly's gathers without changing
    which signals are read.
    """
    n = int(idx.numel())
    start = int(idx[0])
    if n == 1:
        return slice(start, start + 1)
    d = idx[1:] - idx[:-1]
    step = int(d[0])
    if step >= 1 and bool(torch.equal(d, torch.full_like(d, step))):
        return slice(start, start + step * n, step)
    return idx


def hard_bit(z: torch.Tensor) -> torch.Tensor:
    """1[sin(z) > 0] -- periodic, so a latent never saturates out of reach of the gradient."""
    return (torch.sin(z) > 0).to(torch.uint8)


# Residual init: T[p] = a (tt 0b1100, pass input A). Magnitude pi/4, not pi/2 -- at pi/2 the table is
# exactly right but cos(pi/2)=0 kills the gradient. Jitter breaks the per-gate symmetry.
_RES_SIGN = torch.tensor([-1.0, -1.0, 1.0, 1.0])
_JITTER = 0.1

# Working-set ceiling for ONE forward/backward's scratch (see `_Butterfly.merge`). Not a training
# knob: the global batch, the shard protocol and the gradient are identical whatever it is set to --
# it only decides how many of the fixed micro-batches are fused into one set of kernel launches.
# 2 GiB keeps the xxl tier at its original 8-image micro-batch and lets every smaller tier run the
# whole 256-image global batch in one shot.
_PLANE_BUDGET = 2 << 30

# Capture the optimiser step as a CUDA graph when we can (see `_StepGraph`). Off => always eager.
_USE_CUDA_GRAPH = True


def _encode(pix: torch.Tensor, thresholds) -> torch.Tensor:
    """(N, n_pixels) uint8 -> (n_in, N) uint8, byte-major bit p*k+j, matching hw.emit_thermometer."""
    thr = torch.tensor(thresholds, device=pix.device, dtype=torch.int16)
    bits = pix.to(torch.int16).unsqueeze(-1) > thr          # (N, n_pixels, bits)
    return bits.reshape(pix.shape[0], -1).T.contiguous().to(torch.uint8)


class _Plane:
    """Everything one batch width needs, built once and reused for the whole run.

    `acts` (n_sig, B) is the signal plane; every forward writes it end to end (encoder rows, then one
    contiguous slice per layer), so it never needs zeroing. All the source/destination views into it
    are fixed for the life of the plane, so `propagate` does no tensor indexing at all -- it just
    walks a prebuilt list of `out=` kernel arguments:

        steps[l] = (sel, a, b, p, out)   sel  the index_select()s this layer needs (0, 1 or 2 taps
                                              that are real gathers -- when both are, they are
                                              fetched by ONE index_select into a (2w, B) buffer that
                                              `a` and `b` view)
                                         a,b  the two source planes (strided views or gather buffers)
                                         p    the (w, B) active pattern: int64 on a training plane,
                                              where it doubles as the gradient's scatter index; uint8
                                              on an eval plane, which has no gradient to feed
                                         out  this layer's slice of `acts`

    A training plane also carries the stacked body pattern `pb` (each body layer's `p` is a view into
    it, so one scatter_add covers them all), the readout pattern reshaped by class, and the float
    scratch for votes / error / stacked feedback delta.
    """

    __slots__ = ("acts", "enc", "read", "steps", "grad", "pb", "pr3", "vot", "e", "etau",
                 "etau3", "db")

    def __init__(self, net: "_Butterfly", B: int, grad: bool) -> None:
        dev, nb, bw, nc = net.device, net.nb, net.bw, net.spec.n_classes
        self.grad = grad
        self.acts = torch.empty((net.n_sig, B), dtype=torch.uint8, device=dev)
        self.enc = self.acts[: net.n_in]
        self.read = self.acts[net.offs[-2] : net.offs[-1]].view(nc, net.rg, B)

        # the active patterns: body layers stacked (one scatter_add for all of them), readout apart
        dt = torch.int64 if grad else torch.uint8
        self.pb = torch.empty((nb * bw, B), dtype=dt, device=dev)
        pr = torch.empty((net.widths[-1], B), dtype=dt, device=dev)
        self.pr3 = pr.view(nc, net.rg, B) if grad else None
        ps = [self.pb[l * bw : (l + 1) * bw] for l in range(nb)] + [pr]

        self.steps = []
        for l, ((ta, tb), w) in enumerate(zip(net.taps, net.widths)):
            va, vb = isinstance(ta, slice), isinstance(tb, slice)
            if va and vb:
                sel, a, b = (), self.acts[ta], self.acts[tb]
            elif va:
                gb = torch.empty((w, B), dtype=torch.uint8, device=dev)
                sel, a, b = ((tb, gb),), self.acts[ta], gb
            elif vb:
                ga = torch.empty((w, B), dtype=torch.uint8, device=dev)
                sel, a, b = ((ta, ga),), ga, self.acts[tb]
            else:   # both taps are gathers: fetch them with a single index_select
                gab = torch.empty((2 * w, B), dtype=torch.uint8, device=dev)
                sel, a, b = ((torch.cat([ta, tb]), gab),), gab[:w], gab[w:]
            self.steps.append((sel, a, b, ps[l], self.acts[net.offs[l] : net.offs[l + 1]]))

        # float scratch (training only): votes/logits, the error, and the stacked feedback delta
        if grad:
            self.vot = torch.empty((nc, B), device=dev)
            self.e = torch.empty((nc, B), device=dev)
            self.etau = torch.empty((nc, B), device=dev)
            self.etau3 = self.etau.unsqueeze(1).expand(nc, net.rg, B)   # stride 0: never materialised
            self.db = torch.empty((nb * bw, B), device=dev)
        else:
            self.vot = self.e = self.etau = self.etau3 = self.db = None


class _Butterfly:
    """Fixed butterfly fan-in-2 wiring; per-gate 4-entry sin latent (learned); fixed random B."""

    def __init__(self, spec: DatasetSpec, bits: int, width: int, layers: int, readout: int,
                 device: str, g: torch.Generator) -> None:
        if readout % spec.n_classes:
            raise ValueError(f"readout {readout} must be divisible by {spec.n_classes}")
        _log2(width)  # the butterfly needs a power-of-two body; fail loudly
        self.spec = spec
        self.thresholds = even_thresholds(bits)
        self.n_in = spec.n_pixels * bits
        self.device = device
        self.widths = [width] * layers + [readout]
        self.nb, self.bw = layers, width          # body layers, all one width -> stackable
        self.tau = (readout // spec.n_classes) ** 0.5
        self.rg = readout // spec.n_classes       # readout gates per class (contiguous groups)

        # fixed wiring: srcs[l] = (2, w) GLOBAL ids; layer l reads only layer l-1 (or the encoder)
        self.offs = [self.n_in]
        in_dim, in_base = self.n_in, 0
        self.srcs: list[torch.Tensor] = []
        taps = []
        for l, w in enumerate(self.widths):
            bf = _butterfly_src(in_dim, w, l - 1).contiguous()   # (2, w) local into in_dim
            gl = bf + in_base                                    # global ids, still on the host
            taps.append((_tap(gl[0]), _tap(gl[1])))
            self.srcs.append(gl.to(device))
            in_base = self.offs[-1]                              # next layer reads this layer's outs
            self.offs.append(self.offs[-1] + w)
            in_dim = w
        self.taps = [(a if isinstance(a, slice) else a.to(device),
                      b if isinstance(b, slice) else b.to(device)) for a, b in taps]

        # learned: the only parameters in the whole record. Drawn per layer (the draw order IS the
        # record's RNG stream), then stacked body / readout -- two tensors instead of layers+1.
        res = _RES_SIGN.to(device) * (torch.pi / 4)
        zs = [res.expand(w, 4).contiguous() + torch.randn(w, 4, generator=g, device=device) * _JITTER
              for w in self.widths]
        self.z = [torch.cat(zs[:-1], 0) if layers else zs[-1][:0], zs[-1]]
        # fixed random feedback: the backward "model". Never learned, never synthesized. Only B^T is
        # ever used (delta^T = B^T e^T), and only stacked, so that is what we keep.
        Bs = [torch.randn(spec.n_classes, w, generator=g, device=device) / spec.n_classes ** 0.5
              for w in self.widths[:-1]]
        self.Btb = (torch.cat([b.t() for b in Bs], 0).contiguous() if layers
                    else torch.zeros(0, spec.n_classes, device=device))

        # persistent scratch for the binarizer: `_fb` holds sin(z) while the tables are built and
        # cos(z) while the gradient is scaled -- two disjoint windows of the same step, one buffer.
        self._fb = [torch.empty_like(z) for z in self.z]
        self._Tb = [torch.empty(z.shape, dtype=torch.uint8, device=device) for z in self.z]
        self._p2 = torch.tensor([1, 2, 4, 8], dtype=torch.uint8, device=device)
        self._m1 = torch.tensor(-1.0, device=device)   # the onehot(y) subtraction, as one index_put_
        self._planes: dict[tuple[int, bool], _Plane] = {}
        self._ar: dict[int, torch.Tensor] = {}

    @property
    def n_sig(self) -> int:
        return self.offs[-1]

    def plane(self, B: int, grad: bool = False) -> _Plane:
        p = self._planes.get((B, grad))
        if p is None:
            p = self._planes[(B, grad)] = _Plane(self, B, grad)
        return p

    def ar(self, B: int) -> torch.Tensor:
        a = self._ar.get(B)
        if a is None:
            a = self._ar[B] = torch.arange(B, device=self.device)
        return a

    def plane_bytes(self, grad: bool = True) -> int:
        """Device bytes one `_Plane` costs PER IMAGE -- what bounds how wide a forward may be.

        At the xxl tier the readout's int64 scatter index alone is 160 MB per image, which is why
        `micro` exists; at the small tiers a plane is kilobytes and the same micro-batch wastes the
        machine on kernel-launch latency. `merge` turns this number into the answer.
        """
        idx = 8 if grad else 1
        b = self.n_sig                                     # the signal plane
        b += (self.nb * self.bw + self.widths[-1]) * idx   # the active patterns
        for (ta, tb), w in zip(self.taps, self.widths):    # the index_select destinations
            b += w * ((not isinstance(ta, slice)) + (not isinstance(tb, slice)))
        if grad:
            b += self.nb * self.bw * 4 + 3 * self.spec.n_classes * 4   # delta^T, votes/e/e_tau
        return b

    def merge(self, micro: int, accum: int) -> int:
        """How many of the `accum` micro-batches one forward may swallow, largest divisor first.

        The accumulation loop reconstructs the same full-batch gradient however it is cut up, so the
        cut is free to follow the memory: one forward of `micro*g` images instead of `g` forwards of
        `micro`. `g` divides `accum`, so the effective global batch, the per-rank contiguous shard
        and `nglobal` are all exactly what they were -- only the float summation order moves.
        """
        cap = max(1, _PLANE_BUDGET // max(1, self.plane_bytes() * micro))
        return max(d for d in range(1, accum + 1) if accum % d == 0 and d <= cap)

    def _split(self, body: torch.Tensor, read: torch.Tensor) -> list[torch.Tensor]:
        bw = self.bw
        return [body[l * bw : (l + 1) * bw] for l in range(self.nb)] + [read]

    def tables(self) -> list[torch.Tensor]:
        """Per-layer (w, 4) uint8 hard tables -- two tensors' worth, sliced back into layer views.

        Written into persistent buffers: this runs once per optimiser step, and on the captured path
        it has to be allocation-free anyway.
        """
        for z, fb, T in zip(self.z, self._fb, self._Tb):
            torch.sin(z.detach(), out=fb)             # detach: `out=` refuses a grad-tracking input,
            torch.gt(fb, 0, out=T)                    # and this is hard_bit(z), never differentiated
        return self._split(self._Tb[0], self._Tb[1])

    def packed(self) -> list[torch.Tensor]:
        """Per-layer (w, 1) uint8 with bit p = T[p]: the eval forward's whole table in one byte."""
        self.tables()
        p = [(T * self._p2).sum(1, keepdim=True, dtype=torch.uint8) for T in self._Tb]
        return self._split(p[0], p[1])

    def propagate(self, pl: _Plane, tab: list[torch.Tensor]) -> torch.Tensor:
        """`pl.enc` is already loaded; fill the rest. Exact bits; no relaxation anywhere.

        Two kernels per layer on a training plane (`2a+b` straight into the int64 pattern, then
        `T.gather`), three on an eval plane (`2a+b` in uint8, then `(tt >> p) & 1` on the packed
        table), plus one `index_select` per layer that has to gather its taps.
        """
        acts = pl.acts
        if pl.grad:
            for T, (sel, a, b, p, o) in zip(tab, pl.steps):
                for src, dst in sel:
                    torch.index_select(acts, 0, src, out=dst)
                torch.add(b, a, alpha=2, out=p)                  # p = 2a + b, in {0,1,2,3}
                torch.gather(T, 1, p, out=o)                     # T[p], straight into the plane
        else:
            for T, (sel, a, b, p, o) in zip(tab, pl.steps):
                for src, dst in sel:
                    torch.index_select(acts, 0, src, out=dst)
                torch.add(b, a, alpha=2, out=p)
                torch.bitwise_right_shift(T, p, out=o)
                torch.bitwise_and(o, 1, out=o)
        return acts

    def forward(self, enc: torch.Tensor, tab: list[torch.Tensor] | None = None) -> torch.Tensor:
        """enc (n_in, B) uint8 -> acts (n_sig, B) uint8."""
        pl = self.plane(enc.shape[1])
        pl.enc.copy_(enc)
        return self.propagate(pl, self.packed() if tab is None else tab)

    def votes_t(self, pl: _Plane, out: torch.Tensor | None = None) -> torch.Tensor:
        """(n_classes, B) float32 group popcounts -- exact (a group is far under 2^24 bits)."""
        return torch.sum(pl.read, 1, dtype=torch.float32, out=out)

    def layers_np(self) -> list:
        """Extract the hard tables + fixed wiring as np.int64 (idx_a, idx_b, tt) for LutModel.

        This is the hand-off to the exact side of the contract -- what `predict()` computes and what
        `emit_verilog()` describes -- so it deliberately re-derives `hard_bit(z)` into FRESH tensors
        rather than reading `tables()`' persistent buffers. Those buffers are written by the training
        step (and by a CUDA graph replaying it); the extraction must not be able to observe them.
        """
        hard = self._split(hard_bit(self.z[0].detach()), hard_bit(self.z[1].detach()))
        out = []
        for s, t in zip(self.srcs, hard):
            tt = t[:, 0] | (t[:, 1] << 1) | (t[:, 2] << 2) | (t[:, 3] << 3)   # bit p = T[p]
            out.append((s[0].cpu().numpy().astype(np.int64),
                        s[1].cpu().numpy().astype(np.int64),
                        tt.cpu().numpy().astype(np.int64)))
        return out


class _StepGraph:
    """One whole optimiser step's forward + hand-written gradient, captured as a CUDA graph.

    Once the micro-batches are merged, a step is a fixed sequence of a few dozen kernels over fixed
    buffers with fixed shapes -- which is exactly what `torch.cuda.CUDAGraph` wants. At the small
    tiers every one of those kernels is a couple of microseconds of arithmetic behind a couple of
    microseconds of launch, so replaying the whole step as one submission is most of the remaining
    time. Nothing about the computation changes: the same kernels run on the same buffers.

    Only the single-GPU path is captured. The gradient all-reduce sits in the middle of the step and
    NCCL collectives inside a capture need their own stream handling, so under DDP (and on any
    failure at all) the caller keeps the eager path.

    The two things that vary between steps are device buffers the caller fills before `replay`:
    `cols` (which images this step sees) and `nglob` (the divisor, which only shrinks on a short
    final block -- and short blocks take the eager path anyway).
    """

    __slots__ = ("cols", "nglob", "graph", "loss")

    def __init__(self, model, net, enc, y_all, Gacc, Gr3, mb: int, accum: int, device) -> None:
        self.cols = torch.empty(mb * accum, dtype=torch.int64, device=device)
        self.nglob = torch.empty((), dtype=torch.float32, device=device)
        views = [self.cols[a * mb:(a + 1) * mb] for a in range(accum)]
        ybuf = [torch.empty(mb, dtype=y_all.dtype, device=device) for _ in range(accum)]
        net.plane(mb, True)                  # allocate the scratch outside the capture, on this
        net.ar(mb)                           # stream, so the graph only ever records kernels

        def one_step():
            torch._foreach_zero_(Gacc)
            tab = net.tables()               # z is frozen for the whole step
            lt = None
            for v, yb in zip(views, ybuf):
                torch.index_select(y_all, 0, v, out=yb)
                lt = model._accum_grad(net, enc, v, yb, self.nglob, Gacc, Gr3, tab, True)
            for l, z in enumerate(net.z):    # the shared 0.5*cos(z) factor
                torch.cos(z.detach(), out=net._fb[l])
                net._fb[l] *= 0.5
                torch.mul(Gacc[l], net._fb[l], out=z.grad)
            return lt

        self.cols.copy_(torch.arange(mb * accum, device=device) % max(1, y_all.shape[0]))
        self.nglob.fill_(float(mb * accum))
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):               # the capture records a steady state, not a cold one
                one_step()
        torch.cuda.current_stream().wait_stream(warm)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss = one_step()

    @staticmethod
    def build(model, net, enc, y_all, Gacc, Gr3, mb, accum, world, device):
        """The captured step, or None when this run must stay eager."""
        if (world != 1 or not _USE_CUDA_GRAPH or torch.device(device).type != "cuda"
                or y_all.shape[0] < mb * accum):
            return None   # no full-size block will ever appear, so there is nothing to capture
        try:
            return _StepGraph(model, net, enc, y_all, Gacc, Gr3, mb, accum, device)
        except Exception as e:            # capture is an optimisation, never a requirement
            torch.cuda.synchronize(device)   # settle whatever the aborted capture left behind
            print(f"  (cuda graph capture unavailable: {type(e).__name__}: {e}); running eager",
                  flush=True)
            return None

    def replay(self, cols: torch.Tensor, nglobal: int) -> torch.Tensor:
        self.cols.copy_(cols)
        self.nglob.fill_(float(nglobal))
        self.graph.replay()
        return self.loss


class Dfa(LutModel):
    """Produces hard (thresholds, layers); emit/predict/scores/save come from LutModel + lut_sim."""

    def __init__(self, spec: DatasetSpec, bits: int, width: int, layers: int, readout: int,
                 epochs: int, lr: float = 0.01, batch: int = 256, patience: int = 12,
                 micro: int = 8) -> None:
        super().__init__(spec)
        # `micro` = per-GPU micro-batch (small enough for the 20M-gate readout), accumulated to the
        # fixed global `batch` -- same protocol for every size of this method.
        self.cfg = dict(bits=bits, width=width, layers=layers, readout=readout, epochs=epochs,
                        lr=lr, batch=batch, patience=patience, micro=micro)

    def _chunk(self) -> int:
        c = self.cfg
        n_sig = self.spec.n_pixels * c["bits"] + c["width"] * c["layers"] + c["readout"]
        return max(64, min(4096, 2 ** 28 // n_sig))

    def train(self, data: Dataset, *, device: str = "cpu", seed: int = 0) -> None:
        # Multi-GPU is opt-in via `ddp_gpus` (set by the harness). <=1 GPU runs `_worker` inline, so
        # this path is byte-for-byte the old DFA trainer; >1 shards each batch across ranks and
        # all-reduces the hand-written gradient (below), which is exactly the full-batch gradient.
        self._data, self._seed, self._base_device = data, seed, device
        res = ddp.launch(self._worker, getattr(self, "ddp_gpus", 1))
        self.thresholds, self.layers = res["thresholds"], res["layers"]
        self.train_seconds = res["train_seconds"]  # pure training time (no val/measure)
        self._data = None                          # don't keep the dataset pinned to the model
        # tell LutModel where to evaluate: its `lut_sim` has a bit-identical CUDA twin, and it owns
        # the crossover gate (small nets lose to numpy) and the numpy fallback.
        self.sim_device = device

    @torch.no_grad()
    def _worker(self, rank: int, world: int) -> dict:
        data, seed, c = self._data, self._seed, self.cfg
        device = self._base_device if world == 1 else f"cuda:{rank}"
        torch.manual_seed(seed)
        g = torch.Generator(device=device).manual_seed(seed)  # same draws on every rank (seed-only)
        net = _Butterfly(self.spec, c["bits"], c["width"], c["layers"], c["readout"], device, g)

        enc_tr = _encode(_t(data.train_x, device), net.thresholds)   # (n_in, N)
        y_tr = _t(data.train_y, device).long()
        enc_va = _encode(_t(data.val_x, device), net.thresholds)
        y_va = _t(data.val_y, device).long()

        # Adam only ever sees gradients we computed by hand; it never runs a backward pass.
        net.z = [torch.nn.Parameter(z) for z in net.z]
        opt = torch.optim.Adam(net.z, lr=c["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])

        # persistent scratch: the raw gradient accumulator (body stacked / readout), the 0.5*cos(z)
        # factor, and the grad tensors Adam reads -- allocated once, not once per optimiser step (the
        # xxl readout is 80M floats a piece).
        Gacc = [torch.zeros_like(z.data) for z in net.z]
        Gr3 = Gacc[1].view(self.spec.n_classes, net.rg, 4)
        for z in net.z:
            z.grad = torch.empty_like(z)

        n = enc_tr.shape[1]
        step, accum, _ = ddp.accum_plan(c["micro"], world, c["batch"])  # fixed global batch/size
        gmul = net.merge(c["micro"], accum)   # ... executed as fewer, wider forwards where it fits
        step, accum = step * gmul, accum // gmul
        eff = step * accum
        gstep = _StepGraph.build(self, net, enc_tr, y_tr, Gacc, Gr3, step // world, accum,
                                 world, device)
        if rank == 0:
            print(f"  batch {eff} = {accum} x {step} ({step // world}/rank, "
                  f"{net.plane_bytes() * step // world / 2**20:.0f} MiB scratch"
                  f"{', cuda graph' if gstep else ''})", flush=True)
        best, best_z, best_ep = float("inf"), [z.detach().clone() for z in net.z], 0
        train_secs, t_all, loss_t = 0.0, time.time(), None
        loss = float("nan")
        for ep in range(c["epochs"]):
            # the loss is a print-only quantity: compute it on the epochs that print it, and nowhere
            # else (identical printed numbers, four fewer kernels on every other micro-batch)
            want_loss = rank == 0 and (ep % 5 == 0 or ep == c["epochs"] - 1)
            perm = torch.randperm(n, generator=g, device=device)  # identical on every rank
            t0 = time.perf_counter()
            for i in range(0, n, eff):
                sl = perm[i:i + eff]
                if gstep is not None and sl.shape[0] == eff:
                    loss_t = gstep.replay(sl, eff)   # one submission for the whole step
                    opt.step()
                    continue
                micros = [sl[a * step:(a + 1) * step] for a in range(accum)]
                micros = [m for m in micros if m.shape[0] >= world]  # deterministic across ranks
                if not micros:
                    continue
                nglobal = sum((m.shape[0] // world) * world for m in micros)  # global samples used
                torch._foreach_zero_(Gacc)
                tab = net.tables()  # z is frozen for the whole step: binarize the tables once
                for m in micros:  # accumulate the raw hand-written gradient over micro-batches
                    local = ddp.shard(m, rank, world)
                    loss_t = self._accum_grad(net, enc_tr, local, y_tr[local], nglobal, Gacc, Gr3,
                                              tab, want_loss)
                for l, z in enumerate(net.z):  # one all-reduce, then the shared 0.5*cos(z) factor
                    ddp.all_reduce_(Gacc[l])
                    torch.cos(z.detach(), out=net._fb[l])
                    net._fb[l] *= 0.5
                    torch.mul(Gacc[l], net._fb[l], out=z.grad)
                opt.step()
            if device != "cpu":
                torch.cuda.synchronize(device)
            train_secs += time.perf_counter() - t0
            sched.step()

            # rank-0 val loss, broadcast to all -> identical early-stop decision (no DDP deadlock)
            vl = ddp.broadcast_float(self._val_loss(net, enc_va, y_va) if rank == 0 else 0.0)
            if vl < best - 1e-4:
                best, best_ep = vl, ep
                best_z = [z.detach().clone() for z in net.z]
            if want_loss:
                loss = float(loss_t) if loss_t is not None else loss   # the epoch's one host sync
                print(f"  epoch {ep + 1:3d}/{c['epochs']}  loss {loss:.3f}  val loss {vl:.4f}  "
                      f"(best {best:.4f} @ {best_ep + 1})  "
                      f"{(ep + 1) / (time.time() - t_all):.2f} ep/s", flush=True)
            if ep - best_ep >= c["patience"]:
                if rank == 0:
                    print(f"  early stop at epoch {ep + 1}: no gain since {best_ep + 1}", flush=True)
                break

        for z, bz in zip(net.z, best_z):
            z.copy_(bz)
        # hand the hard structure to LutModel: predict/scores/emit all read exactly these arrays
        return {"thresholds": net.thresholds, "layers": net.layers_np(), "train_seconds": train_secs}

    # ---- the DFA update ------------------------------------------------------------------------
    def _accum_grad(self, net: _Butterfly, enc_all: torch.Tensor, cols: torch.Tensor,
                    y: torch.Tensor, nglobal: int, Gacc: list, Gr3: torch.Tensor, tab: list,
                    want_loss: bool = True):
        """Add one micro-batch's raw DFA gradient into `Gacc`. No graph, no .backward().

        The error is normalised by the GLOBAL sample count `nglobal`, so summing these raw `G` over
        every micro-batch AND (in the caller) all-reducing across ranks reconstructs exactly the
        single-process full-batch gradient. The caller applies the shared 0.5*cos(z) factor and steps.

        Everything is carried TRANSPOSED -- (n_classes, B), (w, B) -- because that is the layout the
        scatter and the feedback matmul want, so nothing is transposed or copied on the hot path. The
        body layers share one matmul and one scatter_add; the readout's delta is `e` broadcast over
        each contiguous class group, an expand (stride 0), never a materialised (w, B) copy. The
        returned loss stays a device tensor: the caller syncs once per epoch, not once per step.
        """
        B = int(cols.shape[0])
        pl = net.plane(B, True)
        torch.index_select(enc_all, 1, cols, out=pl.enc)        # gather the images straight in place
        net.propagate(pl, tab)

        # the ONLY global quantity: the output error, broadcast from here to every layer at once
        e, ar = pl.e, net.ar(B)
        torch.div(net.votes_t(pl, out=pl.vot), net.tau, out=pl.vot)          # logits (n_classes, B)
        torch.softmax(pl.vot, 0, out=e)                                      # prob
        loss = -torch.log(e[y, ar] + 1e-12).mean() if want_loss else None    # read before mutating
        e.index_put_((y, ar), net._m1, accumulate=True)                      # e = prob - onehot(y)
        e /= nglobal                                            # mean over the GLOBAL batch

        # only the ACTIVE table entry of each gate gets gradient: G[i,p] += sum_b delta[b,i]
        if net.nb:
            torch.mm(net.Btb, e, out=pl.db)                     # (nb*w, nc) @ (nc, B) = delta^T
            Gacc[0].scatter_add_(1, pl.pb, pl.db)
        # readout: its own local gradient. dlogit_c/d bit_i = 1/tau for i in group c.
        torch.div(e, net.tau, out=pl.etau)
        Gr3.scatter_add_(2, pl.pr3, pl.etau3)
        return loss

    # ---- eval (torch forward; only used for the early-stopping loss) ---------------------------
    @torch.no_grad()
    def _val_loss(self, net: _Butterfly, enc: torch.Tensor, y: torch.Tensor) -> float:
        ch = self._chunk()
        tot = torch.zeros((), dtype=torch.float64, device=enc.device)
        tab = net.packed()
        for i in range(0, enc.shape[1], ch):
            yb = y[i : i + ch]
            nb = int(yb.shape[0])
            pl = net.plane(nb)
            pl.enc.copy_(enc[:, i : i + ch])
            net.propagate(pl, tab)
            logits = net.votes_t(pl) / net.tau                             # (n_classes, nb)
            logp = logits - torch.logsumexp(logits, 0, keepdim=True)
            tot += -logp[yb, net.ar(nb)].sum()
        return float(tot) / enc.shape[1]


def build(spec: DatasetSpec, **point) -> Dfa:
    return Dfa(spec, **point)
