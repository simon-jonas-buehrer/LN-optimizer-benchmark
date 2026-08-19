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

Implementation notes (pure plumbing -- the maths above is untouched):
  * the whole forward runs in uint8: a gate is `(tt >> p) & 1` on a per-gate 4-bit packed truth
    table, so no (w, B) int64 temporary is ever built for the evaluation itself;
  * the signal plane and the per-layer pattern planes are allocated once per batch shape and reused
    (`_Plane`), and the patterns computed by the forward are handed to the gradient instead of being
    recomputed;
  * every source-index vector that happens to be an arithmetic progression -- which is *every* body
    layer's `a` tap, since the butterfly reads j from the layer right below -- degenerates to a
    strided view of the signal plane instead of a gather;
  * error/loss stay on the device: one host sync per epoch, not one per micro-batch.
"""

from __future__ import annotations

import time

import numpy as np
import torch

import ddp
from data import Dataset, DatasetSpec
from hw import even_thresholds
from methods.lut import LutModel, lut_sim

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
    tap is `j + base` (step 1) and the encoder tap is often `2j` (step 2), so this removes roughly
    half of the butterfly's gathers without touching which signals are read.
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


def _encode(pix: torch.Tensor, thresholds) -> torch.Tensor:
    """(N, n_pixels) uint8 -> (n_in, N) uint8, byte-major bit p*k+j, matching hw.emit_thermometer."""
    thr = torch.tensor(thresholds, device=pix.device, dtype=torch.int16)
    bits = pix.to(torch.int16).unsqueeze(-1) > thr          # (N, n_pixels, bits)
    return bits.reshape(pix.shape[0], -1).T.contiguous().to(torch.uint8)


class _Plane:
    """Reusable buffers for one batch width: the signal plane + each layer's active-pattern plane.

    `acts` is written end-to-end every forward (encoder rows, then one slice per layer), so it never
    needs zeroing; `p[l]` is the (w, B) uint8 pattern the forward already computed, handed straight
    to the gradient; `pl[l]` is its int64 twin, allocated only on the training path because
    `scatter_add_` insists on int64 indices.
    """

    __slots__ = ("acts", "p", "pl")

    def __init__(self, net: "_Butterfly", B: int, grad: bool) -> None:
        dev = net.device
        self.acts = torch.empty((net.n_sig, B), dtype=torch.uint8, device=dev)
        self.p = [torch.empty((w, B), dtype=torch.uint8, device=dev) for w in net.widths]
        self.pl = ([torch.empty((w, B), dtype=torch.int64, device=dev) for w in net.widths]
                   if grad else None)


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
        self.tau = (readout // spec.n_classes) ** 0.5
        self.rg = readout // spec.n_classes      # readout gates per class (contiguous groups)

        # fixed wiring: srcs[l] = (2, w) GLOBAL ids; layer l reads only layer l-1 (or the encoder)
        self.offs = [self.n_in]
        in_dim, in_base = self.n_in, 0
        self.srcs: list[torch.Tensor] = []
        self.taps: list[tuple] = []
        for l, w in enumerate(self.widths):
            bf = _butterfly_src(in_dim, w, l - 1).contiguous()   # (2, w) local into in_dim
            gl = bf + in_base                                    # global ids, still on the host
            self.taps.append((_tap(gl[0]), _tap(gl[1])))
            self.srcs.append(gl.to(device))
            in_base = self.offs[-1]                              # next layer reads this layer's outs
            self.offs.append(self.offs[-1] + w)
            in_dim = w
        self.taps = [(a if isinstance(a, slice) else a.to(device),
                      b if isinstance(b, slice) else b.to(device)) for a, b in self.taps]

        # learned: the only parameters in the whole record
        self.z = [
            (_RES_SIGN.to(device) * (torch.pi / 4)).expand(w, 4).contiguous()
            + torch.randn(w, 4, generator=g, device=device) * _JITTER
            for w in self.widths
        ]
        # fixed random feedback: the backward "model". Never learned, never synthesized.
        self.B = [
            torch.randn(spec.n_classes, w, generator=g, device=device) / spec.n_classes ** 0.5
            for w in self.widths[:-1]
        ]
        # the update only ever needs B_l^T (w, n_classes): delta^T = B_l^T e^T, one matmul, no copy
        self.Bt = [b.t().contiguous() for b in self.B]

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

    def tables(self) -> list[torch.Tensor]:
        return [hard_bit(z) for z in self.z]

    def packed(self) -> list[torch.Tensor]:
        """Per-layer (w, 1) uint8 with bit p = T[p]: the forward's whole truth table in one byte."""
        return [((torch.sin(z) > 0).to(torch.uint8) * self._p2).sum(1, keepdim=True,
                                                                    dtype=torch.uint8)
                for z in self.z]

    def propagate(self, pl: _Plane, tt: list[torch.Tensor]) -> torch.Tensor:
        """`pl.acts[:n_in]` is already loaded; fill the rest. Exact bits; no relaxation anywhere."""
        acts = pl.acts
        for l, (ta, tb) in enumerate(self.taps):
            p = pl.p[l]
            torch.bitwise_left_shift(acts[ta], 1, out=p)         # p = 2a + b, in {0,1,2,3}
            torch.bitwise_or(p, acts[tb], out=p)
            o = acts[self.offs[l] : self.offs[l + 1]]
            torch.bitwise_right_shift(tt[l], p, out=o)           # T[p], straight into the plane
            torch.bitwise_and(o, 1, out=o)
        return acts

    def forward(self, enc: torch.Tensor, tt: list[torch.Tensor] | None = None) -> torch.Tensor:
        """enc (n_in, B) uint8 -> acts (n_sig, B) uint8."""
        pl = self.plane(enc.shape[1])
        pl.acts[: self.n_in].copy_(enc)
        return self.propagate(pl, self.packed() if tt is None else tt)

    def votes_t(self, acts: torch.Tensor) -> torch.Tensor:
        """(n_classes, B) float32 group popcounts -- exact (a group is < 2^24 bits)."""
        out = acts[self.offs[-2] : self.offs[-1]]                # (R, B)
        return out.reshape(self.spec.n_classes, -1, out.shape[1]).sum(1, dtype=torch.float32)

    def votes(self, acts: torch.Tensor) -> torch.Tensor:
        return self.votes_t(acts).T

    def layers_np(self) -> list:
        """Extract the hard tables + fixed wiring as np.int64 (idx_a, idx_b, tt) for LutModel."""
        out = []
        for s, t in zip(self.srcs, self.tables()):
            tt = t[:, 0] | (t[:, 1] << 1) | (t[:, 2] << 2) | (t[:, 3] << 3)   # bit p = T[p]
            out.append((s[0].cpu().numpy().astype(np.int64),
                        s[1].cpu().numpy().astype(np.int64),
                        tt.cpu().numpy().astype(np.int64)))
        return out


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
        self._sim_device = None

    def _chunk(self) -> int:
        c = self.cfg
        n_sig = self.spec.n_pixels * c["bits"] + c["width"] * c["layers"] + c["readout"]
        return max(64, min(4096, 2 ** 28 // n_sig))

    # ---- inference: memoized, and on the GPU when we trained on one ----------------------------
    def _counts(self, pix: np.ndarray) -> np.ndarray:
        """The harness asks for predict(val_x) then scores(val_x): simulate that net once, not twice.

        Keyed on the array's identity (a hard reference is kept so the id cannot be recycled) and its
        shape; the counts are a pure function of (thresholds, layers, pix), which are frozen by the
        time the harness starts measuring.
        """
        key = (id(pix), pix.shape)
        if getattr(self, "_cnt_key", None) == key:
            return self._cnt_val
        val = lut_sim(self.thresholds, self.layers, pix, self.spec, device=self._sim_device)
        self._cnt_key, self._cnt_pix, self._cnt_val = key, pix, val
        return val

    def train(self, data: Dataset, *, device: str = "cpu", seed: int = 0) -> None:
        # Multi-GPU is opt-in via `ddp_gpus` (set by the harness). <=1 GPU runs `_worker` inline, so
        # this path is byte-for-byte the old DFA trainer; >1 shards each batch across ranks and
        # all-reduces the hand-written gradient (below), which is exactly the full-batch gradient.
        self._data, self._seed, self._base_device = data, seed, device
        res = ddp.launch(self._worker, getattr(self, "ddp_gpus", 1))
        self.thresholds, self.layers = res["thresholds"], res["layers"]
        self.train_seconds = res["train_seconds"]  # pure training time (no val/measure)
        self._data = None                          # don't pin the dataset in the checkpointed model
        # the exact packed simulator has a bit-identical CUDA twin; use it when we have a GPU
        if str(device) != "cpu" and torch.cuda.is_available():
            self._sim_device = device

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

        # persistent scratch: the raw per-layer gradient accumulator, the 0.5*cos(z) factor, and the
        # grad tensors Adam reads. Allocated once instead of once per optimiser step (the readout
        # tier is 80M floats a piece).
        Gacc = [torch.zeros(w, 4, device=device) for w in net.widths]
        cosb = [torch.empty(w, 4, device=device) for w in net.widths]
        for z in net.z:
            z.grad = torch.empty_like(z)

        n = enc_tr.shape[1]
        step, accum, _ = ddp.accum_plan(c["micro"], world, c["batch"])  # fixed global batch/size
        best, best_z, best_ep = float("inf"), [z.detach().clone() for z in net.z], 0
        train_secs, t_all, loss_t = 0.0, time.time(), None
        loss = float("nan")
        for ep in range(c["epochs"]):
            # the loss is a print-only quantity: compute it on the epochs that print it, and nowhere
            # else (same printed numbers, four fewer kernels on every one of the other micro-batches)
            want_loss = rank == 0 and (ep % 5 == 0 or ep == c["epochs"] - 1)
            perm = torch.randperm(n, generator=g, device=device)  # identical on every rank
            t0 = time.perf_counter()
            for i in range(0, n, step * accum):
                micros = [perm[i + a * step:i + (a + 1) * step] for a in range(accum)]
                micros = [m for m in micros if m.shape[0] >= world]  # deterministic across ranks
                if not micros:
                    continue
                nglobal = sum((m.shape[0] // world) * world for m in micros)  # global samples used
                torch._foreach_zero_(Gacc)
                tt = net.packed()   # z is frozen for the whole step: binarize the tables once
                for m in micros:  # accumulate the raw hand-written gradient over micro-batches
                    local = ddp.shard(m, rank, world)
                    loss_t = self._accum_grad(net, enc_tr, local, y_tr[local], nglobal, Gacc, tt,
                                              want_loss)
                for l, z in enumerate(net.z):  # one all-reduce, then the shared 0.5*cos(z) factor
                    ddp.all_reduce_(Gacc[l])
                    torch.cos(z, out=cosb[l])
                    cosb[l] *= 0.5
                    torch.mul(Gacc[l], cosb[l], out=z.grad)
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
            if rank == 0 and (ep % 5 == 0 or ep == c["epochs"] - 1):
                loss = float(loss_t) if loss_t is not None else loss   # the epoch's only host sync
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
                    y: torch.Tensor, nglobal: int, Gacc: list, tt: list,
                    want_loss: bool = True):
        """Add one micro-batch's raw DFA gradient into `Gacc` (per layer). No graph, no .backward().

        The error is normalised by the GLOBAL sample count `nglobal`, so summing these raw `G` over
        every micro-batch AND (in the caller) all-reducing across ranks reconstructs exactly the
        single-process full-batch gradient. The caller applies the shared 0.5*cos(z) factor and steps.

        Everything is carried TRANSPOSED (n_classes, B) / (w, B): that is the layout the scatter and
        the feedback matmul want, so no transpose or contiguous copy happens on the hot path. The
        returned loss stays a device tensor -- the caller syncs once per epoch, not once per step.
        """
        B = int(cols.shape[0])
        nc = self.spec.n_classes
        pl = net.plane(B, True)
        torch.index_select(enc_all, 1, cols, out=pl.acts[: net.n_in])   # gather straight into place
        acts = net.propagate(pl, tt)

        # the ONLY global quantity: the output error, broadcast from here to every layer at once
        logits = net.votes_t(acts) / net.tau                    # (n_classes, B)
        prob = torch.softmax(logits, 0)
        ar = net.ar(B)
        loss = -torch.log(prob[y, ar] + 1e-12).mean() if want_loss else None
        e = prob.clone()
        e.index_put_((y, ar), net._m1, accumulate=True)          # e = prob - onehot(y)
        e /= nglobal                                            # mean over the GLOBAL batch

        last = len(net.widths) - 1
        for l, w in enumerate(net.widths):
            idx = pl.pl[l]
            idx.copy_(pl.p[l])                                  # scatter wants int64 indices
            if l == last:
                # readout: its own local gradient. dlogit_c/d bit_i = 1/tau for i in group c.
                # class(i) = i // rg, so delta^T is just e^T broadcast over each contiguous group --
                # an expand (stride 0), never a materialised (w, B) copy.
                src = (e / net.tau).unsqueeze(1).expand(nc, net.rg, B)
                Gacc[l].view(nc, net.rg, 4).scatter_add_(2, idx.view(nc, net.rg, B), src)
            else:
                # only the ACTIVE table entry of each gate gets gradient: G[i,p] += sum_b delta[b,i]
                Gacc[l].scatter_add_(1, idx, net.Bt[l] @ e)     # (w, nc) @ (nc, B) = delta^T
        return loss

    # ---- eval (torch forward; only used for the early-stopping loss) ---------------------------
    @torch.no_grad()
    def _val_loss(self, net: _Butterfly, enc: torch.Tensor, y: torch.Tensor) -> float:
        ch = self._chunk()
        tot = torch.zeros((), dtype=torch.float64, device=enc.device)
        tt = net.packed()
        for i in range(0, enc.shape[1], ch):
            yb = y[i : i + ch]
            nb = int(yb.shape[0])
            pl = net.plane(nb)
            pl.acts[: net.n_in].copy_(enc[:, i : i + ch])
            logits = net.votes_t(net.propagate(pl, tt)) / net.tau      # (n_classes, nb)
            logp = logits - torch.logsumexp(logits, 0, keepdim=True)
            tot += -logp[yb, net.ar(nb)].sum()
        return float(tot) / enc.shape[1]


def build(spec: DatasetSpec, **point) -> Dfa:
    return Dfa(spec, **point)
