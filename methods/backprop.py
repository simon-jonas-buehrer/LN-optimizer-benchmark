"""backprop: gradient descent learns each gate's truth table AND its wiring.

Four latent reals per gate -> STE on a sin -> a hard 4-bit truth table. Each input picks among 8
random candidate signals by a learnable logit (argmax forward, softmax gradient). The forward pass
is exact boolean, so the trained (thresholds, layers) go straight to LutModel. Trains to convergence
with early stopping on validation loss.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import ddp
from data import Dataset, DatasetSpec
from hw import even_thresholds
from methods.lut import LutModel, lut_sim

TITLE = "backprop (learned truth tables + learned wiring)"

# >=5 size points, targeting ~1k -> ~20M gates by pre-optimisation gate count (~sum of widths).
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


def points(spec: DatasetSpec) -> list[dict]:
    return [{"name": n, "bits": b, "widths": w, "epochs": 200} for n, (b, w) in _LADDER.items()]


def _t(a, device):
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _hard(z):
    return (torch.sin(z) > 0).to(z.dtype)


def _ste(z):
    sz = torch.sin(z)                       # one sin, not two (`_hard(z)` would recompute it)
    soft = 0.5 + 0.5 * sz
    return (sz > 0).to(z.dtype) + (soft - soft.detach())


def _gs_kernel(g, sig, cand, out_dtype):
    """dL/dsoft = sum_b g[b,i,j] * sig[b, cand[i,j,k]]  -- the hot kernel of the whole method.

    It touches (B, 2, W, cands) elements three times (gather, multiply, reduce over the batch). It
    is a standalone pure function precisely so `torch.compile` can fuse those three passes into one
    kernel that never materialises the candidate tensor at all; see `_compiled_gs`. The reduction
    accumulates in `out_dtype` (fp32) even when the activations are bf16.
    """
    x = sig.index_select(1, cand.reshape(-1)).view(sig.shape[0], *cand.shape)   # (B, 2, W, cands)
    return (g.unsqueeze(-1) * x).sum(0, dtype=out_dtype)


_GS = _gs_kernel                # swapped for the compiled twin by _worker when it is worth it
_COMPILED = None
# Cap on the transient (rows, 2, W, cands) gather inside one _gs_kernel call. The gather and the
# product it feeds are the single largest allocations in the whole trainer, and at the top of the
# ladder (W = 11e6, cands = 8) even ONE image is 176e6 elements -- so the batch has to be walked in
# slices or the 20M-gate tier cannot run at all. Splitting the batch only reorders the sum over b.
_GS_CHUNK = 2 ** 27


def _wire_reduce(g, sig, cand, conn):
    """dL/dconn for the straight-through wiring pick: contract the candidate signals against the
    incoming gradient (in batch slices small enough to fit), then push that through softmax."""
    b = sig.shape[0]
    rows = max(1, min(b, _GS_CHUNK // max(1, 2 * cand.shape[1] * cand.shape[2])))
    if rows >= b:
        gs = _GS(g, sig, cand, conn.dtype)
    else:
        gs = torch.zeros_like(conn)
        for i in range(0, b, rows):
            gs += _GS(g[i:i + rows], sig[i:i + rows], cand, conn.dtype)
    soft = torch.softmax(conn, -1)
    return soft * (gs - (gs * soft).sum(-1, keepdim=True))                       # softmax backward


def _compiled_gs(device, dtype):
    """`torch.compile`d `_gs_kernel`, built once per process, PROBED before it is trusted.

    `torch.compile(...)` itself never raises -- a missing or too-old Triton only surfaces on the
    first call, which would otherwise happen inside the autograd engine mid-training (exactly what
    the cluster's cu124 venv does). So compile it, run it once on a toy input on the real device and
    dtype, and fall back to the eager kernel for good if that probe fails.
    """
    global _COMPILED
    if _COMPILED is None:
        fn = torch.compile(_gs_kernel, dynamic=True)
        try:
            fn(torch.zeros(2, 2, 3, device=device, dtype=dtype),
               torch.zeros(2, 8, device=device, dtype=dtype),
               torch.zeros(2, 3, 4, dtype=torch.long, device=device),
               torch.float32)
            _COMPILED = fn
        except Exception as e:
            print(f"  (torch.compile unusable here: {type(e).__name__}: {str(e)[:120]} "
                  f"-- staying eager)", flush=True)
            _COMPILED = _gs_kernel
    return _COMPILED


class _WireGrad(torch.autograd.Function):
    """Straight-through gradient for the wiring logits. Identity on the forward value.

    It replaces the ``(sig[:, cand] * (soft - soft.detach())).sum(-1)`` term of the original layer,
    which is identically 0.0 (``soft - soft.detach()`` is exactly zero) and exists only so that
    ``conn`` receives ``dL/dsoft[i,j,k] = sum_b g[b,i,j] * sig[b, cand[i,j,k]]``. Written as an
    autograd Function that term costs *nothing* in the forward pass and never keeps the
    (B, 2, W, cands) candidate tensor alive between forward and backward: the gather happens once,
    in the backward, where the reduction needs it. Value, ``dL/dsig`` (exactly zero from this term)
    and ``dL/dconn`` are all bit-identical to the original expression -- with the eager
    ``_wire_reduce``; the compiled twin reorders the reduction and agrees to ~1e-6 relative.
    """

    @staticmethod
    def forward(ctx, picked, sig, conn, cand):
        # identity on `picked` -- adding an explicit zero tensor would only cost an allocation,
        # a fill and an add over (B, 2, W) per layer per step
        ctx.save_for_backward(sig, conn, cand)
        return picked

    @staticmethod
    def backward(ctx, g):
        sig, conn, cand = ctx.saved_tensors
        # `picked` passes its gradient straight on; this term's dL/dsig is exactly 0
        return g, None, _wire_reduce(g, sig, cand, conn), None


class _LutLayer(torch.nn.Module):
    def __init__(self, off, width, cands, g):
        super().__init__()
        self.register_buffer("cand", torch.randint(off, (2, width, cands), generator=g))
        self.conn = torch.nn.Parameter(torch.randn(2, width, cands, generator=g) * 0.1)
        self.table = torch.nn.Parameter(torch.randn(width, 4, generator=g))

    def wires(self):
        return self.cand.gather(2, self.conn.argmax(-1, keepdim=True)).squeeze(-1)

    def forward(self, sig):
        # Hard wiring: take the argmax candidate. The original form,
        #     x = sig[:, cand];  picked = (x * (onehot + (soft - soft.detach()))).sum(-1)
        # has the VALUE of `sig[:, wires]` (every non-argmax product is exactly 0.0), but it made
        # autograd push a full (B, 2, W, cands) gradient back through the gather -- an index
        # scatter-add over `cands` times more elements than there are gates -- and kept that tensor
        # alive for the backward. Splitting the two roles keeps value and gradients bit-identical:
        #   * `index_select` at `wires` carries the (B, 2, W) gradient to the source signals,
        #   * `_WireGrad` passes that through untouched and adds the wiring gradient for `conn`.
        w = self.wires()
        picked = sig.index_select(1, w.reshape(-1)).view(sig.shape[0], 2, w.shape[1])
        train = torch.is_grad_enabled() and self.conn.requires_grad
        if train:
            picked = _WireGrad.apply(picked, sig, self.conn, self.cand)
        xa, xb = picked[:, 0], picked[:, 1]
        c = _ste(self.table) if train else _hard(self.table)   # _ste == _hard under no_grad
        c = c.to(sig.dtype)          # exact: the table is 0.0/1.0, and so is every layer value
        f00, f01, f10, f11 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        return f00 + (f10 - f00) * xa + (f01 - f00) * xb + (f00 - f01 - f10 + f11) * xa * xb

    def truth_table(self):
        c = _hard(self.table).long()
        return c[:, 0] | (c[:, 1] << 1) | (c[:, 2] << 2) | (c[:, 3] << 3)


class _Net(torch.nn.Module):
    def __init__(self, spec, bits, widths, cands, seed):
        super().__init__()
        if widths[-1] % spec.n_classes:
            raise ValueError(f"readout {widths[-1]} not divisible by {spec.n_classes}")
        self.spec, self.bits, self.widths = spec, bits, widths
        self.thresholds = even_thresholds(bits)
        self.act = torch.float32   # activation dtype; see Backprop._act
        g = torch.Generator().manual_seed(seed)
        off = spec.n_pixels * bits
        self.layers = torch.nn.ModuleList()
        for w in widths:
            self.layers.append(_LutLayer(off, w, cands, g))
            off += w

    def encode(self, pix):
        t = torch.tensor(self.thresholds, device=pix.device, dtype=torch.int16)
        return (pix.to(torch.int16).unsqueeze(-1) > t).reshape(pix.shape[0], -1).float()

    def encode_all(self, pix_np, device, rows=4096):
        """Thermometer-encode a whole split ONCE, as uint8, resident on `device`.

        `encode` used to be re-run inside every micro-batch of every epoch -- a
        (B, n_pixels, bits) int16 temporary each time -- for a result that never changes. Caching
        it as uint8 (8x smaller than float32, so even bits=7 MNIST is ~300 MB) and casting only the
        tiny per-batch slice yields exactly the same values and consumes no RNG.
        """
        t = torch.tensor(self.thresholds, dtype=torch.int16)
        out = torch.empty((len(pix_np), self.spec.n_pixels * self.bits),
                          dtype=torch.uint8, device=device)
        for i in range(0, len(pix_np), rows):
            p = torch.from_numpy(np.ascontiguousarray(pix_np[i:i + rows]))
            b = (p.to(torch.int16).unsqueeze(-1) > t).reshape(p.shape[0], -1).to(torch.uint8)
            out[i:i + rows] = b.to(device)
        return out

    def forward(self, sig):
        """`sig`: the already thermometer-encoded input, (B, n_pixels*bits), in `self.act`.

        Every signal in this net is exactly 0.0 or 1.0 and every truth-table constant is 0 or 1, so
        the whole gate cascade is EXACT in bf16 too (all intermediates are integers in [-1, 2]).
        Only the readout popcount needs a wider accumulator, and the gradients -- which are real --
        are the only thing a low activation precision actually costs.
        """
        for layer in self.layers:
            sig = torch.cat([sig, layer(sig)], 1)
        last = sig[:, -self.widths[-1]:]
        return last.reshape(last.shape[0], self.spec.n_classes, -1).sum(-1, dtype=torch.float32) / \
            (self.widths[-1] // self.spec.n_classes) ** 0.5


def _accum_loss(out, tgt, shards, n_micros):
    """Loss of one (possibly fused) accumulation step == sum of the per-micro mean losses / n.

    One micro per step is literally the un-fused expression. k fused micros of equal size reach the
    same number with a single mean reduction; ragged micros (only possible when the split size is
    not a multiple of `micro`) fall back to explicit per-sample weights.
    """
    if len(shards) == 1:
        return F.cross_entropy(out, tgt) / n_micros
    n0 = shards[0].shape[0]
    if all(s.shape[0] == n0 for s in shards):
        return F.cross_entropy(out, tgt) * (len(shards) / n_micros)
    w = torch.cat([torch.full((s.shape[0],), 1.0 / (n_micros * s.shape[0]),
                              dtype=out.dtype, device=out.device) for s in shards])
    return (F.cross_entropy(out, tgt, reduction="none") * w).sum()


class Backprop(LutModel):
    def __init__(self, spec, bits, widths, epochs, lr=0.2, batch=128, cands=8, patience=40, micro=8):
        super().__init__(spec)
        # `micro` is the per-GPU micro-batch (kept small so even the 20M-gate net fits 24 GB); the
        # optimiser sees an accumulated global batch of >=32 (ddp.accum_plan). `batch` is unused now.
        self.cfg = dict(bits=bits, widths=tuple(widths), epochs=epochs, lr=lr, batch=batch,
                        cands=cands, patience=patience, micro=micro)

    # Activation precision. The gate cascade itself is exact in bf16 (all values are 0/1 and all
    # intermediates are integers in [-1, 2]) and the readout popcount accumulates in fp32, so bf16
    # a GIVEN net evaluates to exactly the same logits and validation loss in either precision --
    # it only halves the bytes moved by the dominant wiring-gradient gather/reduce, at the cost of
    # 8-bit-mantissa gradients (which do, over a run, train a slightly different net).
    # Parameters and the Adam state stay fp32 (master weights). "auto" = bf16 on CUDA, fp32 on CPU
    # (CPU bf16 is emulated and slower). Override with MNISTBENCH_BACKPROP_ACT=fp32|bf16|auto.
    ACT = "auto"

    # torch.compile of _gs_kernel fuses its three passes over the (B, 2, W, cands) candidate
    # tensor into one kernel, but costs ~10 s of compilation per process. "auto" pays that only
    # when the run is long enough for it to be noise (the real ladder is 200 epochs x hundreds of
    # steps; a 3-epoch smoke is not). Override with MNISTBENCH_BACKPROP_COMPILE=0|1|auto.
    COMPILE = "auto"
    # Measured break-even: compiling costs ~10 s and saves ~1.4 ms per million candidate elements
    # reduced, so it pays once a run will push ~8e9 of them through _gs_kernel. The 200-epoch
    # ladder is 1e11..1e15; a 3-epoch smoke is ~5e8, and stays eager.
    COMPILE_MIN_WORK = 8e9

    def _use_compile(self, work):
        want = os.environ.get("MNISTBENCH_BACKPROP_COMPILE", self.COMPILE)
        return work >= self.COMPILE_MIN_WORK if want == "auto" else want not in ("0", "off")

    def _act(self, device):
        want = os.environ.get("MNISTBENCH_BACKPROP_ACT", self.ACT)
        if want == "auto":
            want = "bf16" if torch.device(device).type == "cuda" else "fp32"
        return {"fp32": torch.float32, "bf16": torch.bfloat16}[want]

    def _budget(self):
        """Images whose widest (2, W, cands) candidate tensor still fits the ~1 GiB working budget.

        Counted in ELEMENTS, not bytes, so a bf16 run keeps exactly the fp32 batching and simply
        occupies half the memory -- bf16 must never be the thing that makes a tier stop fitting.
        """
        return max(1, 2 ** 28 // (2 * max(self.cfg["widths"]) * self.cfg["cands"]))

    def _chunk(self):
        """Validation batch. Under `no_grad` the layer never builds a (B, 2, W, cands) tensor any
        more (see `_LutLayer.forward`), so the eval working set is just the signal buffer
        (B, n_sig); size the chunk against THAT instead of against the training budget. Bigger
        chunks = far fewer launches per epoch on the wide tiers (l: 104 -> ~950, m: 645 -> 2048).
        """
        n_sig = self.spec.n_pixels * self.cfg["bits"] + sum(self.cfg["widths"])
        return max(64, min(2048, 2 ** 28 // (2 * n_sig)))

    def _fuse(self, step, accum):
        """How many accumulation micro-steps to run in ONE forward/backward.

        Summing k micro-steps into a single batched forward accumulates the same gradient (only the
        float reduction order differs) while paying the per-step launch overhead, the
        softmax/argmax/STE over the whole parameter set and the layer concatenations once instead
        of k times. Capped by the same working-set budget `_chunk` uses, so the biggest tiers keep
        their `micro`-sized footprint -- there `fuse == 1`, i.e. byte-for-byte the un-fused path.
        """
        f = max(1, min(accum, self._budget() // step))
        cap = int(os.environ.get("MNISTBENCH_BACKPROP_FUSE", "0"))   # 0 = auto (the default)
        return f if cap <= 0 else max(1, min(f, cap))   # =1 -> the bit-exact un-fused path

    def train(self, data: Dataset, *, device="cpu", seed=0):
        # Multi-GPU is opt-in via `ddp_gpus` (set by the harness). ddp.launch runs `_worker` on one
        # process per GPU; with <=1 GPU it runs inline -> this path is byte-for-byte the old trainer.
        self._data, self._seed, self._base_device = data, seed, device
        res = ddp.launch(self._worker, getattr(self, "ddp_gpus", 1))
        self.thresholds, self.layers = res["thresholds"], res["layers"]
        self.train_seconds = res["train_seconds"]  # pure training time (no val/measure), for the json
        self.train_samples = res["train_samples"]  # training-example evals until early stop
        self._counts_memo = None
        self._data = None  # only needed to reach the worker; don't pin the dataset afterwards

    def _worker(self, rank, world):
        data, seed, c = self._data, self._seed, self.cfg
        device = self._base_device if world == 1 else f"cuda:{rank}"
        torch.manual_seed(seed)  # identical seed on every rank -> identical init AND identical perm
        m = _Net(self.spec, c["bits"], c["widths"], c["cands"], seed).to(device)
        m.act = self._act(device)
        train_m = DDP(m, device_ids=[rank]) if world > 1 else m  # DDP averages grads across ranks
        opt = torch.optim.Adam(m.parameters(), lr=c["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])
        # thermometer-encode both splits once, on-device, instead of once per micro-batch per epoch
        ex, y = m.encode_all(data.train_x, device), _t(data.train_y, device)
        ev, vy = m.encode_all(data.val_x, device), _t(data.val_y, device)
        n_train, n_val = ex.shape[0], ev.shape[0]
        ch = self._chunk()
        # fixed protocol for every size of this method: per-GPU micro-batch `micro` (small, fits the
        # 20M-gate net), accumulated to the method's global batch `batch` -- constant across sizes.
        step, accum, _ = ddp.accum_plan(c["micro"], world, c["batch"])
        fuse = self._fuse(step, accum)
        global _GS   # set explicitly every run, so one process can train several models
        work = (c["epochs"] * -(-n_train // (step * accum))        # optimiser steps
                * (step * accum // world) * 2 * c["cands"] * sum(c["widths"]))   # elems per step
        _GS = _compiled_gs(device, m.act) if self._use_compile(work) else _gs_kernel
        best, best_state, best_ep, train_secs, nseen = float("inf"), None, 0, 0.0, 0
        for ep in range(c["epochs"]):
            perm = torch.randperm(n_train, device=device)  # same on every rank (same seed)
            t0 = time.perf_counter()
            for i in range(0, n_train, step * accum):
                micros = [perm[i + a * step:i + (a + 1) * step] for a in range(accum)]
                micros = [mb for mb in micros if mb.shape[0] >= world]  # deterministic across ranks
                if not micros:
                    continue
                opt.zero_grad(set_to_none=True)
                groups = [micros[a:a + fuse] for a in range(0, len(micros), fuse)]
                for j, grp in enumerate(groups):
                    shards = [ddp.shard(mb, rank, world) for mb in grp]  # ~micro per rank each
                    local = shards[0] if len(shards) == 1 else torch.cat(shards)
                    sync = nullcontext() if (world == 1 or j == len(groups) - 1) else train_m.no_sync()
                    with sync:  # DDP all-reduces once, on the last (fused) step of the accumulation
                        _accum_loss(train_m(ex[local].to(m.act)), y[local], shards,
                                    len(micros)).backward()
                opt.step()
            if ex.is_cuda:
                torch.cuda.synchronize(device)
            train_secs += time.perf_counter() - t0
            nseen += n_train  # training-example forward passes this epoch (samples looked at)
            sched.step()
            # early stop on val LOSS (forward is already hard = the circuit). Compute it on rank 0
            # only and broadcast, so every rank breaks on the SAME number (GPU reductions differ in
            # the last bits across devices, and a diverging break would deadlock DDP).
            if rank == 0:
                with torch.no_grad():  # no_grad also drops the wiring-gradient term in _LutLayer
                    parts = torch.stack([
                        F.cross_entropy(m(ev[i:i + ch].to(m.act)), vy[i:i + ch], reduction="sum")
                        for i in range(0, n_val, ch)])
                # ONE device->host copy per epoch, then the same python summation order the
                # per-chunk `.item()` loop used -> bit-identical val loss without a sync per chunk
                vl = sum(float(v) for v in parts.cpu()) / n_val
            else:
                vl = 0.0
            vl = ddp.broadcast_float(vl)
            if vl < best - 1e-4:
                best, best_ep = vl, ep
                # snapshot into pre-allocated buffers (copy_ instead of a fresh clone per epoch);
                # the `cand` buffers are constant, so only the parameters need saving
                if best_state is None:
                    best_state = [p.detach().clone() for p in m.parameters()]
                else:
                    with torch.no_grad():
                        for dst, p in zip(best_state, m.parameters()):
                            dst.copy_(p.detach())
            if rank == 0 and (ep % 5 == 0 or ep == c["epochs"] - 1):
                print(f"  epoch {ep + 1:3d}/{c['epochs']}  val loss {vl:.4f}  (best {best:.4f} @ {best_ep + 1})",
                      flush=True)
            if ep - best_ep >= c["patience"]:
                if rank == 0:
                    print(f"  early stop at epoch {ep + 1}", flush=True)
                break
        if best_state is not None:
            with torch.no_grad():
                for p, b in zip(m.parameters(), best_state):
                    p.copy_(b)
        with torch.no_grad():   # one wires() per layer, not two (it argmaxes the whole conn tensor)
            wires = [lay.wires().cpu().numpy() for lay in m.layers]
            layers = [(w[0], w[1], lay.truth_table().cpu().numpy())
                      for w, lay in zip(wires, m.layers)]
        return {"thresholds": m.thresholds, "train_seconds": train_secs, "train_samples": nseen,
                "layers": layers}

    def _counts(self, pix: np.ndarray) -> np.ndarray:
        """On a CUDA run, evaluate through lut_sim's GPU backend (bit-for-bit equal to numpy).

        The repeat-call memo itself lives in LutModel._counts; this only picks the backend, and
        reuses that same memo shape so a CUDA run gets the caching too.
        """
        dev = str(getattr(self, "_base_device", "cpu"))
        if dev == "cpu" or not torch.cuda.is_available():
            return super()._counts(pix)
        memo = self._counts_memo
        if memo is not None:
            p, thr, lay, out = memo
            if (p is pix and thr is self.thresholds and len(lay) == len(self.layers)
                    and all(a is b for a, b in zip(lay, self.layers))):
                return out
        out = lut_sim(self.thresholds, self.layers, pix, self.spec, device=dev)
        self._counts_memo = (pix, self.thresholds, list(self.layers), out)
        return out


def build(spec, **point) -> Backprop:
    return Backprop(spec, **point)
