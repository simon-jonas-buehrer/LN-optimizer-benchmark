"""Quantized-MLP references: shared QAT core for w1.58a4 / w1.58a8 / w4a4.

Conventional dense MLPs put on the same gate axis as the logic nets. Weights are ternary ({-1,0,+1},
"w1.58") or signed 4-bit ("w4"); activations are unsigned 4- or 8-bit. Training is quantization-aware
(fake-quant + STE); at export the model becomes a list of `hw.QLayer` (integer weights/bias + a fixed
requant mul/shift per layer), and predict()/scores()/emit() all derive from that SAME list via
`hw.qmlp_forward` / `hw.emit_quant_mlp`, so the circuit equals predict() bit for bit. Early stopping
on validation loss.

`variant(wmode, abits)` returns (TITLE, points, build); the three thin modules w1_58a4/w1_58a8/w4a4
just call it, so each is one series with its own results folder.

Speed notes (nothing below changes what is computed):
  * inference (`predict`/`scores`) runs the SAME integer recurrence as `hw.qmlp_forward`, but the
    MACs go through a BLAS GEMM in a float dtype that is provably exact for the value range of that
    layer (`_gemm_dtype`); numpy has no BLAS path for int64, so this is worth ~20-50x. It falls back
    to `hw.qmlp_forward` whenever exactness is not provable.
  * training caches the fake-quantized weights/bias for the duration of one optimiser step (they are
    constant across the gradient-accumulation micro-batches) and re-attaches the straight-through
    gradient with `_QW`, which reproduces the autograd of `_quant_w`/`_ste_round` exactly.
"""

from __future__ import annotations

import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import ddp
from data import Dataset, DatasetSpec
from hw import QLayer, emit_quant_mlp, input_activations, qmlp_forward, requantize

_SH = 16  # requant fixed-point shift; mul = round(scale * 2^_SH)

# >=5 sizes by hidden-layer widths. Quantized arithmetic is dense, so these land high on the gate
# axis (they do not reach ~1k gates -- that floor is a finding, not forced).
_LADDER = {
    "xs": (64,),
    "s": (256,),
    "m": (512, 512),
    "l": (1024, 1024),
    "xl": (2048, 2048, 2048),
    "xxl": (4096, 4096, 4096),
}


def _t(a, device):
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _ste_round(x):
    return x + (x.round() - x).detach()


def _quant_w(Wf, wmode):
    """Fake-quant weights with STE. ternary -> {-1,0,1} (TWN); int4 -> {-8..7}."""
    if wmode == "ternary":
        delta = 0.7 * Wf.abs().mean(0, keepdim=True)
        hard = torch.where(Wf > delta, 1.0, torch.where(Wf < -delta, -1.0, 0.0))
        return Wf + (hard - Wf).detach()
    hard = torch.clamp(_ste_round(Wf), -8, 7)  # int4
    return hard


def _quant_w_ste(Wf, wmode):
    """Value + straight-through mask of `_quant_w`, computed once (no autograd graph).

    The value is produced by the *same* float expression as `_quant_w`, so it is bit-identical.
    `mask` is what `_quant_w`'s backward multiplies the incoming gradient by: identity (None) for
    ternary, and the clamp's pass-through region for int4 (`clamp` sends 0 where its input is
    outside [-8, 7], and the STE round in front of it has gradient 1).
    """
    with torch.no_grad():
        if wmode == "ternary":
            delta = 0.7 * Wf.abs().mean(0, keepdim=True)
            hard = torch.where(Wf > delta, 1.0, torch.where(Wf < -delta, -1.0, 0.0))
            return Wf + (hard - Wf), None
        pre = Wf + (Wf.round() - Wf)  # == _ste_round(Wf) forward value
        return torch.clamp(pre, -8, 7), (pre >= -8) & (pre <= 7)


class _QW(torch.autograd.Function):
    """Attach a straight-through gradient to a pre-computed fake-quantized constant.

    `_QW.apply(p, q, mask)` has the forward value of `q` and the backward of the expression `q` was
    computed from (`grad` or `grad * mask` w.r.t. `p`) -- i.e. exactly `_quant_w(p, ...)` /
    `_ste_round(p)`, minus the per-call recomputation of the elementwise chain.
    """

    @staticmethod
    def forward(ctx, p, q, mask):
        ctx.save_for_backward(mask)
        return q.view_as(q)  # fresh tensor object; shares q's storage, no copy

    @staticmethod
    def backward(ctx, g):
        (mask,) = ctx.saved_tensors
        return (g if mask is None else g * mask), None, None


class _Layer(torch.nn.Module):
    def __init__(self, n_in, n_out, wmode, abits, final, g):
        super().__init__()
        self.wmode, self.abits, self.final = wmode, abits, final
        self.W = torch.nn.Parameter(torch.randn(n_in, n_out, generator=g) / n_in ** 0.5)
        self.b = torch.nn.Parameter(torch.zeros(n_out))
        self.log_s = torch.nn.Parameter(torch.zeros(()))  # requant scale (log), non-final only
        self._q = None  # (Wq, wmask, br) cache, valid until the parameters change

    def invalidate(self):
        self._q = None

    def _cache(self):
        q = self._q
        if q is None:
            Wq, wmask = _quant_w_ste(self.W, self.wmode)
            with torch.no_grad():
                br = self.b + (self.b.round() - self.b)  # == _ste_round(self.b) forward value
            q = self._q = (Wq, wmask, br)
        return q

    def forward(self, a):
        Wq, wmask, br = self._cache()
        acc = a @ _QW.apply(self.W, Wq, wmask) + _QW.apply(self.b, br, None)
        if self.final:
            return acc
        y = torch.clamp(_ste_round(acc * self.log_s.exp()), 0, (1 << self.abits) - 1)
        return y

    def export(self) -> QLayer:
        with torch.no_grad():
            Wq = _quant_w(self.W, self.wmode).cpu().numpy().round().astype(np.int64)
            b = _ste_round(self.b).cpu().numpy().round().astype(np.int64)
            if self.final:
                return QLayer(Wq, b, 0, 0, 0, final=True)
            mul = max(1, int(round(float(self.log_s.exp()) * (1 << _SH))))
        return QLayer(Wq, b, mul, _SH, self.abits, final=False)


# ================================================================================================
# Exact integer inference through BLAS
# ================================================================================================

_F32_SAFE = 1 << 23   # every partial sum stays an exactly representable float32 integer
_F64_SAFE = 1 << 52   # ... float64

_CHUNK = 4096         # rows per pass, to bound the activation buffers


def _gemm_dtype(layers, in_amax):
    """Smallest float dtype in which every layer's MAC (and bias add) is exact, else None.

    All operands are integers; a float GEMM is exact as long as every partial sum is an integer
    below 2^mantissa. `sum_i |W_ij| * amax + |b_j|` bounds every partial sum of column j, whatever
    order (or FMA/blocking) BLAS uses.
    """
    dt, amax = np.float32, int(in_amax)
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


def _qmlp_forward_fast(x0, layers, in_amax, cache):
    """Bit-identical to `hw.qmlp_forward`, with the MACs in BLAS. `cache` is a per-model dict."""
    prep = cache.get("prep")
    if prep is None or prep[0] is not layers:
        dt = _gemm_dtype(layers, in_amax)
        prep = (layers, dt, None if dt is None else
                [(np.ascontiguousarray(L.Wq, dt), L.bias.astype(dt)) for L in layers])
        cache["prep"] = prep
    _, dt, packed = prep
    if dt is None:  # provably-exact float GEMM not available -> the reference int64 path
        return qmlp_forward(x0, layers)

    out = np.empty((len(x0), layers[-1].Wq.shape[1]), np.int64)
    for i in range(0, len(x0), _CHUNK):
        x = np.ascontiguousarray(x0[i:i + _CHUNK], dt)
        for L, (Wf, bf) in zip(layers, packed):
            acc = x @ Wf
            acc += bf
            if L.final:
                out[i:i + _CHUNK] = acc.astype(np.int64)
            else:
                x = requantize(acc.astype(np.int64), L.mul, L.sh, L.out_abits).astype(dt)
    return out


class QuantModel:
    """emit / predict / scores / save from an exported hw.QLayer list + input activation bits."""

    def __init__(self, spec: DatasetSpec, wmode: str, abits: int, hidden, epochs=200, lr=0.01,
                 batch=128, patience=20, micro=32):
        self.spec, self.wmode, self.abits, self.hidden = spec, wmode, abits, tuple(hidden)
        # `micro` = per-GPU micro-batch, accumulated to the fixed global `batch` (same across sizes).
        self.cfg = dict(epochs=epochs, lr=lr, batch=batch, patience=patience, micro=micro)
        self.layers: list[QLayer] = []
        self._gemm: dict = {}     # exported-weight GEMM operands, rebuilt when self.layers changes
        self._logits: dict = {}   # {id(pix): (pix, logits)} -- the harness scores the same arrays twice

    def _in(self, pix):
        return input_activations(pix, self.spec, self.abits)

    def _in_np(self, pix, dtype=None):
        """`input_activations` without the int64 temporary.

        Pixels are non-negative, so `pix >> k` in the input dtype has exactly the values of
        `pix.astype(int64) >> k`; only the (wider) container differs.
        """
        if not 1 <= self.abits <= self.spec.pixel_bits:
            raise ValueError(f"in_abits {self.abits} out of range 1..{self.spec.pixel_bits}")
        a = np.right_shift(pix, self.spec.pixel_bits - self.abits)
        return a if dtype is None else a.astype(dtype, copy=False)

    def emit_verilog(self) -> str:
        return emit_quant_mlp(self.layers, self.spec, self.abits)

    def _forward(self, pix):
        """Integer logits for `pix`, memoised on the identity of the array handed in.

        The harness runs `predict(val_x)` and `scores(val_x)` back to back; the cache keeps a strong
        reference to the keyed array, so an id can never be recycled under it.
        """
        hit = self._logits.get(id(pix))
        if hit is not None and hit[0] is pix:
            return hit[1]
        logits = _qmlp_forward_fast(self._in_np(pix), self.layers,
                                    (1 << self.abits) - 1, self._gemm)
        self._logits = {id(pix): (pix, logits)}
        return logits

    def predict(self, pix):
        return self._forward(pix).argmax(1)

    def scores(self, pix):
        return self._forward(pix).astype(float)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"abits": self.abits, "wmode": self.wmode,
                         "layers": [(L.Wq, L.bias, L.mul, L.sh, L.out_abits, L.final)
                                    for L in self.layers]}, f)

    def train(self, data: Dataset, *, device="cpu", seed=0):
        # Multi-GPU is opt-in via `ddp_gpus` (set by the harness). <=1 GPU runs `_worker` inline, so
        # this path stays byte-for-byte the old trainer; >1 spawns one DDP rank per GPU.
        self._data, self._seed, self._base_device = data, seed, device
        res = ddp.launch(self._worker, getattr(self, "ddp_gpus", 1))
        self.layers, self.train_seconds = res["layers"], res["train_seconds"]
        self._gemm, self._logits = {}, {}

    def _worker(self, rank, world):
        data, seed, c = self._data, self._seed, self.cfg
        device = self._base_device if world == 1 else f"cuda:{rank}"
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)  # CPU generator -> identical init on every rank
        dims = [self.spec.n_pixels, *self.hidden, self.spec.n_classes]
        finals = [False] * (len(dims) - 2) + [True]
        net = torch.nn.Sequential(
            *[_Layer(dims[i], dims[i + 1], self.wmode, self.abits, finals[i], g)
              for i in range(len(dims) - 1)]).to(device)
        # final layer's log_s is unused (no requant on the logits), so DDP must tolerate it
        train_net = (DDP(net, device_ids=[rank], find_unused_parameters=True) if world > 1 else net)
        opt = torch.optim.Adam(net.parameters(), lr=c["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])

        x = _t(self._in_np(data.train_x, np.float32), device)
        y = _t(data.train_y, device)
        vx = _t(self._in_np(data.val_x, np.float32), device)
        vy = _t(data.val_y, device)
        step, accum, _ = ddp.accum_plan(c["micro"], world, c["batch"])  # fixed global batch/size
        best, best_state, best_ep, train_secs = float("inf"), None, 0, 0.0
        layers = list(net)
        n_train = x.shape[0]
        for ep in range(c["epochs"]):
            perm = torch.randperm(n_train, device=device)  # same on every rank (same seed)
            t0 = time.perf_counter()
            for i in range(0, n_train, step * accum):
                micros = [perm[i + a * step:i + (a + 1) * step] for a in range(accum)]
                micros = [m for m in micros if m.shape[0] >= world]
                if not micros:
                    continue
                opt.zero_grad(set_to_none=True)
                last, scale = len(micros) - 1, float(len(micros))
                for j, m in enumerate(micros):
                    local = ddp.shard(m, rank, world)
                    sync = nullcontext() if (world == 1 or j == last) else train_net.no_sync()
                    with sync:
                        (F.cross_entropy(train_net(x[local]), y[local]) / scale).backward()
                opt.step()
                for L in layers:  # the fake-quant cache is only valid between optimiser steps
                    L._q = None
            if x.is_cuda:
                torch.cuda.synchronize(device)
            train_secs += time.perf_counter() - t0
            sched.step()
            # rank-0 val loss, broadcast to all -> identical early-stop decision (no DDP deadlock)
            if rank == 0:
                with torch.no_grad():
                    vl = sum(F.cross_entropy(net(vx[i:i + 4096]), vy[i:i + 4096],
                                             reduction="sum").item()
                             for i in range(0, vx.shape[0], 4096)) / vx.shape[0]
            else:
                vl = 0.0
            vl = ddp.broadcast_float(vl)
            if vl < best - 1e-4:
                best, best_ep = vl, ep
                if best_state is None:  # reuse the buffers; a full clone per improvement is churn
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    with torch.no_grad():
                        for k, v in net.state_dict().items():
                            best_state[k].copy_(v)
            if rank == 0 and (ep % 5 == 0 or ep == c["epochs"] - 1):
                print(f"  epoch {ep + 1:3d}/{c['epochs']}  val loss {vl:.4f}  (best {best:.4f} @ {best_ep + 1})",
                      flush=True)
            if ep - best_ep >= c["patience"]:
                if rank == 0:
                    print(f"  early stop at epoch {ep + 1}", flush=True)
                break
        net.load_state_dict(best_state)
        for L in layers:
            L._q = None
        return {"layers": [L.export() for L in net], "train_seconds": train_secs}


def variant(wmode: str, abits: int):
    """Return (TITLE, points, build) for one weight/activation scheme."""
    wtag = "w1.58" if wmode == "ternary" else "w4"
    title = f"{wtag}a{abits} (quantized MLP reference)"

    def points(spec: DatasetSpec) -> list[dict]:
        return [{"name": n, "hidden": h} for n, h in _LADDER.items()]

    def build(spec, **point):
        return QuantModel(spec, wmode, abits, **point)

    return title, points, build
