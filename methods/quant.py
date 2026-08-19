"""Quantized-MLP references: shared QAT core for w1.58a4 / w1.58a8 / w4a4.

Conventional dense MLPs put on the same gate axis as the logic nets. Weights are ternary ({-1,0,+1},
"w1.58") or signed 4-bit ("w4"); activations are unsigned 4- or 8-bit. Training is quantization-aware
(fake-quant + STE); at export the model becomes a list of `hw.QLayer` (integer weights/bias + a fixed
requant mul/shift per layer), and predict()/scores()/emit() all derive from that SAME list via
`hw.qmlp_forward` / `hw.emit_quant_mlp`, so the circuit equals predict() bit for bit. Early stopping
on validation loss.

`variant(wmode, abits)` returns (TITLE, points, build); the three thin modules w1_58a4/w1_58a8/w4a4
just call it, so each is one series with its own results folder.

Speed
-----
The EXPORT/INFERENCE side is bit-exact and must stay that way: `predict`/`scores` go through
`hw.qmlp_forward` (integer recurrence, BLAS GEMM in a provably-exact float dtype, int64 requantize)
and `export()` runs in eager fp32, so the emitted circuit still equals `predict()` exactly.

The TRAINING side is tuned for wallclock and its float arithmetic is deliberately NOT bit-reproducible
against the older trainer -- it reaches slightly different weights along a statistically equivalent
trajectory.  Levers, all switchable for ablation:

  * one fused forward/backward per optimiser step over the whole global batch, instead of
    `accum` micro-batches (the trainer is kernel-launch bound at these shapes)     -- always on
  * `_TF32`         : TF32 tensor cores for the fp32 matmuls on Ampere+
  * `_COMPILE`      : `torch.compile` on the net (fuses the fake-quant/STE elementwise chains)
  * `_FUSED_ADAM`   : the fused CUDA Adam kernel

Protocol is untouched: same epochs / patience / hidden sizes / global batch / permutation stream /
early-stopping criterion, and every training sample is still used in the same 128-sample groups.
"""

from __future__ import annotations

import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import ddp
from data import Dataset, DatasetSpec
from hw import QLayer, emit_quant_mlp, input_activations, qmlp_forward

_SH = 16  # requant fixed-point shift; mul = round(scale * 2^_SH)

# Training-speed switches.
#
# NOTE FOR ANYONE ABLATING THESE: they are module globals resolved from the ENVIRONMENT at import
# time. `ddp.launch` starts the >1-GPU ranks with torch.multiprocessing *spawn*, so each child
# RE-IMPORTS this module and re-reads the environment -- assigning `methods.quant._COMPILE = False`
# in the parent process does NOT reach the DDP children, which will silently keep the default. To
# ablate a multi-GPU run, set the env var (e.g. MNISTBENCH_QUANT_COMPILE=0) before launching, not
# the attribute. (A single-GPU/CPU run executes inline, so there the attribute does work -- which is
# exactly what makes the mistake easy to miss.)
def _flag(name, default=True):
    v = os.environ.get(name)
    return default if v is None else v not in ("0", "false", "False", "")


# Defaults come from measurement on an RTX 3090 (see the module docstring):
#   compile  ~2.0x steady-state per epoch, one-time cost 1-5s -> on
#   fused    ~1.1-1.2x on the wide points                     -> on
#   tf32     measured neutral at these shapes, and it is process-global state that other methods
#            in the same `run.py all` process would inherit   -> off
_TF32 = _flag("MNISTBENCH_QUANT_TF32", False)
_COMPILE = _flag("MNISTBENCH_QUANT_COMPILE", True)
_FUSED_ADAM = _flag("MNISTBENCH_QUANT_FUSED_ADAM", True)

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


class _Layer(torch.nn.Module):
    def __init__(self, n_in, n_out, wmode, abits, final, g):
        super().__init__()
        self.wmode, self.abits, self.final = wmode, abits, final
        self.W = torch.nn.Parameter(torch.randn(n_in, n_out, generator=g) / n_in ** 0.5)
        self.b = torch.nn.Parameter(torch.zeros(n_out))
        self.log_s = torch.nn.Parameter(torch.zeros(()))  # requant scale (log), non-final only

    def forward(self, a):
        # addmm folds the bias add into the GEMM epilogue (one kernel instead of two).
        acc = torch.addmm(_ste_round(self.b), a, _quant_w(self.W, self.wmode))
        if self.final:
            return acc
        y = torch.clamp(_ste_round(acc * self.log_s.exp()), 0, (1 << self.abits) - 1)
        return y

    def export(self) -> QLayer:
        """Eager fp32, no autocast/TF32 influence (elementwise only) -> the exported integers are
        exactly what `hw.emit_quant_mlp` turns into gates and what `predict()` recomputes."""
        with torch.no_grad():
            Wq = _quant_w(self.W.float(), self.wmode).cpu().numpy().round().astype(np.int64)
            b = _ste_round(self.b.float()).cpu().numpy().round().astype(np.int64)
            if self.final:
                return QLayer(Wq, b, 0, 0, 0, final=True)
            mul = max(1, int(round(float(self.log_s.float().exp()) * (1 << _SH))))
        return QLayer(Wq, b, mul, _SH, self.abits, final=False)


class QuantModel:
    """emit / predict / scores / save from an exported hw.QLayer list + input activation bits."""

    def __init__(self, spec: DatasetSpec, wmode: str, abits: int, hidden, epochs=200, lr=0.01,
                 batch=128, patience=20, micro=32):
        self.spec, self.wmode, self.abits, self.hidden = spec, wmode, abits, tuple(hidden)
        # `micro` = per-GPU micro-batch; with `world` GPUs it sets the global batch together with
        # `batch` (kept for compatibility -- the global batch per optimiser step is unchanged).
        self.cfg = dict(epochs=epochs, lr=lr, batch=batch, patience=patience, micro=micro)
        self.layers: list[QLayer] = []
        self._gemm: dict = {}     # hw.qmlp_forward's float weight copies, keyed on self.layers
        self._logits: dict = {}   # {id(pix): (pix, layers, logits)}; the harness scores val_x twice

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
        if hit is not None and hit[0] is pix and hit[1] is self.layers:
            return hit[2]
        logits = qmlp_forward(self._in_np(pix), self.layers, cache=self._gemm)
        self._logits = {id(pix): (pix, self.layers, logits)}
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
        # Multi-GPU is opt-in via `ddp_gpus` (set by the harness). <=1 GPU runs `_worker` inline;
        # >1 spawns one DDP rank per GPU.
        self._data, self._seed, self._base_device = data, seed, device
        res = ddp.launch(self._worker, getattr(self, "ddp_gpus", 1))
        self.layers, self.train_seconds = res["layers"], res["train_seconds"]
        self.train_samples = res["train_samples"]  # training-example evals until early stop
        self._gemm, self._logits = {}, {}

    def _worker(self, rank, world):
        data, seed, c = self._data, self._seed, self.cfg
        device = self._base_device if world == 1 else f"cuda:{rank}"
        cuda = str(device).startswith("cuda")
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)  # CPU generator -> identical init on every rank
        dims = [self.spec.n_pixels, *self.hidden, self.spec.n_classes]
        finals = [False] * (len(dims) - 2) + [True]
        net = torch.nn.Sequential(
            *[_Layer(dims[i], dims[i + 1], self.wmode, self.abits, finals[i], g)
              for i in range(len(dims) - 1)]).to(device)

        # TF32 is process-global state, and `run.py all` runs several methods in one process, so it
        # is scoped strictly to the training loop and restored afterwards. EXPORT DELIBERATELY
        # HAPPENS OUTSIDE THIS SCOPE: the exported integers define the circuit, so no backend
        # precision flag may be in force while they are computed.
        tf32_was = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        if _TF32 and cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        try:
            net, train_secs, nseen = self._fit(net, rank, world, device, cuda, c)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32_was
        # eager, fp32, TF32 restored -> `export` is exactly the integer model the emitter turns
        # into gates and `predict()` recomputes.
        return {"layers": [L.export() for L in net], "train_seconds": train_secs,
                "train_samples": nseen}

    def _fit(self, net, rank, world, device, cuda, c):
        """Train in place and return (net, pure_training_seconds). Training numerics are free to
        drift (compile / TF32 / fused Adam / one fused batch per step); only `export` is exact."""
        x = _t(self._in_np(self._data.train_x, np.float32), device)
        y = _t(self._data.train_y, device)
        vx = _t(self._in_np(self._data.val_x, np.float32), device)
        vy = _t(self._data.val_y, device)

        core = net
        if _COMPILE:  # cuda -> triton, cpu -> the cpp/OpenMP backend
            # Inductor fuses the fake-quant/STE elementwise chains (~1.2-1.6x per epoch).
            # torch.compile fails LAZILY -- at the first call, not at wrap time -- and its error
            # can be misleading: a missing `setuptools` (which triton imports at runtime) surfaces
            # as "Cannot find a working triton installation". So probe it once here and fall back
            # to eager instead of letting the run die mid-training on some future node.
            try:
                cand = torch.compile(net, dynamic=True)
                with torch.no_grad():
                    cand(x[:2])
                core = cand
            except Exception as e:
                if rank == 0:
                    print(f"  [compile] unavailable, running eager ({type(e).__name__}: "
                          f"{str(e).splitlines()[0][:120]})", flush=True)
                core = net
        # final layer's log_s is unused (no requant on the logits), so DDP must tolerate it
        train_net = (DDP(core, device_ids=[rank], find_unused_parameters=True) if world > 1 else core)
        opt = torch.optim.Adam(net.parameters(), lr=c["lr"],
                               **({"fused": True} if (_FUSED_ADAM and cuda) else {}))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])
        step, accum, gstep = ddp.accum_plan(c["micro"], world, c["batch"])  # gstep = global batch
        best, best_state, best_ep, train_secs, nseen = float("inf"), None, 0, 0.0, 0
        n_train = x.shape[0]
        for ep in range(c["epochs"]):
            perm = torch.randperm(n_train, device=device)  # same on every rank (same seed)
            t0 = time.perf_counter()
            for i in range(0, n_train, gstep):
                idx = perm[i:i + gstep]
                if idx.shape[0] < world:
                    continue
                local = ddp.shard(idx, rank, world)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(train_net(x[local]), y[local]).backward()
                opt.step()
            if cuda:
                torch.cuda.synchronize(device)
            train_secs += time.perf_counter() - t0
            nseen += n_train  # training-example forward passes this epoch (samples looked at)
            sched.step()
            # rank-0 val loss, broadcast to all -> identical early-stop decision (no DDP deadlock)
            if rank == 0:
                with torch.no_grad():
                    vl = sum(F.cross_entropy(core(vx[i:i + 4096]), vy[i:i + 4096],
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
        if best_state is not None:      # None only when epochs == 0 (a structure-only probe)
            net.load_state_dict(best_state)
        return net, train_secs, nseen


def variant(wmode: str, abits: int):
    """Return (TITLE, points, build) for one weight/activation scheme."""
    wtag = "w1.58" if wmode == "ternary" else "w4"
    title = f"{wtag}a{abits} (quantized MLP reference)"

    def points(spec: DatasetSpec) -> list[dict]:
        return [{"name": n, "hidden": h} for n, h in _LADDER.items()]

    def build(spec, **point):
        return QuantModel(spec, wmode, abits, **point)

    return title, points, build
