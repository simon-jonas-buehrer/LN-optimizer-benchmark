"""Quantized-MLP references: shared QAT core for w1.58a4 / w1.58a8 / w4a4.

Conventional dense MLPs put on the same gate axis as the logic nets. Weights are ternary ({-1,0,+1},
"w1.58") or signed 4-bit ("w4"); activations are unsigned 4- or 8-bit. Training is quantization-aware
(fake-quant + STE); at export the model becomes a list of `hw.QLayer` (integer weights/bias + a fixed
requant mul/shift per layer), and predict()/scores()/emit() all derive from that SAME list via
`hw.qmlp_forward` / `hw.emit_quant_mlp`, so the circuit equals predict() bit for bit. Early stopping
on validation loss.

`variant(wmode, abits)` returns (TITLE, points, build); the three thin modules w1_58a4/w1_58a8/w4a4
just call it, so each is one series with its own results folder.
"""

from __future__ import annotations

import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import ddp
from data import Dataset, DatasetSpec
from hw import QLayer, emit_quant_mlp, input_activations, qmlp_forward

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


class _Layer(torch.nn.Module):
    def __init__(self, n_in, n_out, wmode, abits, final, g):
        super().__init__()
        self.wmode, self.abits, self.final = wmode, abits, final
        self.W = torch.nn.Parameter(torch.randn(n_in, n_out, generator=g) / n_in ** 0.5)
        self.b = torch.nn.Parameter(torch.zeros(n_out))
        self.log_s = torch.nn.Parameter(torch.zeros(()))  # requant scale (log), non-final only

    def forward(self, a):
        acc = a @ _quant_w(self.W, self.wmode) + _ste_round(self.b)
        if self.final:
            return acc
        y = torch.clamp(_ste_round(acc * self.log_s.exp()), 0, (1 << self.abits) - 1)
        return y

    def export(self) -> QLayer:
        Wq = _quant_w(self.W, self.wmode).detach().cpu().numpy().round().astype(np.int64)
        b = _ste_round(self.b).detach().cpu().numpy().round().astype(np.int64)
        if self.final:
            return QLayer(Wq, b, 0, 0, 0, final=True)
        mul = max(1, int(round(float(self.log_s.exp()) * (1 << _SH))))
        return QLayer(Wq, b, mul, _SH, self.abits, final=False)


class QuantModel:
    """emit / predict / scores / save from an exported hw.QLayer list + input activation bits."""

    def __init__(self, spec: DatasetSpec, wmode: str, abits: int, hidden, epochs=200, lr=0.01,
                 batch=128, patience=20):
        self.spec, self.wmode, self.abits, self.hidden = spec, wmode, abits, tuple(hidden)
        self.cfg = dict(epochs=epochs, lr=lr, batch=batch, patience=patience)
        self.layers: list[QLayer] = []

    def _in(self, pix):
        return input_activations(pix, self.spec, self.abits)

    def emit_verilog(self) -> str:
        return emit_quant_mlp(self.layers, self.spec, self.abits)

    def predict(self, pix):
        return qmlp_forward(self._in(pix), self.layers).argmax(1)

    def scores(self, pix):
        return qmlp_forward(self._in(pix), self.layers).astype(float)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"abits": self.abits, "wmode": self.wmode,
                         "layers": [(L.Wq, L.bias, L.mul, L.sh, L.out_abits, L.final)
                                    for L in self.layers]}, f)

    def train(self, data: Dataset, *, device="cpu", seed=0):
        # Multi-GPU is opt-in via `ddp_gpus` (set by the harness). <=1 GPU runs `_worker` inline, so
        # this path stays byte-for-byte the old trainer; >1 spawns one DDP rank per GPU.
        self._data, self._seed, self._base_device = data, seed, device
        self.layers = ddp.launch(self._worker, getattr(self, "ddp_gpus", 1))["layers"]

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

        x = _t(self._in(data.train_x).astype(np.float32), device)
        y = _t(data.train_y, device)
        vx = _t(self._in(data.val_x).astype(np.float32), device)
        vy = _t(data.val_y, device)
        best, best_state, best_ep = float("inf"), None, 0
        for ep in range(c["epochs"]):
            perm = torch.randperm(x.shape[0], device=device)  # same on every rank (same seed)
            for i in range(0, x.shape[0], c["batch"]):
                idx = perm[i:i + c["batch"]]
                if idx.shape[0] < world:
                    continue
                local = ddp.shard(idx, rank, world)
                loss = F.cross_entropy(train_net(x[local]), y[local])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
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
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if rank == 0 and (ep % 5 == 0 or ep == c["epochs"] - 1):
                print(f"  epoch {ep + 1:3d}/{c['epochs']}  val loss {vl:.4f}  (best {best:.4f} @ {best_ep + 1})",
                      flush=True)
            if ep - best_ep >= c["patience"]:
                if rank == 0:
                    print(f"  early stop at epoch {ep + 1}", flush=True)
                break
        net.load_state_dict(best_state)
        return {"layers": [L.export() for L in net]}


def variant(wmode: str, abits: int):
    """Return (TITLE, points, build) for one weight/activation scheme."""
    wtag = "w1.58" if wmode == "ternary" else "w4"
    title = f"{wtag}a{abits} (quantized MLP reference)"

    def points(spec: DatasetSpec) -> list[dict]:
        return [{"name": n, "hidden": h} for n, h in _LADDER.items()]

    def build(spec, **point):
        return QuantModel(spec, wmode, abits, **point)

    return title, points, build
