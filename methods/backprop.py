"""backprop: gradient descent learns each gate's truth table AND its wiring.

Four latent reals per gate -> STE on a sin -> a hard 4-bit truth table. Each input picks among 8
random candidate signals by a learnable logit (argmax forward, softmax gradient). The forward pass
is exact boolean, so the trained (thresholds, layers) go straight to LutModel. Trains to convergence
with early stopping on validation loss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from data import Dataset, DatasetSpec
from hw import even_thresholds
from methods.lut import LutModel

TITLE = "backprop (learned truth tables + learned wiring)"

# >=5 size points, targeting ~1k -> ~20M gates by pre-optimisation gate count (~sum of widths).
_LADDER = {
    "xs": (1, (700, 300)),
    "s": (1, (4000, 2000)),
    "m": (3, (26000, 13000)),
    "l": (3, (160000, 80000, 40000)),
    "xl": (7, (1_000_000, 500_000, 250_000)),
    "xxl": (7, (11_000_000, 6_000_000, 3_000_000)),
}


def points(spec: DatasetSpec) -> list[dict]:
    return [{"name": n, "bits": b, "widths": w, "epochs": 200} for n, (b, w) in _LADDER.items()]


def _t(a, device):
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _hard(z):
    return (torch.sin(z) > 0).to(z.dtype)


def _ste(z):
    soft = 0.5 + 0.5 * torch.sin(z)
    return _hard(z) + (soft - soft.detach())


class _LutLayer(torch.nn.Module):
    def __init__(self, off, width, cands, g):
        super().__init__()
        self.register_buffer("cand", torch.randint(off, (2, width, cands), generator=g))
        self.conn = torch.nn.Parameter(torch.randn(2, width, cands, generator=g) * 0.1)
        self.table = torch.nn.Parameter(torch.randn(width, 4, generator=g))

    def wires(self):
        return self.cand.gather(2, self.conn.argmax(-1, keepdim=True)).squeeze(-1)

    def forward(self, sig):
        x = sig[:, self.cand]
        soft = torch.softmax(self.conn, -1)
        sel = torch.zeros_like(soft).scatter_(-1, self.conn.argmax(-1, keepdim=True), 1.0)
        picked = (x * (sel + (soft - soft.detach()))).sum(-1)
        xa, xb = picked[:, 0], picked[:, 1]
        c = _ste(self.table)
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
        g = torch.Generator().manual_seed(seed)
        off = spec.n_pixels * bits
        self.layers = torch.nn.ModuleList()
        for w in widths:
            self.layers.append(_LutLayer(off, w, cands, g))
            off += w

    def encode(self, pix):
        t = torch.tensor(self.thresholds, device=pix.device, dtype=torch.int16)
        return (pix.to(torch.int16).unsqueeze(-1) > t).reshape(pix.shape[0], -1).float()

    def forward(self, pix):
        sig = self.encode(pix)
        for layer in self.layers:
            sig = torch.cat([sig, layer(sig)], 1)
        last = sig[:, -self.widths[-1]:]
        return last.reshape(last.shape[0], self.spec.n_classes, -1).sum(-1) / \
            (self.widths[-1] // self.spec.n_classes) ** 0.5


class Backprop(LutModel):
    def __init__(self, spec, bits, widths, epochs, lr=0.2, batch=128, cands=8, patience=40):
        super().__init__(spec)
        self.cfg = dict(bits=bits, widths=tuple(widths), epochs=epochs, lr=lr, batch=batch,
                        cands=cands, patience=patience)

    def _chunk(self):
        return max(64, min(2048, 2 ** 28 // (2 * max(self.cfg["widths"]) * self.cfg["cands"])))

    def train(self, data: Dataset, *, device="cpu", seed=0):
        torch.manual_seed(seed)
        c = self.cfg
        m = _Net(self.spec, c["bits"], c["widths"], c["cands"], seed).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=c["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])
        x, y = _t(data.train_x, device), _t(data.train_y, device)
        vx, vy = _t(data.val_x, device), _t(data.val_y, device)
        ch = self._chunk()
        best, best_state, best_ep = float("inf"), None, 0
        for ep in range(c["epochs"]):
            perm = torch.randperm(x.shape[0], device=device)
            for i in range(0, x.shape[0], c["batch"]):
                idx = perm[i:i + c["batch"]]
                loss = F.cross_entropy(m(x[idx]), y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
            with torch.no_grad():  # early stop on val LOSS (forward is already hard = the circuit)
                vl = sum(F.cross_entropy(m(vx[i:i + ch]), vy[i:i + ch], reduction="sum").item()
                         for i in range(0, vx.shape[0], ch)) / vx.shape[0]
            if vl < best - 1e-4:
                best, best_ep = vl, ep
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            if ep % 5 == 0 or ep == c["epochs"] - 1:
                print(f"  epoch {ep + 1:3d}/{c['epochs']}  val loss {vl:.4f}  (best {best:.4f} @ {best_ep + 1})",
                      flush=True)
            if ep - best_ep >= c["patience"]:
                print(f"  early stop at epoch {ep + 1}", flush=True)
                break
        m.load_state_dict(best_state)
        self.thresholds = m.thresholds
        self.layers = [(lay.wires()[0].cpu().numpy(), lay.wires()[1].cpu().numpy(),
                        lay.truth_table().cpu().numpy()) for lay in m.layers]


def build(spec, **point) -> Backprop:
    return Backprop(spec, **point)
