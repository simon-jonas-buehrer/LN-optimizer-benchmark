"""backprop: gradient descent learns each gate's truth table AND its wiring.

Four latent reals per gate -> STE on a sin -> a hard 4-bit truth table. Each input picks among 8
random candidate signals by a learnable logit (argmax forward, softmax gradient). The forward pass
is exact boolean, so the trained (thresholds, layers) go straight to LutModel. Trains to convergence
with early stopping on validation loss.
"""

from __future__ import annotations

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
        # Hard wiring: take the argmax candidate. The old form was
        #     x = sig[:, cand];  picked = (x * (onehot + (soft - soft.detach()))).sum(-1)
        # whose VALUE is exactly `sig[:, wires]` (every non-argmax product is exactly 0.0 and
        # `soft - soft.detach()` is exactly 0.0), but which made autograd push a full
        # (B, 2, W, cands) gradient back through the gather -- an index scatter-add over `cands`
        # times more elements than there are gates. Splitting the two roles keeps both the value
        # and the gradient identical:
        #   * `sig[:, wires]` carries the (B, 2, W) gradient to the source signals;
        #   * the detached candidate term carries the wiring gradient to `conn` and contributes
        #     exactly 0.0 to the value and exactly 0.0 to the gradient wrt `sig`.
        picked = sig[:, self.wires()]                                    # (B, 2, W), exact
        train = torch.is_grad_enabled() and self.conn.requires_grad
        if train:
            soft = torch.softmax(self.conn, -1)
            picked = picked + (sig.detach()[:, self.cand] * (soft - soft.detach())).sum(-1)
        xa, xb = picked[:, 0], picked[:, 1]
        c = _ste(self.table) if train else _hard(self.table)   # _ste == _hard under no_grad
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
        """`sig`: the already thermometer-encoded float input, (B, n_pixels*bits)."""
        for layer in self.layers:
            sig = torch.cat([sig, layer(sig)], 1)
        last = sig[:, -self.widths[-1]:]
        return last.reshape(last.shape[0], self.spec.n_classes, -1).sum(-1) / \
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
        self._sim_cache: dict = {}

    def _chunk(self):
        return max(64, min(2048, 2 ** 28 // (2 * max(self.cfg["widths"]) * self.cfg["cands"])))

    def _fuse(self, step, accum):
        """How many accumulation micro-steps to run in ONE forward/backward.

        Summing k micro-steps into a single batched forward accumulates the same gradient (only the
        float reduction order differs) while paying the per-step launch overhead, the
        softmax/argmax/STE over the whole parameter set and the layer concatenations once instead
        of k times. Capped by the same activation budget `_chunk` uses, so the big tiers keep their
        `micro`-sized memory footprint -- there `fuse == 1`, i.e. byte-for-byte the un-fused path.
        """
        per = 2 * self.cfg["cands"] * sum(self.cfg["widths"])   # activation elems per sample
        return max(1, min(accum, max(1, 2 ** 28 // max(1, per)) // step))

    def train(self, data: Dataset, *, device="cpu", seed=0):
        # Multi-GPU is opt-in via `ddp_gpus` (set by the harness). ddp.launch runs `_worker` on one
        # process per GPU; with <=1 GPU it runs inline -> this path is byte-for-byte the old trainer.
        self._data, self._seed, self._base_device = data, seed, device
        res = ddp.launch(self._worker, getattr(self, "ddp_gpus", 1))
        self.thresholds, self.layers = res["thresholds"], res["layers"]
        self.train_seconds = res["train_seconds"]  # pure training time (no val/measure), for the json
        self._sim_cache = {}
        self._data = None  # only needed to reach the worker; don't pin the dataset afterwards

    def _worker(self, rank, world):
        data, seed, c = self._data, self._seed, self.cfg
        device = self._base_device if world == 1 else f"cuda:{rank}"
        torch.manual_seed(seed)  # identical seed on every rank -> identical init AND identical perm
        m = _Net(self.spec, c["bits"], c["widths"], c["cands"], seed).to(device)
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
        best, best_state, best_ep, train_secs = float("inf"), None, 0, 0.0
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
                        _accum_loss(train_m(ex[local].float()), y[local], shards,
                                    len(micros)).backward()
                opt.step()
            if ex.is_cuda:
                torch.cuda.synchronize(device)
            train_secs += time.perf_counter() - t0
            sched.step()
            # early stop on val LOSS (forward is already hard = the circuit). Compute it on rank 0
            # only and broadcast, so every rank breaks on the SAME number (GPU reductions differ in
            # the last bits across devices, and a diverging break would deadlock DDP).
            if rank == 0:
                with torch.no_grad():  # no_grad also drops the wiring-gradient term in _LutLayer
                    parts = torch.stack([
                        F.cross_entropy(m(ev[i:i + ch].float()), vy[i:i + ch], reduction="sum")
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
        with torch.no_grad():
            layers = [(lay.wires()[0].cpu().numpy(), lay.wires()[1].cpu().numpy(),
                       lay.truth_table().cpu().numpy()) for lay in m.layers]
        return {"thresholds": m.thresholds, "train_seconds": train_secs, "layers": layers}

    # ------------------------------------------------------------------------------------------
    # Evaluation. The harness asks for predict(val_x) and then scores(val_x) (same for test), i.e.
    # it runs the identical packed simulation over the identical array twice. Memoise on the
    # identity of that array (a reference is kept alive, so the id cannot be recycled) and, on a
    # CUDA run, use lut_sim's bit-for-bit-equal GPU backend.
    # ------------------------------------------------------------------------------------------
    def _counts(self, pix: np.ndarray) -> np.ndarray:
        cache = getattr(self, "_sim_cache", None)
        if cache is None:
            cache = self._sim_cache = {}
        hit = cache.get(id(pix))
        if hit is not None and hit[0] is pix:
            return hit[1]
        dev = str(getattr(self, "_base_device", "cpu"))
        out = lut_sim(self.thresholds, self.layers, pix, self.spec,
                      device=dev if dev != "cpu" and torch.cuda.is_available() else None)
        if len(cache) >= 2:
            cache.clear()
        cache[id(pix)] = (pix, out)
        return out


def build(spec, **point) -> Backprop:
    return Backprop(spec, **point)
