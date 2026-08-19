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

        # fixed wiring: srcs[l] = (2, w) GLOBAL ids; layer l reads only layer l-1 (or the encoder)
        self.offs = [self.n_in]
        in_dim, in_base = self.n_in, 0
        self.srcs: list[torch.Tensor] = []
        for l, w in enumerate(self.widths):
            bf = _butterfly_src(in_dim, w, l - 1).contiguous()   # (2, w) local into in_dim
            self.srcs.append((bf + in_base).to(device))
            in_base = self.offs[-1]                              # next layer reads this layer's outs
            self.offs.append(self.offs[-1] + w)
            in_dim = w

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

    @property
    def n_sig(self) -> int:
        return self.offs[-1]

    def tables(self) -> list[torch.Tensor]:
        return [hard_bit(z) for z in self.z]

    def forward(self, enc: torch.Tensor, T: list[torch.Tensor] | None = None) -> torch.Tensor:
        """enc (n_in, B) uint8 -> acts (n_sig, B) uint8. Exact bits; no relaxation anywhere."""
        T = self.tables() if T is None else T
        acts = torch.zeros((self.n_sig, enc.shape[1]), dtype=torch.uint8, device=enc.device)
        acts[: self.n_in] = enc
        for l, s in enumerate(self.srcs):
            p = (acts[s[0]].long() << 1) | acts[s[1]].long()     # (w, B) in {0,1,2,3}
            acts[self.offs[l] : self.offs[l + 1]] = T[l].gather(1, p)
        return acts

    def votes(self, acts: torch.Tensor) -> torch.Tensor:
        out = acts[self.offs[-2] : self.offs[-1]]                # (R, B)
        return out.reshape(self.spec.n_classes, -1, out.shape[1]).sum(1).T.float()  # (B, n_classes)

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
                 epochs: int, lr: float = 0.01, batch: int = 256, patience: int = 12) -> None:
        super().__init__(spec)
        self.cfg = dict(bits=bits, width=width, layers=layers, readout=readout, epochs=epochs,
                        lr=lr, batch=batch, patience=patience)

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

        n = enc_tr.shape[1]
        steps = max(1, n // c["batch"])
        best, best_z, best_ep, t0 = float("inf"), [z.detach().clone() for z in net.z], 0, time.time()
        for ep in range(c["epochs"]):
            perm = torch.randperm(n, generator=g, device=device)  # identical on every rank
            for i in range(steps):
                idx = perm[i * c["batch"] : (i + 1) * c["batch"]]
                if idx.shape[0] < world:
                    continue
                local = ddp.shard(idx, rank, world)
                loss = self._step(net, enc_tr[:, local], y_tr[local], opt, local.shape[0] * world)
            sched.step()

            # rank-0 val loss, broadcast to all -> identical early-stop decision (no DDP deadlock)
            vl = ddp.broadcast_float(self._val_loss(net, enc_va, y_va) if rank == 0 else 0.0)
            if vl < best - 1e-4:
                best, best_ep = vl, ep
                best_z = [z.detach().clone() for z in net.z]
            if rank == 0 and (ep % 5 == 0 or ep == c["epochs"] - 1):
                print(f"  epoch {ep + 1:3d}/{c['epochs']}  loss {loss:.3f}  val loss {vl:.4f}  "
                      f"(best {best:.4f} @ {best_ep + 1})  "
                      f"{(ep + 1) / (time.time() - t0):.2f} ep/s", flush=True)
            if ep - best_ep >= c["patience"]:
                if rank == 0:
                    print(f"  early stop at epoch {ep + 1}: no gain since {best_ep + 1}", flush=True)
                break

        for z, bz in zip(net.z, best_z):
            z.copy_(bz)
        # hand the hard structure to LutModel: predict/scores/emit all read exactly these arrays
        return {"thresholds": net.thresholds, "layers": net.layers_np()}

    # ---- the DFA update ------------------------------------------------------------------------
    def _step(self, net: _Butterfly, enc: torch.Tensor, y: torch.Tensor, opt, nglobal: int) -> float:
        """One direct-feedback-alignment step. No graph, no .backward(), no cross-layer signal.

        Under DDP each rank runs this on its slice of the global batch; normalising the error by the
        GLOBAL count `nglobal` and SUM-all-reducing each layer's raw gradient `G` reconstructs exactly
        the single-process full-batch gradient (all-reduce is a no-op when not distributed)."""
        B = enc.shape[1]
        nc = self.spec.n_classes
        T = net.tables()
        acts = net.forward(enc, T)

        # the ONLY global quantity: the output error, broadcast from here to every layer at once
        logits = net.votes(acts) / net.tau                      # (B, n_classes)
        prob = torch.softmax(logits, 1)
        ar = torch.arange(B, device=enc.device)
        loss = -torch.log(prob[ar, y] + 1e-12).mean()
        e = prob.clone()
        e[ar, y] -= 1.0
        e /= nglobal                                            # mean over the GLOBAL batch

        for l, w in enumerate(net.widths):
            s = net.srcs[l]
            p = (acts[s[0]].long() << 1) | acts[s[1]].long()    # (w, B) active pattern, recomputed
            if l == len(net.widths) - 1:
                # readout: its own local gradient. dlogit_c/d bit_i = 1/tau for i in group c.
                cls_of = torch.arange(w, device=enc.device) // (w // nc)
                delta = e[:, cls_of] / net.tau                  # (B, w)
            else:
                delta = e @ net.B[l]                            # (B, w) -- the direct projection
            # only the ACTIVE table entry of each gate gets gradient: G[i,p] = sum_b delta[b,i]
            G = torch.zeros(w * 4, device=enc.device)
            idx = torch.arange(w, device=enc.device)[:, None] * 4 + p   # (w, B)
            G.scatter_add_(0, idx.reshape(-1), delta.t().reshape(-1))
            ddp.all_reduce_(G)  # sum this rank's slice into the full-batch gradient (no-op if 1 GPU)
            net.z[l].grad = G.view(w, 4) * (0.5 * torch.cos(net.z[l]))

        opt.step()
        return loss.item()

    # ---- eval (torch forward; only used for the early-stopping loss) ---------------------------
    @torch.no_grad()
    def _val_loss(self, net: _Butterfly, enc: torch.Tensor, y: torch.Tensor) -> float:
        ch, tot = self._chunk(), 0.0
        for i in range(0, enc.shape[1], ch):
            yb = y[i : i + ch]
            logits = net.votes(net.forward(enc[:, i : i + ch])) / net.tau
            logp = logits - torch.logsumexp(logits, 1, keepdim=True)
            tot += -logp[torch.arange(len(yb), device=logits.device), yb].sum().item()
        return tot / enc.shape[1]


def build(spec: DatasetSpec, **point) -> Dfa:
    return Dfa(spec, **point)
