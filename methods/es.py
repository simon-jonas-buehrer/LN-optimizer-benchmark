"""es: Evolution Strategies over CONTINUOUS per-LUT weights.

The requested contrast to a GA that evolves binary truth-table bits by crossover: here every 2-input
gate carries four continuous real weights (one per truth-table entry) -- a soft gate -- and an
OpenAI-ES loop (antithetic Gaussian perturbations, rank-shaped update) moves the mean. Wiring is
fixed random (chosen once from the seed); only the continuous weights change. At emit time each
gate's four reals are HARDENED to a boolean truth table (bit k = 1 iff weight_k > 0), and predict()
runs that exact hard net via LutModel -- so the measured circuit is the hard net, not the soft one.
No torch: fitness is the numpy packed simulator. Trains to convergence with early stopping on val
loss.
"""

from __future__ import annotations

import numpy as np

from data import Dataset, DatasetSpec
from hw import even_thresholds
from methods.lut import LutModel, lut_sim

TITLE = "es (evolution strategies, continuous per-LUT weights)"

# >=5 points, ~1k -> ~20M gates by pre-opt gate count (= sum of widths). ES is gradient-free and
# per-eval expensive, so the big tiers are heavy (the plateau is itself a result).
_LADDER = {
    "xs": (1, (700, 300)),
    "s": (1, (4000, 2000)),
    "m": (3, (26000, 13000)),
    "l": (3, (160000, 80000, 40000)),
    "xl": (7, (1_000_000, 500_000, 250_000)),
    "xxl": (7, (11_000_000, 6_000_000, 3_000_000)),
}


def points(spec: DatasetSpec) -> list[dict]:
    return [{"name": n, "bits": b, "widths": w, "gens": 3000} for n, (b, w) in _LADDER.items()]


def _wiring(widths, n_in, rng):
    """Fixed random 2-input wiring: gate reads two strictly-earlier signal ids. Returns per-layer
    (idx_a, idx_b) and the per-layer signal offsets."""
    wires, off = [], n_in
    for w in widths:
        a = rng.integers(0, off, w, dtype=np.int64)
        b = rng.integers(0, off, w, dtype=np.int64)
        wires.append((a, b))
        off += w
    return wires


def _layers(theta, wires, widths):
    """Harden the flat genome theta (G,4) into (idx_a, idx_b, tt) per layer."""
    tt = ((theta[:, 0] > 0) | ((theta[:, 1] > 0) << 1) | ((theta[:, 2] > 0) << 2) |
          ((theta[:, 3] > 0) << 3)).astype(np.int64)
    out, o = [], 0
    for (a, b), w in zip(wires, widths):
        out.append((a, b, tt[o:o + w]))
        o += w
    return out


def _ranks(f):
    """Centered rank utilities in [-0.5, 0.5] (OpenAI-ES fitness shaping)."""
    order = np.argsort(np.argsort(f))
    return order / (len(f) - 1) - 0.5


class ES(LutModel):
    def __init__(self, spec, bits, widths, gens, pop=40, sigma=0.1, lr=0.1, batch=2048,
                 patience=20, eval_every=10):
        super().__init__(spec)
        if widths[-1] % spec.n_classes:
            raise ValueError(f"readout {widths[-1]} not divisible by {spec.n_classes}")
        self.bits, self.widths = bits, tuple(widths)
        self.cfg = dict(gens=gens, pop=pop, sigma=sigma, lr=lr, batch=batch,
                        patience=patience, eval_every=eval_every)

    def _val_loss(self, theta, wires, vx, vy) -> float:
        counts = lut_sim(even_thresholds(self.bits), _layers(theta, wires, self.widths), vx, self.spec)
        z = counts - counts.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return float(-logp[np.arange(len(vy)), vy].mean())

    def train(self, data: Dataset, *, device="cpu", seed=0):
        c = self.cfg
        rng = np.random.default_rng(seed)
        thr = even_thresholds(self.bits)
        n_in = self.spec.n_pixels * self.bits
        wires = _wiring(self.widths, n_in, rng)
        G = sum(self.widths)
        theta = rng.standard_normal((G, 4)).astype(np.float32) * 0.1

        def fit(th, xb, yb):  # negative CE on a minibatch of the hardened net (higher = better)
            counts = lut_sim(thr, _layers(th, wires, self.widths), xb, self.spec)
            z = counts - counts.max(1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(1, keepdims=True))
            return float(logp[np.arange(len(yb)), yb].mean())

        best, best_theta, best_gen = float("inf"), theta.copy(), 0
        half = c["pop"] // 2
        for gen in range(c["gens"]):
            sel = rng.integers(0, len(data.train_x), c["batch"])
            xb, yb = data.train_x[sel], data.train_y[sel]
            seeds = rng.integers(0, 2**31, half)
            fs, eps_signs = [], []
            for s in seeds:  # antithetic pairs; regenerate eps from its seed to keep memory O(G)
                eps = np.random.default_rng(s).standard_normal((G, 4)).astype(np.float32)
                fs.append(fit(theta + c["sigma"] * eps, xb, yb)); eps_signs.append((s, +1))
                fs.append(fit(theta - c["sigma"] * eps, xb, yb)); eps_signs.append((s, -1))
            u = _ranks(np.array(fs))
            grad = np.zeros_like(theta)
            for (s, sign), ui in zip(eps_signs, u):
                eps = np.random.default_rng(s).standard_normal((G, 4)).astype(np.float32)
                grad += (ui * sign) * eps
            theta += c["lr"] / (c["pop"] * c["sigma"]) * grad

            if gen % c["eval_every"] == 0 or gen == c["gens"] - 1:
                vl = self._val_loss(theta, wires, data.val_x, data.val_y)
                if vl < best - 1e-4:
                    best, best_gen, best_theta = vl, gen, theta.copy()
                print(f"  gen {gen + 1:5d}/{c['gens']}  val loss {vl:.4f}  (best {best:.4f} @ {best_gen + 1})",
                      flush=True)
                if gen - best_gen >= c["patience"] * c["eval_every"]:
                    print(f"  early stop at gen {gen + 1}", flush=True)
                    break

        self.thresholds = thr
        self.layers = _layers(best_theta, wires, self.widths)


def build(spec, **point) -> ES:
    return ES(spec, **point)
