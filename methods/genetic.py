"""genetic: learn the wiring of a fixed all-NAND net by mutation hill-climbing.

Every gate is a NAND. NAND is functionally complete, so this search space contains every circuit
the LUT net can express; the only free parameters are which two strictly-earlier signals each gate
reads. No gradients.

    for each generation:
        make k-1 mutants of the current wiring (rewire `mut` gate endpoints at random)
        score all k (the incumbent included) on the same large minibatch by MARGIN
        keep the best

Margin (true-class readout count minus the best wrong class) is the fitness rather than accuracy:
minibatch accuracy only moves when a prediction flips, so almost every single-wire mutation scores
the same and the search random-walks; the margin moves whenever any vote moves, turning the plateau
into a slope. The selection batch must be big or one rewired wire drowns in sampling noise.

Fitness and validation come straight from methods.lut.lut_sim (the packed simulator that mirrors the
emitted netlist), so predict()==emit by construction. Each point trains until validation loss stops
improving; `gens` is a ceiling, not a target.
"""

from __future__ import annotations

import time

import numpy as np

from data import Dataset, DatasetSpec
from hw import even_thresholds
from methods.lut import LutModel, lut_sim

TITLE = "genetic (learned wiring, all gates NAND)"

NAND_TT = 0b0111  # bit (2a+b) of ~(a & b): f(0,0)=f(0,1)=f(1,0)=1, f(1,1)=0

# >=5 size points, ~1k -> ~20M gates by pre-optimisation gate count (~sum of layer widths). xs..l are
# the original record's shapes; xl/xxl extend the curve upward. The hill-climber weakens as the net
# grows (a random rewiring moves the margin less), so the big tiers are expected to plateau -- the
# flattening is the result. Every readout width (widths[-1]) is divisible by 10.
_LADDER = {
    "xs": (1, (256, 256, 160)),
    "s": (1, (1024, 1024, 320)),
    "m": (3, (2048, 2048, 2048, 640)),
    "l": (3, (4096, 4096, 4096, 4096, 1280)),
    "xl": (7, (250_000, 250_000, 250_000, 250_000, 25_000)),
    "xxl": (7, (5_000_000, 5_000_000, 5_000_000, 5_000_000, 500_000)),
}


def points(spec: DatasetSpec) -> list[dict]:
    return [{"name": n, "bits": b, "widths": w, "gens": 2_000_000} for n, (b, w) in _LADDER.items()]


def _mutate(srcs, widths, offs, mut, rng):
    """Rewire `mut` random gate endpoints (a fresh copy). Layers chosen with P(l) ~ width."""
    m = [s.copy() for s in srcs]
    w = np.asarray(widths, float)
    for l in rng.choice(len(widths), size=mut, p=w / w.sum()):
        m[l][rng.integers(2), rng.integers(widths[l])] = rng.integers(offs[l])
    return m


def _margin(counts, y):
    """Readout count for the true class minus the best other class, averaged. A slope, not a cliff."""
    idx = np.arange(len(y))
    true = counts[idx, y]
    other = counts.copy()
    other[idx, y] = -1  # counts are >= 0, so this excludes the true class from the max
    return float((true - other.max(1)).mean())


def _val_ce(counts, y):
    """Cross-entropy of a softmax over the integer readout counts (used as logits)."""
    z = counts.astype(float)
    z -= z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


class Genetic(LutModel):
    def __init__(self, spec, bits, widths, gens, k=8, mut=1, batch=16384,
                 eval_every=5000, patience=20):
        super().__init__(spec)
        if widths[-1] % spec.n_classes:
            raise ValueError(f"readout {widths[-1]} not divisible by {spec.n_classes}")
        self.cfg = dict(bits=bits, widths=tuple(widths), gens=gens, k=k, mut=mut,
                        batch=batch, eval_every=eval_every, patience=patience)

    def train(self, data: Dataset, *, device="cpu", seed=0):
        c, spec = self.cfg, self.spec
        rng = np.random.default_rng(seed)
        widths = c["widths"]
        thresholds = even_thresholds(c["bits"])
        # signal-id layout: encoder owns 0..n_in-1, then layer 0's gates, then layer 1's, ...
        offs = [spec.n_pixels * c["bits"]]
        for w in widths:
            offs.append(offs[-1] + w)
        # a gate reads any strictly-earlier signal -> the graph is acyclic by construction
        srcs = [rng.integers(0, offs[l], size=(2, w), dtype=np.int64) for l, w in enumerate(widths)]

        tt = [np.full(w, NAND_TT, np.int64) for w in widths]  # constant: only the wiring is learned
        layers = lambda g: [(gi[0], gi[1], ti) for gi, ti in zip(g, tt)]
        counts = lambda g, x: lut_sim(thresholds, layers(g), x, spec)  # (N, n_classes) via the sim

        best_ce, best_srcs, stale = float("inf"), [s.copy() for s in srcs], 0
        t0 = time.time()
        for gen in range(c["gens"]):
            idx = rng.integers(0, len(data.train_x), size=c["batch"])
            bx, by = data.train_x[idx], data.train_y[idx]
            best_fit = _margin(counts(srcs, bx), by)  # incumbent, scored on this batch
            winner = None
            for _ in range(c["k"] - 1):
                mutant = _mutate(srcs, widths, offs, c["mut"], rng)
                fit = _margin(counts(mutant, bx), by)
                if fit > best_fit:  # strictly better, so the incumbent survives ties
                    best_fit, winner = fit, mutant
            if winner is not None:
                srcs = winner

            if (gen + 1) % c["eval_every"] == 0 or gen + 1 == c["gens"]:
                ce = _val_ce(counts(srcs, data.val_x), data.val_y)  # lut_sim chunks the val set
                if ce < best_ce - 1e-4:
                    best_ce, best_srcs, stale = ce, [s.copy() for s in srcs], 0
                else:
                    stale += 1
                print(f"  gen {gen + 1:7d}/{c['gens']}  margin {best_fit:+.3f}  val ce {ce:.4f}  "
                      f"(best {best_ce:.4f}, stale {stale})  "
                      f"{(gen + 1) / (time.time() - t0):.0f} gen/s", flush=True)
                if stale >= c["patience"]:  # converged: no new best in patience*eval_every gens
                    print(f"  early stop at gen {gen + 1}: converged (best ce {best_ce:.4f})",
                          flush=True)
                    break

        self.thresholds = thresholds
        self.layers = layers(best_srcs)


def build(spec, **point) -> Genetic:
    return Genetic(spec, **point)
