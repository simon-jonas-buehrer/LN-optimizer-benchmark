"""Gradient-accumulation planning for the single-GPU trainers.

Every method trains in ONE process on ONE device. The gradient methods still need a global batch
that is fixed per method (part of the protocol) while the *micro*-batch that actually goes through
one forward is bounded by device memory -- the 1M-gate tiers only fit a handful of images at a
time. ``accum_plan`` turns that pair into ``(step, accum, eff)``.

This is all that survived of the old ``ddp.py``: with one rank there is no sharding, no all-reduce
and no broadcast, so the rest of it was identity functions.
"""

from __future__ import annotations

import os

import torch


def _micro_cap(micro: int, device=None) -> int:
    """Largest micro-batch we are willing to grow to, from free device memory measured right now.

    ``accum_plan`` is called after the model and optimiser are on the device but before any
    activation is allocated, so ``total - free`` is the resident model+optimiser state. Growing the
    micro-batch k-fold grows activations k-fold, and we do not know the per-sample activation cost --
    so we assume the worst case that one micro-batch's activations are as large as EVERYTHING
    currently resident, and only allow k copies of that inside half the free memory. On the small
    tiers (model ~1 GB of 24 GB) that permits a full merge; on the 1M-gate tiers, where the resident
    state already fills the card, it returns ``micro`` and the memory bound is exactly as before.
    """
    env = os.environ.get("MNISTBENCH_MICRO_MAX")
    if env:
        return max(micro, int(env))
    try:
        if not torch.cuda.is_available():
            return micro                      # CPU run: keep the caller's footprint
        free, total = torch.cuda.mem_get_info(device)
    except Exception:
        return micro
    resident = max(total - free, 1)
    return micro * max(1, int(free * 0.5) // resident)


def accum_plan(micro: int, eff_min: int = 32, *, micro_max: int | None = None):
    """Plan gradient accumulation so a tiny micro-batch still reaches a decent global batch.

    Returns ``(step, accum, eff)`` where ``step`` samples flow through one forward/backward,
    ``accum`` such steps are summed before each optimiser step, and ``eff = step*accum`` is the
    effective global batch -- always >= ``eff_min``.

    The accumulation loop and the micro-batch are the same loop, so splitting the global batch into
    ``accum`` forward/backwards buys nothing but kernel launches. When the free device memory allows
    it (``_micro_cap``, or an explicit ``micro_max``) the step is widened to the largest multiple of
    ``micro`` that still divides ``eff`` -- so ``eff`` is unchanged, every step stays equal-sized,
    and ``accum`` drops (to 1 when the whole global batch fits).

    NOTE this is a deliberate, signed-off trainer change, not a pure reordering: a global batch that
    is not a multiple of ``step`` is now one short step instead of several, and a mean over the
    concatenation is not the average of the per-piece means. It is a better estimator of the same
    objective, but the numbers move. A caller that pins ``micro_max=micro`` keeps the old plan.
    """
    step = micro
    accum = max(1, -(-eff_min // step))  # ceil(eff_min/step)
    eff = step * accum
    if accum > 1:
        cap = _micro_cap(micro) if micro_max is None else max(micro, micro_max)
        d = max((k for k in range(1, accum + 1) if accum % k == 0 and micro * k <= cap), default=1)
        step, accum = micro * d, accum // d
    return step, accum, eff
