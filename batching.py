"""Device sizing for the single-GPU trainers: what dtype to compute in, and how big a working
buffer this card can actually afford.

Every method trains in ONE process on ONE device. The gradient methods still need a global batch
that is fixed per method (part of the protocol) while the *micro*-batch that actually goes through
one forward is bounded by device memory -- the 1M-gate tiers only fit a handful of images at a
time. ``accum_plan`` turns that pair into ``(step, accum, eff)``.

``accum_plan`` is all that survived of the old ``ddp.py``: with one rank there is no sharding, no
all-reduce and no broadcast, so the rest of it was identity functions. ``budget`` and ``act_dtype``
are new, and exist because the jobs no longer run on one known card: an RTX 2080 Ti has 11 GB and no
bfloat16, where the RTX 3090 the constants were tuned on has 24 GB and does. Both degrade to exactly
the old behaviour on a big Ampere card, so nothing about a 3090 run changes.
"""

from __future__ import annotations

import os

import torch


def free_bytes(device=None) -> int | None:
    """Free memory on the CUDA device right now, or None if there is no usable CUDA."""
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.mem_get_info(device)[0]
    except Exception:
        return None


def budget(cap: int, fraction: float = 0.5, device=None) -> int:
    """A working-buffer ceiling of ``cap`` bytes, lowered to ``fraction`` of what is actually free.

    The scratch budgets in the methods were picked on a 24 GB RTX 3090 and hardcoded. On a card with
    half the memory the same constant is no longer a ceiling but a promise to OOM, so ask the device.
    Call this AFTER the model, optimiser and encoded dataset are resident -- then "free" is genuinely
    what the working buffer may take, and ``fraction`` is headroom for the transients around it.

    Returns ``cap`` unchanged when the device cannot be asked, and (on any card with room to spare)
    when ``fraction`` of free memory is above ``cap`` -- so this never *raises* a tuned budget.
    """
    free = free_bytes(device)
    return cap if free is None else max(1, min(cap, int(free * fraction)))


def act_dtype(device, want: str = "auto") -> "torch.dtype":
    """Activation dtype for the gate cascades: ``fp32``, ``bf16``, or ``auto``.

    ``auto`` means bf16 on hardware that really has it and fp32 everywhere else. Turing (RTX 2080 Ti,
    sm_75) has no bfloat16 unit: torch will accept the tensors and emulate, which is both slow and
    not what the bf16 path was measured to be. CPU bf16 is emulated too. fp16 is deliberately NOT
    the fallback -- bf16 was chosen for fp32's exponent range, and the trainers run without a grad
    scaler, so half's range is exactly the thing they would trip over.
    """
    if want == "auto":
        ok = False
        if torch.device(device).type == "cuda":
            try:
                ok = torch.cuda.get_device_capability(device)[0] >= 8
            except Exception:
                ok = False
        want = "bf16" if ok else "fp32"
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[want]


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
