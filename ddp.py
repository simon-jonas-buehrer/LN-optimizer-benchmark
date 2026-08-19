"""Single-node multi-GPU training helpers (torch DDP via torch.multiprocessing.spawn).

Only the gradient methods (backprop, dfa, quant) use this. A method's ``train`` calls::

    result = ddp.launch(self._worker, gpus)

where ``worker(rank, world)`` runs on ``cuda:rank`` inside an initialised process group and returns a
picklable result; only rank 0's result is kept and handed back to the caller. If ``gpus <= 1`` (or
CUDA is unavailable) ``worker(0, 1)`` runs INLINE, in-process -- byte-for-byte the old single-process
path, so a 1-GPU (or CPU) run is exactly what it was before this file existed. The harness stays
single-process: it calls ``train`` once and measures the one resulting hard net.

``worker`` and everything it closes over are pickled to the child processes, so it must be a
top-level function or a bound method of a picklable object. The methods pass ``self._worker`` and
stash the (picklable) dataset/seed on ``self`` first; ``self`` before training carries only spec+cfg.

Shard convention (used by the methods, not enforced here): every rank sees the SAME per-epoch
permutation (seed the generator identically) and takes a contiguous equal slice of each global
minibatch, so the averaged/all-reduced gradient equals the single-process gradient over that same
global batch -- results track the 1-GPU run up to floating-point reduction order.
"""

from __future__ import annotations

import os
import pickle
import socket
import tempfile

import torch


def available_gpus() -> int:
    """Visible CUDA devices, or 0 if CUDA can't initialise (e.g. driver too old)."""
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _entry(rank: int, world: int, worker, out_path: str, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
    try:
        res = worker(rank, world)
        if rank == 0:
            with open(out_path, "wb") as f:
                pickle.dump(res, f)
        torch.distributed.barrier()  # keep ranks alive until rank 0 has written the result
    finally:
        torch.distributed.destroy_process_group()


def launch(worker, gpus: int | None):
    """Run ``worker(rank, world)`` across ``gpus`` GPUs on this node; return rank 0's result.

    ``gpus <= 1`` (or no usable CUDA) => run ``worker(0, 1)`` inline, identical to the old path.
    """
    n = available_gpus()
    world = min(int(gpus or 1), n) if n else 1
    if world <= 1:
        return worker(0, 1)
    out = tempfile.NamedTemporaryFile(prefix="ddp_", suffix=".pkl", delete=False).name
    try:
        torch.multiprocessing.spawn(
            _entry, args=(world, worker, out, _free_port()), nprocs=world, join=True)
        with open(out, "rb") as f:
            return pickle.load(f)
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


def _micro_cap(micro: int, device=None) -> int:
    """Largest micro-batch we are willing to grow to, from free device memory measured right now.

    ``accum_plan`` is called after the model and optimiser are on the device but before any
    activation is allocated, so ``total - free`` is the resident model+optimiser state. Growing the
    micro-batch k-fold grows activations k-fold, and we do not know the per-sample activation cost --
    so we assume the worst case that one micro-batch's activations are as large as EVERYTHING
    currently resident, and only allow k copies of that inside half the free memory. On the small
    tiers (model ~1 GB of 24 GB) that permits a full merge; on the 20M-gate tiers, where the resident
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


def accum_plan(micro: int, world: int, eff_min: int = 32, *, micro_max: int | None = None):
    """Plan gradient accumulation so a tiny per-GPU micro-batch still reaches a decent global batch.

    Returns ``(step, accum, eff)`` where ``step`` global samples flow through one forward/backward,
    ``accum`` such steps are summed before each optimiser step, and ``eff = step*accum`` is the
    effective global batch -- always >= ``eff_min``. With 4 GPUs and ``micro=8`` and ``eff_min=32``
    that is already ``accum=1``; on 1 GPU accum grows to reach ``eff_min``.

    On a SINGLE rank the accumulation loop and the micro-batch are the same loop, so splitting the
    global batch into ``accum`` forward/backwards buys nothing but kernel launches. When the free
    device memory allows it (``_micro_cap``, or an explicit ``micro_max``) the step is widened to
    the largest multiple of ``micro`` that still divides ``eff`` -- so ``eff`` is unchanged, every
    step stays equal-sized, and ``accum`` drops (to 1 when the whole global batch fits).

    NOTE this is a deliberate, signed-off trainer change, not a pure reordering: a global batch that
    is not a multiple of ``step`` is now one short step instead of several, and a mean over the
    concatenation is not the average of the per-piece means. It is a better estimator of the same
    objective, but the numbers move. ``world > 1`` is untouched (byte-for-byte the old plan), and so
    is any caller that pins ``micro_max=micro``.
    """
    step = micro * world
    accum = max(1, -(-eff_min // step))  # ceil(eff_min/step)
    eff = step * accum
    if world == 1 and accum > 1:
        cap = _micro_cap(micro) if micro_max is None else max(micro, micro_max)
        d = max((k for k in range(1, accum + 1) if accum % k == 0 and micro * k <= cap), default=1)
        step, accum = micro * d, accum // d
    return step, accum, eff


def shard(idx: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """Contiguous equal-ish slice of a global minibatch's indices for this rank.

    Equal split of the first ``(len // world) * world`` items; any remainder < world is dropped so
    every rank does the same number of backward passes (a must for DDP) and the averaged gradient is
    exactly the mean over the kept samples.
    """
    n = (idx.shape[0] // world) * world
    if n == 0:
        return idx[:0]
    per = n // world
    return idx[rank * per:(rank + 1) * per]


def is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def all_reduce_(t: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce when distributed; no-op otherwise. Returns ``t``."""
    if is_dist():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t


def broadcast_float(x: float) -> float:
    """Return rank 0's value of ``x`` on every rank (no-op when not distributed).

    Used for the validation loss so the early-stop decision is made from ONE number on all ranks --
    GPU reductions aren't bit-identical across devices, and a diverging break would deadlock DDP.
    """
    if not is_dist():
        return x
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    t = torch.tensor([x], dtype=torch.float64, device=dev)
    torch.distributed.broadcast(t, src=0)
    return float(t.item())
