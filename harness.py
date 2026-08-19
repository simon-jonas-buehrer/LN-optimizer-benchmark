"""The harness: train a model, emit SystemVerilog, let yosys+ABC map it to gates, measure it.

A method lives in `methods/<name>.py` and exposes:

    points(spec) -> list[dict]              # >=5 size points on this method's curve, per dataset
    build(spec, **point) -> model

where `model` (duck-typed, no base class) has:

    train(data, *, device, seed)            # fit on data.train_*/val_*; data.test_* is off limits
    emit_verilog() -> str                   # the trained model as `module top(...)` (uses self.spec)
    predict(pix) -> (N,) classes            # EXACT function emit_verilog describes (cross-checked)
    scores(pix) -> (N, n_classes) | None    # optional, for the loss/perplexity axis
    save(path)                              # optional, write a checkpoint

The harness sets `model.spec` before training. Per point it writes, under
`results/<dataset>/<method>/`:

    <point>.s<seed>.sv       emitted SystemVerilog
    <point>.s<seed>.ckpt     trained checkpoint (if the model implements save)
    <point>.s<seed>.json     accuracy, loss, perplexity, nand/inv/gate count, sky130 GE, config

Area (headline) is the raw 2-input gate count (NAND2+INV) of the fast-ABC netlist; accuracy is read
off that same netlist; sky130 GE is measured too when a liberty is configured.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from data import Dataset, DatasetSpec, to_bits

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHECK_SAMPLES = 512

# ================================================================================================
# Synthesis (yosys + ABC). Configuration is environment-only, so the public repo carries no machine
# or cluster paths. Set MNISTBENCH_YOSYS (else $MNISTBENCH_EDA/bin/yosys, else PATH) and, only for
# the optional sky130-GE metric, MNISTBENCH_LIBERTY (else derived from $MNISTBENCH_EDA).
# ================================================================================================

_EDA = os.environ.get("MNISTBENCH_EDA")
_LIB_REL = "share/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
YOSYS = (os.environ.get("MNISTBENCH_YOSYS")
         or (str(Path(_EDA) / "bin/yosys") if _EDA else None) or shutil.which("yosys") or "yosys")
LIBERTY = os.environ.get("MNISTBENCH_LIBERTY") or (str(Path(_EDA) / _LIB_REL) if _EDA else "")
NAND2_AREA_UM2 = 3.7536  # sky130_fd_sc_hd__nand2_1, the GE unit

# One FAST, deliberately-imperfect script for every model (strash + one dc2 + map): the "fast area
# optimizer". Frozen -- same effort for everyone, so it is not a leaderboard knob.
FAST = "strash;dc2;map"
_RESYN2 = "balance;rewrite;refactor;balance;rewrite;rewrite,-z;balance;refactor,-z;rewrite,-z;balance"
OPT = f"strash;{_RESYN2};dc2;{_RESYN2};resub,-K,8;dc2;{_RESYN2};map"  # high-effort, optional only


@dataclass
class Nand:
    netlist: dict
    nand: int
    inv: int

    @property
    def gates(self) -> int:
        return self.nand + self.inv


def has_liberty() -> bool:
    return bool(LIBERTY)


def _run(cmds: str, cwd: str, timeout: int) -> str:
    p = subprocess.run([YOSYS, "-p", cmds], capture_output=True, text=True, cwd=cwd, timeout=timeout)
    log = p.stdout + p.stderr
    if "ABC script did not complete" in log or "cmd error" in log:
        err = "\n".join(l for l in log.splitlines() if "cmd error" in l)
        raise RuntimeError(f"ABC aborted:\n{err}")
    if p.returncode != 0:
        raise RuntimeError(f"yosys exit {p.returncode}:\n{log[-3000:]}")
    return log


def synth_nand(sv: Path, *, script: str = FAST, top: str = "top", timeout: int = 14400) -> Nand:
    """Map to NAND2 + INV only and return the netlist (for exact simulation) and gate counts."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "netlist.json"
        cmds = (f"read_verilog -sv {sv.resolve()}; synth -top {top} -flatten -noabc; opt -full; "
                f"abc -g NAND -script +{script}; opt_clean; stat; write_json {out}")
        log = _run(cmds, td, timeout)
        netlist = json.loads(out.read_text())
    tail = log[log.rfind("Printing statistics"):]
    counts: dict[str, int] = {}
    for n, cell in re.findall(r"^\s+(\d+)\s+\$_(\w+)_\s*$", tail, re.M):
        counts[cell] = counts.get(cell, 0) + int(n)
    stray = set(counts) - {"NAND", "NOT"}
    if stray:
        raise RuntimeError(f"netlist is not NAND-only, found {stray}")
    return Nand(netlist, counts.get("NAND", 0), counts.get("NOT", 0))


def synth_ge(sv: Path, *, script: str = FAST, top: str = "top", timeout: int = 14400) -> tuple[float, float, int]:
    """Optional: map to sky130 cells -> (gate equivalents, area um^2, cell count). Needs a liberty."""
    if not LIBERTY:
        raise RuntimeError("no sky130 liberty configured (set MNISTBENCH_LIBERTY)")
    cmds = (f"read_verilog -sv {sv.resolve()}; synth -top {top} -flatten -noabc; opt -full; "
            f"abc -liberty {LIBERTY} -script +{script}; opt_clean; stat -liberty {LIBERTY}")
    with tempfile.TemporaryDirectory() as td:
        log = _run(cmds, td, timeout)
    tail = log[log.rfind("Printing statistics"):]
    m = re.search(r"Chip area for (?:top )?module '\\?" + re.escape(top) + r"':\s*([\d.]+)", tail)
    if not m:
        raise RuntimeError(f"no chip area in stat output:\n{tail[-2000:]}")
    area = float(m.group(1))
    rows = [(int(n), name) for n, name in re.findall(r"^\s+(\d+)\s+\S+\s+(\S+)\s*$", tail, re.M)]
    total = next((n for n, name in rows if name == "cells"), 0)
    return area / NAND2_AREA_UM2, area, total


# ================================================================================================
# NAND netlist simulation (pure numpy): the accuracy axis comes from running the circuit.
# ================================================================================================

CONST0, CONST1, WORD = 0, 1, 64


@dataclass
class NandNet:
    n_in: int
    n_sig: int
    src_a: list
    src_b: list
    offs: list
    out_sig: list

    @property
    def depth(self) -> int:
        return len(self.src_a)


def from_json(nl: dict, spec: DatasetSpec, top: str = "top") -> NandNet:
    mod = nl["modules"][top]
    in_bits, out_bits = [], []
    for port in mod["ports"].values():
        (in_bits if port["direction"] == "input" else out_bits).extend(port["bits"])
    if len(in_bits) != spec.port_bits:
        raise RuntimeError(f"top has {len(in_bits)} input bits, expected {spec.port_bits}")
    sig = {"0": CONST0, "1": CONST1, "x": CONST0, "z": CONST0}
    for i, b in enumerate(in_bits):
        sig[b] = 2 + i
    n_in = len(in_bits)
    cells = []
    for name, c in mod["cells"].items():
        conn = c["connections"]
        if c["type"] == "$_NAND_":
            cells.append((conn["Y"][0], conn["A"][0], conn["B"][0]))
        elif c["type"] == "$_NOT_":
            cells.append((conn["Y"][0], conn["A"][0], conn["A"][0]))
        else:
            raise RuntimeError(f"cell {name} is {c['type']}; must be NAND-only")
    driver = {y: i for i, (y, _, _) in enumerate(cells)}
    if len(driver) != len(cells):
        raise RuntimeError("a net is driven by two gates")
    level, state = [0] * len(cells), [0] * len(cells)
    for root in range(len(cells)):
        if state[root]:
            continue
        stack = [(root, False)]
        while stack:
            g, done = stack.pop()
            if done:
                lv = 0
                for net in cells[g][1:]:
                    src = driver.get(net)
                    if src is not None:
                        lv = max(lv, level[src] + 1)
                level[g], state[g] = lv, 2
            elif state[g] == 0:
                state[g] = 1
                stack.append((g, True))
                for net in cells[g][1:]:
                    src = driver.get(net)
                    if src is not None and state[src] == 0:
                        stack.append((src, False))
            elif state[g] == 1:
                raise RuntimeError("combinational loop in the netlist")
    depth = max(level) + 1 if cells else 0
    buckets = [[] for _ in range(depth)]
    for g, lv in enumerate(level):
        buckets[lv].append(g)
    offs, nxt = [2 + n_in], 2 + n_in
    for lv in range(depth):
        for g in buckets[lv]:
            sig[cells[g][0]] = nxt
            nxt += 1
        offs.append(nxt)

    def sid(net):
        if net not in sig:
            raise RuntimeError(f"net {net!r} is read but never driven")
        return sig[net]

    src_a = [np.array([sid(cells[g][1]) for g in b], np.int64) for b in buckets]
    src_b = [np.array([sid(cells[g][2]) for g in b], np.int64) for b in buckets]
    return NandNet(n_in, nxt, src_a, src_b, offs, [sid(b) for b in out_bits])


def run(net: NandNet, x_bits: np.ndarray, spec: DatasetSpec, chunk: int = 4096) -> np.ndarray:
    preds = []
    for i in range(0, len(x_bits), chunk):
        xb = x_bits[i:i + chunk]
        n = len(xb)
        pad = (-n) % WORD
        if pad:
            xb = np.concatenate([xb, np.zeros((pad, xb.shape[1]), np.uint8)])
        packed = np.packbits(np.ascontiguousarray(xb.T), axis=1, bitorder="little").view(np.uint64)
        acts = np.zeros((net.n_sig, packed.shape[1]), np.uint64)
        acts[CONST1] = np.uint64(0xFFFFFFFFFFFFFFFF)
        acts[2:2 + net.n_in] = packed
        for lv, (a, b) in enumerate(zip(net.src_a, net.src_b)):
            acts[net.offs[lv]:net.offs[lv + 1]] = ~(acts[a] & acts[b])
        out = np.unpackbits(acts[net.out_sig].view(np.uint8), axis=1, bitorder="little")
        cls = (out.astype(np.int64) << np.arange(len(net.out_sig), np.int64)[:, None]).sum(0)
        preds.append(cls[:n])
    pred = np.concatenate(preds)
    if (pred >= spec.n_classes).any():
        raise RuntimeError(f"circuit produced class >= {spec.n_classes} -- argmax broken")
    return pred


# ================================================================================================
# Loss / perplexity
# ================================================================================================

def _cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(z)), y].mean())


def _fit_temperature(scores: np.ndarray, y: np.ndarray) -> float:
    lo, hi = 1e-3, 1e2
    for _ in range(6):
        grid = np.geomspace(lo, hi, 40)
        j = int(np.argmin([_cross_entropy(scores / t, y) for t in grid]))
        lo, hi = grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]
    return float(np.sqrt(lo * hi))


# ================================================================================================
# Run one point / one method
# ================================================================================================

def load_method(name: str) -> ModuleType:
    mod = importlib.import_module(f"methods.{name}")
    for attr in ("points", "build"):
        if not hasattr(mod, attr):
            raise SystemExit(f"methods/{name}.py must define {attr}")
    return mod


def _stem(method: str, dataset: str, point: str, seed: int) -> Path:
    d = RESULTS / dataset / method
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{point}.s{seed}"


def measure(sv: Path, data: Dataset) -> tuple[dict, NandNet]:
    spec = data.spec
    t0 = time.time()
    nand = synth_nand(sv, script=FAST)
    net = from_json(nand.netlist, spec)
    print(f"[gates] {nand.gates:,} 2-input gates ({nand.nand:,} NAND + {nand.inv:,} INV), "
          f"depth {net.depth}, {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    acc = float((run(net, to_bits(data.test_x, spec), spec) == data.test_y).mean()) * 100
    print(f"[sim  ] test acc {acc:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
    m = {"gates": nand.gates, "nand": nand.nand, "inv": nand.inv, "depth": net.depth,
         "test_acc": round(acc, 2)}
    if has_liberty():
        try:
            ge, area, cells = synth_ge(sv, script=FAST)
            m |= {"ge": round(ge, 1), "area_um2": round(area, 1), "cells": cells}
            print(f"[ge   ] {ge:,.0f} GE ({cells:,} cells)", flush=True)
        except RuntimeError as e:
            print(f"[ge   ] skipped ({e})", flush=True)
    return m, net


def run_point(mod: ModuleType, point: dict, data: Dataset, *, device: str, seed: int) -> dict:
    spec = data.spec
    cfg = {k: v for k, v in point.items() if k != "name"}
    print(f"\n=== {spec.name}/{mod.__name__.split('.')[-1]}/{point['name']}  {cfg}  seed={seed}",
          flush=True)
    model = mod.build(spec, **cfg)
    model.spec = spec
    stem = _stem(mod.__name__.split(".")[-1], spec.name, point["name"], seed)

    t0 = time.time()
    model.train(data, device=device, seed=seed)
    train_s = time.time() - t0
    print(f"[train] {train_s:.0f}s", flush=True)

    if hasattr(model, "save"):
        model.save(str(stem) + ".ckpt")
        print(f"[ckpt ] {stem.name}.ckpt", flush=True)

    sv = Path(str(stem) + ".sv")
    sv.write_text(model.emit_verilog())
    print(f"[emit ] {sv.name}, {sv.stat().st_size / 1e6:.1f} MB", flush=True)

    m, net = measure(sv, data)

    hw = run(net, to_bits(data.test_x[:CHECK_SAMPLES], spec), spec)
    py = np.asarray(model.predict(data.test_x[:CHECK_SAMPLES]))
    if not (hw == py).all():
        bad = np.flatnonzero(hw != py)
        raise SystemExit(f"REJECTED: circuit != model on {len(bad)}/{CHECK_SAMPLES} images "
                         f"(e.g. {bad[:5].tolist()}). emit_verilog must equal predict.")

    val_acc = float((np.asarray(model.predict(data.val_x)) == data.val_y).mean()) * 100
    out = {"name": point["name"], "method": mod.__name__.split(".")[-1], "dataset": spec.name,
           "seed": seed, "config": cfg, **m, "val_acc": round(val_acc, 2),
           "train_s": round(train_s), "device": device}

    scores = getattr(model, "scores", lambda _p: None)
    sc_va, sc_te = scores(data.val_x), scores(data.test_x)
    if sc_te is not None:
        t = _fit_temperature(np.asarray(sc_va, float), data.val_y)
        ce = _cross_entropy(np.asarray(sc_te, float) / t, data.test_y)
        out |= {"test_ce": round(ce, 4), "test_ppl": round(float(np.exp(ce)), 4),
                "ce_temp": round(float(t), 4)}

    Path(str(stem) + ".json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[write] {stem.name}.json", flush=True)
    return out


def run_method(name: str, data: Dataset, *, device: str, seed: int,
               only: list[str] | None, force: bool) -> None:
    mod = load_method(name)
    for point in mod.points(data.spec):
        if only and point["name"] not in only:
            continue
        stem = _stem(name, data.spec.name, point["name"], seed)
        if Path(str(stem) + ".json").exists() and not force:
            print(f"=== {data.spec.name}/{name}/{point['name']}: done, skipping (--force)", flush=True)
            continue
        run_point(mod, point, data, device=device, seed=seed)


def rescore_method(name: str, data: Dataset, seed: int) -> None:
    """Re-measure stored .sv artifacts without retraining (both axes come from the Verilog)."""
    d = RESULTS / data.spec.name / name
    for sv in sorted(d.glob(f"*.s{seed}.sv")):
        stem = sv.with_suffix("")
        print(f"\n=== rescore {sv.name}", flush=True)
        m, _ = measure(sv, data)
        jp = Path(str(stem) + ".json")
        base = json.loads(jp.read_text()) if jp.exists() else {"name": stem.name}
        base.update(m)
        jp.write_text(json.dumps(base, indent=2) + "\n")
