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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from data import Dataset, DatasetSpec, to_bits

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHECK_SAMPLES = 512   # historical floor only: run_point now cross-checks the FULL test set
                      # (the circuit predictions it compares against are computed for test_acc anyway)

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
# How yosys hands the mapped netlist back: "blif" (default, ~7x smaller and streamed) or "json".
# Both parse to a byte-identical NandNet; this only changes the writer and the reader.
NETLIST_FMT = os.environ.get("MNISTBENCH_NETLIST", "blif")
# Extra concurrent yosys processes for the optional sky130-GE syntheses (0 = fully serial, the old
# behaviour; 2 = all three syntheses at once). yosys is single-threaded, so overlapping them is free
# in CPU but costs one full yosys heap each -- hence the memory guard in `_synth_jobs`.
SYNTH_JOBS = int(os.environ.get("MNISTBENCH_SYNTH_JOBS", "1"))
# Measured: a 1.66 MB .sv (145,300 NAND cells) peaks at 1.66 GB of yosys RSS, i.e. ~1 GB per MB of
# emitted Verilog. Used only to decide whether a second/third process fits, never for the result.
SYNTH_GB_PER_SV_MB = float(os.environ.get("MNISTBENCH_SYNTH_GB_PER_MB", "1.1"))


def _mem_budget_gb() -> float:
    """Bytes this process may reasonably use, from the cgroup limit if there is one, else MemAvailable."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(path).read_text().strip()
            if v.isdigit() and int(v) < (1 << 50):
                return int(v) / 1e9
        except OSError:
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def _synth_jobs(sv: Path) -> int:
    """How many EXTRA yosys processes to keep live next to the NAND synthesis of `sv`."""
    jobs = SYNTH_JOBS
    if jobs <= 0:
        return 0
    budget = _mem_budget_gb()
    if budget <= 0:
        return jobs                      # unknown limit: trust the configured value
    try:
        est = sv.stat().st_size / 1e6 * SYNTH_GB_PER_SV_MB
    except OSError:
        return jobs
    while jobs > 0 and (jobs + 1) * est > 0.75 * budget:
        jobs -= 1
    return jobs


# One FAST, deliberately-imperfect script for every model (strash + one dc2 + map): the "fast area
# optimizer". Frozen -- same effort for everyone, so it is not a leaderboard knob.
FAST = "strash;dc2;map"
# Sky130 mapping with NO ABC logic optimisation (strash canonicalises, map picks cells): the
# "before ABC" point. Compared against FAST ("after ABC", one dc2 pass) it shows how much slack ABC
# can still squeeze out of each trained net -- i.e. which methods leave more compressible logic.
GE_MAP = "strash;map"
_RESYN2 = "balance;rewrite;refactor;balance;rewrite;rewrite,-z;balance;refactor,-z;rewrite,-z;balance"
OPT = f"strash;{_RESYN2};dc2;{_RESYN2};resub,-K,8;dc2;{_RESYN2};map"  # high-effort, optional only


@dataclass
class Nand:
    netlist: dict | None      # parsed write_json output; None when the netlist came back as BLIF
    nand: int
    inv: int
    net: "NandNet | None" = None   # ready-made NandNet (the BLIF path builds it while parsing)

    @property
    def gates(self) -> int:
        return self.nand + self.inv


def has_liberty() -> bool:
    return bool(LIBERTY)


class _Later:
    """Serial stand-in for a Future: runs the call on .result() instead of in a worker thread."""

    def __init__(self, fn, *a, **kw):
        self._call = (fn, a, kw)

    def result(self):
        fn, a, kw = self._call
        return fn(*a, **kw)


def _run(cmds: str, cwd: str, timeout: int) -> str:
    p = subprocess.run([YOSYS, "-p", cmds], capture_output=True, text=True, cwd=cwd, timeout=timeout)
    log = p.stdout + p.stderr
    if "ABC script did not complete" in log or "cmd error" in log:
        err = "\n".join(l for l in log.splitlines() if "cmd error" in l)
        raise RuntimeError(f"ABC aborted:\n{err}")
    if p.returncode != 0:
        raise RuntimeError(f"yosys exit {p.returncode}:\n{log[-3000:]}")
    return log


def synth_nand(sv: Path, *, script: str = FAST, top: str = "top", timeout: int = 14400,
               spec: DatasetSpec | None = None) -> Nand:
    """Map to NAND2 + INV only and return the netlist (for exact simulation) and gate counts.

    With `spec` given (and MNISTBENCH_NETLIST left at "blif") the netlist comes back as BLIF and is
    parsed straight into a `NandNet`: same synthesis, same `stat`, same gate counts, byte-identical
    `NandNet` -- but ~7x less text and no dict-of-dicts, which is what the biggest tiers need. Called
    without `spec` it is exactly the old json path and still fills `.netlist`.
    """
    blif = spec is not None and NETLIST_FMT == "blif"
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / ("netlist.blif" if blif else "netlist.json")
        writer = f"write_blif {out}" if blif else f"write_json {out}"
        cmds = (f"read_verilog -sv {sv.resolve()}; synth -top {top} -flatten -noabc; opt -full; "
                f"abc -g NAND -script +{script}; opt_clean; stat; {writer}")
        log = _run(cmds, td, timeout)
        netlist, net = (None, from_blif(out, spec, top)) if blif else (json.loads(out.read_text()), None)
    tail = log[log.rfind("Printing statistics"):]
    counts: dict[str, int] = {}
    for n, cell in re.findall(r"^\s+(\d+)\s+\$_(\w+)_\s*$", tail, re.M):
        counts[cell] = counts.get(cell, 0) + int(n)
    stray = set(counts) - {"NAND", "NOT"}
    if stray:
        raise RuntimeError(f"netlist is not NAND-only, found {stray}")
    return Nand(netlist, counts.get("NAND", 0), counts.get("NOT", 0), net)


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


def _build_net(in_bits, out_bits, cells, spec: DatasetSpec) -> NandNet:
    """Shared core of every netlist reader: port bits + an ORDERED list of (y, a, b) net keys.

    `cells[i]` is gate i as (output net, input net, input net) -- an inverter repeats its input, the
    way `$_NOT_` is read. Net keys may be anything hashable; "0"/"1" are the constants. Everything
    downstream (levelisation, the per-level signal numbering, `src_a`/`src_b`) lives here, so any
    two readers that hand over the same port bits and the same cell ORDER produce byte-identical
    `NandNet`s by construction.
    """
    if len(in_bits) != spec.port_bits:
        raise RuntimeError(f"top has {len(in_bits)} input bits, expected {spec.port_bits}")
    sig = {"0": CONST0, "1": CONST1, "x": CONST0, "z": CONST0}
    for i, b in enumerate(in_bits):
        sig[b] = 2 + i
    n_in = len(in_bits)
    driver = {y: i for i, (y, _, _) in enumerate(cells)}
    if len(driver) != len(cells):
        raise RuntimeError("a net is driven by two gates")
    # Iterative DFS on two flat predecessor lists (-1 = primary input / constant) instead of
    # re-slicing the cell tuples and re-hashing the net names on every visit; a finished node is
    # pushed back as its bitwise complement so no (node, flag) tuple is allocated per visit.
    n_cells = len(cells)
    pa = [driver.get(c[1], -1) for c in cells]
    pb = [driver.get(c[2], -1) for c in cells]
    level, state = [0] * n_cells, bytearray(n_cells)
    for root in range(n_cells):
        if state[root]:
            continue
        stack = [root]
        while stack:
            g = stack.pop()
            if g < 0:                                   # post-order visit of ~g
                g = ~g
                x, y = pa[g], pb[g]
                lv = level[x] + 1 if x >= 0 else 0
                if y >= 0 and level[y] + 1 > lv:
                    lv = level[y] + 1
                level[g], state[g] = lv, 2
            elif state[g] == 0:
                state[g] = 1
                stack.append(~g)
                x, y = pa[g], pb[g]
                if x >= 0 and state[x] == 0:
                    stack.append(x)
                if y >= 0 and state[y] == 0:
                    stack.append(y)
            elif state[g] == 1:
                raise RuntimeError("combinational loop in the netlist")
    depth = max(level) + 1 if cells else 0
    buckets = [[] for _ in range(depth)]
    for g, lv in enumerate(level):
        buckets[lv].append(g)
    offs, nxt, order = [2 + n_in], 2 + n_in, []
    for lv in range(depth):
        for g in buckets[lv]:
            sig[cells[g][0]] = nxt
            nxt += 1
        order.extend(buckets[lv])
        offs.append(nxt)

    def sid(net):
        if net not in sig:
            raise RuntimeError(f"net {net!r} is read but never driven")
        return sig[net]

    # One flat level-ordered pass with a direct dict lookup per pin (instead of ~2 python calls per
    # pin through sid()), then slice the levels out of it -- the arrays are contiguous by level.
    base = 2 + n_in
    try:
        a_ids = np.fromiter((sig[cells[g][1]] for g in order), np.int64, len(order))
        b_ids = np.fromiter((sig[cells[g][2]] for g in order), np.int64, len(order))
    except KeyError as e:
        raise RuntimeError(f"net {e.args[0]!r} is read but never driven") from None
    src_a = [a_ids[offs[lv] - base:offs[lv + 1] - base] for lv in range(depth)]
    src_b = [b_ids[offs[lv] - base:offs[lv + 1] - base] for lv in range(depth)]
    return NandNet(n_in, nxt, src_a, src_b, offs, [sid(b) for b in out_bits])


def from_json(nl: dict, spec: DatasetSpec, top: str = "top") -> NandNet:
    mod = nl["modules"][top]
    in_bits, out_bits = [], []
    for port in mod["ports"].values():
        (in_bits if port["direction"] == "input" else out_bits).extend(port["bits"])
    cells = []
    for name, c in mod["cells"].items():
        conn = c["connections"]
        if c["type"] == "$_NAND_":
            cells.append((conn["Y"][0], conn["A"][0], conn["B"][0]))
        elif c["type"] == "$_NOT_":
            cells.append((conn["Y"][0], conn["A"][0], conn["A"][0]))
        else:
            raise RuntimeError(f"cell {name} is {c['type']}; must be NAND-only")
    return _build_net(in_bits, out_bits, cells, spec)


_BLIF_CONST = {"$false": "0", "$true": "1", "$undef": "0"}


def from_blif(path, spec: DatasetSpec, top: str = "top") -> NandNet:
    """Same `NandNet` as `from_json`, read from yosys `write_blif` instead.

    BLIF is ~7x smaller than the json for the same netlist and is line-oriented, so it parses
    without ever holding the whole file (let alone a dict-of-dicts) in memory -- which is what makes
    the biggest tiers reachable at all. yosys writes the cells in the same module order for both
    writers, so handing `_build_net` the same port bits and the same cell order reproduces the json
    net exactly (asserted field-by-field in the tests).

    Recognised `.names` blocks, which is exactly what an all-NAND netlist can contain:
      0 inputs, no row / row `1`       -> the constants $false / $undef, $true
      1 input,  row `0 1` / `1 1`      -> inverter / buffer (a buffer is an alias, not a gate:
                                          the json has no cell for it either)
      2 inputs, rows `0- 1` and `-0 1` -> NAND
    """
    ids: dict = {}          # net name -> small int key; the names themselves are never kept

    def nid(name):
        i = ids.get(name)
        if i is None:
            i = ids[name] = len(ids)
        return i

    in_bits, out_bits, cells, alias = [], [], [], {}
    model = None
    pend_ins, pend_out, rows = None, None, None

    def flush():
        """Turn the finished .names block into a cell, an alias, or a constant."""
        if pend_out is None:
            return
        k = len(pend_ins)
        if k == 0:
            v = "1" if rows == ["1"] else "0"
            if pend_out != v:            # `.names $true` is the constant itself, not an alias of it
                alias[pend_out] = v
        elif k == 1:
            if rows == ["0 1"]:
                cells.append((pend_out, pend_ins[0], pend_ins[0]))       # inverter
            elif rows == ["1 1"]:
                if pend_out != pend_ins[0]:
                    alias[pend_out] = pend_ins[0]                        # buffer: pure connection
            else:
                raise RuntimeError(f"blif: unsupported 1-input .names rows {rows}")
        elif k == 2:
            if sorted(rows) != ["-0 1", "0- 1"]:
                raise RuntimeError(f"blif: 2-input .names is not a NAND, rows {rows}")
            cells.append((pend_out, pend_ins[0], pend_ins[1]))
        else:
            raise RuntimeError(f"blif: {k}-input .names; the netlist must be NAND-only")

    with open(path, "r") as f:
        buf = ""
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith("\\"):
                buf += line[:-1] + " "
                continue
            line, buf = buf + line, ""
            if line.startswith("."):
                flush()
                pend_ins, pend_out, rows = None, None, None
                tok = line.split()
                cmd = tok[0]
                if cmd == ".model":
                    model = tok[1] if len(tok) > 1 else None
                elif cmd == ".inputs":
                    in_bits.extend(nid(t) for t in tok[1:])
                elif cmd == ".outputs":
                    out_bits.extend(nid(t) for t in tok[1:])
                elif cmd == ".names":
                    pend_ins = [_BLIF_CONST.get(t) or nid(t) for t in tok[1:-1]]
                    pend_out = _BLIF_CONST.get(tok[-1]) or nid(tok[-1])
                    rows = []
                elif cmd == ".end":
                    break
                else:
                    raise RuntimeError(f"blif: unexpected directive {cmd}")
            elif rows is not None:
                rows.append(line)
            else:
                raise RuntimeError(f"blif: stray line {line!r}")
        flush()
    if model is not None and model != top:
        raise RuntimeError(f"blif: model is {model!r}, expected {top!r}")

    if alias:   # resolve buffer/constant chains with path compression, then rewrite the cell pins
        limit = len(alias) + 1

        def root(k):
            seen = []
            while k in alias:
                seen.append(k)
                k = alias[k]
                if len(seen) > limit:
                    raise RuntimeError("combinational loop in the netlist")
            for s_ in seen:
                alias[s_] = k
            return k
        cells = [(y, root(a), root(b)) for y, a, b in cells]
        in_bits = [root(b) for b in in_bits]
        out_bits = [root(b) for b in out_bits]
    return _build_net(in_bits, out_bits, cells, spec)


FULL64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_ACTS_BUDGET = 2 << 30   # cap the (n_sig, words) signal buffer; chunking is result-transparent


def _fit(packed: np.ndarray, rows: int, words: int) -> np.ndarray:
    """Zero-extend a (rows, w) packed block to exactly `words` uint64 columns."""
    if packed.shape[1] == words:
        return packed
    out = np.zeros((rows, words * 8), np.uint8)
    src = packed.view(np.uint8)
    out[:, :src.shape[1]] = src
    return out.view(np.uint64)


def _pack_bits(x_bits: np.ndarray, words: int) -> np.ndarray:
    """(n, port_bits) uint8 bits -> (port_bits, words) uint64, 64 images per word."""
    packed = np.packbits(np.ascontiguousarray(x_bits.T), axis=1, bitorder="little")
    return _fit(packed, x_bits.shape[1], words)


def _pack_pix(pix: np.ndarray, spec: DatasetSpec, words: int) -> np.ndarray:
    """(n, n_pixels) uint8 -> (port_bits, words) uint64: the same bits `to_bits` produces, packed.

    Identical to `_pack_bits(to_bits(pix, spec), words)` but ~5x faster and without the (N, port_bits)
    intermediate: the shift/mask writes the transposed layout directly, so nothing has to byte-
    transpose a multi-MB bit array afterwards."""
    n = len(pix)
    pb = spec.pixel_bits
    pt = np.ascontiguousarray(pix.T)                              # (n_pixels, n)
    sh = np.arange(pb, dtype=np.uint8)[None, :, None]
    b = ((pt[:, None, :] >> sh) & np.uint8(1)).reshape(spec.port_bits, n)
    packed = np.packbits(np.ascontiguousarray(b), axis=1, bitorder="little")
    return _fit(packed, spec.port_bits, words)


def _words_for(net: NandNet, chunk: int, n_total: int) -> int:
    words = max(1, (min(chunk, n_total) + WORD - 1) // WORD) if n_total else 1
    while words > 1 and net.n_sig * words * 8 > _ACTS_BUDGET:
        words //= 2
    return words


def _simulate(net: NandNet, spec: DatasetSpec, chunks, n_total: int, words: int) -> np.ndarray:
    """Shared packed NAND simulation core. `chunks` yields (packed (port_bits, <=words), n)."""
    acts = np.empty((net.n_sig, words), np.uint64)   # allocated once, not per chunk
    levels = list(zip(net.src_a, net.src_b, net.offs, net.offs[1:]))
    weights = np.arange(len(net.out_sig), dtype=np.int64)[:, None]
    pred = np.empty(n_total, np.int64)
    pos = 0
    for packed, n in chunks:
        acts[CONST0] = 0
        acts[CONST1] = FULL64
        acts[2:2 + net.n_in] = packed
        for a, b, o0, o1 in levels:
            t = acts[o0:o1]                          # write the NAND straight into its own rows
            np.bitwise_and(acts[a], acts[b], out=t)
            np.invert(t, out=t)
        out = np.unpackbits(acts[net.out_sig].view(np.uint8), axis=1, bitorder="little")
        cls = (out.astype(np.int64) << weights).sum(0)
        pred[pos:pos + n] = cls[:n]
        pos += n
    if (pred >= spec.n_classes).any():
        raise RuntimeError(f"circuit produced class >= {spec.n_classes} -- argmax broken")
    return pred


def run(net: NandNet, x_bits: np.ndarray, spec: DatasetSpec, chunk: int = 4096) -> np.ndarray:
    n_total = len(x_bits)
    words = _words_for(net, chunk, n_total)
    step = words * WORD
    chunks = ((_pack_bits(x_bits[i:i + step], words), len(x_bits[i:i + step]))
              for i in range(0, n_total, step))
    return _simulate(net, spec, chunks, n_total, words)


def run_pix(net: NandNet, pix: np.ndarray, spec: DatasetSpec, chunk: int = 4096) -> np.ndarray:
    """`run(net, to_bits(pix, spec), spec)` without ever materialising the bit array (same result)."""
    n_total = len(pix)
    words = _words_for(net, chunk, n_total)
    step = words * WORD
    chunks = ((_pack_pix(pix[i:i + step], spec, words), len(pix[i:i + step]))
              for i in range(0, n_total, step))
    return _simulate(net, spec, chunks, n_total, words)


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


def measure_full(sv: Path, data: Dataset) -> tuple[dict, NandNet, np.ndarray]:
    """`measure` plus the circuit's predictions over the whole test set (they are computed
    for `test_acc` anyway, and `run_point` cross-checks the model against every one of them)."""
    spec = data.spec
    # yosys is single-threaded, so the (optional) sky130-GE runs -- which are independent full
    # syntheses of the same .sv -- overlap with the NAND synthesis + simulation instead of following
    # it. Same processes, same scripts, same numbers; only the wall clock changes. MNISTBENCH_SYNTH_
    # JOBS caps how many extra yosys processes may be live (default 1, i.e. 2 in total; 2 overlaps
    # all three, measured 2.6x); `_synth_jobs` drops back to serial when the .sv is big enough that
    # the extra yosys heaps would not fit.
    pool, ge_jobs = None, []
    if has_liberty():
        jobs = _synth_jobs(sv)
        if jobs > 0:
            pool = ThreadPoolExecutor(max_workers=jobs)
            submit = pool.submit
        else:
            submit = _Later                       # serial fallback, run on .result()
        ge_jobs = [submit(synth_ge, sv, script=FAST), submit(synth_ge, sv, script=GE_MAP)]
    try:
        return _measure(sv, data, spec, ge_jobs)
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


def measure(sv: Path, data: Dataset) -> tuple[dict, NandNet]:
    m, net, _ = measure_full(sv, data)
    return m, net


def _measure(sv: Path, data: Dataset, spec: DatasetSpec,
             ge_jobs: list) -> tuple[dict, NandNet, np.ndarray]:
    t0 = time.time()
    nand = synth_nand(sv, script=FAST, spec=spec)
    net = nand.net if nand.net is not None else from_json(nand.netlist, spec)
    nand.netlist = nand.net = None   # free the netlist objects before the simulator allocates
    print(f"[gates] {nand.gates:,} 2-input gates ({nand.nand:,} NAND + {nand.inv:,} INV), "
          f"depth {net.depth}, {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    pred = run_pix(net, data.test_x, spec)          # the circuit's decision per test image
    acc = float((pred == data.test_y).mean()) * 100
    print(f"[sim  ] test acc {acc:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
    m = {"gates": nand.gates, "nand": nand.nand, "inv": nand.inv, "depth": net.depth,
         "test_acc": round(acc, 2)}
    if ge_jobs:
        try:
            # sky130 gate-equivalents, mapped WITHOUT and WITH ABC's dc2 optimisation. The ratio is
            # the "compressibility after training": how much ABC still removes per method.
            ge_post, area_post, cells_post = ge_jobs[0].result()           # after ABC
            ge_pre, area_pre, cells_pre = ge_jobs[1].result()              # before ABC (map only)
            ratio = round(ge_pre / max(ge_post, 1e-9), 3)
            m |= {"ge": round(ge_post, 1), "area_um2": round(area_post, 1), "cells": cells_post,
                  "ge_pre_abc": round(ge_pre, 1), "ge_post_abc": round(ge_post, 1),
                  "cells_pre_abc": cells_pre, "cells_post_abc": cells_post,
                  "ge_abc_ratio": ratio}  # >1 means ABC compressed it that many x
            print(f"[ge   ] sky130 pre-ABC {ge_pre:,.0f} -> post-ABC {ge_post:,.0f} GE  "
                  f"({ratio:.2f}x, {cells_post:,} cells)", flush=True)
        except RuntimeError as e:
            print(f"[ge   ] skipped ({e})", flush=True)
    return m, net, pred


def run_point(mod: ModuleType, point: dict, data: Dataset, *, device: str, seed: int,
              gpus: int = 1) -> dict:
    """Train one point, emit it, synthesize it, and measure the SYNTHESIZED CIRCUIT.

    The reported (gates, accuracy, loss) triple has to be one real operating point of the emitted
    ASIC, so every axis is tied back to the netlist:

      * `gates`/`depth` come from the netlist itself;
      * `test_acc` is the netlist simulated over the whole test set -- it is the hardware number,
        never the model's;
      * `predict()` is checked against that circuit on EVERY test image (not a sample), so `val_acc`
        -- which has no netlist to run against, the val split being the model's own -- is produced
        by a function proven equal to the hardware on all 10,000 test images;
      * `argmax(scores())` is checked against the same circuit predictions, which is what ties the
        CE / perplexity axis to the emitted hardware rather than to the trainer.

    How a method gets there is its own business: training may use a soft/relaxed net and evaluate
    the hardened one -- that gap is by design and is not checked here. The only pairing that must
    hold exactly is eval <-> emitted circuit.

    RESIDUAL LIMITATION, on purpose: the netlist's only output is the class index (`cls`), so the
    harness can validate the ARGMAX of `scores()` against hardware but never the magnitudes of the
    scores. `test_ce` / `test_ppl` therefore rest on `scores()` being the circuit's readout by
    CONSTRUCTION -- `methods.lut.lut_sim` mirroring `hw.emit_lutnet` for the logic nets, the exact
    integer forward for the quantized MLPs, `_score_int` for forest. Small float differences in
    those magnitudes are fine; a structural divergence would not be caught here, only its effect on
    the argmax would. Emitting the readout counts from the circuit would fix that and is deliberately
    NOT done: it would change the design and the gate count, which is the headline measurement.
    """
    spec = data.spec
    cfg = {k: v for k, v in point.items() if k != "name"}
    print(f"\n=== {spec.name}/{mod.__name__.split('.')[-1]}/{point['name']}  {cfg}  seed={seed}",
          flush=True)
    model = mod.build(spec, **cfg)
    model.spec = spec
    model.ddp_gpus = gpus  # torch methods use one DDP rank per GPU; the rest ignore it
    stem = _stem(mod.__name__.split(".")[-1], spec.name, point["name"], seed)

    t0 = time.time()
    model.train(data, device=device, seed=seed)
    train_wall_s = time.time() - t0
    # PURE training time: only the training compute the method measured internally (no validation, no
    # data staging, no synth/measure). This is the axis for "how fast does each method train".
    train_s = float(getattr(model, "train_seconds", train_wall_s))
    # samples the trainer looked at before early-stopping (training-example forward passes):
    # epochs*Ntrain for the gradient methods, gens*pop(or k)*batch for es/genetic, rounds*Ntrain
    # for forest. Pairs with train_s as "how long / how much data until it converged".
    train_samples = int(getattr(model, "train_samples", 0))
    print(f"[train] {train_s:.0f}s pure ({train_wall_s:.0f}s wall incl. val)  "
          f"{train_samples:,} samples seen", flush=True)

    if hasattr(model, "save"):
        model.save(str(stem) + ".ckpt")
        print(f"[ckpt ] {stem.name}.ckpt", flush=True)

    sv = Path(str(stem) + ".sv")
    sv.write_text(model.emit_verilog())
    print(f"[emit ] {sv.name}, {sv.stat().st_size / 1e6:.1f} MB", flush=True)

    m, net, hw = measure_full(sv, data)          # hw: the circuit's class per test image
    scores = getattr(model, "scores", lambda _p: None)
    n_test = len(hw)

    # Every test image, not a sample of them. The test-set calls are kept adjacent so a model that
    # memoises its forward (LutModel._counts, the quant cache) serves predict+scores from one pass.
    t0 = time.time()
    py = np.asarray(model.predict(data.test_x))
    if not (hw == py).all():
        bad = np.flatnonzero(hw != py)
        raise SystemExit(f"REJECTED: circuit != model on {len(bad)}/{n_test} test images "
                         f"(e.g. {bad[:5].tolist()}). emit_verilog must equal predict.")
    sc_te = scores(data.test_x)
    if sc_te is not None:
        # argmax only -- the circuit reports a class, not counts, so magnitudes are unverifiable
        # here (see the docstring). Both sides break ties toward the lowest class, as the emitted
        # argmax does, so an equal-score tie is not a mismatch.
        sa = np.asarray(sc_te, float).argmax(1)
        if not (sa == hw).all():
            bad = np.flatnonzero(sa != hw)
            raise SystemExit(f"REJECTED: argmax(scores) != circuit on {len(bad)}/{n_test} test "
                             f"images (e.g. {bad[:5].tolist()}). the CE axis must describe the same "
                             f"circuit as the accuracy axis.")
    print(f"[check] model == circuit on all {n_test:,} test images"
          f"{'' if sc_te is None else ' (predict and argmax(scores))'}  "
          f"({time.time() - t0:.0f}s)", flush=True)

    val_acc = float((np.asarray(model.predict(data.val_x)) == data.val_y).mean()) * 100
    sc_va = scores(data.val_x)
    out = {"name": point["name"], "method": mod.__name__.split(".")[-1], "dataset": spec.name,
           "seed": seed, "config": cfg, **m, "val_acc": round(val_acc, 2),
           "train_s": round(train_s, 1), "train_wall_s": round(train_wall_s, 1),
           "train_samples": train_samples, "device": device}

    if sc_te is not None:
        t = _fit_temperature(np.asarray(sc_va, float), data.val_y)
        ce = _cross_entropy(np.asarray(sc_te, float) / t, data.test_y)
        out |= {"test_ce": round(ce, 4), "test_ppl": round(float(np.exp(ce)), 4),
                "ce_temp": round(float(t), 4)}

    Path(str(stem) + ".json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[write] {stem.name}.json", flush=True)
    return out


def run_method(name: str, data: Dataset, *, device: str, seed: int,
               only: list[str] | None, force: bool, gpus: int = 1) -> None:
    mod = load_method(name)
    for point in mod.points(data.spec):
        if only and point["name"] not in only:
            continue
        stem = _stem(name, data.spec.name, point["name"], seed)
        if Path(str(stem) + ".json").exists() and not force:
            print(f"=== {data.spec.name}/{name}/{point['name']}: done, skipping (--force)", flush=True)
            continue
        run_point(mod, point, data, device=device, seed=seed, gpus=gpus)


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
