# Reproducing every number

The repo runs anywhere with Python + yosys. Training uses a GPU; **measuring** (emit → yosys/ABC →
NAND-netlist simulation) is integer-exact numpy and does not depend on the torch build at all.

## 1. Python

```bash
pip install -e .                  # numpy + torch + matplotlib, loose pins
```

### The environment of record

`pyproject.toml` keeps loose pins so the repo also runs on newer drivers and on CPU. A different
torch build trains a numerically different net, so to reproduce the *published* numbers use the
exact environment every result in `results/` came from — [`requirements-lock.txt`](../requirements-lock.txt):

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
pip install -r requirements-lock.txt
```

Python 3.11.15, torch 2.6.0+cu124, numpy 2.4.6, on 4× NVIDIA RTX 3090 (24 GB), driver 535 / CUDA
12.2, Debian 12. yosys 0.68+ (conda-forge, git sha1 `36d0660ff`) with its bundled ABC.

## 2. yosys + ABC (+ optional sky130)

Not pip-installable. One conda env:

```bash
conda create -n eda -c conda-forge -c litex-hub yosys open_pdks.sky130a
export MNISTBENCH_YOSYS=/path/to/eda/bin/yosys
# optional: only needed for the secondary sky130 gate-equivalents metric
export MNISTBENCH_LIBERTY=/path/to/eda/.../sky130_fd_sc_hd__tt_025C_1v80.lib
# or, to derive both from the env prefix:
export MNISTBENCH_EDA=/path/to/eda
```

The default headline area (2-input gate count) needs only `MNISTBENCH_YOSYS`.

**The GE metric costs 3× the measurement.** With a liberty configured, every point is synthesized
three times (NAND for the headline number, sky130 map-only, sky130 post-ABC). Synthesis is the
wall-clock bottleneck at the large tiers, so the recommended order is: sweep with
`MNISTBENCH_YOSYS` only, then add GE afterwards from the stored `.sv`, without retraining:

```bash
MNISTBENCH_EDA=/path/to/eda python run.py rescore backprop --dataset mnist
```

## 3. Run

```bash
python run.py all --device cuda           # the whole matrix, resumable
# or one cell, or spread across 4 GPUs by launching several:
python run.py run backprop --dataset mnist --device cuda:0 &
python run.py run forest   --dataset cifar10 --device cuda:1 &
python run.py plots                        # the four figures + results/leaderboard.md
```

Results land in `results/<dataset>/<method>/<point>.s<seed>.{sv,ckpt,json}`. Only the `.json` files
and the figures are committed; the `.sv` and `.ckpt` are regenerable and gitignored. A point whose
`.json` exists is skipped unless `--force`, so the matrix survives interruption. `rescore`
re-derives both axes from a stored `.sv` without retraining.

### When a point fails

By default one failing point does **not** take the rest of the ladder with it: the traceback is
printed, written to `results/<dataset>/<method>/<point>.s<seed>.failed.txt`, and the run continues
with the next size. No `.json` is written for a failed point, so a rejected or unmeasurable model
can never reach the plots; the process still exits non-zero with a list of what failed. Pass
`--halt-on-error` to stop at the first failure instead.

## Budget & caveats

* **GPU trains, CPU synthesizes.** yosys+ABC is single-threaded per point and CPU-bound; the GPUs
  do not help there. The largest tiers dominate wall-clock even under the fast script.
* **Synthesis memory is the hard ceiling, not GPU memory.** Measured with this repo's emitters:
  a 3.9 MB `.sv` peaks at ~3 GB of yosys RSS, a 15.9 MB `.sv` at ~14 GB — roughly **1 GB of RAM per
  MB of emitted Verilog**, and both time and memory grow faster than linearly in the gate count.
  Size the machine (or the SLURM `--mem`) against the `.sv`, not against the model.
* **24 GB GPU ceiling.** The biggest logic nets may not fit a single 3090 in training; cap such a
  method at the largest size that fits and note it — the *union* of methods still spans the range.
* **Quantized references** sit at the high-gate end and do not reach ~1k gates (an arithmetic
  floor); their curves start higher, which is reported, not forced.
* **CIFAR10 is hard for logic nets** — low absolute accuracy is expected and is itself a finding.
* **`train_s` is not cross-method comparable.** It is wall-clock on whatever device that method
  ran on (4×3090 DDP for backprop/dfa/quant, 1×3090 for es/genetic, CPU for forest). It is a
  within-method scaling quantity; comparing it across methods compares hardware allocations, not
  optimizers.

## Cluster

Cluster-specific `sbatch` wrappers live in the gitignored `.local/` (see `.local/README.md`); they
just `exec python run.py ...` inside the SLURM job, so cluster and local runs are the same code.
