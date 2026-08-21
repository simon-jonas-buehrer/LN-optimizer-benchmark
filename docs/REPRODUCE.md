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
# Every method trains in ONE process on ONE device, so several GPUs means several processes --
# one method x dataset cell each:
python run.py run backprop --dataset mnist   --device cuda:0 &
python run.py run forest   --dataset cifar10 --device cpu    &
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
* **Synthesis memory is the hard ceiling, not GPU memory** — and `.sv` size does not predict it.
  Measured peak yosys RSS per MB of emitted Verilog, across this repo's own emitters:

  | method | point | `.sv` | peak RSS | GB per MB |
  |---|---|---|---|---|
  | forest | xl | 7.5 MB | 3.6 GB | 0.5 |
  | es | l | 11.2 MB | 11.1 GB | 1.0 |
  | backprop | l | 11.2 MB | 11.4 GB | 1.0 |
  | genetic | l | 0.8 MB | 1.1 GB | 1.4 |
  | w1_58a4 | xs | 0.8 MB | 9.0 GB | 11.2 |
  | dfa | l | 4.0 MB | 51.0 GB | 12.8 |

  A 25x spread, and the two most expensive rows are among the *smallest* files. The reason is that
  Verilog bytes are not netlist size: a wide popcount is a couple of compact lines that elaborate
  into an enormous adder network. dfa's readout sums `readout/n_classes` = 9,000 bits per class at
  the `l` tier, and the quantized references sum a full integer MAC per output — both cost orders of
  magnitude more RAM than their file size suggests, while forest's many small independent trees cost
  less. Size `--mem` against the *elaborated* design (how wide are the popcounts, how many gate
  instances before optimisation), never against the byte count, and leave a large margin: both time
  and memory grow faster than linearly in the gate count.
* **24 GB GPU ceiling.** The biggest logic nets may not fit a single 3090 in training; cap such a
  method at the largest size that fits and note it — the *union* of methods still spans the range.
* **Why each ladder stops where it does.** The tiers are bounded by what can be trained AND
  measured inside one 48 h job, which was established by measurement rather than assumed:

  | method | top tier | what bounds it |
  |---|---|---|
  | backprop, es, dfa | `xl` | `xxl` = 20M LUT nodes needs ~2 TB of yosys RSS; nodes have 515 GB |
  | genetic | `l` | early stopping needs 100,000 generations minimum = 74 h floor at the old `xl` |
  | quant | `m` | `l` needs ~790 GB on CIFAR10, `xl`/`xxl` need ~2 TB / ~8 TB |
  | forest | `xxl` | nothing -- it is cheap throughout, and its own early stopping caps it at ~635k gates |

  Two of these are ceilings of the *method*, not of the hardware, and are findings in themselves:
  dfa's readout replicates rather than grows above `m` (only `width/2` distinct input pairs exist,
  so at most `8*width` distinct readout gates), and forest's boosting early-stops at 56 of 1,200
  rounds, so its top tiers converge to the same circuit. Both were measured, not extrapolated.

* **Quantized references** sit at the high-gate end and do not reach ~1k gates (an arithmetic
  floor); their curves start higher, which is reported, not forced.
* **CIFAR10 is hard for logic nets** — low absolute accuracy is expected and is itself a finding.
* **`train_s` is not cross-method comparable.** It is wall-clock on whatever device that method
  ran on (one RTX 3090 for every method except forest, which is CPU). It is a within-method
  scaling quantity; comparing it across methods compares hardware allocations, not optimizers.

## Cluster

Cluster-specific `sbatch` wrappers live in the gitignored `.local/` (see `.local/README.md`); they
just `exec python run.py ...` inside the SLURM job, so cluster and local runs are the same code.
The GPU sweep is a four-task array holding one GPU each, which schedules where a single four-GPU
allocation would not.
