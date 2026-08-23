# Reproducing every number

The repo runs anywhere with Python + yosys. Training uses a GPU; **measuring** (emit → yosys/ABC →
NAND-netlist simulation) is integer-exact numpy and does not depend on the torch build at all.

## 1. Python — uv

```bash
uv sync            # creates .venv from uv.lock: Python 3.11, torch 2.6.0+cu124, numpy 2.4.6
uv run run.py …    # every command below; re-checks the environment first, so `uv sync` is optional
```

`uv.lock` **is** the environment of record — the exact set of packages every number in `results/`
came from, resolved for Linux x86_64 / Python 3.11. Nothing to activate: `uv run` runs against it.

Two things in [`pyproject.toml`](../pyproject.toml) make the lock reproduce rather than drift:

* **torch comes from the cu124 wheel index**, not PyPI (`[tool.uv.sources]` + an explicit
  `[[tool.uv.index]]`). The GPUs here are RTX 3090s on driver 535 / CUDA 12.2; a default PyPI torch
  is built against a newer CUDA and fails to load with *driver too old*.
* **`constraint-dependencies` pins the resolve** to torch 2.6.0 / numpy 2.4.6 / matplotlib 3.11.1,
  while `dependencies` stays loose (`torch>=2.2`) so the pip route below still works on CPU and on
  newer drivers.

Python 3.11.15, torch 2.6.0+cu124, numpy 2.4.6, on 4× NVIDIA RTX 3090 (24 GB), driver 535 / CUDA
12.2, Debian 12. yosys 0.68+ (conda-forge, git sha1 `36d0660ff`) with its bundled ABC.

### Without uv

`pip install -e .` gets the loose pins — portable, but a different torch build trains a numerically
different net, so it does not reproduce the published numbers. The same environment by hand, from
[`requirements-lock.txt`](../requirements-lock.txt):

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
pip install -r requirements-lock.txt
```

That file and `uv.lock` are the same environment, package for package.

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
MNISTBENCH_EDA=/path/to/eda uv run run.py rescore backprop --dataset mnist
```

## 3. Run

```bash
uv run run.py all --device cuda           # the whole matrix, resumable
# Every method trains in ONE process on ONE device, so several GPUs means several processes --
# one method x dataset cell each:
uv run run.py run backprop --dataset mnist   --device cuda:0 &
uv run run.py run forest   --dataset cifar10 --device cpu    &
uv run run.py plots                        # the four figures + results/leaderboard.md
```

`uv run` syncs the environment before each command, which is a few milliseconds when nothing
changed. To skip that check — inside a batch job, or when several processes share one `.venv` and
must not race to rewrite it — use `uv run --no-sync --frozen run.py …` and sync once up front.

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

## Running it somewhere else

Nothing here assumes a particular machine, scheduler or site. The whole sweep is

```bash
uv run run.py all --phase train --device cuda:0     # -> .ckpt / .train.json
uv run run.py all --phase emit                      # -> .sv
uv run run.py all --phase synth                     # -> .pre_opt.sv / .post_opt.sv / .json
```

and `uv run run.py all` runs all three in one process on one machine.

The phases are separable precisely so they can go to different hardware, because they want very
different things: training wants a GPU and a few GB of host RAM, emitting wants neither and takes
seconds, and yosys wants no GPU at all and — for the widest quantized points — hundreds of GB. Fused
into one allocation, every GPU worker would also have to reserve the worst synthesis footprint.

Each phase resumes by skipping points whose own artifact already exists and passing over points
whose input is not ready, so all three are safe to re-run, to interrupt, and to run concurrently
with each other. That is all a batch script needs: run the same commands under whatever scheduler
you have. For several GPUs, launch one `--phase train --device cuda:N` per card; each writes to a
different point and they do not interact.

Set `MNISTBENCH_RESULTS` to redirect the output tree, `MNISTBENCH_YOSYS` to a yosys binary, and
`MNISTBENCH_LIBERTY` (or `MNISTBENCH_EDA`) to a sky130 liberty for the optional gate-equivalent
axis. Without a liberty the sweep runs exactly as before and simply reports no GE.
