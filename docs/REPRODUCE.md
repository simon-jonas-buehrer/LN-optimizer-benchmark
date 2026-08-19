# Reproducing every number

The repo runs anywhere with Python + yosys; the two paper runs (a 4×RTX3090 box, then the cluster)
execute the *same* code — only the environment and scheduling differ.

## 1. Python

```bash
pip install -e .          # numpy + torch + matplotlib
```

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

## 3. Run

```bash
python run.py all --device cuda           # the whole matrix, resumable
# or one cell, or spread across 4 GPUs by launching several:
python run.py run backprop --dataset mnist --device cuda:0 &
python run.py run forest   --dataset cifar10 --device cuda:1 &
python run.py plots                        # the four figures + results/leaderboard.md
```

Results land in `results/<dataset>/<method>/<point>.s<seed>.{sv,ckpt,json}`. A point whose `.json`
exists is skipped unless `--force`, so the matrix survives interruption. `rescore` re-derives both
axes from a stored `.sv` without retraining.

## Budget & caveats

* **GPU trains, CPU synthesizes.** yosys+ABC is single-threaded per point and CPU-bound; the 3090s do
  not help there. The largest (~20M-gate) points dominate wall-clock even under the fast script.
* **24 GB ceiling.** The biggest logic nets may not fit a single 3090's memory in training; cap such a
  method at the largest size that fits and note it — the *union* of methods still spans 1k→20M.
* **Quantized references** sit at the high-gate end and do not reach ~1k gates (an arithmetic floor);
  their curves start higher, which is reported, not forced.
* **CIFAR10 is hard for logic nets** — low absolute accuracy is expected and is itself a finding.

## Cluster

Cluster-specific `sbatch` wrappers live in the gitignored `.local/` (see `.local/README.md`); they
just `exec python run.py ...` inside the SLURM job, so cluster and local runs are the same code.
