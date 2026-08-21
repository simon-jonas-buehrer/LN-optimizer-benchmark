"""CLI + experiment matrix.

    python run.py train <method> --dataset mnist [--point s] [--device cuda]   # GPU half
    python run.py synth <method> --dataset mnist [--point s]                   # yosys half, no GPU
    python run.py run   <method> --dataset mnist [--point s] [--device cuda]   # both, one process
    python run.py all [--datasets ...] [--methods ...] [--seeds 0 1] [--phase train|synth|all]
    python run.py rescore <method> --dataset mnist [--seed 0]      # re-measure stored .sv, no retrain
    python run.py plots                                            # the four figures + leaderboard

The five logic-net optimizers and three quantized references, on MNIST and CIFAR10; results land in
results/<dataset>/<method>/<point>.s<seed>.{sv,ckpt,train.json,json}.

TRAIN AND SYNTH ARE SEPARABLE ON PURPOSE. They want opposite machines: training needs a GPU and a
few GB of host RAM, while yosys needs no GPU and up to ~350 GB of host RAM. `train` writes the
.ckpt and .sv; `synth` reloads the .ckpt, synthesizes the .sv, and cross-checks the netlist against
that reloaded model before writing the measured .json. Run them as one process with `run` on a
single machine, or as two differently-shaped batch jobs on a cluster.

Resumable in both halves: `train` skips a point that already has its .ckpt/.sv, `synth` skips one
that already has its .json and passes over one that is not trained yet. Every method uses ONE
device; for several GPUs, launch several `train --device cuda:N` in parallel (the gitignored
.local/ has sbatch wrappers for the cluster).
"""

from __future__ import annotations

import argparse

import data as D
import harness

METHODS = ["backprop", "genetic", "dfa", "forest", "es", "w1_58a4", "w1_58a8", "w4a4"]


def main():
    ap = argparse.ArgumentParser(prog="run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for cmd, helptext in (("run", "train + measure one method on one dataset (one process)"),
                          ("train", "train one method on one dataset: writes .ckpt/.sv, no yosys"),
                          ("synth", "synthesize + measure a trained method: no GPU, no training")):
        r = sub.add_parser(cmd, help=helptext)
        r.add_argument("method", choices=METHODS)
        r.add_argument("--dataset", choices=D.NAMES, required=True)
        r.add_argument("--point", action="append")
        r.add_argument("--seed", type=int, default=0)
        if cmd != "synth":
            r.add_argument("--device", default="cpu")
        r.add_argument("--force", action="store_true")
        r.add_argument("--halt-on-error", action="store_true",
                       help="stop at the first failing point instead of finishing the ladder")

    a = sub.add_parser("all", help="run the whole matrix (method x dataset x seed)")
    a.add_argument("--datasets", nargs="+", choices=D.NAMES, default=list(D.NAMES))
    a.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    a.add_argument("--seeds", nargs="+", type=int, default=[0])
    a.add_argument("--device", default="cpu")
    a.add_argument("--phase", choices=["all", "train", "synth"], default="all",
                   help="run only the GPU half, only the yosys half, or both")
    a.add_argument("--force", action="store_true")
    a.add_argument("--halt-on-error", action="store_true",
                   help="stop at the first failing point instead of finishing the rest of the matrix")

    s = sub.add_parser("rescore", help="re-measure stored .sv without retraining")
    s.add_argument("method", choices=METHODS)
    s.add_argument("--dataset", choices=D.NAMES, required=True)
    s.add_argument("--seed", type=int, default=0)

    sub.add_parser("plots", help="rebuild the four figures + leaderboard")

    args = ap.parse_args()
    if args.cmd == "plots":
        import plots
        return plots.main()

    if args.cmd == "all":
        # One method x dataset cell that fails must not cost the rest of the matrix: collect the
        # failures, keep going, and report them all at the end with a non-zero exit.
        failures = []
        for ds in args.datasets:
            dat = D.load(ds)
            print(f"\n##### {ds}: train {dat.train_x.shape}, val {dat.val_x.shape}, "
                  f"test {dat.test_x.shape}", flush=True)
            for seed in args.seeds:
                for method in args.methods:
                    try:
                        harness.run_method(method, dat, device=args.device, seed=seed,
                                           only=None, force=args.force, phase=args.phase,
                                           keep_going=not args.halt_on_error)
                    except SystemExit as e:
                        if args.halt_on_error:
                            raise
                        print(f"!!! {e}", flush=True)
                        failures.append(str(e))
        if failures:
            raise SystemExit("\n".join(["INCOMPLETE MATRIX:", *failures]))
        return

    dat = D.load(args.dataset)
    print(f"{args.dataset}: train {dat.train_x.shape}, val {dat.val_x.shape}, test {dat.test_x.shape}",
          flush=True)
    if args.cmd == "rescore":
        return harness.rescore_method(args.method, dat, args.seed)
    phase = {"run": "all", "train": "train", "synth": "synth"}[args.cmd]
    harness.run_method(args.method, dat, device=getattr(args, "device", "cpu"), seed=args.seed,
                       only=args.point, force=args.force, phase=phase,
                       keep_going=not args.halt_on_error)


if __name__ == "__main__":
    main()
