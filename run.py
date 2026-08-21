"""CLI + experiment matrix.

    python run.py run <method> --dataset mnist [--point s] [--seed 0] [--device cuda] [--force]
    python run.py all [--datasets mnist cifar10] [--methods backprop ...] [--seeds 0 1] [--device cuda]
    python run.py rescore <method> --dataset mnist [--seed 0]      # re-measure stored .sv, no retrain
    python run.py plots                                            # the four figures + leaderboard

The five logic-net optimizers and three quantized references, on MNIST and CIFAR10; results land in
results/<dataset>/<method>/<point>.s<seed>.{sv,ckpt,json}. Resumable: a point whose .json exists is
skipped unless --force. Every method trains on ONE device; to use several GPUs, launch several
`run --device cuda:N` in parallel (the gitignored .local/ has sbatch wrappers for the cluster).
"""

from __future__ import annotations

import argparse

import data as D
import harness

METHODS = ["backprop", "genetic", "dfa", "forest", "es", "w1_58a4", "w1_58a8", "w4a4"]


def main():
    ap = argparse.ArgumentParser(prog="run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="train + measure one method on one dataset")
    r.add_argument("method", choices=METHODS)
    r.add_argument("--dataset", choices=D.NAMES, required=True)
    r.add_argument("--point", action="append")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="cpu")
    r.add_argument("--force", action="store_true")
    r.add_argument("--halt-on-error", action="store_true",
                   help="stop at the first failing point instead of finishing the rest of the ladder")

    a = sub.add_parser("all", help="run the whole matrix (method x dataset x seed)")
    a.add_argument("--datasets", nargs="+", choices=D.NAMES, default=list(D.NAMES))
    a.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    a.add_argument("--seeds", nargs="+", type=int, default=[0])
    a.add_argument("--device", default="cpu")
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
                                           only=None, force=args.force,
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
    harness.run_method(args.method, dat, device=args.device, seed=args.seed,
                       only=args.point, force=args.force,
                       keep_going=not args.halt_on_error)


if __name__ == "__main__":
    main()
