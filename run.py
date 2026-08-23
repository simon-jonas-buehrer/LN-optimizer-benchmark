"""CLI + experiment matrix.

    python run.py train <method> --dataset mnist [--point s] [--device cuda]   # GPU, .ckpt only
    python run.py emit  <method> --dataset mnist [--point s]                   # .ckpt -> .sv
    python run.py synth <method> --dataset mnist [--point s]                   # yosys half, no GPU
    python run.py run   <method> --dataset mnist [--point s] [--device cuda]   # all three, one proc
    python run.py all [--datasets ...] [--methods ...] [--phase train|emit|synth|all]
    python run.py rescore <method> --dataset mnist [--seed 0]      # re-measure stored .sv, no retrain
    python run.py plots                                            # the four figures + leaderboard

The five logic-net optimizers and three quantized references, on MNIST and CIFAR10; results land in
results/<dataset>/<method>/<point>.s<seed>.{sv,ckpt,train.json,json}.

THREE PHASES, ON PURPOSE, BECAUSE THEY WANT THREE DIFFERENT MACHINES.

  train   GPU + a few GB of host RAM     -> .ckpt, .train.json
  emit    no GPU, seconds, ~1 GB         -> .sv          (a pure function of .ckpt + the emitter)
  synth   no GPU, minutes to hours, up to ~900 GB of host RAM
                                         -> .pre_opt.sv, .post_opt.sv, .json

`emit` is separate from `train` because the checkpoint is the only expensive artifact: improving
the RTL then costs one cheap pass over results/ rather than a retrain, and re-emitting every point
from ONE emitter is what keeps a ladder's gate counts comparable when training spans a code change.
`synth` reloads the .ckpt, synthesizes the .sv, and cross-checks the netlist against that reloaded
model on every test image before writing the measured .json.

Resumable in all three: each skips a point whose own product already exists and passes over one
whose input is not ready yet. Every method uses ONE device; for several GPUs, launch several
`train --device cuda:N` in parallel -- they write different points and do not interact.

Nothing here is tied to a scheduler or a site: a batch script only has to run these same commands.
See docs/REPRODUCE.md, and the MNISTBENCH_* environment variables in harness.py for redirecting the
output tree and pointing at a yosys/liberty of your own.
"""

from __future__ import annotations

import argparse

import data as D
import harness

METHODS = ["backprop", "genetic", "dfa", "forest", "es", "w1_58a4", "w1_58a8", "w4a4"]


def main():
    ap = argparse.ArgumentParser(prog="run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for cmd, helptext in (("run", "train + emit + measure one method on one dataset (one process)"),
                          ("train", "train one method on one dataset: writes .ckpt, no RTL, no yosys"),
                          ("emit", "write the .sv for a trained method from its .ckpt: no GPU, no yosys"),
                          ("synth", "synthesize + measure an emitted method: no GPU, no training")):
        r = sub.add_parser(cmd, help=helptext)
        r.add_argument("method", choices=METHODS)
        r.add_argument("--dataset", choices=D.NAMES, required=True)
        r.add_argument("--point", action="append")
        r.add_argument("--seed", type=int, default=0)
        if cmd in ("run", "train"):
            r.add_argument("--device", default="cpu")
        r.add_argument("--force", action="store_true")
        r.add_argument("--halt-on-error", action="store_true",
                       help="stop at the first failing point instead of finishing the ladder")

    a = sub.add_parser("all", help="run the whole matrix (method x dataset x seed)")
    a.add_argument("--datasets", nargs="+", choices=D.NAMES, default=list(D.NAMES))
    a.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    a.add_argument("--seeds", nargs="+", type=int, default=[0])
    a.add_argument("--device", default="cpu")
    a.add_argument("--phase", choices=["all", "train", "emit", "synth"], default="all",
                   help="run only the GPU phase, only the RTL phase, only the yosys phase, or all")
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
    phase = {"run": "all", "train": "train", "emit": "emit", "synth": "synth"}[args.cmd]
    harness.run_method(args.method, dat, device=getattr(args, "device", "cpu"), seed=args.seed,
                       only=args.point, force=args.force, phase=phase,
                       keep_going=not args.halt_on_error)


if __name__ == "__main__":
    main()
