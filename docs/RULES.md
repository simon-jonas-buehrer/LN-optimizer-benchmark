# The measurement protocol

Every model is judged as a circuit. We train it, emit SystemVerilog, and measure the circuit.

## The axes

| axis | what it is | where it comes from |
|---|---|---|
| x | circuit size, in **2-input gates** | `yosys` + `ABC` map the emitted Verilog to a NAND2+INV netlist with one fast script; the area is the raw gate count (NAND + INV) |
| y | test accuracy | the same NAND netlist is simulated gate-by-gate on all test images |

There is also a **loss** curve (temperature-calibrated cross-entropy over the readout's per-class
scores) and **perplexity** = exp(CE). Optionally, when a sky130 liberty is configured, we also record
**gate equivalents** (sky130 area / NAND2 area) as a secondary metric — but the headline area is the
2-input gate count, so the default pipeline needs only yosys+ABC, no PDK.

Nothing a model reports about itself is used. `predict()` exists only so the harness can check that the
python model and the emitted circuit are the same function; if they disagree on even one of 512 sampled
images, the point is rejected.

## The circuit contract

Derived from the dataset's `DatasetSpec`:

```verilog
module top (input [port_bits-1:0] pix, output [cls_bits-1:0] cls);   // combinational; no clock, no memory
```

* `pix[pixel_bits*p +: pixel_bits]` is byte `p`. MNIST: 784 bytes (`pix[6271:0]`). CIFAR10: 3072 raw
  RGB bytes, interleaved R,G,B per pixel (`pix[24575:0]`).
* `cls` is the predicted class (`cls_bits = ceil(log2(n_classes))`, = 4 for 10 classes).
* Purely combinational. Everything between the ports is counted: the encoder, the logic, the readout,
  the argmax. No free preprocessing, no free softmax.

## The optimizer is frozen

Every model is synthesized with the **same fast ABC script** (`harness.FAST = strash;dc2;map`): one
strash, one `dc2`, and a technology map to NAND2+INV. It is deliberately fast and imperfect — it folds
away dead and constant logic and does a cheap resubstitution, but does not run the multi-pass `resyn2`
sweep — so a ~20M-gate design still maps in bounded time and every method gets exactly the same effort.
The script is a constant, not a tunable.

Because the optimizer deletes whatever you wasted (dead gates, constant logic, an unread pixel), **you
pay for the circuit you need, not the one you wrote.**

## What a method may and may not do

* Train on `data.train_*`; tune / early-stop on `data.val_*`. **Never fit on the test set.**
* Any optimizer, any architecture, any seed, any compute. The axes are gates and accuracy.
* Every method trains to convergence with early stopping on **validation loss**, and reports **≥5
  size points** spanning as much of the ~1k → ~20M gate range as it reaches (sizes are targeted by the
  *pre-optimization* gate estimate; the reduced count is only known after ABC).

## Reproducing

See [REPRODUCE.md](REPRODUCE.md). `python run.py rescore <method> --dataset <ds>` re-measures the
stored `.sv` without retraining — both axes come from the Verilog alone.
