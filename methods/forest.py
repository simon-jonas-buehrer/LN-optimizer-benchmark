"""forest: SAMME-boosted decision trees over thermometer bits, emitted straight to gates.

A decision tree over binary features IS a boolean function: each root-to-leaf path is a conjunction
of literals, so leaf indicators are a shared-prefix AND network and the paths reaching a class form
a DNF. Boost a forest, weight each tree by an integer, sum per class and argmax -- the whole model
is a circuit. No gradient, no LUT net, no search; this method emits its OWN `top` module (it does
not use the fan-in-2 LutModel).

Three things make it cheap in silicon:
  * reach wires: reach(child) = reach(parent) & +/-literal is 2 gates per internal node, so a tree
    costs 2(L-1) gates regardless of shape -- area tracks leaves, not depth (depth is free).
  * the class indicator partitions: each leaf carries one class, so the per-class ORs are disjoint
    and a single reach network scores every class.
  * bit-plane popcount: w_t is a constant and the class bit is one wire, so a zero weight-bit costs
    no hardware at all.

Weights are quantized INSIDE the boosting loop (alpha -> int of `wbits` before the sample-weight
update), so later trees fit the residual of the circuit-exact ensemble and predict() == the netlist
bit-for-bit. Boosting rounds are added until validation cross-entropy stops improving; `trees` is a
ceiling.
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from data import Dataset, DatasetSpec
from hw import even_thresholds

TITLE = "forest (SAMME-boosted decision trees, integer-weighted vote)"

# `bits` thermometer bits/pixel (hw.even_thresholds), `leaves` the per-tree leaf budget (the only
# capacity knob -- depth is free), `wbits` the integer tree-weight width, `trees` the boosting-round
# CEILING (early stopping on val loss keeps fewer). Small tiers match the original record's shapes;
# leaves/trees then climb so the pre-opt gate estimate (~3*trees*leaves) spans ~1k -> ~20M.
_LADDER = {
    "xxs": dict(bits=3, leaves=8,    wbits=2, trees=5),
    "xs":  dict(bits=7, leaves=16,   wbits=2, trees=9),
    "s":   dict(bits=3, leaves=128,  wbits=3, trees=13),
    "m":   dict(bits=7, leaves=128,  wbits=2, trees=40),
    "l":   dict(bits=7, leaves=512,  wbits=2, trees=120),
    "xl":  dict(bits=7, leaves=2048, wbits=2, trees=300),
    "xxl": dict(bits=7, leaves=4096, wbits=2, trees=1200),
}

NEG_INF = float("-inf")


# ==================================================================================================
# Tree builder: leaf-wise (best-first) on weighted Gini. Over binary features a split search is one
# GEMM (wyoh.T @ X counts, per class, the weighted samples with each bit set). Four parallel arrays
# over nodes: feat (GLOBAL bit index, -1 at a leaf), left, right (child ids), cls (leaf class).
# ==================================================================================================
def build_tree(X, y, w, wyoh, max_leaves, min_leaf, K) -> dict:
    feat: list[int] = []
    left: list[int] = []
    right: list[int] = []
    cls: list[int] = []

    def add_leaf(idx):
        cw = np.bincount(y[idx], weights=w[idx], minlength=K)
        nid = len(feat)
        feat.append(-1)
        left.append(-1)
        right.append(-1)
        cls.append(int(cw.argmax()))
        return nid, cw

    def best_split(idx, cw):
        Xg = X[idx].astype(np.float32)                       # (n, F)
        b1 = wyoh[idx].T @ Xg                                # (K, F) weighted count of bit==1
        raw1 = Xg.sum(0)
        n = idx.size
        cntR = b1                                            # bit==1 goes right
        cntL = cw[:, None] - b1                              # bit==0 goes left
        valid = (raw1 >= min_leaf) & (n - raw1 >= min_leaf)
        score = (cntL ** 2).sum(0) / np.clip(cntL.sum(0), 1e-12, None) \
            + (cntR ** 2).sum(0) / np.clip(cntR.sum(0), 1e-12, None)
        score = np.where(valid, score, NEG_INF)
        base = float((cw ** 2).sum()) / max(float(cw.sum()), 1e-12)
        best = int(score.argmax())
        gain = float(score[best]) - base
        if not bool(valid[best]) or gain <= 1e-9:
            return None
        bit = Xg[:, best] > 0
        return gain, best, idx[~bit], idx[bit]

    heap: list = []
    ctr = 0

    def consider(nid, idx, cw):
        nonlocal ctr
        if idx.size < 2 * min_leaf:
            return
        bs = best_split(idx, cw)
        if bs is not None:
            heapq.heappush(heap, (-bs[0], ctr, nid, idx, bs))
            ctr += 1

    root_idx = np.arange(X.shape[0])
    root, cw0 = add_leaf(root_idx)
    consider(root, root_idx, cw0)
    leaves = 1
    while heap and leaves < max_leaves:
        _, _, nid, idx, bs = heapq.heappop(heap)
        _, gf, lidx, ridx = bs
        lc, lcw = add_leaf(lidx)
        rc, rcw = add_leaf(ridx)
        feat[nid], left[nid], right[nid] = gf, lc, rc         # the popped leaf becomes internal
        leaves += 1
        consider(lc, lidx, lcw)
        consider(rc, ridx, rcw)

    return {"feat": np.array(feat, np.int64), "left": np.array(left, np.int64),
            "right": np.array(right, np.int64), "cls": np.array(cls, np.int64)}


def _route(tree: dict, X) -> np.ndarray:
    """Route every row to its leaf; return the leaf node id per row."""
    feat, left, right = tree["feat"], tree["left"], tree["right"]
    node = np.zeros(X.shape[0], np.int64)
    ar = np.arange(X.shape[0])
    for _ in range(len(feat)):
        f = feat[node]
        leaf = f < 0
        if leaf.all():
            break
        bit = X[ar, np.maximum(f, 0)] > 0
        node = np.where(leaf, node, np.where(bit, right[node], left[node]))
    return node


def _fit_boost(X, y, K, *, n_trees, max_leaves, min_leaf, lr, qscale, wbits,
               evalset=None, patience=30) -> tuple[list, list]:
    """SAMME. If qscale is given, alpha is QUANTIZED to an int in [1, 2^wbits-1] inside the loop and
    the quantized alpha drives the weight update, so later trees correct the circuit-exact residual.
    With evalset, stop when integer-vote validation cross-entropy stops improving and return the
    prefix (the first t trees ARE the round-t ensemble) with the best val loss."""
    N = X.shape[0]
    w = np.full(N, 1.0 / N)
    trees: list = []
    alphas: list = []
    if evalset is not None:
        Xv, yv = evalset
        vscore = np.zeros((len(yv), K), np.int64)
        sum_w, best_ce, best_t = 0, float("inf"), 0

    loop_t0, val_secs = time.perf_counter(), 0.0  # train_secs = whole loop minus the val evals
    for t in range(n_trees):
        yoh = np.zeros((N, K))
        yoh[np.arange(N), y] = 1.0
        tree = build_tree(X, y, w, yoh * w[:, None], max_leaves, min_leaf, K)
        miss = (tree["cls"][_route(tree, X)] != y).astype(np.float64)
        err = float((w * miss).sum() / w.sum())

        if err >= 1.0 - 1.0 / K:                              # worse than random: drop it
            w = np.full(N, 1.0 / N)
            continue
        if err <= 1e-12:
            alpha = (math.log((1 - 1e-12) / 1e-12) + math.log(K - 1)) * lr
            reset = True
        else:
            alpha = (math.log((1 - err) / err) + math.log(K - 1)) * lr
            reset = False

        if qscale is not None:                                # the integer the CIRCUIT will use
            a_int = int(np.clip(round(alpha * qscale), 1, 2 ** wbits - 1))
            alpha_eff, keep = a_int / qscale, a_int
        else:
            alpha_eff, keep = alpha, alpha

        if reset:
            w = np.full(N, 1.0 / N)
        else:
            w = w * np.exp(alpha_eff * miss)
            w = w / w.sum()

        trees.append(tree)
        alphas.append(keep)

        if evalset is not None:                               # early stop on the INTEGER vote's CE
            v0 = time.perf_counter()
            vscore[np.arange(len(yv)), tree["cls"][_route(tree, Xv)]] += keep
            sum_w += keep
            p = vscore[np.arange(len(yv)), yv] / sum_w        # = scores() of the true class
            ce = float(-np.log(np.clip(p, 1e-12, None)).mean())
            acc = 100.0 * float((vscore.argmax(1) == yv).mean())
            if ce < best_ce - 1e-6:
                best_ce, best_t = ce, len(trees)
            print(f"  tree {t:4d} | err {err:.4f} a {alpha:5.2f} -> {keep} | "
                  f"val ce {ce:.4f} acc {acc:5.2f} (best ce {best_ce:.4f} @ {best_t})", flush=True)
            val_secs += time.perf_counter() - v0
            if len(trees) - best_t >= patience:
                print(f"  early stop at round {t}", flush=True)
                break

    train_secs = (time.perf_counter() - loop_t0) - val_secs
    if evalset is not None:
        return trees[:best_t], alphas[:best_t], train_secs
    return trees, alphas, train_secs


class Forest:
    def __init__(self, spec: DatasetSpec, trees: int, leaves: int, wbits: int, bits: int,
                 min_leaf: int = 5, lr: float = 0.3, patience: int = 30):
        self.spec = spec
        self.n_trees, self.max_leaves, self.wbits, self.bits = trees, leaves, wbits, bits
        self.min_leaf, self.lr, self.patience = min_leaf, lr, patience
        self.thresholds = even_thresholds(bits)
        self.trees: list = []
        self.w: list[int] = []

    # ---- what a feature id MEANS: g = p*k + j <-> (pixel p, threshold index j) -------------------
    def _split(self, g: int) -> tuple[int, int]:
        k = len(self.thresholds)
        return g // k, int(self.thresholds[g % k])

    def _encode(self, pix: np.ndarray) -> np.ndarray:
        """(N, n_pixels) uint8 -> (N, n_pixels*k) thermometer bits, strict `>` as in the Verilog."""
        cols = [(pix > int(t)).astype(np.uint8) for t in self.thresholds]
        return np.stack(cols, axis=2).reshape(len(pix), -1)

    def _feat_expr(self, g: int) -> str:
        p, t = self._split(g)
        pb = self.spec.pixel_bits
        if t == (1 << (pb - 1)) - 1:            # pix > 2^(pb-1)-1 is the MSB: a free wire, 0 gates
            return f"pix[{p * pb + pb - 1}]"
        return f"(pix[{p * pb} +: {pb}] > {pb}'d{t})"

    # ---- exact rewrite: merge same-class sibling leaves (accuracy-neutral) -----------------------
    def _collapse(self, tree: dict) -> dict:
        feat, left, right, cls = (tree[k].copy() for k in ("feat", "left", "right", "cls"))
        changed = True
        while changed:
            changed = False
            for n in range(len(feat)):
                if feat[n] < 0:
                    continue
                l, r = left[n], right[n]
                if feat[l] < 0 and feat[r] < 0 and cls[l] == cls[r]:
                    feat[n], left[n], right[n], cls[n] = -1, -1, -1, cls[l]
                    changed = True
        return {"feat": feat, "left": left, "right": right, "cls": cls}

    # ---- the circuit's integers, in python (no float touches the decision path) ------------------
    def _score_int(self, pix: np.ndarray) -> np.ndarray:
        F = self._encode(pix)
        s = np.zeros((len(pix), self.spec.n_classes), np.int64)
        ar = np.arange(len(pix))
        for tree, w in zip(self.trees, self.w):
            s[ar, tree["cls"][_route(tree, F)]] += w
        return s

    def predict(self, pix: np.ndarray) -> np.ndarray:
        return self._score_int(pix).argmax(1)               # ties -> lowest class, as argmax emits

    def scores(self, pix: np.ndarray) -> np.ndarray:
        # score / sum(w) is a probability distribution (each tree adds w to one class); a strictly
        # increasing affine map, so its argmax (ties included) is the one predict() takes.
        return self._score_int(pix) / float(sum(self.w))

    # ---- training --------------------------------------------------------------------------------
    def train(self, data: Dataset, *, device: str = "cpu", seed: int = 0) -> None:
        np.random.seed(seed)                                # tree building is deterministic anyway
        K = self.spec.n_classes
        X = self._encode(data.train_x)
        y = data.train_y.astype(np.int64)
        Xv = self._encode(data.val_x)
        yv = data.val_y.astype(np.int64)

        # pass 1: unquantized, only to see where alpha lands (nothing here is hand-tuned). A short
        # prefix is enough to size the resolution; correctness never depends on qscale.
        kw = dict(max_leaves=self.max_leaves, min_leaf=self.min_leaf, lr=self.lr, wbits=self.wbits)
        _, a0, ts1 = _fit_boost(X, y, K, n_trees=min(self.n_trees, 50), qscale=None, **kw)
        p95 = float(np.percentile(np.asarray(a0, float), 95)) if a0 else 1.0
        qscale = (2 ** self.wbits - 1) / max(p95, 1e-9)
        print(f"[quant] alpha p95 {p95:.3f} -> wscale {qscale:.3f} "
              f"(alpha -> int in [1, {2 ** self.wbits - 1}])", flush=True)

        # pass 2: the real fit, circuit integers in the loop, early stop on validation loss
        trees, w, ts2 = _fit_boost(X, y, K, n_trees=self.n_trees, qscale=qscale,
                                   evalset=(Xv, yv), patience=self.patience, **kw)
        self.trees = [self._collapse(t) for t in trees]
        self.w = [int(a) for a in w]
        self.train_seconds = ts1 + ts2  # pure boosting time (both passes), excluding val CE
        assert self.trees, "no tree survived boosting"
        assert min(self.w) >= 1, f"weights must be >= 1 (unsigned scores), got {min(self.w)}"

        nl = sum(int((t["feat"] < 0).sum()) for t in self.trees)
        acc = 100.0 * float((self.predict(data.val_x) == data.val_y).mean())
        print(f"[forest] {len(self.trees)} trees, {nl:,} leaves, weights {self.w} | val {acc:.2f}%",
              flush=True)

    # ---- emission: this method's OWN top module --------------------------------------------------
    def _class_exprs(self, tree: dict, ti: int, reach) -> dict[int, str]:
        """Per-class indicator expression at the root, iterative post-order. Whole one-class
        subtrees reuse their reach wire; otherwise ORs of the children's per-class exprs."""
        feat, left, right, cls = tree["feat"], tree["left"], tree["right"], tree["cls"]
        expr: dict[int, dict[int, str]] = {}
        stack = [(0, False)]
        while stack:
            n, done = stack.pop()
            if feat[n] < 0:
                expr[n] = {int(cls[n]): reach(n)}
            elif not done:
                stack.append((n, True))
                stack.append((int(left[n]), False))
                stack.append((int(right[n]), False))
            else:
                el, er = expr[int(left[n])], expr[int(right[n])]
                keys = set(el) | set(er)
                if len(keys) == 1:                            # whole subtree one class: reuse reach
                    expr[n] = {next(iter(keys)): reach(n)}
                else:
                    m = {}
                    for c in keys:
                        a, b = el.get(c), er.get(c)
                        m[c] = a if b is None else b if a is None else f"({a} | {b})"
                    expr[n] = m
        return expr[0]

    def emit_verilog(self) -> str:
        spec = self.spec
        K, pb = spec.n_classes, spec.pixel_bits
        cw_bits = spec.cls_bits
        W = int(sum(self.w)).bit_length()
        assert sum(self.w) < 2 ** W, "score width too narrow -- would truncate silently"

        body: list[str] = []

        # features: only the ones some node splits on (opt_clean would drop the rest anyway).
        used = sorted({int(f) for t in self.trees for f in t["feat"] if f >= 0})
        body.append(f"  // thermometer features used: {len(used)} of "
                    f"{spec.n_pixels * len(self.thresholds)}")
        for g in used:
            body.append(f"  wire f{g} = {self._feat_expr(g)};")

        # per-tree reach network + class indicators
        vind: list[dict[int, str]] = []
        for ti, tree in enumerate(self.trees):
            feat, left, right = tree["feat"], tree["left"], tree["right"]
            nl = int((feat < 0).sum())
            body.append(f"  // tree {ti}: {nl} leaves, weight {self.w[ti]}")

            def reach(n: int, ti=ti) -> str:
                return "1'b1" if n == 0 else f"r{ti}_{n}"

            stack = [0]
            while stack:                                      # reach wires, top-down
                n = stack.pop()
                if feat[n] < 0:
                    continue
                f = f"f{int(feat[n])}"
                for child, lit in ((int(left[n]), f"~{f}"), (int(right[n]), f)):
                    rhs = lit if n == 0 else f"{reach(n)} & {lit}"
                    body.append(f"  wire {reach(child)} = {rhs};")
                    stack.append(child)

            vi: dict[int, str] = {}
            for c, e in self._class_exprs(tree, ti, reach).items():
                body.append(f"  wire v{ti}_{c} = {e};")
                vi[c] = f"v{ti}_{c}"
            vind.append(vi)

        # head: per-class bit-plane popcount. A zero weight-bit costs nothing.
        body.append(f"  // head: per-class bit-plane popcount, {W}-bit unsigned scores")
        body.append(f"  logic [{W - 1}:0] score [0:{K - 1}];")
        for c in range(K):
            planes = []
            for b in range(self.wbits):
                terms = [vind[t][c] for t in range(len(self.trees))
                         if (self.w[t] >> b) & 1 and c in vind[t]]
                if not terms:
                    continue
                body.append(f"  logic [{W - 1}:0] p{b}_c{c};")
                body.append(f"  assign p{b}_c{c} = {' + '.join(terms)};")
                planes.append(f"{1 << b} * p{b}_c{c}" if b else f"p{b}_c{c}")
            rhs = " + ".join(planes) if planes else f"{W}'d0"
            body.append(f"  assign score[{c}] = {rhs};")

        # argmax: strict >, ascending c -> ties to the lowest class
        body.append(f"  logic [{W - 1}:0] best;")
        body.append("  always_comb begin")
        body.append("    best = score[0];")
        body.append(f"    cls  = {cw_bits}'d0;")
        for c in range(1, K):
            body.append(f"    if (score[{c}] > best) begin best = score[{c}]; "
                        f"cls = {cw_bits}'d{c}; end")
        body.append("  end")

        nl = sum(int((t["feat"] < 0).sum()) for t in self.trees)
        return (f"// generated by methods/forest -- {spec.name}, {len(self.trees)} SAMME trees, "
                f"{nl} leaves,\n"
                f"// {len(self.thresholds)} thermometer bits/pixel, {self.wbits}-bit tree weights, "
                f"{W}-bit scores\n"
                f"module top (input [{spec.port_bits - 1}:0] pix, "
                f"output logic [{cw_bits - 1}:0] cls);\n\n"
                + "\n".join(body) + "\nendmodule\n")

    # ---- checkpoint ------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        d = {"w": np.array(self.w, np.int64), "thresholds": np.array(self.thresholds, np.int64),
             "bits": self.bits, "wbits": self.wbits, "n_trees": len(self.trees)}
        for i, t in enumerate(self.trees):
            for k in ("feat", "left", "right", "cls"):
                d[f"t{i}_{k}"] = t[k]
        with open(path, "wb") as f:
            np.savez(f, **d)


def points(spec: DatasetSpec) -> list[dict]:
    return [{"name": n, **cfg} for n, cfg in _LADDER.items()]


def build(spec: DatasetSpec, **point) -> Forest:
    return Forest(spec, **point)
