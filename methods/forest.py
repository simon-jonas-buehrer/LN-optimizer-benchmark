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
import os
import time
from concurrent.futures import ThreadPoolExecutor

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

# The split search is memory-bound, not flop-bound (a class-grouped GEMM cuts flops 5x and measured
# only 1.15x, so it is not used). Everything below attacks traffic instead.
#
# _CDT is the working precision of the count blocks and the GEMM. float32 halves the traffic of
# both the widening and the GEMM and roughly doubles the search; the counts themselves stay EXACT
# (row K is an integer below 2^24, so `min_leaf` validity never turns on rounding) and only the
# weighted rows carry ~1e-7 relative error, which can flip a near-tie between two features.
# Measured over 8 configs that is worth -0.01 pp val / +0.03 pp test: noise, in both directions.
#
# NOTE it is confined to the SEARCH. Nothing float32 reaches the trained model: a node emits only
# int(cw.argmax()) as its leaf class, the SAMME sample weights and the alpha -> int quantisation
# stay float64 in _fit_boost, and self.w holds python ints. predict() and emit_verilog() therefore
# still read one source of truth -- self.trees (int64 feat/left/right/cls) and self.w (ints) --
# so the harness's circuit == predict() cross-check is unaffected by anything in here.
_CDT = np.float32
# Rows are gathered into a preallocated buffer in chunks that stay hot in L2 (a fresh multi-GB temp
# per node is mostly page faults). The widening is the one hot phase numpy does NOT thread, so it is
# split over a small pool; the slices are disjoint and it is a plain copy, so the bytes match.
_GATHER_CHUNK = 512
_PAR_MIN_ELEMS = 1 << 22          # below this the pool round-trip costs more than it saves


def _num_threads() -> int:
    """Match whatever BLAS was told to use, so benchmarks stay comparable and we never
    oversubscribe: OMP_NUM_THREADS if set, else the cpu count, capped."""
    for var in ("FOREST_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        v = os.environ.get(var, "").strip()
        if v.isdigit() and int(v) > 0:
            return min(int(v), 16)
    return min(os.cpu_count() or 1, 16)


_NTHREAD = _num_threads()
_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=max(1, _NTHREAD - 1),
                                   thread_name_prefix="forest-gather")
    return _POOL


class _Scratch:
    """Buffers reused by every split search: the (n, F) widened GEMM operand and the (n, K+1)
    weight block. Allocating/first-touching these per node costs more than the GEMM itself."""

    __slots__ = ("f", "w", "cl", "sq")

    def __init__(self, N: int, F: int, K: int):
        self.f = np.empty((N, F), _CDT)
        self.w = np.zeros((N, K + 1), _CDT)
        self.w[:, K] = 1.0                                   # the ones row that yields raw1 free
        self.cl = np.empty((K, F), _CDT)                     # cntL
        self.sq = np.empty((K, F), _CDT)                     # squares, per side


def _gather_rows(out: np.ndarray, X: np.ndarray, idx: np.ndarray, lo: int, hi: int) -> None:
    for c in range(lo, hi, _GATHER_CHUNK):
        j = idx[c:min(c + _GATHER_CHUNK, hi)]
        np.copyto(out[c:c + j.size], X[j], casting="unsafe")


def _gather_block(out: np.ndarray, X: np.ndarray, idx: np.ndarray) -> None:
    """out[:] = X[idx] widened to `out`'s float dtype, chunked so the uint8 temp stays in cache and
    (for big blocks) split over the thread pool -- np.copyto drops the GIL."""
    n = idx.size
    if _NTHREAD > 1 and n * X.shape[1] >= _PAR_MIN_ELEMS:
        step = -(-n // _NTHREAD)
        futs = [_pool().submit(_gather_rows, out, X, idx, c, min(c + step, n))
                for c in range(step, n, step)]
        _gather_rows(out, X, idx, 0, min(step, n))
        for f in futs:
            f.result()
        return
    _gather_rows(out, X, idx, 0, n)


# ==================================================================================================
# Tree builder: leaf-wise (best-first) on weighted Gini. Over binary features a split search is one
# GEMM (wyoh.T @ X counts, per class, the weighted samples with each bit set). Four parallel arrays
# over nodes: feat (GLOBAL bit index, -1 at a leaf), left, right (child ids), cls (leaf class).
#
# Three things make the search cheap:
#   * HISTOGRAM SUBTRACTION. A node's (K+1, F) count block is the sum of its two children's, so only
#     the SMALLER child is ever counted by GEMM; the larger one is the parent minus the sibling.
#     Measured on MNIST bits=7/leaves=128: 649k counted rows instead of 1.72M, a 2.66x cut. The
#     parent's block is recycled in place as the larger child's, so one buffer is born per split.
#   * The GEMM operand is pre-widened to the weight dtype. Mixed `float @ uint8`/`float64 @ float32`
#     makes numpy widen the whole (n, F) block anyway, so doing it once into a reused buffer is the
#     same numbers and ~3.5x faster than letting matmul do it into a fresh temp.
#   * A ones column appended to the weight block turns the plain bit count `raw1` into row K of the
#     same GEMM instead of a second full pass. raw1 stays an EXACT integer under subtraction (both
#     operands are exact integers below 2^53), so `min_leaf` validity is never decided by rounding.
# The weighted rows do drift down a chain of subtractions, which can flip a near-tie between two
# features -- a different, equally-trained forest of the same shape, not a worse one. Column pruning
# (features that can no longer pass min_leaf are dead in the whole subtree, ~35% of the work) was
# tried and rejected: the np.ix_ column gather it needs is 7-14x slower per element than the row
# gather, so it costs more than it saves.
# ==================================================================================================
def build_tree(X, y, w, wyoh, max_leaves, min_leaf, K, scratch: "_Scratch | None" = None) -> dict:
    N, F = X.shape
    if scratch is None:
        scratch = _Scratch(N, F, K)
    fbuf, wbuf = scratch.f, scratch.w
    clbuf, sqbuf = scratch.cl, scratch.sq
    spare: list = []                                         # recycled (K+1, F) count blocks
    feat: list[int] = []
    left: list[int] = []
    right: list[int] = []
    cls: list[int] = []

    def add_leaf(idx):
        cw = np.bincount(y[idx], weights=w[idx], minlength=K).astype(_CDT)
        nid = len(feat)
        feat.append(-1)
        left.append(-1)
        right.append(-1)
        cls.append(int(cw.argmax()))
        return nid, cw

    def counts(idx, out):
        """out[:] = (K+1, F): rows 0..K-1 the weighted per-class count of bit==1, row K the plain
        count. This is the only place that touches the sample matrix."""
        n = idx.size
        Xg = fbuf[:n]
        _gather_block(Xg, X, idx)
        Wx = wbuf[:n]                                        # (n, K+1), last column all ones
        Wx[:, :K] = wyoh[idx]
        np.matmul(Wx.T, Xg, out=out)

    def best_split(idx, cw, ext):
        n = idx.size
        b1 = ext[:K]                                         # weighted count of bit==1, per class
        raw1 = ext[K]                                        # plain count of bit==1
        cntR = b1                                            # bit==1 goes right
        cntL = np.subtract(cw[:, None], b1, out=clbuf)       # bit==0 goes left
        valid = (raw1 >= min_leaf) & (n - raw1 >= min_leaf)
        score = np.square(cntL, out=sqbuf).sum(0) / np.clip(cntL.sum(0), 1e-12, None) \
            + np.square(cntR, out=sqbuf).sum(0) / np.clip(cntR.sum(0), 1e-12, None)
        score = np.where(valid, score, NEG_INF)
        base = float((cw ** 2).sum()) / max(float(cw.sum()), 1e-12)
        best = int(score.argmax())
        gain = float(score[best]) - base
        if not bool(valid[best]) or gain <= 1e-9:
            return None
        bit = X[idx, best] > 0                               # one column, no block needed
        return gain, best, idx[~bit], idx[bit]

    heap: list = []
    ctr = 0

    def consider(nid, idx, cw, ext) -> bool:
        """Push the node's best split; True if it took ownership of `ext`."""
        nonlocal ctr
        if idx.size < 2 * min_leaf:
            return False
        bs = best_split(idx, cw, ext)
        if bs is None:
            return False
        heapq.heappush(heap, (-bs[0], ctr, nid, idx, ext, bs))
        ctr += 1
        return True

    root_idx = np.arange(N)
    root, cw0 = add_leaf(root_idx)
    ext = spare.pop() if spare else np.empty((K + 1, F), _CDT)
    counts(root_idx, ext)
    if not consider(root, root_idx, cw0, ext):
        spare.append(ext)
    leaves = 1
    while heap and leaves < max_leaves:
        _, _, nid, idx, ext, bs = heapq.heappop(heap)
        _, gf, lidx, ridx = bs
        lc, lcw = add_leaf(lidx)
        rc, rcw = add_leaf(ridx)
        feat[nid], left[nid], right[nid] = gf, lc, rc         # the popped leaf becomes internal
        leaves += 1
        if lidx.size < 2 * min_leaf and ridx.size < 2 * min_leaf:
            spare.append(ext)                                 # neither child can split: no counting
            continue
        small = spare.pop() if spare else np.empty((K + 1, F), _CDT)
        if lidx.size <= ridx.size:                            # count the cheaper side only
            counts(lidx, small)
            lext, rext = small, np.subtract(ext, small, out=ext)
        else:
            counts(ridx, small)
            rext, lext = small, np.subtract(ext, small, out=ext)
        if not consider(lc, lidx, lcw, lext):
            spare.append(lext)
        if not consider(rc, ridx, rcw, rext):
            spare.append(rext)

    return {"feat": np.array(feat, np.int64), "left": np.array(left, np.int64),
            "right": np.array(right, np.int64), "cls": np.array(cls, np.int64)}


def _route(tree: dict, X) -> np.ndarray:
    """Route every row to its leaf; return the leaf node id per row.

    Same descent as a per-level `np.where` sweep, but rows that have already landed on a leaf are
    dropped from the working set, so the cost is sum(depth per row) rather than N * max_depth --
    leaf-wise trees are very unbalanced, so most rows stop early."""
    feat, left, right = tree["feat"], tree["left"], tree["right"]
    node = np.zeros(X.shape[0], np.int64)
    act = np.arange(X.shape[0])
    while act.size:
        nd = node[act]
        f = feat[nd]
        live = f >= 0
        if not live.all():
            act = act[live]
            if act.size == 0:
                break
            nd, f = nd[live], f[live]
        bit = X[act, f] > 0
        node[act] = np.where(bit, right[nd], left[nd])
    return node


def _fit_boost(X, y, K, *, n_trees, max_leaves, min_leaf, lr, qscale, wbits,
               evalset=None, patience=30, scratch=None) -> tuple[list, list]:
    """SAMME. If qscale is given, alpha is QUANTIZED to an int in [1, 2^wbits-1] inside the loop and
    the quantized alpha drives the weight update, so later trees correct the circuit-exact residual.
    With evalset, stop when integer-vote validation cross-entropy stops improving and return the
    prefix (the first t trees ARE the round-t ensemble) with the best val loss."""
    N = X.shape[0]
    if scratch is None:
        scratch = _Scratch(N, X.shape[1], K)
    yoh = np.zeros((N, K))                                   # constant across rounds: build once
    yoh[np.arange(N), y] = 1.0
    w = np.full(N, 1.0 / N)
    trees: list = []
    alphas: list = []
    if evalset is not None:
        Xv, yv = evalset
        var = np.arange(len(yv))
        vscore = np.zeros((len(yv), K), np.int64)
        sum_w, best_ce, best_t = 0, float("inf"), 0

    loop_t0, val_secs = time.perf_counter(), 0.0  # train_secs = whole loop minus the val evals
    for t in range(n_trees):
        tree = build_tree(X, y, w, yoh * w[:, None], max_leaves, min_leaf, K, scratch)
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
            vscore[var, tree["cls"][_route(tree, Xv)]] += keep
            sum_w += keep
            p = vscore[var, yv] / sum_w                       # = scores() of the true class
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
        # the EFFECTIVE settings, defaults included, so the published .json records what actually
        # ran rather than only what the ladder point happened to name. Read by harness.run_point.
        self.cfg = dict(trees=trees, leaves=leaves, wbits=wbits, bits=bits,
                        min_leaf=min_leaf, lr=lr, patience=patience)
        self.thresholds = even_thresholds(bits)
        self.trees: list = []
        self.w: list[int] = []
        self._score_memo = None      # (pix, trees list, w snapshot, n_trees, integer scores)

    # ---- what a feature id MEANS: g = p*k + j <-> (pixel p, threshold index j) -------------------
    def _split(self, g: int) -> tuple[int, int]:
        k = len(self.thresholds)
        return g // k, int(self.thresholds[g % k])

    def _encode(self, pix: np.ndarray) -> np.ndarray:
        """(N, n_pixels) uint8 -> (N, n_pixels*k) thermometer bits, strict `>` as in the Verilog."""
        k = len(self.thresholds)
        out = np.empty((len(pix), pix.shape[1], k), np.uint8)   # one buffer, no stack copy
        for j, t in enumerate(self.thresholds):
            out[:, :, j] = pix > int(t)
        return out.reshape(len(pix), -1)

    def _feat_expr(self, g: int) -> str:
        p, t = self._split(g)
        pb = self.spec.pixel_bits
        if t == (1 << (pb - 1)) - 1:            # pix > 2^(pb-1)-1 is the MSB: a free wire, 0 gates
            return f"pix[{p * pb + pb - 1}]"
        return f"(pix[{p * pb} +: {pb}] > {pb}'d{t})"

    # ---- exact rewrite: merge same-class sibling leaves (accuracy-neutral) -----------------------
    def _collapse(self, tree: dict) -> dict:
        feat, left, right, cls = (tree[k].copy() for k in ("feat", "left", "right", "cls"))
        while True:                                           # same monotone fixpoint, one numpy
            inter = feat >= 0                                 # pass per collapse level
            if not inter.any():
                break
            l = np.where(inter, left, 0)
            r = np.where(inter, right, 0)
            m = inter & (feat[l] < 0) & (feat[r] < 0) & (cls[l] == cls[r])
            if not m.any():
                break
            cls[m] = cls[l[m]]
            feat[m] = -1
            left[m] = -1
            right[m] = -1
        return {"feat": feat, "left": left, "right": right, "cls": cls}

    # ---- the circuit's integers, in python (no float touches the decision path) ------------------
    def _score_bits(self, F: np.ndarray) -> np.ndarray:
        """The integer vote over already-encoded thermometer bits."""
        s = np.zeros((F.shape[0], self.spec.n_classes), np.int64)
        ar = np.arange(F.shape[0])
        for tree, w in zip(self.trees, self.w):
            s[ar, tree["cls"][_route(tree, F)]] += w
        return s

    def _score_int(self, pix: np.ndarray) -> np.ndarray:
        """The harness asks for predict(val_x) and then scores(val_x) -- the same array object --
        so keep the last integer vote. The memo only hits when the input is the SAME array object
        (a strong ref is kept, so no id can be recycled under us) and the ensemble is untouched:
        same list object, same length, same weights. Anything else recomputes."""
        memo = self._score_memo
        if (memo is not None and memo[0] is pix and memo[1] is self.trees
                and memo[3] == len(self.trees) and memo[2] == self.w):
            return memo[4]
        s = self._score_bits(self._encode(pix))
        self._score_memo = (pix, self.trees, list(self.w), len(self.trees), s)
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
        self._score_memo = None                             # any refit invalidates the memo
        K = self.spec.n_classes
        X = self._encode(data.train_x)
        y = data.train_y.astype(np.int64)
        Xv = self._encode(data.val_x)
        yv = data.val_y.astype(np.int64)

        # pass 1: unquantized, only to see where alpha lands (nothing here is hand-tuned). A short
        # prefix is enough to size the resolution; correctness never depends on qscale.
        kw = dict(max_leaves=self.max_leaves, min_leaf=self.min_leaf, lr=self.lr, wbits=self.wbits,
                  scratch=_Scratch(X.shape[0], X.shape[1], K))
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
        # samples looked at until early stop: each kept boosting round scans the full train set
        self.train_samples = len(self.trees) * X.shape[0]
        assert self.trees, "no tree survived boosting"
        assert min(self.w) >= 1, f"weights must be >= 1 (unsigned scores), got {min(self.w)}"

        nl = sum(int((t["feat"] < 0).sum()) for t in self.trees)
        acc = 100.0 * float((self._score_bits(Xv).argmax(1) == data.val_y).mean())  # Xv is encoded
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


def load(spec: DatasetSpec, path: str) -> Forest:
    """Rebuild the boosted forest from a `.ckpt` written by `Forest.save`.

    (trees, w, thresholds) is the whole circuit -- `emit_verilog`, `predict` and `scores` read only
    those -- so the synthesis phase reloads and cross-checks without re-boosting. `leaves` is a
    training cap and is not stored; it is irrelevant to an already-grown forest.
    """
    with np.load(path) as d:
        m = Forest(spec, trees=int(d["n_trees"]), leaves=1,
                   wbits=int(d["wbits"]), bits=int(d["bits"]))
        m.thresholds = [int(t) for t in d["thresholds"]]
        m.w = [int(x) for x in d["w"]]
        m.trees = [{k: d[f"t{i}_{k}"] for k in ("feat", "left", "right", "cls")}
                   for i in range(int(d["n_trees"]))]
    return m


def points(spec: DatasetSpec) -> list[dict]:
    return [{"name": n, **cfg} for n, cfg in _LADDER.items()]


def build(spec: DatasetSpec, **point) -> Forest:
    return Forest(spec, **point)
