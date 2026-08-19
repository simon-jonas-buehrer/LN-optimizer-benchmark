"""The paper's four figures + leaderboard, from results/<dataset>/<method>/*.json.

For each dataset (MNIST, CIFAR10): accuracy vs circuit size, and cross-entropy loss vs circuit size,
log-x, with a per-method power-law fit extended (dashed) past the largest measured point. Circuit
size is the raw 2-input gate count after the fast ABC pass. Multiple seeds are averaged per point.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# pretty series labels (avoid importing method modules, which pull in torch)
LABELS = {"backprop": "backprop", "genetic": "genetic", "dfa": "dfa", "forest": "forest",
          "es": "es", "w1_58a4": "w1.58a4", "w1_58a8": "w1.58a8", "w4a4": "w4a4"}
ORDER = ["backprop", "genetic", "dfa", "forest", "es", "w1_58a4", "w1_58a8", "w4a4"]

SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#b9b8b4", "#fcfcfb"


def load(dataset: str) -> dict[str, list[dict]]:
    """method -> list of points, averaged over seeds (by point name)."""
    by_method: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for jf in sorted((RESULTS / dataset).glob("*/*.json")):
        p = json.loads(jf.read_text())
        if "gates" in p and "test_acc" in p:
            by_method[p["method"]][p["name"]].append(p)
    out = {}
    for method, pts in by_method.items():
        rows = []
        for name, group in pts.items():
            r = {"name": name, "gates": float(np.mean([g["gates"] for g in group])),
                 "test_acc": float(np.mean([g["test_acc"] for g in group]))}
            ce = [g["test_ce"] for g in group if g.get("test_ce") is not None]
            if ce:
                r["test_ce"] = float(np.mean(ce))
            rows.append(r)
        out[method] = sorted(rows, key=lambda r: r["gates"])
    return out


def _frontier(points: list[dict]) -> list[dict]:
    """Each method reduced to its own frontier (drop points it beats on both axes)."""
    keep = [p for p in points if not any(
        q["gates"] <= p["gates"] and q["test_acc"] >= p["test_acc"]
        and (q["gates"], -q["test_acc"]) != (p["gates"], -p["test_acc"]) for q in points)]
    return sorted(keep, key=lambda p: p["gates"])


def _powerlaw(x, y):
    b, a = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(a)), float(b)


def _ax(plt, title, ylabel):
    import matplotlib.ticker as ticker
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1., 2., 5.), numticks=30))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v/1e9:g}B" if v >= 1e9 else f"{v/1e6:g}M" if v >= 1e6
        else f"{v/1e3:g}k" if v >= 1e3 else f"{v:g}"))
    ax.set_xlabel("circuit size  (2-input gates)", color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    ax.grid(True, which="both", color=MUTED, alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    return fig, ax


def _style(method):
    i = ORDER.index(method) if method in ORDER else len(ORDER)
    return SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]


def _save(fig, out):
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"wrote {out} (+ .pdf)")


def plot_dataset(dataset: str, data: dict, extrapolate_to=2e7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # accuracy
    fig, ax = _ax(plt, f"{dataset.upper()} accuracy vs circuit size", f"{dataset.upper()} test accuracy  (%)")
    for method in [m for m in ORDER if m in data] + [m for m in data if m not in ORDER]:
        ps = _frontier(data[method])
        if not ps:
            continue
        c, mk = _style(method)
        ge = np.array([p["gates"] for p in ps], float)
        acc = np.array([p["test_acc"] for p in ps], float)
        ax.plot(ge, acc, color=c, lw=2, marker=mk, ms=8, mec=SURFACE, mew=2, zorder=3,
                label=LABELS.get(method, method))
        err = np.clip(100 - acc, 1e-3, None)
        if len(ps) >= 2:
            A, b = _powerlaw(ge, err)
            xs = np.geomspace(ge[-1], extrapolate_to, 50)
            ax.plot(xs, 100 - A * xs**b, color=c, lw=1.6, ls=(0, (5, 3)), alpha=0.7, zorder=2)
    ax.legend(frameon=False, loc="lower right", fontsize=8, labelcolor=INK2, ncol=2)
    _save(fig, RESULTS / f"{dataset}_acc.png")

    # loss (log-log)
    lp = {m: [p for p in ps if p.get("test_ce") is not None] for m, ps in data.items()}
    if any(lp.values()):
        fig, ax = _ax(plt, f"{dataset.upper()} loss vs circuit size  (log-log)",
                      f"{dataset.upper()} test cross-entropy  (log)")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1., 2., 3., 5.), numticks=30))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        for method in [m for m in ORDER if lp.get(m)] + [m for m in lp if m not in ORDER and lp.get(m)]:
            ps = sorted(lp[method], key=lambda p: p["gates"])
            c, mk = _style(method)
            ge = np.array([p["gates"] for p in ps], float)
            ce = np.array([p["test_ce"] for p in ps], float)
            ax.plot(ge, ce, color=c, lw=2, marker=mk, ms=8, mec=SURFACE, mew=2, zorder=3,
                    label=LABELS.get(method, method))
            if len(ps) >= 2:
                A, b = _powerlaw(ge, ce)
                xs = np.geomspace(ge[-1], extrapolate_to, 50)
                ax.plot(xs, A * xs**b, color=c, lw=1.6, ls=(0, (5, 3)), alpha=0.7, zorder=2)
        ax.legend(frameon=False, loc="lower left", fontsize=8, labelcolor=INK2, ncol=2)
        _save(fig, RESULTS / f"{dataset}_loss.png")


def table(dataset: str, data: dict) -> str:
    rows = []
    for method, ps in data.items():
        for p in ps:
            rows.append((method, p))
    rows.sort(key=lambda r: (-r[1]["test_acc"], r[1]["gates"]))
    lines = [f"### {dataset.upper()}", "",
             "| method | point | 2-input gates | test acc | test CE |",
             "|---|---|---|---|---|"]
    for method, p in rows:
        ce = f"{p['test_ce']:.3f}" if p.get("test_ce") is not None else "--"
        lines.append(f"| `{LABELS.get(method, method)}` | {p['name']} | {p['gates']:,.0f} "
                     f"| **{p['test_acc']:.2f}%** | {ce} |")
    return "\n".join(lines) + "\n"


def main():
    md = []
    any_data = False
    for dataset in ("mnist", "cifar10"):
        data = load(dataset)
        if not data:
            continue
        any_data = True
        plot_dataset(dataset, data)
        md.append(table(dataset, data))
    if not any_data:
        raise SystemExit("no results yet -- run `python run.py all` first")
    (RESULTS / "leaderboard.md").write_text("\n".join(md))
    print(f"wrote {RESULTS / 'leaderboard.md'}")


if __name__ == "__main__":
    main()
