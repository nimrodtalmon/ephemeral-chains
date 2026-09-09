"""The two maps on shared axes (slack x cap heterogeneity), from the exact
small-instance grids: left, what governance can move (steerability per
group, results/landscape_governance.csv); right, what agents can take
(fraction with a profitable honorable misreport per channel,
results/landscape.csv). Bottleneck rule; manipulation under uniform
weights; averaged over stake heterogeneity and instances.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import RESULTS_DIR


def honorable(r):
    if r.role == "app":
        return not ((r.coord == "gasprice" and r.factor > 1) or (r.coord == "stake" and r.factor < 1))
    return not ((r.coord == "gasprice" and r.factor < 1) or (r.coord in ("gas", "stake") and r.factor > 1))


def heat(ax, table, title, vmax):
    im = ax.imshow(table.values, cmap="viridis", vmin=0, vmax=vmax, origin="lower", aspect="auto")
    ax.set_xticks(range(table.shape[1])); ax.set_xticklabels([f"{c:g}" for c in table.columns])
    ax.set_yticks(range(table.shape[0])); ax.set_yticklabels([f"{r:g}" for r in table.index])
    ax.set_xlabel("slack (supply / demand scale)"); ax.set_ylabel("cap heterogeneity")
    ax.set_title(title, fontsize=9)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v < 0.6 * vmax else "black")
    return im


def main(out=RESULTS_DIR / "two_maps.pdf"):
    g = pd.read_csv(RESULTS_DIR / "landscape_governance.csv")
    g = g[g.rule == "min"]
    d = pd.read_csv(RESULTS_DIR / "landscape.csv")
    d = d[(d.rule == "min") & (d.weights == "uniform")]
    d = d[d.apply(honorable, axis=1)]
    keys = ["slack", "cap_sigma", "stake", "seed", "role", "agent", "coord"]
    best = d.groupby(keys).gain.max().reset_index()
    best["prof"] = best.gain > 1e-9
    channels = [("app", "gas", "applications: inflate demand"),
                ("op", "gasprice", "operators: raise ask"),
                ("op", "stake", "operators: under-lock stake")]
    groups = [("steer_app", "application service"), ("steer_op", "operator yield"),
              ("steer_sys", "fee volume")]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.0))
    for ax, (col, name) in zip(axes[0], groups):
        t = g.pivot_table(index="cap_sigma", columns="slack", values=col, aggfunc="mean")
        heat(ax, t, f"steerability: {name}", vmax=1.0)
    for ax, (role, coord, name) in zip(axes[1], channels):
        sub = best[(best.role == role) & (best.coord == coord)]
        t = sub.pivot_table(index="cap_sigma", columns="slack", values="prof", aggfunc="mean")
        heat(ax, t, f"manipulability: {name}", vmax=1.0)
    axes[0, 0].annotate("What governance can move", xy=(-0.35, 1.18), xycoords="axes fraction",
                        fontsize=11, fontweight="bold")
    axes[1, 0].annotate("What agents can take", xy=(-0.35, 1.18), xycoords="axes fraction",
                        fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
