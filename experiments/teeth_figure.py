"""When governance and manipulation have teeth.

Left: feature sensitivity on sampled large instances (Spearman rank
correlation between instance features and each outcome; results/map_large.csv).
Right: the three switches, exactly (results/axes.csv): the ask channel
against cap heterogeneity, and the demand and stake channels against slack.
"""
from __future__ import annotations
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from experiments.common import RESULTS_DIR

FEATS = [("supply_demand", "slack (supply/demand)"), ("cap_cv", "cap heterogeneity"),
         ("capacity_cv", "capacity heterogeneity"), ("stake_cv", "stake heterogeneity"),
         ("ask_cv", "ask heterogeneity")]
OUTS = [("steer_napp", "steer\napps"), ("steer_nop", "steer\nops"),
        ("steer_nsys", "steer\nfees"), ("manip_app_gas_rel", "gain\napp demand"),
        ("manip_op_gasprice_rel", "gain\nop ask"), ("manip_op_stake_rel", "gain\nop stake"),
        ("manip_op_gas_rel", "gain\nop capacity")]


def main(out=RESULTS_DIR / "teeth.pdf"):
    m = pd.read_csv(RESULTS_DIR / "map_large.csv")
    for g in ("napp", "nop", "nsys"):
        cs = [c for c in m.columns if c.startswith(g + "_")]
        m[f"steer_{g}"] = m[cs].max(axis=1) - m[cs].min(axis=1)
    R = np.array([[spearmanr(m[f], m[o])[0] for o, _ in OUTS] for f, _ in FEATS])
    a = pd.read_csv(RESULTS_DIR / "axes.csv")

    fig = plt.figure(figsize=(9.8, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1, 1], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(R, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(OUTS))); ax.set_xticklabels([l for _, l in OUTS], fontsize=6.5)
    ax.set_yticks(range(len(FEATS))); ax.set_yticklabels([l for _, l in FEATS], fontsize=8)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            ax.text(j, i, f"{R[i,j]:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(R[i, j]) > 0.4 else "black")
    ax.set_title("(a) which feature moves what (Spearman, large instances)", fontsize=8, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=0)

    def curve(ax, sub, col, label, color, mk, logx=False):
        levels = sorted(sub.level.unique())
        M = np.array([[sub[(sub.level == l) & (sub.seed == s)][col].iloc[0] for l in levels]
                      for s in sorted(sub.seed.unique())])
        mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(M.shape[0])
        ax.plot(levels, mu, marker=mk, color=color, label=label, lw=1.4, ms=4)
        ax.fill_between(levels, mu - se, mu + se, color=color, alpha=0.15, lw=0)
        if logx: ax.set_xscale("log")

    ax1 = fig.add_subplot(gs[1]); sub = a[a.axis == "cap"]
    curve(ax1, sub, "manip_op_gasprice_frac", "operators: raise ask", "#d62728", "s")
    ax1.set_xlabel("cap heterogeneity", fontsize=8); ax1.set_ylabel("fraction who profit", fontsize=8)
    ax1.set_title("(b) the price switch (exact)", fontsize=8, loc="left")
    ax2 = fig.add_subplot(gs[2]); sub = a[a.axis == "slack"]
    curve(ax2, sub, "manip_app_gas_frac", "applications: inflate demand", "#1f77b4", "o", logx=True)
    curve(ax2, sub, "manip_op_stake_frac", "operators: under-lock stake", "#9467bd", "^", logx=True)
    ax2.set_xlabel("slack (capacity scale)", fontsize=8); ax2.set_ylabel("fraction who profit", fontsize=8)
    ax2.set_title("(c) the quantity switch (exact)", fontsize=8, loc="left")
    for x in (ax1, ax2):
        x.grid(alpha=0.25); x.tick_params(labelsize=7); x.legend(fontsize=6.5, frameon=False)
    ax1.set_ylim(0, 1); ax2.set_ylim(0, 1)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
