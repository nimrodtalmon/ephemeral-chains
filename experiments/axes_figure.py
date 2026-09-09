"""Draw the two maps along their axes from results/axes.csv."""
from __future__ import annotations
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiments.common import RESULTS_DIR


def band(ax, x, ys, label, color, marker):
    m = ys.mean(axis=0); s = ys.std(axis=0, ddof=1) / np.sqrt(ys.shape[0])
    ax.plot(x, m, marker=marker, color=color, label=label, lw=1.4, ms=4)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)


def main(out=RESULTS_DIR / "two_maps.pdf"):
    d = pd.read_csv(RESULTS_DIR / "axes.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), sharex="col")
    axes_spec = [("cap", "cap heterogeneity (log-scale noise)", False),
                 ("slack", "slack (capacity scale)", True)]
    steer = [("steer_napp", "application service", "#1f77b4", "o"),
             ("steer_nop", "operator yield", "#ff7f0e", "s"),
             ("steer_nsys", "fee volume", "#2ca02c", "^")]
    manip = [("manip_app_gas_frac", "applications: inflate demand", "#1f77b4", "o"),
             ("manip_op_gasprice_frac", "operators: raise ask", "#d62728", "s"),
             ("manip_op_stake_frac", "operators: under-lock stake", "#9467bd", "^")]
    for j, (axis, xlabel, logx) in enumerate(axes_spec):
        sub = d[d.axis == axis]
        levels = sorted(sub.level.unique())
        def mat(col):
            return np.array([[sub[(sub.level == l) & (sub.seed == s)][col].iloc[0] for l in levels]
                             for s in sorted(sub.seed.unique())])
        for col, label, color, mk in steer:
            band(axes[0, j], levels, mat(col), label, color, mk)
        for col, label, color, mk in manip:
            band(axes[1, j], levels, mat(col), label, color, mk)
        if logx:
            axes[1, j].set_xscale("log")
        axes[1, j].set_xlabel(xlabel)
    axes[0, 0].set_ylabel("steerability\n(range over simplex)")
    axes[1, 0].set_ylabel("manipulability\n(fraction who profit)")
    for ax in axes.flat:
        ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    axes[0, 0].set_title("what governance can move", fontsize=9, loc="left")
    axes[1, 0].set_title("what agents can take", fontsize=9, loc="left")
    axes[0, 1].legend(fontsize=7, frameon=False, loc="upper left")
    axes[1, 1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
