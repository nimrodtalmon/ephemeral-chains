"""Draw the map of instances from ``results/map.csv``.

Top row: feature axes (supply/demand on a log scale vs. cap
heterogeneity). Bottom row: behavioral embedding -- classical MDS of the
governance response profiles (Euclidean distance between the 63-vectors
of group utilities over the simplex grid). Left column colored by
operator manipulability (mean relative unilateral gain, clipped), right
column by fee steerability (range of normalized fee volume over the
simplex).
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
from sklearn.manifold import MDS

from experiments.common import RESULTS_DIR


def main(path=RESULTS_DIR / "map.csv", out=RESULTS_DIR / "instance_map.pdf") -> None:
    d = pd.read_csv(path)
    prof_cols = [c for c in d.columns if c.startswith(("napp_", "nop_", "nsys_"))]
    P = d[prof_cols].to_numpy(float)
    # Standardize each group's block so the (tiny-scale) operator block is
    # not drowned by the application and fee blocks.
    for g in ("napp", "nop", "nsys"):
        idx = [k for k, c in enumerate(prof_cols) if c.startswith(g + "_")]
        block = P[:, idx]
        P[:, idx] = (block - block.mean()) / (block.std() + 1e-12)
    # Steerability per group = range over the simplex.
    for g in ("napp", "nop", "nsys"):
        cols = [c for c in prof_cols if c.startswith(g + "_")]
        d[f"steer_{g}"] = d[cols].max(axis=1) - d[cols].min(axis=1)
    emb = MDS(n_components=2, metric=True, random_state=0, n_init=4,
              init="classical_mds", normalized_stress="auto").fit_transform(P)
    d["mds_x"], d["mds_y"] = emb[:, 0], emb[:, 1]

    colorings = [
        ("manip_op_frac", "operators with a profitable misreport", (0, 1)),
        ("manip_app_frac", "applications with a profitable misreport", (0, 1)),
        ("steer_nsys", "fee steerability (range over simplex)", (0, None)),
        ("steer_napp", "service steerability (range over simplex)", (0, None)),
    ]
    fig, axes = plt.subplots(2, len(colorings), figsize=(4.2 * len(colorings), 7.6))
    for j, (col, label, (lo, hi)) in enumerate(colorings):
        c = d[col].clip(lower=lo, upper=hi) if hi is not None else d[col]
        for i, (x, y, xl, yl, logx) in enumerate([
            (d.supply_demand, d.cap_cv, "supply / demand", "cap heterogeneity (CV)", True),
            (d.mds_x, d.mds_y, "MDS 1", "MDS 2", False),
        ]):
            ax = axes[i, j]
            sc = ax.scatter(x, y, c=c, s=18, cmap="viridis", edgecolor="none")
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            if i == 0:
                ax.set_title(label, fontsize=9)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    axes[0, 0].set_ylabel("feature axes\ncap heterogeneity (CV)")
    axes[1, 0].set_ylabel("behavioral embedding\nMDS 2")
    fig.tight_layout()
    fig.savefig(out)
    d[["seed", "mds_x", "mds_y"] + [c for c in d.columns if c.startswith("steer_")]].to_csv(
        RESULTS_DIR / "map_embedding.csv", index=False)
    print(f"{len(d)} instances -> {out}")


if __name__ == "__main__":
    main()
