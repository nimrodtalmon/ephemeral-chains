"""Experiment: the effect of the governance weights.

Sweeps the governance weights over a grid on the 3-simplex and, per
instance, records the realized (normalized) group utilities and the
chain structure at every weight vector. Instances are drawn from a grid
of regimes varying (i) the supply/demand balance, via a shift of the
operator-capacity distribution, and (ii) price heterogeneity, via the
price noise scale. Two per-instance summary statistics quantify
*steerability* -- how much the weights actually change outcomes:

* ``steer_<group>``: range (max - min) over the simplex of that group's
  mean normalized utility;
* ``steer_structure``: range over the simplex of the number of chains.

Normalization bounds are fixed per instance (analytic), so values are
comparable across weight vectors.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from statistics import mean

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np

from ephemeral_chains.instance import GeneratorConfig, generate
from ephemeral_chains.model import (
    THROUGHPUT_RULES,
    NormalizationBounds,
    Weights,
)
from ephemeral_chains.solver import solve_local_search
from experiments.common import base_parser, write_csv

BASE = GeneratorConfig(n_apps=20, n_ops=10)
CAPACITY_SCALES = [0.5, 1.0, 2.0, 4.0]  # multiplies operator capacity
PRICE_SIGMAS = [0.1, 0.4, 0.8]  # price heterogeneity
SIMPLEX_STEP = 0.2  # weight grid resolution (21 points)


def simplex_grid(step: float):
    n = round(1 / step)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            yield Weights(app=i / n, op=j / n, sys=k / n)


def regime_config(capacity_scale: float, price_sigma: float) -> GeneratorConfig:
    # Multiplying log-normal capacities by s == shifting the location by ln s.
    return replace(
        BASE,
        cap_mu=BASE.cap_mu + float(np.log(capacity_scale)),
        price_sigma=price_sigma,
    )


def main() -> None:
    args = base_parser(__doc__).parse_args()
    rule = THROUGHPUT_RULES["min"]
    point_rows, summary_rows = [], []

    for cap_scale in CAPACITY_SCALES:
        for price_sigma in PRICE_SIGMAS:
            config = regime_config(cap_scale, price_sigma)
            for i in range(args.instances):
                seed = args.seed + i
                inst = generate(config, seed)
                bounds = NormalizationBounds.analytic(inst)
                supply = sum(o.gas for o in inst.ops)
                demand = sum(a.gas for a in inst.apps)

                per_point = []
                for w in simplex_grid(SIMPLEX_STEP):
                    res = solve_local_search(inst, rule, w, bounds=bounds, seed=seed)
                    ev = res.evaluation
                    napp = mean(
                        ev.app_utils[a] / inst.apps[a].gas
                        for a in range(inst.n_apps)
                    )
                    nop = mean(u / bounds.q_op for u in ev.op_utils)
                    nsys = ev.sys_util / bounds.q_sys
                    n_chains = len(res.assignment.chains())
                    per_point.append((w, napp, nop, nsys, n_chains))
                    point_rows.append(dict(
                        cap_scale=cap_scale, price_sigma=price_sigma, seed=seed,
                        w_app=round(w.app, 2), w_op=round(w.op, 2),
                        w_sys=round(w.sys, 2),
                        napp=napp, nop=nop, nsys=nsys, n_chains=n_chains,
                    ))

                def rng(idx: int) -> float:
                    vals = [p[idx] for p in per_point]
                    return max(vals) - min(vals)

                summary_rows.append(dict(
                    cap_scale=cap_scale, price_sigma=price_sigma, seed=seed,
                    supply_demand_ratio=supply / demand,
                    steer_app=rng(1), steer_op=rng(2), steer_sys=rng(3),
                    steer_structure=rng(4),
                ))

    write_csv(args.out or "governance_points.csv", point_rows,
              ["cap_scale", "price_sigma", "seed", "w_app", "w_op", "w_sys",
               "napp", "nop", "nsys", "n_chains"])
    write_csv("governance_summary.csv", summary_rows,
              ["cap_scale", "price_sigma", "seed", "supply_demand_ratio",
               "steer_app", "steer_op", "steer_sys", "steer_structure"])


if __name__ == "__main__":
    main()
