"""Experiments 2--4.

* ``sensitivity_normalization``: how the optimal assignment and the group
  utilities move when the normalization bounds Q are scaled around the
  analytic values.
* ``sensitivity_throughput``: how the optimal chain structure (number of
  chains, chain sizes, served demand) differs across throughput rules on
  the same instances.
* ``misreporting``: maximal unilateral gain of a sampled agent over a
  multiplicative grid of misreports of one declaration coordinate.

Run any of them as ``python experiments/suite.py <name> [options]``.
"""

from __future__ import annotations

import sys
from statistics import mean

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np

from ephemeral_chains.instance import App, GeneratorConfig, Op, generate
from ephemeral_chains.model import (
    THROUGHPUT_RULES,
    NormalizationBounds,
    Weights,
    chain_state,
)
from ephemeral_chains.solver import solve_local_search
from experiments.common import base_parser, write_csv

WEIGHTS = Weights(app=1 / 3, op=1 / 3, sys=1 / 3)
CONFIG = GeneratorConfig(n_apps=20, n_ops=10)
GRID = [0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0]  # multiplicative misreports / scales


def sensitivity_normalization(args) -> None:
    rows = []
    for i in range(args.instances):
        seed = args.seed + i
        inst = generate(CONFIG, seed)
        for scale in GRID:
            bounds = NormalizationBounds.analytic(inst, scale=scale)
            res = solve_local_search(inst, THROUGHPUT_RULES["min"], WEIGHTS,
                                     bounds=bounds, seed=seed)
            ev = res.evaluation
            rows.append(dict(
                seed=seed, scale=scale, objective=ev.objective,
                mean_app_util=mean(ev.app_utils), mean_op_util=mean(ev.op_utils),
                sys_util=ev.sys_util,
                n_chains=len(res.assignment.chains()),
            ))
    write_csv(args.out or "sensitivity_normalization.csv", rows,
              ["seed", "scale", "objective", "mean_app_util", "mean_op_util",
               "sys_util", "n_chains"])


def sensitivity_throughput(args) -> None:
    rows = []
    for i in range(args.instances):
        seed = args.seed + i
        inst = generate(CONFIG, seed)
        for rule_name, rule in THROUGHPUT_RULES.items():
            res = solve_local_search(inst, rule, WEIGHTS, seed=seed)
            chains = res.assignment.chains()
            states = [chain_state(inst, res.assignment, c, rule) for c in chains]
            states = [s for s in states if s is not None]
            served = sum(s.gas for s in states)
            demand = sum(a.gas for a in inst.apps)
            rows.append(dict(
                seed=seed, rule=rule_name, objective=res.evaluation.objective,
                n_chains=len(states),
                mean_ops_per_chain=(mean(len(s.ops) for s in states)
                                    if states else 0.0),
                served_fraction=served / demand,
            ))
    write_csv(args.out or "sensitivity_throughput.csv", rows,
              ["seed", "rule", "objective", "n_chains", "mean_ops_per_chain",
               "served_fraction"])


def misreporting(args) -> None:
    """Max unilateral gain over a grid of misreports, per role and coordinate."""
    rows = []
    rng = np.random.default_rng(args.seed)
    rule = THROUGHPUT_RULES["min"]
    for i in range(args.instances):
        seed = args.seed + i
        inst = generate(CONFIG, seed)
        truthful = solve_local_search(inst, rule, WEIGHTS, seed=seed)

        for role in ("app", "op"):
            idx = int(rng.integers(0, inst.n_apps if role == "app" else inst.n_ops))
            true_util = (truthful.evaluation.app_utils[idx] if role == "app"
                         else truthful.evaluation.op_utils[idx])
            for coord in ("gas", "gasprice"):
                best_gain = 0.0
                for factor in GRID:
                    if factor == 1.0:
                        continue
                    if role == "app":
                        agent = inst.apps[idx]
                        mis = App(**{**agent.__dict__, coord: getattr(agent, coord) * factor})
                        mis_inst = inst.with_app(idx, mis)
                    else:
                        agent = inst.ops[idx]
                        mis = Op(**{**agent.__dict__, coord: getattr(agent, coord) * factor})
                        mis_inst = inst.with_op(idx, mis)
                    res = solve_local_search(mis_inst, rule, WEIGHTS, seed=seed,
                                             warm_start=truthful.assignment)
                    # Utility of the (truthfully valued) agent under the
                    # misreported outcome: recompute served gas / yield from
                    # the realized assignment but value it truthfully.
                    if role == "app":
                        util = res.evaluation.app_utils[idx]
                        util = min(util, agent.gas)  # gas beyond true demand is valueless
                    else:
                        util = res.evaluation.op_utils[idx]
                    best_gain = max(best_gain, util - true_util)
                rows.append(dict(
                    seed=seed, role=role, agent=idx, coord=coord,
                    true_util=true_util, max_gain=best_gain,
                    rel_gain=(best_gain / true_util if true_util > 0 else ""),
                ))
    write_csv(args.out or "misreporting.csv", rows,
              ["seed", "role", "agent", "coord", "true_util", "max_gain",
               "rel_gain"])


EXPERIMENTS = {
    "sensitivity_normalization": sensitivity_normalization,
    "sensitivity_throughput": sensitivity_throughput,
    "misreporting": misreporting,
}


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    args = parser.parse_args()
    EXPERIMENTS[args.experiment](args)


if __name__ == "__main__":
    main()
