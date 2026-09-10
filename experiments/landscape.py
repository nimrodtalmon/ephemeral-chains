"""Manipulation landscape: gains over a factorial grid of instance features.

Tests the predictions of the paper's structural section (Section
"Where Manipulation Gains Come From") on instances small enough to solve
exactly (4 applications, 3 operators), so measured gains are properties
of the mechanism, not of a heuristic solver.

Grid: slack (operator-capacity scale) x cap heterogeneity (log-scale
noise on application caps; 0 = homogeneous caps) x stake heterogeneity
(uniform vs. Pareto) x throughput rule x governance weights. For every
instance, every agent, and every coordinate (gas, gasprice, stake), the
instance is re-solved exactly under a multiplicative grid of unilateral
misreports and the truthfully-valued gain of every misreport is
recorded (one row per agent, coordinate, and factor), so that any
restriction of the strategy set can be applied at analysis time.

Writes one CSV per cell to ``results/landscape/`` (so long runs can be
resumed) and merges them into ``results/landscape.csv``.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from ephemeral_chains.instance import App, GeneratorConfig, Op, generate
from ephemeral_chains.model import THROUGHPUT_RULES, Weights, chain_state
from ephemeral_chains.solver import solve_exact
from experiments.common import RESULTS_DIR, base_parser, write_csv

BASE = GeneratorConfig(n_apps=4, n_ops=3)
SLACKS = [0.5, 4.0]  # operator-capacity scale (shift of cap_mu)
CAP_SIGMAS = [0.0, 0.8]  # heterogeneity of application caps
STAKES = {"uniform": 1e9, "pareto": 1.5}  # Pareto tail index (huge = uniform)
RULES = ["min", "sum"]
WEIGHTS = {
    "uniform": Weights(app=1 / 3, op=1 / 3, sys=1 / 3),
    "sys": Weights(app=0.0, op=0.0, sys=1.0),
}
# Exploratory grid (final runs will widen it): two levels per feature and
# two weight vectors.
FACTORS = [0.25, 0.5, 0.8, 1.25, 2.0, 4.0]
COORDS = ["gas", "gasprice", "stake"]


def cell_config(slack: float, cap_sigma: float, stake_alpha: float) -> GeneratorConfig:
    return GeneratorConfig(
        n_apps=BASE.n_apps, n_ops=BASE.n_ops,
        cap_mu=BASE.cap_mu + float(np.log(slack)),
        app_price_sigma=cap_sigma,
        price_gas_corr=0.0 if cap_sigma == 0.0 else BASE.price_gas_corr,
        stake_pareto_alpha=stake_alpha,
        # Small instances: keep application stake requirements below the
        # typical chain stake, so stake feasibility is a constraint rather
        # than the whole story.
        app_stake_scale=BASE.stake_scale / 2,
    )


def true_utility(role, idx, agent, factor, coord, evaluation) -> float:
    """Utility of the misreporting agent valued at its true declaration."""
    if role == "app":
        return min(evaluation.app_utils[idx], agent.gas)
    util = evaluation.op_utils[idx]  # fee per unit of *declared* stake
    if coord == "stake":
        return util * factor  # fee received / true stake
    return util


def run_cell(slack, cap_sigma, stake_name, rule_name, weight_name, n_instances, seed0):
    rule = THROUGHPUT_RULES[rule_name]
    weights = WEIGHTS[weight_name]
    config = cell_config(slack, cap_sigma, STAKES[stake_name])
    rows = []
    for i in range(n_instances):
        seed = seed0 + i
        inst = generate(config, seed)
        truthful = solve_exact(inst, rule, weights)
        demand = sum(a.gas for a in inst.apps)
        supply = sum(o.gas for o in inst.ops)
        for role in ("app", "op"):
            agents = inst.apps if role == "app" else inst.ops
            for idx, agent in enumerate(agents):
                ev = truthful.evaluation
                true_util = (min(ev.app_utils[idx], agent.gas) if role == "app"
                             else ev.op_utils[idx])
                for coord in COORDS:
                    for factor in FACTORS:
                        mis = (App if role == "app" else Op)(
                            **{**agent.__dict__, coord: getattr(agent, coord) * factor})
                        mis_inst = (inst.with_app(idx, mis) if role == "app"
                                    else inst.with_op(idx, mis))
                        res = solve_exact(mis_inst, rule, weights)
                        util = true_utility(role, idx, agent, factor, coord, res.evaluation)
                        rows.append(dict(
                            slack=slack, cap_sigma=cap_sigma, stake=stake_name,
                            rule=rule_name, weights=weight_name, seed=seed,
                            supply_demand=supply / demand,
                            role=role, agent=idx, coord=coord, factor=factor,
                            true_util=true_util,
                            service=(true_util / agent.gas if role == "app" else ""),
                            gain=util - true_util,
                            rel_gain=((util - true_util) / true_util if true_util > 0 else ""),
                        ))
    return rows


FIELDS = ["slack", "cap_sigma", "stake", "rule", "weights", "seed", "supply_demand",
          "role", "agent", "coord", "factor", "true_util", "service", "gain", "rel_gain"]


def main() -> None:
    parser = base_parser(__doc__)
    parser.set_defaults(instances=4)
    parser.add_argument("--max-cells", type=int, default=None,
                        help="stop after this many not-yet-computed cells")
    parser.add_argument("--rules", nargs="*", default=None, help="restrict to these rules")
    args = parser.parse_args()
    cell_dir = RESULTS_DIR / "landscape"
    cell_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for slack in SLACKS:
        for cap_sigma in CAP_SIGMAS:
            for stake_name in STAKES:
                for rule_name in (args.rules or RULES):
                    for weight_name in WEIGHTS:
                        name = f"s{slack}_c{cap_sigma}_{stake_name}_{rule_name}_{weight_name}" + (f"_seed{args.seed}" if args.seed else "") + ".csv"
                        if (cell_dir / name).exists():
                            continue
                        rows = run_cell(slack, cap_sigma, stake_name, rule_name,
                                        weight_name, args.instances, args.seed)
                        write_csv(f"landscape/{name}", rows, FIELDS)
                        done += 1
                        if args.max_cells and done >= args.max_cells:
                            return
    merged = []
    for path in sorted(cell_dir.glob("*.csv")):
        with open(path) as handle:
            lines = handle.read().splitlines()
        merged.extend(lines[1:])
    with open(RESULTS_DIR / "landscape.csv", "w") as handle:
        handle.write(",".join(FIELDS) + "\n" + "\n".join(merged) + "\n")
    print(f"merged {len(merged)} rows into results/landscape.csv")


if __name__ == "__main__":
    main()
