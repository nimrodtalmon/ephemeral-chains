"""Governance steerability on the same exact-solvable grid as the
manipulation landscape (``experiments/landscape.py``).

For every cell of slack x cap heterogeneity x stake heterogeneity x
throughput rule, and every instance, the governance weights are swept
over a grid on the simplex and each group's mean normalized utility is
recorded at every point; steerability is the range over the simplex.
Solved exactly, so the two maps share coordinates and solver.
"""

from __future__ import annotations

import pathlib
import sys
from statistics import mean

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ephemeral_chains.instance import generate
from ephemeral_chains.model import THROUGHPUT_RULES, NormalizationBounds
from ephemeral_chains.solver import solve_exact
from experiments.common import base_parser, write_csv
from experiments.governance import SIMPLEX_STEP, simplex_grid
from experiments.landscape import CAP_SIGMAS, RULES, SLACKS, STAKES, cell_config


def main() -> None:
    parser = base_parser(__doc__)
    parser.set_defaults(instances=3)
    args = parser.parse_args()
    point_rows, summary_rows = [], []
    for slack in SLACKS:
        for cap_sigma in CAP_SIGMAS:
            for stake_name, alpha in STAKES.items():
                config = cell_config(slack, cap_sigma, alpha)
                for rule_name in RULES:
                    rule = THROUGHPUT_RULES[rule_name]
                    for i in range(args.instances):
                        seed = args.seed + i
                        inst = generate(config, seed)
                        bounds = NormalizationBounds.ideal(inst, rule, lambda i, r, w, b: solve_exact(i, r, w, bounds=b))
                        per_point = []
                        for w in simplex_grid(SIMPLEX_STEP):
                            ev = solve_exact(inst, rule, w, bounds=bounds).evaluation
                            napp = mean(ev.app_utils[a] / inst.apps[a].gas
                                        for a in range(inst.n_apps))
                            nop = mean(u / bounds.q_op for u in ev.op_utils)
                            nsys = ev.sys_util / bounds.q_sys
                            per_point.append((napp, nop, nsys))
                            point_rows.append(dict(
                                slack=slack, cap_sigma=cap_sigma, stake=stake_name,
                                rule=rule_name, seed=seed,
                                w_app=round(w.app, 2), w_op=round(w.op, 2),
                                w_sys=round(w.sys, 2), napp=napp, nop=nop, nsys=nsys))
                        rng = lambda k: (max(p[k] for p in per_point)
                                         - min(p[k] for p in per_point))
                        summary_rows.append(dict(
                            slack=slack, cap_sigma=cap_sigma, stake=stake_name,
                            rule=rule_name, seed=seed,
                            steer_app=rng(0), steer_op=rng(1), steer_sys=rng(2)))
    write_csv("landscape_governance_points.csv", point_rows,
              ["slack", "cap_sigma", "stake", "rule", "seed", "w_app", "w_op",
               "w_sys", "napp", "nop", "nsys"])
    write_csv("landscape_governance.csv", summary_rows,
              ["slack", "cap_sigma", "stake", "rule", "seed",
               "steer_app", "steer_op", "steer_sys"])


if __name__ == "__main__":
    main()
