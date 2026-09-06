"""Experiment 1: scalability and optimality gap.

For small instances, compare local search against exact enumeration
(optimality gap); for larger instances, report runtime and solution
value of local search across sizes and throughput rules.
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from ephemeral_chains.instance import GeneratorConfig, generate
from ephemeral_chains.model import THROUGHPUT_RULES, Weights
from ephemeral_chains.solver import solve_exact, solve_local_search
from experiments.common import base_parser, write_csv

SMALL_SIZES = [(3, 2), (3, 3), (4, 3)]  # exact tractable
LARGE_SIZES = [(10, 5), (20, 10), (50, 25), (100, 50), (200, 100)]
WEIGHTS = Weights(app=1 / 3, op=1 / 3, sys=1 / 3)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    rows = []

    for rule_name, rule in THROUGHPUT_RULES.items():
        for n_apps, n_ops in SMALL_SIZES:
            for i in range(args.instances):
                seed = args.seed + i
                inst = generate(GeneratorConfig(n_apps=n_apps, n_ops=n_ops), seed)
                exact = solve_exact(inst, rule, WEIGHTS)
                ls = solve_local_search(inst, rule, WEIGHTS, seed=seed)
                gap = (
                    0.0
                    if exact.evaluation.objective == 0
                    else 1 - ls.evaluation.objective / exact.evaluation.objective
                )
                rows.append(
                    dict(
                        rule=rule_name, n_apps=n_apps, n_ops=n_ops, seed=seed,
                        regime="small", objective=ls.evaluation.objective,
                        exact_objective=exact.evaluation.objective, gap=gap,
                        seconds=ls.seconds, exact_seconds=exact.seconds,
                        evaluated=ls.evaluated,
                    )
                )

        for n_apps, n_ops in LARGE_SIZES:
            for i in range(args.instances):
                seed = args.seed + i
                inst = generate(GeneratorConfig(n_apps=n_apps, n_ops=n_ops), seed)
                ls = solve_local_search(inst, rule, WEIGHTS, seed=seed)
                rows.append(
                    dict(
                        rule=rule_name, n_apps=n_apps, n_ops=n_ops, seed=seed,
                        regime="large", objective=ls.evaluation.objective,
                        exact_objective="", gap="", seconds=ls.seconds,
                        exact_seconds="", evaluated=ls.evaluated,
                    )
                )

    write_csv(
        args.out or "scalability.csv", rows,
        ["rule", "n_apps", "n_ops", "seed", "regime", "objective",
         "exact_objective", "gap", "seconds", "exact_seconds", "evaluated"],
    )


if __name__ == "__main__":
    main()
