"""Experiment: operator price manipulation vs. pivotality.

Hypothesis (paper, Section 6): price misreports have no ground truth, so
the only defense is competition -- an operator that over-asks risks being
replaced. That defense should fail exactly for *pivotal* operators. We
test this by measuring, per sampled operator, (i) its pivotality -- the
relative drop in the optimal objective when the operator is excluded --
and (ii) its maximal utility gain from a grid of price misreports, and
correlating the two across instances spanning scarce-to-abundant
capacity regimes.

An operator is excluded by raising its ask above every application's
price cap, which makes every chain containing it price-infeasible while
keeping indices stable.
"""

from __future__ import annotations

import sys
from dataclasses import replace

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np

from ephemeral_chains.instance import GeneratorConfig, Op, generate
from ephemeral_chains.model import THROUGHPUT_RULES, Weights
from ephemeral_chains.solver import solve_local_search
from experiments.common import base_parser, write_csv

BASE = GeneratorConfig(n_apps=20, n_ops=10)
CAPACITY_SCALES = [0.5, 1.0, 2.0]
OPS_PER_INSTANCE = 3
PRICE_FACTORS = [0.25, 0.5, 0.8, 1.25, 2.0, 4.0]
WEIGHTS = Weights(app=1 / 3, op=1 / 3, sys=1 / 3)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    rule = THROUGHPUT_RULES["min"]
    rows = []

    for cap_scale in CAPACITY_SCALES:
        config = replace(BASE, cap_mu=BASE.cap_mu + float(np.log(cap_scale)))
        for i in range(args.instances):
            seed = args.seed + i
            inst = generate(config, seed)
            rng = np.random.default_rng(seed)
            truthful = solve_local_search(inst, rule, WEIGHTS, seed=seed, restarts=32)
            obj = truthful.evaluation.objective
            slack = sum(o.gas for o in inst.ops) / sum(a.gas for a in inst.apps)
            exclusion_price = 10.0 * max(a.gasprice for a in inst.apps)

            for idx in rng.choice(inst.n_ops, OPS_PER_INSTANCE, replace=False):
                idx = int(idx)
                op = inst.ops[idx]
                # Pivotality: relative objective drop when the operator is out.
                excluded = inst.with_op(idx, Op(gas=op.gas, stake=op.stake,
                                               gasprice=exclusion_price))
                without = solve_local_search(excluded, rule, WEIGHTS, seed=seed,
                                             restarts=32,
                                             warm_start=truthful.assignment)
                pivotality = (obj - without.evaluation.objective) / obj if obj > 0 else 0.0

                true_util = truthful.evaluation.op_utils[idx]
                best_gain = 0.0
                for factor in PRICE_FACTORS:
                    mis = inst.with_op(idx, Op(gas=op.gas, stake=op.stake,
                                               gasprice=op.gasprice * factor))
                    res = solve_local_search(mis, rule, WEIGHTS, seed=seed,
                                             restarts=32,
                                             warm_start=truthful.assignment)
                    best_gain = max(best_gain, res.evaluation.op_utils[idx] - true_util)

                rows.append(dict(
                    cap_scale=cap_scale, seed=seed, op=idx,
                    slack=slack, pivotality=pivotality,
                    true_util=true_util, max_price_gain=best_gain,
                ))

    write_csv(args.out or "pivotality.csv", rows,
              ["cap_scale", "seed", "op", "slack", "pivotality",
               "true_util", "max_price_gain"])


if __name__ == "__main__":
    main()
