"""Joint deviation: all operators inflate their asks by a common factor.

Exact solver on the small grid (bottleneck rule, uniform weights, cap
heterogeneity 0.8, Pareto stakes, scarce and abundant capacity). Reports
mean true operator utility, application utility, and fee volume under
the joint deviation relative to the truthful profile.
"""

from __future__ import annotations

import pathlib
import sys
from statistics import mean

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ephemeral_chains.instance import Op, generate
from ephemeral_chains.model import THROUGHPUT_RULES
from ephemeral_chains.solver import solve_exact
from experiments.common import base_parser, write_csv
from experiments.landscape import STAKES, WEIGHTS, cell_config


def main() -> None:
    parser = base_parser(__doc__)
    parser.set_defaults(instances=10)
    args = parser.parse_args()
    rule = THROUGHPUT_RULES["min"]
    rows = []
    for slack in (0.5, 4.0):
        config = cell_config(slack, 0.8, STAKES["pareto"])
        for i in range(args.instances):
            seed = args.seed + i
            inst = generate(config, seed)
            base = solve_exact(inst, rule, WEIGHTS["uniform"]).evaluation
            for factor in (1.0, 1.25, 1.5, 2.0, 4.0):
                dev = inst
                for j, o in enumerate(inst.ops):
                    dev = dev.with_op(j, Op(**{**o.__dict__, "gasprice": o.gasprice * factor}))
                ev = solve_exact(dev, rule, WEIGHTS["uniform"]).evaluation
                rows.append(dict(slack=slack, seed=seed, factor=factor,
                                 op=mean(ev.op_utils), op_truthful=mean(base.op_utils),
                                 app=mean(min(u, a.gas) for u, a in zip(ev.app_utils, inst.apps)),
                                 app_truthful=mean(base.app_utils),
                                 sys=ev.sys_util, sys_truthful=base.sys_util))
    write_csv(args.out or "joint.csv", rows, list(rows[0].keys()))


if __name__ == "__main__":
    main()
