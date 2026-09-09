"""The two maps along their two axes.

For each level of cap heterogeneity (at balanced capacity) and each level
of slack (at heterogeneous caps), exact small instances are solved (i)
over the governance simplex, giving each group's steerability, and (ii)
under every honorable unilateral misreport at uniform weights, giving
each channel's manipulability. Same instances, same solver, so the two
maps can be drawn on one pair of axes. Resumable per level.
"""

from __future__ import annotations

import csv
import pathlib
import sys
from statistics import mean

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from ephemeral_chains.instance import App, Op, generate
from ephemeral_chains.model import THROUGHPUT_RULES, NormalizationBounds
from ephemeral_chains.solver import solve_exact
from experiments.common import RESULTS_DIR, base_parser
from experiments.governance import SIMPLEX_STEP, simplex_grid
from experiments.landscape import COORDS, FACTORS, STAKES, WEIGHTS, cell_config

CAP_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SLACK_LEVELS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RULE = THROUGHPUT_RULES["min"]


def honorable(role, coord, factor):
    if role == "app":
        return not ((coord == "gasprice" and factor > 1) or (coord == "stake" and factor < 1))
    return not ((coord == "gasprice" and factor < 1) or (coord in ("gas", "stake") and factor > 1))


def measure(inst):
    bounds = NormalizationBounds.ideal(inst, RULE, lambda i, r, w, b: solve_exact(i, r, w, bounds=b))
    prof = {"napp": [], "nop": [], "nsys": []}
    for w in simplex_grid(SIMPLEX_STEP):
        ev = solve_exact(inst, RULE, w, bounds=bounds).evaluation
        prof["napp"].append(mean(ev.app_utils[a] / inst.apps[a].gas for a in range(inst.n_apps)))
        prof["nop"].append(mean(u / bounds.q_op for u in ev.op_utils))
        prof["nsys"].append(ev.sys_util / bounds.q_sys)
    out = {f"steer_{k}": max(v) - min(v) for k, v in prof.items()}
    truthful = solve_exact(inst, RULE, WEIGHTS["uniform"]).evaluation
    channels = {("app", "gas"): [], ("op", "gasprice"): [], ("op", "stake"): [], ("op", "gas"): []}
    for role in ("app", "op"):
        agents = inst.apps if role == "app" else inst.ops
        for idx, agent in enumerate(agents):
            true_util = (min(truthful.app_utils[idx], agent.gas) if role == "app"
                         else truthful.op_utils[idx])
            for coord in COORDS:
                if (role, coord) not in channels:
                    continue
                best = 0.0
                for factor in FACTORS:
                    if not honorable(role, coord, factor):
                        continue
                    mis = (App if role == "app" else Op)(
                        **{**agent.__dict__, coord: getattr(agent, coord) * factor})
                    mis_inst = inst.with_app(idx, mis) if role == "app" else inst.with_op(idx, mis)
                    ev = solve_exact(mis_inst, RULE, WEIGHTS["uniform"]).evaluation
                    util = (min(ev.app_utils[idx], agent.gas) if role == "app"
                            else ev.op_utils[idx] * (factor if coord == "stake" else 1.0))
                    best = max(best, util - true_util)
                channels[(role, coord)].append((best, true_util))
    for (role, coord), g in channels.items():
        out[f"manip_{role}_{coord}_frac"] = mean(b > 1e-9 for b, _ in g)
        out[f"manip_{role}_{coord}_rel"] = mean(b / t for b, t in g if t > 0) if any(t > 0 for _, t in g) else 0.0
    return out


def main() -> None:
    parser = base_parser(__doc__)
    parser.set_defaults(instances=10)
    parser.add_argument("--max-levels", type=int, default=None)
    args = parser.parse_args()
    path = RESULTS_DIR / (args.out or "axes.csv")
    done = set()
    if path.exists():
        with open(path) as h:
            done = {(r["axis"], float(r["level"]), int(r["seed"])) for r in csv.DictReader(h)}
    levels = [("cap", c, 1.0, c) for c in CAP_LEVELS] + [("slack", s, s, 0.8) for s in SLACK_LEVELS]
    n_levels = 0
    for axis, level, slack, cap_sigma in levels:
        if all((axis, level, args.seed + i) in done for i in range(args.instances)):
            continue
        rows = []
        for i in range(args.instances):
            seed = args.seed + i
            if (axis, level, seed) in done:
                continue
            inst = generate(cell_config(slack, cap_sigma, STAKES["pareto"]), seed)
            rows.append(dict(axis=axis, level=level, slack=slack, cap_sigma=cap_sigma, seed=seed,
                             **measure(inst)))
        new = not path.exists()
        with open(path, "a", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"{axis}={level} done", flush=True)
        n_levels += 1
        if args.max_levels and n_levels >= args.max_levels:
            return


if __name__ == "__main__":
    main()
