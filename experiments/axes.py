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


def measure(inst, weights=None):
    weights = weights or WEIGHTS["uniform"]
    bounds = NormalizationBounds.ideal(inst, RULE, lambda i, r, w, b: solve_exact(i, r, w, bounds=b))
    prof = {"napp": [], "nop": [], "nsys": []}
    for w in simplex_grid(SIMPLEX_STEP):
        ev = solve_exact(inst, RULE, w, bounds=bounds).evaluation
        prof["napp"].append(mean(ev.app_utils[a] / inst.apps[a].gas for a in range(inst.n_apps)))
        prof["nop"].append(mean(u / bounds.q_op for u in ev.op_utils))
        prof["nsys"].append(ev.sys_util / bounds.q_sys)
    # steerability = range of the ideal-point-normalized aggregate; for
    # applications the ideal point is the maximum over the simplex (attained
    # at the application corner).
    q_app = max(prof["napp"]) or 1.0
    out = {"steer_napp": (max(prof["napp"]) - min(prof["napp"])) / q_app,
           "steer_nop": max(prof["nop"]) - min(prof["nop"]),
           "steer_nsys": max(prof["nsys"]) - min(prof["nsys"])}
    # the same ranges restricted to the app--sys edge (lambda_op = 0)
    edge = [i for i, w in enumerate(simplex_grid(SIMPLEX_STEP)) if w.op == 0.0]
    for k in ("napp", "nsys"):
        vals = [prof[k][i] for i in edge]
        out[f"edge_{k}"] = (max(vals) - min(vals)) / (q_app if k == "napp" else 1.0)
    truthful = solve_exact(inst, RULE, weights).evaluation
    # theory ceilings for the application demand channel (Prop. gains):
    # fraction of applications not fully served, and mean unserved fraction
    served = [min(truthful.app_utils[a], inst.apps[a].gas) / inst.apps[a].gas for a in range(inst.n_apps)]
    out["app_not_full_frac"] = mean(f < 1 - 1e-9 for f in served)
    out["app_unserved_mean"] = mean(1 - f for f in served)
    # full service feasible? (premise of the homogeneous-caps corollary)
    out["full_service_feasible"] = float(q_app > 1 - 1e-9)
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
                    ev = solve_exact(mis_inst, RULE, weights).evaluation
                    util = (min(ev.app_utils[idx], agent.gas) if role == "app"
                            else ev.op_utils[idx] * (factor if coord == "stake" else 1.0))
                    best = max(best, util - true_util)
                channels[(role, coord)].append((best, true_util))
    for (role, coord), g in channels.items():
        out[f"manip_{role}_{coord}_frac"] = mean(b > 1e-9 for b, _ in g)
        out[f"manip_{role}_{coord}_rel"] = mean(b / t for b, t in g if t > 0) if any(t > 0 for _, t in g) else 0.0
        if role == "app":  # gain as a fraction of true demand (bounded by unserved fraction)
            out["manip_app_gas_dem"] = mean(b / inst.apps[i].gas for i, (b, _) in enumerate(g))
    return out


def main() -> None:
    parser = base_parser(__doc__)
    parser.set_defaults(instances=10)
    parser.add_argument("--max-levels", type=int, default=None)
    parser.add_argument("--weights", default="uniform", choices=list(WEIGHTS), help="weights for the manipulation measures")
    parser.add_argument("--axis", default=None, choices=["cap", "slack"])
    args = parser.parse_args()
    path = RESULTS_DIR / (args.out or "axes.csv")
    done = set()
    if path.exists():
        with open(path) as h:
            done = {(r["axis"], float(r["level"]), int(r["seed"])) for r in csv.DictReader(h)}
    levels = [("cap", c, 1.0, c) for c in CAP_LEVELS] + [("slack", s, s, 0.8) for s in SLACK_LEVELS]
    if args.axis:
        levels = [l for l in levels if l[0] == args.axis]
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
                             **measure(inst, WEIGHTS[args.weights])))
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
