"""Best-response dynamics in the declaration game.

Starting from truthful declarations, agents best-respond in round-robin
order (applications, then operators) over a finite strategy set: keep
the truthful declaration, or misreport one coordinate (gas, gasprice,
stake) by a factor from a fixed grid, restricted to declarations the
agent could honor: applications never raise their cap or lower their
stake requirement, operators never lower their ask or overdeclare
capacity or stake. An agent switches only to a
strictly better strategy (ties keep the current one). The process stops
at a fixed point (a pure equilibrium of the restricted declaration game)
or after ``--rounds`` rounds; a repeated profile is reported as a cycle.

For each run we record convergence, the number of rounds, and each
group's mean true utility at the final profile relative to the truthful
profile. Instances are solved exactly (5 applications, 3 operators).
"""

from __future__ import annotations

import pathlib
import sys
from statistics import mean

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ephemeral_chains.instance import App, Op, generate
from ephemeral_chains.model import THROUGHPUT_RULES
from ephemeral_chains.solver import solve_exact
import csv
from experiments.common import base_parser, write_csv, RESULTS_DIR
from dataclasses import replace

from experiments.landscape import COORDS, FACTORS, STAKES, WEIGHTS, cell_config

N_APPS, N_OPS = 4, 3  # exploratory; final runs use 5x3 or larger

CELLS = [(slack, cap_sigma, "pareto", "min", w)
         for slack in (0.5, 4.0) for cap_sigma in (0.0, 0.8)
         for w in ("uniform", "sys")]
# Strategies respect each agent's real requirements: an application will
# not accept a price above its true cap or a chain with less stake than
# it really requires; an operator will not serve below its true ask,
# cannot lock more stake than it has, and cannot deliver more gas than
# its capacity. Everything else -- demand in either direction, and
# tightening one's own constraints -- is on the table.
def strategies(role):
    out = [("gas", 1.0)]
    for coord in COORDS:
        for f in FACTORS:
            if role == "app" and coord == "gasprice" and f > 1.0:
                continue
            if role == "app" and coord == "stake" and f < 1.0:
                continue
            if role == "op" and coord == "gasprice" and f < 1.0:
                continue
            if role == "op" and coord in ("gas", "stake") and f > 1.0:
                continue
            out.append((coord, f))
    return out


APP_STRATEGIES, OP_STRATEGIES = strategies("app"), strategies("op")


def declared(agent, strategy):
    coord, factor = strategy
    cls = App if isinstance(agent, App) else Op
    return cls(**{**agent.__dict__, coord: getattr(agent, coord) * factor})


def profile_instance(truth, profile):
    inst = truth
    for idx, strat in enumerate(profile[:truth.n_apps]):
        inst = inst.with_app(idx, declared(truth.apps[idx], strat))
    for j, strat in enumerate(profile[truth.n_apps:]):
        inst = inst.with_op(j, declared(truth.ops[j], strat))
    return inst


def true_utils(truth, profile, evaluation):
    """True utilities of all agents under a declared profile."""
    apps = [min(evaluation.app_utils[a], truth.apps[a].gas) for a in range(truth.n_apps)]
    ops = []
    for j in range(truth.n_ops):
        coord, factor = profile[truth.n_apps + j]
        u = evaluation.op_utils[j]
        ops.append(u * factor if coord == "stake" else u)
    return apps, ops


def run(truth, rule, weights, rounds):
    n = truth.n_apps + truth.n_ops
    profile = [("gas", 1.0)] * n
    seen = {tuple(profile)}
    ev = solve_exact(truth, rule, weights).evaluation
    truthful_apps, truthful_ops = true_utils(truth, profile, ev)
    outcome, n_rounds = "max_rounds", rounds
    for r in range(1, rounds + 1):
        changed = False
        for i in range(n):
            cur_inst = profile_instance(truth, profile)
            cur_ev = solve_exact(cur_inst, rule, weights).evaluation
            cur_apps, cur_ops = true_utils(truth, profile, cur_ev)
            cur = cur_apps[i] if i < truth.n_apps else cur_ops[i - truth.n_apps]
            best, best_strat = cur, profile[i]
            strategies = APP_STRATEGIES if i < truth.n_apps else OP_STRATEGIES
            for strat in strategies:
                if strat == profile[i]:
                    continue
                trial = list(profile)
                trial[i] = strat
                ev_t = solve_exact(profile_instance(truth, trial), rule, weights).evaluation
                a_u, o_u = true_utils(truth, trial, ev_t)
                u = a_u[i] if i < truth.n_apps else o_u[i - truth.n_apps]
                if u > best + 1e-9:
                    best, best_strat = u, strat
            if best_strat != profile[i]:
                profile[i] = best_strat
                changed = True
        if not changed:
            outcome, n_rounds = "fixed_point", r
            break
        key = tuple(profile)
        if key in seen:
            outcome, n_rounds = "cycle", r
            break
        seen.add(key)
    final_ev = solve_exact(profile_instance(truth, profile), rule, weights).evaluation
    f_apps, f_ops = true_utils(truth, profile, final_ev)
    n_dev = sum(1 for s in profile if s[1] != 1.0)
    return dict(
        outcome=outcome, rounds=n_rounds, n_deviators=n_dev,
        truthful_app=mean(truthful_apps), final_app=mean(f_apps),
        truthful_op=mean(truthful_ops), final_op=mean(f_ops),
        truthful_sys=ev.sys_util, final_sys=final_ev.sys_util,
        apps_worse=sum(1 for a, b in zip(truthful_apps, f_apps) if b < a - 1e-9),
        ops_worse=sum(1 for a, b in zip(truthful_ops, f_ops) if b < a - 1e-9),
    )


def main() -> None:
    parser = base_parser(__doc__)
    parser.set_defaults(instances=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--cell", type=int, default=None, help="run a single cell index")
    parser.add_argument("--axes", action="store_true",
                        help="run along the two axes of experiments/axes.py (same instances, uniform weights)")
    args = parser.parse_args()
    rows = []
    out_name = args.out or ("dynamics_axes.csv" if args.axes else "dynamics.csv")
    out_path = RESULTS_DIR / out_name
    done = set()
    if out_path.exists():
        with open(out_path) as h:
            done = {(float(r["slack"]), float(r["cap_sigma"]), int(r["seed"])) for r in csv.DictReader(h)}
    cells = CELLS if args.cell is None else [CELLS[args.cell]]
    if args.axes:
        from experiments.axes import CAP_LEVELS, SLACK_LEVELS
        cells = ([(1.0, c, "pareto", "min", "uniform") for c in CAP_LEVELS]
                 + [(s, 0.8, "pareto", "min", "uniform") for s in SLACK_LEVELS])
    for slack, cap_sigma, stake_name, rule_name, w_name in cells:
        config = replace(cell_config(slack, cap_sigma, STAKES[stake_name]),
                         n_apps=N_APPS, n_ops=N_OPS)
        for i in range(args.instances):
            seed = args.seed + i
            if (float(slack), float(cap_sigma), seed) in done:
                continue
            truth = generate(config, seed)
            row = run(truth, THROUGHPUT_RULES[rule_name], WEIGHTS[w_name], args.rounds)
            rows.append(dict(slack=slack, cap_sigma=cap_sigma, stake=stake_name,
                             rule=rule_name, weights=w_name, seed=seed, **row))
            print(rows[-1], flush=True)
            new = not out_path.exists()
            with open(out_path, "a", newline="") as h:
                w = csv.DictWriter(h, fieldnames=list(rows[-1].keys()))
                if new:
                    w.writeheader()
                w.writerow(rows[-1])
    print(f"{len(rows)} new rows -> {out_path}")


if __name__ == "__main__":
    main()
