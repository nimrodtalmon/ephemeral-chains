"""A map of instances (after the "map of elections").

Samples instances with features drawn continuously (slack, cap and ask
heterogeneity, stake heterogeneity) and computes, for each, exactly:

* its *governance response profile* -- the three group utilities at every
  point of a grid on the weight simplex; and
* its *manipulability* -- the maximal truthfully-valued unilateral gain of
  every agent and coordinate (under uniform weights), summarized per role.

The profile defines a behavioral distance between instances (two
instances are close if governance moves them alike); ``map_figure.py``
embeds it in the plane and colors the map by manipulability, next to the
plain feature-axes view. Resumable: one CSV row per instance is appended.
"""

from __future__ import annotations

import csv
import pathlib
import sys
from statistics import mean

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from ephemeral_chains.instance import App, GeneratorConfig, Op, generate
from ephemeral_chains.model import THROUGHPUT_RULES, NormalizationBounds
from ephemeral_chains.solver import solve_exact, solve_local_search
from experiments.common import RESULTS_DIR, base_parser
from experiments.governance import SIMPLEX_STEP, simplex_grid
from experiments.landscape import BASE, COORDS, FACTORS, WEIGHTS

RULE = "min"
AGENTS_PER_ROLE = 5  # sampled per instance when solving heuristically


def sample_config(rng: np.random.Generator, n_apps: int, n_ops: int) -> tuple[GeneratorConfig, dict]:
    slack = float(np.exp(rng.uniform(np.log(0.25), np.log(8.0))))
    cap_sigma = float(rng.uniform(0.0, 1.0))
    ask_sigma = float(rng.uniform(0.0, 0.6))
    stake_alpha = float(np.exp(rng.uniform(np.log(1.2), np.log(50.0))))
    config = GeneratorConfig(
        n_apps=n_apps, n_ops=n_ops,
        cap_mu=BASE.cap_mu + float(np.log(slack)),
        app_price_sigma=cap_sigma, op_price_sigma=ask_sigma,
        price_gas_corr=0.0 if cap_sigma == 0.0 else BASE.price_gas_corr,
        stake_pareto_alpha=stake_alpha,
        app_stake_scale=BASE.stake_scale / 2 if n_ops <= 3 else None,
    )
    return config, dict(slack=slack, cap_sigma=cap_sigma, ask_sigma=ask_sigma,
                        stake_alpha=stake_alpha)


def features(inst) -> dict:
    caps = np.array([a.gasprice for a in inst.apps]); asks = np.array([o.gasprice for o in inst.ops])
    stakes = np.array([o.stake for o in inst.ops]); capac = np.array([o.gas for o in inst.ops])
    compat = np.mean([[o <= a for o in asks] for a in caps])
    return dict(
        supply_demand=sum(capac) / sum(a.gas for a in inst.apps),
        cap_cv=caps.std() / caps.mean(), ask_cv=asks.std() / asks.mean(),
        stake_cv=stakes.std() / stakes.mean(), capacity_cv=capac.std() / capac.mean(),
        price_compat=float(compat),
        stake_cover=float(np.mean([a.stake <= stakes.sum() for a in inst.apps])),
    )


def manipulability(inst, rule, weights, solve, rng) -> dict:
    """Per-channel manipulability under honorable misreports.

    Channels: application demand, operator ask, operator stake, operator
    capacity (under-declared). For each, the fraction of sampled agents
    with a profitable misreport and the mean relative gain.
    """
    truthful_res = solve(inst)
    truthful = truthful_res.evaluation
    channels = {("app", "gas"): [], ("op", "gasprice"): [], ("op", "stake"): [], ("op", "gas"): []}
    for role in ("app", "op"):
        agents = inst.apps if role == "app" else inst.ops
        n = len(agents)
        idxs = rng.choice(n, size=min(AGENTS_PER_ROLE, n), replace=False) if AGENTS_PER_ROLE else range(n)
        for idx in idxs:
            agent = agents[idx]
            true_util = (min(truthful.app_utils[idx], agent.gas) if role == "app"
                         else truthful.op_utils[idx])
            for coord in COORDS:
                if (role, coord) not in channels:
                    continue
                best = 0.0
                for factor in FACTORS:
                    if role == "app" and ((coord == "gasprice" and factor > 1)
                                          or (coord == "stake" and factor < 1)):
                        continue
                    if role == "op" and ((coord == "gasprice" and factor < 1)
                                         or (coord in ("gas", "stake") and factor > 1)):
                        continue
                    mis = (App if role == "app" else Op)(
                        **{**agent.__dict__, coord: getattr(agent, coord) * factor})
                    mis_inst = (inst.with_app(idx, mis) if role == "app"
                                else inst.with_op(idx, mis))
                    ev = solve(mis_inst, warm_start=truthful_res.assignment).evaluation
                    util = (min(ev.app_utils[idx], agent.gas) if role == "app"
                            else ev.op_utils[idx] * (factor if coord == "stake" else 1.0))
                    best = max(best, util - true_util)
                channels[(role, coord)].append((best, true_util))
    out = {}
    for (role, coord), g in channels.items():
        out[f"manip_{role}_{coord}_frac"] = mean(b > 1e-9 for b, _ in g)
        rel = [b / t for b, t in g if t > 0]
        out[f"manip_{role}_{coord}_rel"] = mean(rel) if rel else 0.0
    return out

def main() -> None:
    parser = base_parser(__doc__)
    parser.set_defaults(instances=300)
    parser.add_argument("--size", default="4x3", help="n_apps x n_ops; exact solver up to 4x3, local search above")
    args = parser.parse_args()
    n_apps, n_ops = (int(v) for v in args.size.split("x"))
    exact = n_apps * n_ops <= 12
    path = RESULTS_DIR / (args.out or "map.csv")
    rule = THROUGHPUT_RULES[RULE]
    done = set()
    if path.exists():
        with open(path) as handle:
            done = {int(r["seed"]) for r in csv.DictReader(handle)}
    writer, handle = None, None
    for i in range(args.instances):
        seed = args.seed + i
        if seed in done:
            continue
        rng = np.random.default_rng(seed)
        config, params = sample_config(rng, n_apps, n_ops)
        inst = generate(config, seed)
        bounds = NormalizationBounds.ideal(inst, rule, lambda i, r, w, b: (solve_exact(i, r, w, bounds=b) if exact else solve_local_search(i, r, w, bounds=b, seed=seed, restarts=32)))
        row = dict(seed=seed, n_apps=n_apps, n_ops=n_ops, exact=exact, **params, **features(inst))
        for w in simplex_grid(SIMPLEX_STEP):
            ev = (solve_exact(inst, rule, w, bounds=bounds) if exact else
                  solve_local_search(inst, rule, w, bounds=bounds, seed=seed)).evaluation
            tag = f"{round(w.app, 1)}_{round(w.op, 1)}_{round(w.sys, 1)}"
            row[f"napp_{tag}"] = mean(ev.app_utils[a] / inst.apps[a].gas for a in range(inst.n_apps))
            row[f"nop_{tag}"] = mean(u / bounds.q_op for u in ev.op_utils)
            row[f"nsys_{tag}"] = ev.sys_util / bounds.q_sys
        if exact:
            solve = lambda i, warm_start=None: solve_exact(i, rule, WEIGHTS["uniform"])
        else:
            solve = lambda i, warm_start=None: solve_local_search(
                i, rule, WEIGHTS["uniform"], seed=seed, warm_start=warm_start)
        row.update(manipulability(inst, rule, WEIGHTS["uniform"], solve, rng))
        if writer is None:
            new = not path.exists()
            handle = open(path, "a", newline="")
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if new:
                writer.writeheader()
        writer.writerow(row)
        handle.flush()
        print(f"instance {seed} done", flush=True)
    if handle:
        handle.close()


if __name__ == "__main__":
    main()
