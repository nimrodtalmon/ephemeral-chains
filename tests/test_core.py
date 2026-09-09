"""Tests: model invariants, the paper's worked example, and solver sanity."""

import numpy as np
import pytest

from ephemeral_chains.instance import App, GeneratorConfig, Instance, Op, generate
from ephemeral_chains.model import (
    Assignment,
    NormalizationBounds,
    Weights,
    canonical_key,
    evaluate,
    is_feasible,
    optimal_prices,
    t_min,
    t_overhead_harmonic,
    t_sum,
)
from ephemeral_chains.solver import solve_exact, solve_local_search


def paper_example() -> Instance:
    """The instance of Section 3.6 of the paper."""
    return Instance(
        apps=[
            App(gas=60, stake=10, gasprice=5),
            App(gas=30, stake=10, gasprice=4),
            App(gas=20, stake=30, gasprice=6),
        ],
        ops=[
            Op(gas=50, stake=20, gasprice=2),
            Op(gas=40, stake=20, gasprice=3),
            Op(gas=30, stake=30, gasprice=2),
        ],
    )


def paper_assignment() -> Assignment:
    """Chains c1 = ({a1,a2},{o1,o2}), c2 = ({a3},{o3})."""
    return Assignment(app_chain=(0, 0, 1), op_chain=(0, 0, 1))


def test_example_feasible_under_both_rules():
    inst = paper_example()
    asg = paper_assignment()
    assert is_feasible(inst, asg, t_sum)
    assert is_feasible(inst, asg, t_min)


def test_example_quantities_match_paper():
    inst = paper_example()
    asg = paper_assignment()
    w = Weights(app=0, op=0, sys=1.0)
    # Under T_sum with sys-weight, optimal price of c1 is the cap 4.
    ev = evaluate(inst, asg, t_sum, w, prices={0: 3.5, 1: 4.0})
    # c1: Gas = min(90, 90) = 90, fully served apps.
    assert ev.app_utils[0] == pytest.approx(60)
    assert ev.app_utils[1] == pytest.approx(30)
    # Fee_c1 = 3.5 * 90 = 315; yield = 315/40.
    assert ev.op_utils[0] == pytest.approx(315 / 40)
    assert ev.op_utils[1] == pytest.approx(315 / 40)
    # c2 unaffected: Gas = 20, Fee = 80, yield = 80/30.
    assert ev.op_utils[2] == pytest.approx(80 / 30)

    ev_min = evaluate(inst, asg, t_min, w, prices={0: 3.5, 1: 4.0})
    # c1 under T_min: Supply = 40, rationing 60:30.
    assert ev_min.app_utils[0] == pytest.approx(60 / 90 * 40)
    assert ev_min.app_utils[1] == pytest.approx(30 / 90 * 40)
    assert ev_min.op_utils[0] == pytest.approx(3.5 * 40 / 40)


def test_overhead_rule_between_min_and_sum():
    caps = [50.0, 40.0, 30.0]
    assert t_min(caps) <= t_overhead_harmonic(caps) <= t_sum(caps)
    assert t_overhead_harmonic([7.0]) == pytest.approx(7.0)  # f(1) = 1


def test_price_infeasible_chain_detected():
    inst = Instance(
        apps=[App(gas=10, stake=1, gasprice=2)],
        ops=[Op(gas=10, stake=1, gasprice=3)],  # ask above cap
    )
    asg = Assignment(app_chain=(0,), op_chain=(0,))
    assert not is_feasible(inst, asg, t_sum)


def test_stake_infeasible_chain_detected():
    inst = Instance(
        apps=[App(gas=10, stake=100, gasprice=5)],
        ops=[Op(gas=10, stake=1, gasprice=1)],
    )
    asg = Assignment(app_chain=(0,), op_chain=(0,))
    assert not is_feasible(inst, asg, t_sum)


def test_canonical_key_collapses_relabeling():
    a1 = Assignment(app_chain=(0, 1), op_chain=(0, 1))
    a2 = Assignment(app_chain=(1, 0), op_chain=(1, 0))
    a3 = Assignment(app_chain=(0, 1), op_chain=(1, 0))
    assert canonical_key(a1) == canonical_key(a2)
    assert canonical_key(a1) != canonical_key(a3)


def test_optimal_price_is_upper_endpoint_when_sys_weighted():
    inst = paper_example()
    asg = paper_assignment()
    prices = optimal_prices(inst, asg, t_sum, Weights(app=0, op=0, sys=1.0))
    assert prices[0] == pytest.approx(4.0)  # min app cap on c1
    assert prices[1] == pytest.approx(6.0)


def test_pooling_proposition_additive_uniform_prices():
    """Proposition (pooling): T_sum + lambda_sys=1 + uniform app prices
    -> the single pooled chain is optimal."""
    rng = np.random.default_rng(7)
    for _ in range(5):
        n_a, n_o = 3, 2
        apps = [
            App(gas=float(rng.uniform(1, 20)), stake=1.0, gasprice=5.0)
            for _ in range(n_a)
        ]
        ops = [
            Op(
                gas=float(rng.uniform(1, 20)),
                stake=10.0,
                gasprice=float(rng.uniform(1, 5)),
            )
            for _ in range(n_o)
        ]
        inst = Instance(apps=apps, ops=ops)
        res = solve_exact(inst, t_sum, Weights(app=0, op=0, sys=1.0))
        pooled = Assignment(
            app_chain=tuple([0] * n_a), op_chain=tuple([0] * n_o)
        )
        pooled_ev = evaluate(inst, pooled, t_sum, Weights(app=0, op=0, sys=1.0))
        assert res.evaluation.sys_util == pytest.approx(
            pooled_ev.sys_util, rel=1e-9
        )


def test_local_search_matches_exact_on_small_instances():
    cfg = GeneratorConfig(n_apps=3, n_ops=3)
    for seed in range(4):
        inst = generate(cfg, seed=seed)
        w = Weights(app=0.4, op=0.3, sys=0.3)
        exact = solve_exact(inst, t_min, w)
        ls = solve_local_search(inst, t_min, w, seed=seed, restarts=12)
        assert ls.evaluation.objective <= exact.evaluation.objective + 1e-9
        assert ls.evaluation.objective >= 0.95 * exact.evaluation.objective


def test_generator_reproducible():
    cfg = GeneratorConfig()
    assert generate(cfg, seed=1) == generate(cfg, seed=1)
    assert generate(cfg, seed=1) != generate(cfg, seed=2)


def test_normalization_bounds_upper_bound_utilities():
    cfg = GeneratorConfig(n_apps=4, n_ops=3)
    inst = generate(cfg, seed=3)
    w = Weights(app=1 / 3, op=1 / 3, sys=1 / 3)
    bounds = NormalizationBounds.analytic(inst)
    res = solve_local_search(inst, t_sum, w, seed=0)
    ev = res.evaluation
    assert ev.sys_util <= bounds.q_sys + 1e-9
    assert all(u <= bounds.q_op + 1e-9 for u in ev.op_utils)
    assert 0.0 <= ev.objective <= 1.0
