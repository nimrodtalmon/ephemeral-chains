"""Optional Nevergrad-based solver (same interface as :mod:`solver`).

Encodes an assignment as one categorical variable per agent (a chain
index or "unassigned") and lets a Nevergrad optimizer search the space;
infeasible assignments score 0 (the empty assignment's value), and chain
prices are resolved analytically as everywhere else. Kept for continuity
with the earlier version of this work; the memoized local search in
:mod:`solver` is the default.
"""

from __future__ import annotations

import time
from typing import Optional

from .instance import Instance
from .model import (
    Assignment,
    NormalizationBounds,
    ThroughputRule,
    Weights,
    evaluate,
    is_feasible,
)
from .solver import SolveResult


def solve_nevergrad(
    instance: Instance,
    rule: ThroughputRule,
    weights: Weights,
    bounds: Optional[NormalizationBounds] = None,
    seed: int = 0,
    budget: int = 4000,
    optimizer: str = "NGOpt",
) -> SolveResult:
    import nevergrad as ng  # imported lazily: optional dependency

    start = time.perf_counter()
    n_chains = max(1, min(instance.n_apps, instance.n_ops))
    choices = list(range(n_chains)) + [-1]  # -1 encodes "unassigned"

    param = ng.p.Instrumentation(
        apps=ng.p.Choice(choices, repetitions=instance.n_apps),
        ops=ng.p.Choice(choices, repetitions=instance.n_ops),
    )
    param.random_state.seed(seed)

    def to_assignment(apps, ops) -> Assignment:
        return Assignment(
            app_chain=tuple(None if c == -1 else int(c) for c in apps),
            op_chain=tuple(None if c == -1 else int(c) for c in ops),
        )

    def loss(apps, ops) -> float:
        assignment = to_assignment(apps, ops)
        if not is_feasible(instance, assignment, rule):
            return 0.0  # value of the empty assignment
        return -evaluate(instance, assignment, rule, weights, bounds).objective

    opt = ng.optimizers.registry[optimizer](parametrization=param, budget=budget)
    recommendation = opt.minimize(lambda *a, **kw: loss(**kw))
    best = to_assignment(**recommendation.kwargs)
    if not is_feasible(instance, best, rule):
        from .model import empty_assignment

        best = empty_assignment(instance)

    return SolveResult(
        assignment=best,
        evaluation=evaluate(instance, best, rule, weights, bounds),
        seconds=time.perf_counter() - start,
        evaluated=budget,
        exact=False,
    )
