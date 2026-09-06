"""Solvers for the outer (combinatorial) level.

Two solvers share one interface:

* :func:`solve_exact` -- exhaustive enumeration over canonical
  assignments; exponential, for small instances and for measuring the
  optimality gap of the heuristic.
* :func:`solve_local_search` -- randomized first-improvement local search
  with restarts, memoized on :func:`~ephemeral_chains.model.canonical_key`;
  the practical solver.

Both resolve chain prices analytically through the inner level
(:func:`~ephemeral_chains.model.optimal_prices`) and return a
:class:`SolveResult`.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .instance import Instance
from .model import (
    Assignment,
    Evaluation,
    NormalizationBounds,
    ThroughputRule,
    Weights,
    canonical_key,
    empty_assignment,
    evaluate,
    is_feasible,
)


@dataclass(frozen=True)
class SolveResult:
    assignment: Assignment
    evaluation: Evaluation
    seconds: float
    evaluated: int  # distinct canonical assignments evaluated
    exact: bool


def _evaluate_if_feasible(
    instance: Instance,
    assignment: Assignment,
    rule: ThroughputRule,
    weights: Weights,
    bounds: Optional[NormalizationBounds],
) -> Optional[Evaluation]:
    if not is_feasible(instance, assignment, rule):
        return None
    return evaluate(instance, assignment, rule, weights, bounds)


def solve_exact(
    instance: Instance,
    rule: ThroughputRule,
    weights: Weights,
    bounds: Optional[NormalizationBounds] = None,
) -> SolveResult:
    """Enumerate all assignments up to chain relabeling.

    Every agent independently takes a chain index in ``{None, 0, .., K-1}``
    with ``K = min(n_apps, n_ops)``; canonicalization collapses relabelings.
    Feasible only for very small instances (agents <= ~8).
    """
    start = time.perf_counter()
    n_chains = min(instance.n_apps, instance.n_ops)
    labels: List[Optional[int]] = [None] + list(range(n_chains))

    best: Optional[Tuple[Evaluation, Assignment]] = None
    seen: Set = set()
    for app_choice in itertools.product(labels, repeat=instance.n_apps):
        for op_choice in itertools.product(labels, repeat=instance.n_ops):
            assignment = Assignment(app_chain=app_choice, op_chain=op_choice)
            key = canonical_key(assignment)
            if key in seen:
                continue
            seen.add(key)
            ev = _evaluate_if_feasible(instance, assignment, rule, weights, bounds)
            if ev is not None and (best is None or ev.objective > best[0].objective):
                best = (ev, assignment)

    assert best is not None  # the empty assignment is always feasible
    return SolveResult(
        assignment=best[1],
        evaluation=best[0],
        seconds=time.perf_counter() - start,
        evaluated=len(seen),
        exact=True,
    )


def _random_feasible_start(
    instance: Instance, rule: ThroughputRule, rng: np.random.Generator
) -> Assignment:
    """A random feasible assignment: random op partition, greedy app placement."""
    n_chains = max(1, instance.n_ops // 2)
    op_chain = tuple(int(rng.integers(0, n_chains)) for _ in range(instance.n_ops))
    app_chain: List[Optional[int]] = []
    for a in range(instance.n_apps):
        placed: Optional[int] = None
        for c in rng.permutation(n_chains):
            candidate = Assignment(
                app_chain=tuple(app_chain + [int(c)] + [None] * (instance.n_apps - a - 1)),
                op_chain=op_chain,
            )
            if is_feasible(instance, candidate, rule):
                placed = int(c)
                break
        app_chain.append(placed)
    return Assignment(app_chain=tuple(app_chain), op_chain=op_chain)


def _neighbors(
    assignment: Assignment, n_chains: int, rng: np.random.Generator, k: int
) -> List[Assignment]:
    """k random moves: single-agent relocations plus chain-opening pairs.

    Single-agent moves cannot open a chain (a chain is open only with both
    an application and an operator on it), so a quarter of the moves place
    one application and one operator together on a random chain.
    """
    moves: List[Assignment] = []
    n_apps = len(assignment.app_chain)
    n_ops = len(assignment.op_chain)
    for _ in range(max(1, k // 4)):  # pair-opening moves
        a = int(rng.integers(0, n_apps))
        o = int(rng.integers(0, n_ops))
        c = int(rng.integers(0, n_chains))
        app_chain = list(assignment.app_chain)
        op_chain = list(assignment.op_chain)
        app_chain[a] = c
        op_chain[o] = c
        moves.append(Assignment(app_chain=tuple(app_chain), op_chain=tuple(op_chain)))
    for _ in range(k):
        if rng.random() < n_apps / (n_apps + n_ops):
            i = int(rng.integers(0, n_apps))
            new_c = rng.choice([None] + list(range(n_chains)))
            app_chain = list(assignment.app_chain)
            app_chain[i] = None if new_c is None else int(new_c)
            moves.append(
                Assignment(app_chain=tuple(app_chain), op_chain=assignment.op_chain)
            )
        else:
            i = int(rng.integers(0, n_ops))
            new_c = rng.choice([None] + list(range(n_chains)))
            op_chain = list(assignment.op_chain)
            op_chain[i] = None if new_c is None else int(new_c)
            moves.append(
                Assignment(app_chain=assignment.app_chain, op_chain=tuple(op_chain))
            )
    return moves


def solve_local_search(
    instance: Instance,
    rule: ThroughputRule,
    weights: Weights,
    bounds: Optional[NormalizationBounds] = None,
    seed: int = 0,
    restarts: int = 8,
    iters: int = 2000,
    neighborhood: int = 16,
    warm_start: Optional[Assignment] = None,
) -> SolveResult:
    """Randomized first-improvement local search with restarts.

    Memoizes evaluations on the canonical key, so relabeled duplicates
    cost nothing; ``warm_start`` supports cross-epoch re-optimization.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    n_chains = max(1, min(instance.n_apps, instance.n_ops))

    cache: Dict = {}

    def cached_eval(assignment: Assignment) -> Optional[Evaluation]:
        key = canonical_key(assignment)
        if key not in cache:
            cache[key] = _evaluate_if_feasible(
                instance, assignment, rule, weights, bounds
            )
        return cache[key]

    best_assignment = warm_start or empty_assignment(instance)
    best_eval = cached_eval(best_assignment)
    if best_eval is None:  # infeasible warm start: fall back to empty
        best_assignment = empty_assignment(instance)
        best_eval = cached_eval(best_assignment)

    starts: List[Assignment] = [best_assignment] + [
        _random_feasible_start(instance, rule, rng) for _ in range(restarts)
    ]

    for current in starts:
        current_eval = cached_eval(current)
        if current_eval is None:
            continue
        for _ in range(iters):
            improved = False
            for neighbor in _neighbors(current, n_chains, rng, neighborhood):
                ev = cached_eval(neighbor)
                if ev is not None and ev.objective > current_eval.objective + 1e-12:
                    current, current_eval = neighbor, ev
                    improved = True
                    break
            if not improved:
                break
        if current_eval.objective > best_eval.objective:
            best_assignment, best_eval = current, current_eval

    return SolveResult(
        assignment=best_assignment,
        evaluation=best_eval,
        seconds=time.perf_counter() - start,
        evaluated=len(cache),
        exact=False,
    )
