"""The core model: assignments, feasibility, utilities, and the objective.

Mirrors Section 3 of the paper. An :class:`Assignment` maps each agent to a
chain index or ``None``; chains are identified up to relabeling (see
:func:`canonical_key`). The clearing price of each chain is resolved
analytically by the inner level of the bilevel decomposition
(:func:`optimal_prices`): the objective is affine in each chain price, so
the per-chain optimum is at an endpoint of the feasible interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from .instance import Instance

# A throughput-aggregation rule maps the capacities of a chain's operators
# to the chain supply (paper: T).
ThroughputRule = Callable[[Sequence[float]], float]


def t_min(capacities: Sequence[float]) -> float:
    """Bottleneck rule T_min."""
    return min(capacities)


def t_sum(capacities: Sequence[float]) -> float:
    """Additive rule T_sum."""
    return sum(capacities)


def make_t_overhead(f: Callable[[int], float]) -> ThroughputRule:
    """Overhead-adjusted rule T_f for a nonincreasing f with f(1)=1."""

    def rule(capacities: Sequence[float]) -> float:
        return f(len(capacities)) * sum(capacities)

    rule.__name__ = "t_overhead"
    return rule


def t_overhead_harmonic(capacities: Sequence[float]) -> float:
    """The default overhead rule used in the paper: f(k) = 1 / (1 + log2 k)."""
    import math

    k = len(capacities)
    return sum(capacities) / (1.0 + math.log2(k))


THROUGHPUT_RULES: Dict[str, ThroughputRule] = {
    "min": t_min,
    "sum": t_sum,
    "overhead": t_overhead_harmonic,
}


@dataclass(frozen=True)
class Assignment:
    """Chain membership: for each agent, a chain index or None (unassigned)."""

    app_chain: Tuple[Optional[int], ...]
    op_chain: Tuple[Optional[int], ...]

    def chains(self) -> List[int]:
        """Indices of chains that have at least one member."""
        used = {c for c in self.app_chain if c is not None} | {
            c for c in self.op_chain if c is not None
        }
        return sorted(used)

    def apps_of(self, chain: int) -> List[int]:
        return [i for i, c in enumerate(self.app_chain) if c == chain]

    def ops_of(self, chain: int) -> List[int]:
        return [i for i, c in enumerate(self.op_chain) if c == chain]


def canonical_key(assignment: Assignment) -> Tuple[FrozenSet[Tuple[str, int]], ...]:
    """A key identifying the assignment up to chain relabeling.

    Two assignments that differ only in chain indices share a key; the
    solver memoizes on it (paper, Section 4.2).
    """
    groups: Dict[int, set] = {}
    for i, c in enumerate(assignment.app_chain):
        if c is not None:
            groups.setdefault(c, set()).add(("a", i))
    for i, c in enumerate(assignment.op_chain):
        if c is not None:
            groups.setdefault(c, set()).add(("o", i))
    return tuple(sorted((frozenset(g) for g in groups.values()), key=sorted))


@dataclass(frozen=True)
class ChainState:
    """Derived quantities of one open chain (paper, Section 3.2)."""

    apps: Tuple[int, ...]
    ops: Tuple[int, ...]
    stake: float
    demand: float
    supply: float
    gas: float
    price_lo: float  # max operator ask
    price_hi: float  # min application cap


def chain_state(
    instance: Instance, assignment: Assignment, chain: int, rule: ThroughputRule
) -> Optional[ChainState]:
    """Derived quantities of ``chain``; None if the chain is not open."""
    apps = assignment.apps_of(chain)
    ops = assignment.ops_of(chain)
    if not apps or not ops:
        return None
    demand = sum(instance.apps[a].gas for a in apps)
    supply = rule([instance.ops[o].gas for o in ops])
    return ChainState(
        apps=tuple(apps),
        ops=tuple(ops),
        stake=sum(instance.ops[o].stake for o in ops),
        demand=demand,
        supply=supply,
        gas=min(demand, supply),
        price_lo=max(instance.ops[o].gasprice for o in ops),
        price_hi=min(instance.apps[a].gasprice for a in apps),
    )


def is_feasible(
    instance: Instance, assignment: Assignment, rule: ThroughputRule
) -> bool:
    """Stake and price feasibility of every open chain (paper, Section 3.3)."""
    for c in assignment.chains():
        state = chain_state(instance, assignment, c, rule)
        if state is None:  # a chain missing one side is closed, not infeasible
            continue
        if state.price_lo > state.price_hi:
            return False
        if state.stake < max(instance.apps[a].stake for a in state.apps):
            return False
    return True


@dataclass(frozen=True)
class Weights:
    """Governance weights (nonnegative, summing to one)."""

    app: float
    op: float
    sys: float

    def __post_init__(self) -> None:
        total = self.app + self.op + self.sys
        if min(self.app, self.op, self.sys) < 0 or abs(total - 1.0) > 1e-9:
            raise ValueError(f"Weights must be nonnegative and sum to 1: {self}")


@dataclass(frozen=True)
class NormalizationBounds:
    """Upper bounds used for min--max normalization (paper, Section 3.5).

    ``analytic`` gives loose closed-form bounds; ``ideal`` (the default
    used by the solvers) normalizes each group by its ideal point, i.e.,
    the best value that group can attain on the instance, obtained by
    solving with that group's weight set to one. Since a single-group
    objective is invariant to normalization, the ideal point is
    well defined without bounds.
    """

    q_op: float
    q_sys: float

    @staticmethod
    def ideal(instance: Instance, rule: "ThroughputRule", solve) -> "NormalizationBounds":
        """Ideal-point bounds; ``solve(instance, rule, weights, bounds)`` is a solver."""
        loose = NormalizationBounds.analytic(instance)
        ev_op = solve(instance, rule, Weights(app=0.0, op=1.0, sys=0.0), loose).evaluation
        ev_sys = solve(instance, rule, Weights(app=0.0, op=0.0, sys=1.0), loose).evaluation
        q_op = sum(ev_op.op_utils) / instance.n_ops
        q_sys = ev_sys.sys_util
        return NormalizationBounds(q_op=q_op if q_op > 0 else loose.q_op,
                                   q_sys=q_sys if q_sys > 0 else loose.q_sys)

    @staticmethod
    def analytic(instance: Instance, scale: float = 1.0) -> "NormalizationBounds":
        max_price = max(a.gasprice for a in instance.apps)
        total_gas = sum(a.gas for a in instance.apps)
        min_op_stake = min(o.stake for o in instance.ops)
        return NormalizationBounds(
            q_op=scale * max_price * total_gas / min_op_stake,
            q_sys=scale * max_price * total_gas,
        )


@dataclass(frozen=True)
class Evaluation:
    """Objective value and the per-group and per-agent breakdown."""

    objective: float
    app_utils: Tuple[float, ...]  # raw U^app_a
    op_utils: Tuple[float, ...]  # raw U^op_o
    sys_util: float  # raw U^sys
    prices: Dict[int, float]  # clearing price per open chain


def optimal_prices(
    instance: Instance,
    assignment: Assignment,
    rule: ThroughputRule,
    weights: Weights,
) -> Dict[int, float]:
    """Inner level of the bilevel decomposition.

    The objective is affine in each chain's price with coefficient
    proportional to ``lambda_op * Gas_c / Stake_c`` (per-operator yield
    terms) plus ``lambda_sys * Gas_c`` (fee volume), both nonnegative, so
    the optimum is the upper endpoint whenever either weight is positive;
    we evaluate both endpoints for robustness.
    """
    prices: Dict[int, float] = {}
    for c in assignment.chains():
        state = chain_state(instance, assignment, c, rule)
        if state is None:
            continue
        # Affine in p_c -> optimum at an endpoint; pick the better one.
        best_p, best_v = None, float("-inf")
        for p in (state.price_lo, state.price_hi):
            fee = p * state.gas
            value = weights.sys * fee + weights.op * fee / state.stake
            if value > best_v:
                best_p, best_v = p, value
        prices[c] = best_p
    return prices


def evaluate(
    instance: Instance,
    assignment: Assignment,
    rule: ThroughputRule,
    weights: Weights,
    bounds: Optional[NormalizationBounds] = None,
    prices: Optional[Dict[int, float]] = None,
) -> Evaluation:
    """Objective value of a feasible assignment (prices resolved if absent)."""
    if bounds is None:
        bounds = NormalizationBounds.analytic(instance)
    if prices is None:
        prices = optimal_prices(instance, assignment, rule, weights)

    app_utils = [0.0] * instance.n_apps
    op_utils = [0.0] * instance.n_ops
    sys_util = 0.0

    for c in assignment.chains():
        state = chain_state(instance, assignment, c, rule)
        if state is None:
            continue
        price = prices[c]
        fee = price * state.gas
        sys_util += fee
        for a in state.apps:
            app_utils[a] = instance.apps[a].gas / state.demand * state.gas
        for o in state.ops:
            op_utils[o] = fee / state.stake  # yield per unit stake

    norm_app = sum(
        app_utils[a] / instance.apps[a].gas for a in range(instance.n_apps)
    ) / instance.n_apps
    norm_op = sum(u / bounds.q_op for u in op_utils) / instance.n_ops
    norm_sys = sys_util / bounds.q_sys

    objective = weights.app * norm_app + weights.op * norm_op + weights.sys * norm_sys
    return Evaluation(
        objective=objective,
        app_utils=tuple(app_utils),
        op_utils=tuple(op_utils),
        sys_util=sys_util,
        prices=dict(prices),
    )


def empty_assignment(instance: Instance) -> Assignment:
    """The all-unassigned assignment (always feasible, objective 0)."""
    return Assignment(
        app_chain=tuple([None] * instance.n_apps),
        op_chain=tuple([None] * instance.n_ops),
    )
