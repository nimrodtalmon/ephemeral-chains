"""Problem instances: agents, declarations, and random generators.

Generators produce heavy-tailed, correlated instances (log-normal gas,
Pareto stakes, prices correlated with size), matching the experimental
setup described in the paper. All randomness flows through a
numpy Generator passed explicitly, so every instance is reproducible
from (generator parameters, seed).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class App:
    """An application's declaration."""

    gas: float  # gas demand, > 0
    stake: float  # required stake, > 0
    gasprice: float  # maximum acceptable gas price, > 0

    def __post_init__(self) -> None:
        if min(self.gas, self.stake, self.gasprice) <= 0:
            raise ValueError(f"App declarations must be positive: {self}")


@dataclass(frozen=True)
class Op:
    """An operator's declaration."""

    gas: float  # gas capacity, > 0
    stake: float  # available stake, > 0
    gasprice: float  # minimum acceptable gas price, > 0

    def __post_init__(self) -> None:
        if min(self.gas, self.stake, self.gasprice) <= 0:
            raise ValueError(f"Op declarations must be positive: {self}")


@dataclass(frozen=True)
class Instance:
    """A single-epoch problem instance."""

    apps: List[App]
    ops: List[Op]

    @property
    def n_apps(self) -> int:
        return len(self.apps)

    @property
    def n_ops(self) -> int:
        return len(self.ops)

    def with_app(self, index: int, app: App) -> "Instance":
        """Copy of the instance with one application's declaration replaced.

        Used by the misreporting experiments.
        """
        apps = list(self.apps)
        apps[index] = app
        return replace(self, apps=apps)

    def with_op(self, index: int, op: Op) -> "Instance":
        """Copy of the instance with one operator's declaration replaced."""
        ops = list(self.ops)
        ops[index] = op
        return replace(self, ops=ops)


@dataclass(frozen=True)
class GeneratorConfig:
    """Parameters of the random-instance generator.

    Gas quantities are log-normal (heavy-tailed); stakes are Pareto;
    application price caps are positively correlated with declared gas
    through ``price_gas_corr`` (large applications tolerate higher
    prices), and operator asks sit below application caps on average so
    instances are typically, but not always, price-feasible.
    """

    n_apps: int = 20
    n_ops: int = 10
    gas_mu: float = 3.0  # log-normal location for app gas
    gas_sigma: float = 1.0  # log-normal scale (tail heaviness)
    cap_mu: float = 4.0  # log-normal location for op capacity
    cap_sigma: float = 0.8
    stake_pareto_alpha: float = 1.5  # Pareto tail index for stakes
    stake_scale: float = 10.0
    app_price_base: float = 5.0
    op_price_base: float = 3.0
    price_sigma: float = 0.4  # log-scale noise on prices
    price_gas_corr: float = 0.3  # strength of price--size correlation


def generate(config: GeneratorConfig, seed: int) -> Instance:
    """Draw a random instance from ``config`` with the given seed."""
    rng = np.random.default_rng(seed)

    app_gas = rng.lognormal(config.gas_mu, config.gas_sigma, config.n_apps)
    # Price caps correlated with (standardized) log-gas.
    z = (np.log(app_gas) - config.gas_mu) / config.gas_sigma
    app_price = config.app_price_base * np.exp(
        config.price_gas_corr * z
        + config.price_sigma * rng.standard_normal(config.n_apps)
    )
    app_stake = config.stake_scale * (
        1.0 + rng.pareto(config.stake_pareto_alpha, config.n_apps)
    )

    op_gas = rng.lognormal(config.cap_mu, config.cap_sigma, config.n_ops)
    op_price = config.op_price_base * np.exp(
        config.price_sigma * rng.standard_normal(config.n_ops)
    )
    op_stake = config.stake_scale * (
        1.0 + rng.pareto(config.stake_pareto_alpha, config.n_ops)
    )

    apps = [
        App(gas=float(g), stake=float(s), gasprice=float(p))
        for g, s, p in zip(app_gas, app_stake, app_price)
    ]
    ops = [
        Op(gas=float(g), stake=float(s), gasprice=float(p))
        for g, s, p in zip(op_gas, op_stake, op_price)
    ]
    return Instance(apps=apps, ops=ops)


def generate_uniform(
    n_apps: int, n_ops: int, seed: int, high: float = 100.0
) -> Instance:
    """Baseline uniform-random generator (the weak validation regime the
    paper improves on); kept for comparison experiments."""
    rng = np.random.default_rng(seed)

    def u(n: int) -> np.ndarray:
        return rng.uniform(1.0, high, n)

    apps = [
        App(gas=float(g), stake=float(s), gasprice=float(p))
        for g, s, p in zip(u(n_apps), u(n_apps), u(n_apps))
    ]
    ops = [
        Op(gas=float(g), stake=float(s), gasprice=float(p))
        for g, s, p in zip(u(n_ops), u(n_ops), u(n_ops))
    ]
    return Instance(apps=apps, ops=ops)
