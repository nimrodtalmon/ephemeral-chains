# Ephemeral Chains

Code accompanying the paper *Ephemeral Chains for Adaptive Multichain
Blockchains* (under submission). It models chain formation in a multichain
blockchain as a governance-weighted multiagent resource-allocation problem
and provides solvers and the paper's experiments.

## Layout

- `ephemeral_chains/instance.py` — problem instances and random generators
  (heavy-tailed, correlated; plus a uniform baseline). All randomness is
  seeded.
- `ephemeral_chains/model.py` — the core model: throughput-aggregation
  rules (bottleneck / additive / overhead-adjusted), feasibility,
  utilities, normalization, the governance-weighted objective, and the
  analytic inner-price optimum of the bilevel decomposition.
- `ephemeral_chains/solver.py` — exact enumeration over canonical
  assignments (small instances) and memoized local search with restarts
  (the practical solver).
- `ephemeral_chains/ng_solver.py` — optional Nevergrad-based solver
  (requires `pip install .[nevergrad]`).
- `experiments/` — the paper's experiments; each script writes CSVs to
  `results/` and is reproducible from its `--seed` argument.
- `tests/` — invariants, the paper's worked example, and the pooling
  proposition checked against exact enumeration.

## Usage

```bash
pip install -e .[dev]
pytest
python experiments/scalability.py --help
```
