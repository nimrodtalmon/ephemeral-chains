"""Experiment: what blunts operator price manipulation.

(1) lambda-dependence: maximal unilateral price-misreport gain of sampled
operators under four governance settings (equal, app-, op-, sys-corner).
(2) joint deviation: all operators inflate asks by a common factor; mean
operator utility vs. the truthful profile.

Findings (paper, Section 8.5): under sys-oriented weights the median
relative gain is zero; joint inflation self-defeats at factors >= 2.
"""
# Reproducibility note: this file records the protocol run for the paper;
# see results/defense.csv for the reported data (base seed 0).
