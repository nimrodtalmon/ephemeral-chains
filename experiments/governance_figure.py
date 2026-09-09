
import sys; sys.path.insert(0, ".")
from dataclasses import replace
import numpy as np
from ephemeral_chains.instance import GeneratorConfig, generate
from ephemeral_chains.model import THROUGHPUT_RULES, NormalizationBounds, Weights
from ephemeral_chains.solver import solve_local_search
import csv

BASE = GeneratorConfig(n_apps=20, n_ops=10, price_sigma=0.4)
rule = THROUGHPUT_RULES["min"]
rows = []
for cap_scale in (0.5, 4.0):
    cfg = replace(BASE, cap_mu=BASE.cap_mu + float(np.log(cap_scale)))
    for i in range(24):
        inst = generate(cfg, seed=i)
        bounds = NormalizationBounds.ideal(inst, rule, lambda i, r, w, b: solve_local_search(i, r, w, bounds=b))
        n = 5
        for a in range(n+1):
            for o in range(n+1-a):
                w = Weights(app=a/n, op=o/n, sys=(n-a-o)/n)
                res = solve_local_search(inst, rule, w, bounds=bounds, seed=i, restarts=16)
                ev = res.evaluation
                napp = sum(ev.app_utils[j]/inst.apps[j].gas for j in range(inst.n_apps))/inst.n_apps
                nsys = ev.sys_util / bounds.q_sys
                nop = sum(ev.op_utils)/len(ev.op_utils)
                rows.append((cap_scale, i, w.app, w.op, w.sys, napp, nsys, nop))
with open("results/governance_figure_data.csv","w",newline="") as f:
    wcsv = csv.writer(f); wcsv.writerow(["cap_scale","seed","w_app","w_op","w_sys","napp","nsys","nop"]); wcsv.writerows(rows)
print("rows:", len(rows))
