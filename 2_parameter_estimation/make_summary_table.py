"""
One comparison table for the paper, over every model that has results.

    python make_summary_table.py            # reads results/<model>/paper/table{1,2,3}_*.csv

Writes results/summary_table.tex (LaTeX, booktabs + multirow) and results/summary_table.csv.
Panel A: parameter recovery (bias, RMSE, r) per model, parameter and method.
Panel B: per model and method, the share of datasets whose estimate has a likelihood at least
         as high as the true parameters (independent referee) and the median time per dataset.
Run make_figures.py for each model first.
"""
import argparse
import os

import pandas as pd

MODEL_NAMES = {"ddm": "DDM", "urgency": "Urgency", "ou": "OU", "dmc": "DMC", "collapse": "Collapsing bounds",
               "urgency3": "Urgency, 3 cond.", "ou3": "OU, 3 cond.",
               "ddm3": "DDM, 3 cond.", "collapse3": "Collapsing bounds, 3 cond."}
PARAM_TEX = {"v": "$v$", "a": "$a$", "t0": "$t_0$", "v0": "$v_0$", "v1": "$v_1$", "tau": r"$\tau$",
             "beta": r"$\beta$", "mu0": r"$\mu_0$", "A": "$A$", "T_inf": r"$T_\infty$",
             "a1": "$a_1$", "a2": "$a_2$", "a3": "$a_3$"}
METHODS = ["TINN", "PyDDM", "BayesFlow"]


def fmt(x):
    s = f"{x:.3f}"
    return "0.000" if s == "-0.000" else s.replace("-", "$-$")


def load(results):
    out = {}
    for m in MODEL_NAMES:
        p = os.path.join(results, m, "paper")
        if os.path.exists(os.path.join(p, "table1_recovery.csv")):
            out[m] = {i: pd.read_csv(os.path.join(p, f)) for i, f in
                      ((1, "table1_recovery.csv"), (2, "table2_cost.csv"), (3, "table3_likelihood.csv"))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    tabs = load(args.results)
    methods = [k for k in METHODS if any(f"{k} RMSE" in t[1] for t in tabs.values())]
    print("models:", list(tabs), " methods:", methods)

    rows = []
    for m, t in tabs.items():
        for _, r in t[1].iterrows():
            row = {"model": MODEL_NAMES[m], "parameter": r["parameter"]}
            for k in methods:
                for s in ("bias", "RMSE", "r"):
                    row[f"{k} {s}"] = r.get(f"{k} {s}")
            rows.append(row)
    rec = pd.DataFrame(rows)
    rows = []
    for m, t in tabs.items():
        for k in methods:
            c, l = t[2].set_index("method"), t[3].set_index("method")
            if k in c.index:
                rows.append({"model": MODEL_NAMES[m], "method": k,
                             "% datasets NLL(est) <= NLL(true)": l.loc[k, "% datasets dNLL <= 0.001"],
                             "median s / dataset": c.loc[k, "median s / dataset"],
                             "one-off training (s)": c.loc[k, "one-off training (s)"]})
    cost = pd.DataFrame(rows)
    rec.to_csv(os.path.join(args.results, "summary_table.csv"), index=False)
    cost.to_csv(os.path.join(args.results, "summary_table_panelB.csv"), index=False)

    # ---------------- LaTeX
    n = 3 * len(methods)
    L = [r"% needs \usepackage{booktabs, multirow}", r"\begin{table}[t]", r"\centering\small",
         r"\caption{Comparison of TINN with " + " and ".join(methods[1:]) +
         r" on the same simulated datasets (50 datasets $\times$ 1000 trials per model).}",
         r"\label{tab:comparison}",
         r"\begin{tabular}{ll" + "rrr" * len(methods) + "}", r"\toprule",
         r"\multicolumn{" + str(n + 2) + r"}{l}{\textit{A. Parameter recovery}}\\",
         " & & " + " & ".join(r"\multicolumn{3}{c}{" + k + "}" for k in methods) + r"\\",
         "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(methods))),
         "Model & Param. & " + " & ".join(["Bias & RMSE & $r$"] * len(methods)) + r"\\", r"\midrule"]
    for model, g in rec.groupby("model", sort=False):
        for j, (_, r) in enumerate(g.iterrows()):
            first = rf"\multirow{{{len(g)}}}{{*}}{{{model}}}" if j == 0 else ""
            vals = " & ".join(" & ".join(fmt(r[f"{k} {s}"]) for s in ("bias", "RMSE", "r")) for k in methods)
            L.append(f"{first} & {PARAM_TEX.get(r['parameter'], r['parameter'])} & {vals}" + r"\\")
        L.append(r"\addlinespace")
    L += [r"\midrule",
          r"\multicolumn{" + str(n + 2) + r"}{l}{\textit{B. Likelihood and cost}}\\",
          " & & " + " & ".join(r"\multicolumn{3}{c}{" + k + "}" for k in methods) + r"\\",
          "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(methods))),
          "Model & & " + " & ".join([r"\multicolumn{2}{r}{\% $\mathrm{NLL}_{\hat\theta}\le\mathrm{NLL}_{\theta}$} & s/dataset"]
                                    * len(methods)) + r"\\", r"\midrule"]
    for model, g in cost.groupby("model", sort=False):
        g = g.set_index("method")
        cells = []
        for k in methods:
            if k in g.index:
                t = g.loc[k, "median s / dataset"]
                tt = f"{t:.0f}" if t >= 10 else f"{t:.1f}"
                if g.loc[k, "one-off training (s)"] > 0:
                    tt += rf"$^\dagger$"
                cells.append(rf"\multicolumn{{2}}{{r}}{{{g.loc[k, '% datasets NLL(est) <= NLL(true)']:.0f}}} & {tt}")
            else:
                cells.append(r"\multicolumn{2}{r}{--} & --")
        L.append(f"{model} & & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip\footnotesize",
          r"$r$: correlation between true and estimated values. "
          r"\% $\mathrm{NLL}_{\hat\theta}\le\mathrm{NLL}_{\theta}$: share of datasets in which the estimate "
          r"explains the data at least as well as the generating parameters (within $10^{-3}$ per trial), "
          r"judged by a referee independent of all methods (closed-form density for the DDM, an independent "
          r"Crank--Nicolson solver otherwise). s/dataset: median compute time per dataset on one CPU core."
          + (r" $^\dagger$plus a one-off training cost." if "BayesFlow" in methods else ""),
          r"\end{table}"]
    path = os.path.join(args.results, "summary_table.tex")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nsaved", path)


if __name__ == "__main__":
    main()
