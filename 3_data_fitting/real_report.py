"""
Tables and density figure for the real-data fits of fit_real.py.

    python real_report.py --model ddm3 --results results_real/ddm3 --data real.csv

Reads <results>/tinn.csv and/or <results>/pyddm.csv (one row per subject) and writes to
<results>/paper/:
  table_<method>.tex   parameters per instruction (mean over subjects) with G2, AIC, BIC
  table_fit.tex/.csv   both methods side by side: G2, AIC, BIC and per-subject comparison
  densities.pdf/.png   data histograms with the fitted densities, one panel per instruction
                       (black and white; correct responses up, errors down)
G2 = 2 * sum of the negative log-likelihoods (independent referee), AIC = G2 + 2 k S,
BIC = G2 + k * sum_s ln n_s  (k parameters per subject, S subjects fitted separately).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2_parameter_estimation"))   # common, make_figures
sys.path.insert(0, os.path.join(HERE, "..", "tinn"))
from common import params_of  # noqa: E402

COND = ["Speed", "Neutral", "Accuracy"]
TITLE = {"ddm3": "standard DDM", "urgency3": "urgency model", "ou3": "Ornstein-Uhlenbeck model",
         "collapse3": "collapsing-bound model"}
TEX = {"v": "Drift rate $v$", "v0": "Drift rate $v_0$", "v1": "Urgency $v_1$", "tau": r"Time constant $\tau$",
       "beta": r"Leak $\beta$", "T_inf": r"Collapse time $T_\infty$", "a": "Bound $a$",
       "t0": "Non-decision time $t_0$"}
STYLE = {"TINN": dict(color="#000000", ls="-"), "PyDDM": dict(color="#4d4d4d", ls="--")}
FILES = {"TINN": "tinn.csv", "PyDDM": "pyddm.csv"}


def load(results):
    return {m: pd.read_csv(os.path.join(results, f)) for m, f in FILES.items()
            if os.path.exists(os.path.join(results, f))}


def criteria(df, k):
    g2 = 2 * df["nll"].sum()
    return g2, g2 + 2 * k * len(df), g2 + k * np.log(df["n"]).sum()


def latex_params(model, method, df):
    """table in the layout of the paper: one row per instruction, shared parameters merged"""
    shared = [p for p in params_of(model) if p not in ("a1", "a2", "a3", "t0")]
    cols = shared + ["a", "t0"]
    m = df[params_of(model)].mean()
    k = len(params_of(model))
    g2, aic, bic = criteria(df, k)
    who = f"mean over {len(df)} participants fitted separately" if len(df) > 1 else "pooled data"
    L = [r"% needs \usepackage{multirow}", r"\begin{table}[!ht]", r"\centering",
         rf"\caption{{Fitted parameter values for the {TITLE[model]} under speed, neutral, and accuracy "
         rf"instructions ({method}, {who}).}}",
         rf"\label{{table:{model}_{method.lower()}}}",
         r"\begin{tabular}{|l|" + "c" * len(cols) + "|}", r"\hline",
         " & " + " & ".join(TEX[c] for c in cols) + r" \\", r"\hline"]
    for i, cname in enumerate(COND):
        cells = []
        for c in cols:
            if c == "a":
                cells.append(f"{m[f'a{i + 1}']:.5f}")
            elif i == 0:
                cells.append(rf"\multirow{{3}}{{*}}{{{m[c]:.5f}}}")
            else:
                cells.append("")
        L.append(f"{cname} & " + " & ".join(cells) + r" \\")
    L.append(r"\hline")
    for name, val in (("$G^2$", g2), ("AIC", aic), ("BIC", bic)):
        L += [f"{name} & " + rf"\multicolumn{{{len(cols)}}}{{c|}}{{{val:.5f}}}" + r" \\", r"\hline"]
    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def fit_table(model, fits):
    k = len(params_of(model))
    rows = []
    for meth, df in fits.items():
        g2, aic, bic = criteria(df, k)
        rows.append({"method": meth, "subjects": len(df), "G2": g2, "AIC": aic, "BIC": bic,
                     "median s / subject": df["time_s"].median()})
    out = pd.DataFrame(rows)
    if len(fits) == 2:
        j = fits["TINN"].merge(fits["PyDDM"], on="subj", suffixes=("_t", "_p"))
        d = (j["nll_t"] - j["nll_p"]) / j["n_t"]
        out["subjects with higher likelihood than the other method"] = [
            int((d < -1e-3).sum()), int((d > 1e-3).sum())]
    return out


def densities_figure(model, fits, data, out, t_max=None, dt=1e-3):
    from make_figures import densities
    t_max = t_max or float(np.ceil(data["rt"].max() * 10) / 10)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    bins = np.linspace(0, t_max, 50)
    for k, ax in enumerate(axes):
        dk = data[data["cond"] == k]
        rt, ch = dk["rt"].to_numpy(), dk["choice"].to_numpy()
        w = np.full(len(rt), 1.0 / (len(rt) * (bins[1] - bins[0])))
        ax.hist(rt[ch == 1], bins=bins, weights=w[ch == 1], color="#d9d9d9", label="data")
        ax.hist(rt[ch == 0], bins=bins, weights=-w[ch == 0], color="#d9d9d9")
        for meth, df in fits.items():
            up = lo = tg = None
            ntot = 0
            for _, r in df.iterrows():
                n = r[f"n{k}"]
                g, u, l = densities(model, {p: float(r[p]) for p in params_of(model)}, t_max, k, dt)
                if tg is None:
                    tg, up, lo = g, n * u, n * l
                else:
                    up, lo = up + n * np.interp(tg, g, u), lo + n * np.interp(tg, g, l)
                ntot += n
            ax.plot(tg, up / ntot, lw=1.4, label=meth, **STYLE[meth])
            ax.plot(tg, -lo / ntot, lw=1.4, **STYLE[meth])
        ax.axhline(0, color="#6b6b6b", lw=0.6)
        ax.set_title(COND[k])
        ax.set_xlabel("RT (s)")
        ax.set_xlim(0, t_max)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("density (correct +, error −)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"densities.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--no-densities", action="store_true")
    args = ap.parse_args()
    out = os.path.join(args.results, "paper")
    os.makedirs(out, exist_ok=True)
    fits = load(args.results)
    if not fits:
        raise SystemExit("no tinn.csv or pyddm.csv in " + args.results)
    if len(fits) == 2:                       # compare on the subjects both methods have fitted
        common = set(fits["TINN"]["subj"].astype(str)) & set(fits["PyDDM"]["subj"].astype(str))
        fits = {m: df[df["subj"].astype(str).isin(common)] for m, df in fits.items()}
    data = pd.read_csv(args.data)
    data = data[data["subj"].astype(str).isin(set(next(iter(fits.values()))["subj"].astype(str)))]

    for meth, df in fits.items():
        tex = latex_params(args.model, meth, df)
        open(os.path.join(out, f"table_{meth.lower()}.tex"), "w").write(tex)
        print(tex)
        df.to_csv(os.path.join(out, f"per_subject_{meth.lower()}.csv"), index=False)
    ft = fit_table(args.model, fits)
    ft.to_csv(os.path.join(out, "table_fit.csv"), index=False)
    print(ft.round(3).to_string(index=False))
    if not args.no_densities:
        densities_figure(args.model, fits, data, out)
    print("saved in", out)


if __name__ == "__main__":
    main()
