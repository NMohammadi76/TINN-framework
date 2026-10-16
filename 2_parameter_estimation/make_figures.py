"""
Suggested tables and figures for the TINN / PyDDM / BayesFlow comparison.

    python make_figures.py --results results/ou     # uses whatever of {tinn,pyddm,bayesflow}.csv exist there

Works for all five models; the model is read from <results>/test_datasets.npz.

Tables (results/paper/, .csv and LaTeX booktabs .tex):
  table1_recovery    bias, RMSE, r per parameter and method (+ 95% coverage for BayesFlow)
  table2_cost        time per dataset, one-off training, total time for all datasets
  table3_likelihood  exact NLL of each estimate minus NLL at the true parameters
                     (independent referee: analytic Navarro-Fuss series for the DDM, an
                     independent Crank-Nicolson solver for urgency / OU / DMC, and the same
                     solver in coordinates scaled by the bound for collapsing bounds;
                     ../tinn/fd_check.py), and share of datasets
                     where the estimate is at least as good as the truth
Figures (results/paper/, .png and .pdf):
  fig1_recovery      true vs estimated, one row per method, one column per parameter
  fig2_agreement     TINN estimate vs PyDDM estimate (two maximum-likelihood solvers)
  fig3_identifiability  absolute error vs observed error rate: where estimates are hard
  fig4_cost          cumulative compute time vs number of datasets
  fig5_example_fits  data histograms with the RT densities implied by each method's estimate
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import load_test_datasets, params_of, ranges_of, base_of, bound_values, MODELS  # noqa: E402

# black and white: methods differ by gray level, marker fill and line style
COLORS = {"TINN": "#000000", "PyDDM": "#4d4d4d", "BayesFlow": "#8c8c8c"}
FACES = {"TINN": "#000000", "PyDDM": "#ffffff", "BayesFlow": "#8c8c8c"}
MARKERS = {"TINN": "o", "PyDDM": "s", "BayesFlow": "^"}
LINES = {"TINN": "-", "PyDDM": "--", "BayesFlow": ":"}
LABELS = {"v": "drift $v$", "a": "bound $a$", "t0": "non-decision $t_0$ (s)",
          "v0": "drift $v_0$", "v1": "urgency $v_1$", "tau": r"time constant $\tau$ (s)",
          "beta": r"leak $\beta$", "mu0": r"drift $\mu_0$", "A": "automatic amplitude $A$",
          "T_inf": r"collapse time $T_\infty$ (s)",
          "a1": "bound $a_1$", "a2": "bound $a_2$", "a3": "bound $a_3$"}
COND_NAMES = {"dmc": ["congruent", "incongruent"],
              **{m: ["condition 1", "condition 2", "condition 3"] for m in ("ddm3", "urgency3", "ou3", "collapse3")}}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "legend.frameon": False})


def ddm_logpdf(t, upper, v, a):
    """Exact log density of decision time t at the upper/lower bound (+/- a, start 0, sigma 1):
    Navarro & Fuss (2009) series; small-time form for t / (2a)^2 < 2.5, large-time form otherwise."""
    t = np.asarray(t, float)
    A, w = 2.0 * a, 0.5
    vv = np.where(upper, -v, v)            # upper bound = lower bound of the mirrored process
    u = np.maximum(t, 1e-300) / A ** 2
    k = np.arange(-10, 11)[:, None]
    small = (w + 2 * k) * np.exp(-((w + 2 * k) ** 2 - w ** 2) / (2 * u))
    log_small = -0.5 * np.log(2 * np.pi * u ** 3) - w ** 2 / (2 * u) + np.log(np.maximum(small.sum(0), 1e-300))
    kk = np.arange(1, 11)[:, None]
    large = np.pi * (kk * np.exp(-kk ** 2 * np.pi ** 2 * u / 2) * np.sin(kk * np.pi * w)).sum(0)
    log_large = np.log(np.maximum(large, 1e-300))
    log_g = np.where(u < 2.5, log_small, log_large)
    out = -2 * np.log(A) - vv * A * w - vv ** 2 * t / 2 + log_g
    return np.where(t > 0, out, -np.inf)


def exact_nll(rt, ch, v, a, t0):
    """exact negative log-likelihood per trial (independent of all three fitting methods)."""
    return -float(np.mean(ddm_logpdf(rt - t0, ch == 1, v, a)))


def referee_nll(model, rt, ch, cond, values, dx=1e-3, dt=1e-3):
    """NLL per trial of `values` on one dataset, computed independently of all three fitting
    methods: closed form for the DDM; for the models with fixed bounds our own Crank-Nicolson
    solver (../fd_check.py, 401 grid points, dt = 0.2 ms; it matches the closed form to ~1e-4
    per trial on the DDM), which, unlike PyDDM, keeps solving until the slowest trial; for the
    collapsing bounds the same solver in coordinates scaled by the bound."""
    if model == "ddm":
        return exact_nll(rt, ch, values["v"], values["a"], values["t0"])
    if "base" in MODELS[model]:          # urgency3 / ou3: base model with a = a_k on condition k
        cond = np.asarray(cond).astype(int)
        tot = 0.0
        for k in range(len(MODELS[model]["conditions"])):
            s = cond == k
            tot += s.sum() * referee_nll(base_of(model), rt[s], ch[s], np.zeros(s.sum(), int),
                                         bound_values(model, values, k), dx, dt)
        return tot / len(rt)
    if values["t0"] >= rt.min():
        return np.inf
    if model in ("urgency", "ou", "dmc", "collapse"):
        import sys
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinn"))
        from models import MODELS as TINN_MODELS
        import fd_check
        m = TINN_MODELS[model]()
        p = {k: torch.tensor(float(v)) for k, v in values.items() if k != "t0"}
        p.update({k: torch.tensor(float(v)) for k, v in m.fixed.items()})
        return float(fd_check.nll(m, p, float(values["t0"]), rt, ch, cond))
    from fit_pyddm import nll_at
    return float(nll_at(model, values, rt, ch, cond, dx, dt))


def _referee_job(job):
    model, rt, ch, cond, values, dx = job
    import torch
    torch.set_num_threads(1)
    return referee_nll(model, rt, ch, cond, values, dx, dx)


def referee_all(jobs, d, dx, workers, cache_path):
    """referee NLL for every (method, dataset) in `jobs`, in parallel. Results are kept in
    cache_path, so a rerun (or a new method added later) only computes what is new."""
    import sys
    import time
    from concurrent.futures import ProcessPoolExecutor
    key = lambda name, i, v: (name, int(i), tuple(round(v[k], 10) for k in PARAMS))  # noqa: E731
    cache = {}
    if os.path.exists(cache_path):
        for r in pd.read_csv(cache_path).to_dict("records"):
            cache[key(r["method"], r["dataset"], r)] = r["nll"]
    todo = [(n, i) for (n, i), v in jobs.items() if key(n, i, v) not in cache]
    print(f"likelihood referee: {len(jobs) - len(todo)} cached, {len(todo)} to compute "
          f"with {workers} processes", flush=True)
    if todo:
        args = [(MODEL, d["rt"][i], d["choice"][i], d["cond"][i], jobs[(n, i)], dx) for n, i in todo]
        t, step = time.time(), max(1, len(todo) // 20)
        with ProcessPoolExecutor(workers) as ex:
            for j, ((n, i), v) in enumerate(zip(todo, ex.map(_referee_job, args, chunksize=1))):
                cache[key(n, i, jobs[(n, i)])] = v
                if (j + 1) % step == 0 or j + 1 == len(todo):
                    print(f"  {j + 1}/{len(todo)}  {(time.time() - t) / 60:.1f} min", flush=True)
                    rows = [{"method": k[0], "dataset": k[1], **dict(zip(PARAMS, k[2])), "nll": val}
                            for k, val in cache.items()]
                    pd.DataFrame(rows).to_csv(cache_path, index=False)
        sys.stdout.flush()
    return {(n, i): cache[key(n, i, v)] for (n, i), v in jobs.items()}


def densities(model, values, t_max, cond_index=0, dt=1e-3):
    """RT grid and densities at the upper and lower bound (they integrate to P(choice))."""
    if model == "ddm":
        tg = np.linspace(0, t_max, 600)
        return (tg, np.exp(ddm_logpdf(tg - values["t0"], True, values["v"], values["a"])),
                np.exp(ddm_logpdf(tg - values["t0"], False, values["v"], values["a"])))
    from fit_pyddm import build
    m = build(model, None, float(np.ceil(t_max / dt) * dt), 2 * dt, dt, values=values)
    conds = ({"cong": MODELS[model]["conditions"][cond_index]} if model == "dmc" else
             {"bc": float(cond_index)} if "base" in MODELS[model] else {})
    sol = m.solve(conditions=conds)
    return m.t_domain(), sol.pdf("correct"), sol.pdf("error")


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def to_latex(df, path, caption):
    # written by hand: pandas' to_latex needs a recent jinja2, which is often missing or old
    def cell(x):
        if isinstance(x, float):
            return "--" if x != x else f"{x:.3f}"
        s = str(x)
        for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
            s = s.replace(a, b)
        return s
    L = [r"\begin{table}", r"\centering", r"\caption{" + cell(caption) + "}",
         r"\begin{tabular}{" + "l" * df.shape[1] + "}", r"\toprule",
         " & ".join(cell(c) for c in df.columns) + r" \\", r"\midrule"]
    L += [" & ".join(cell(x) for x in row) + r" \\" for row in df.itertuples(index=False)]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# set in main() from the dataset's model
MODEL, PARAMS, RANGES = "ddm", params_of("ddm"), ranges_of("ddm")


# ---------------------------------------------------------------- tables
def table_recovery(methods):
    rows = []
    for k in PARAMS:
        row = {"parameter": k}
        for name, df in methods.items():
            err = df[f"{k}_est"] - df[f"{k}_true"]
            row[f"{name} bias"] = err.mean()
            row[f"{name} RMSE"] = np.sqrt((err ** 2).mean())
            row[f"{name} r"] = np.corrcoef(df[f"{k}_true"], df[f"{k}_est"])[0, 1]
            if f"{k}_q025" in df:
                row[f"{name} 95% cov."] = ((df[f"{k}_q025"] <= df[f"{k}_true"]) &
                                           (df[f"{k}_true"] <= df[f"{k}_q975"])).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def table_cost(methods):
    rows = []
    for name, df in methods.items():
        one_off = float(df["train_sim_s"].iloc[0] + df["train_fit_s"].iloc[0]) if "train_fit_s" in df else 0.0
        rows.append({"method": name,
                     "median s / dataset": df["time_s"].median(),
                     "IQR s / dataset": df["time_s"].quantile(0.75) - df["time_s"].quantile(0.25),
                     "one-off training (s)": one_off,
                     f"total for {len(df)} datasets (min)": (one_off + df["time_s"].sum()) / 60})
    return pd.DataFrame(rows)


def table_likelihood(dnll):
    """dNLL = inf when the estimate gives zero likelihood to some trial (e.g. t0 >= fastest RT)."""
    rows = []
    for name, v in dnll.items():
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        rows.append({"method": name,
                     "mean dNLL / trial": v[ok].mean() if ok.any() else np.nan,
                     "median dNLL / trial": np.median(v),
                     "% datasets dNLL <= 0.001": 100 * np.mean(v <= 1e-3),
                     "# zero-likelihood estimates": int((~ok).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def fig_recovery(methods, out):
    fig, axes = plt.subplots(len(methods), len(PARAMS), figsize=(2.6 * len(PARAMS), 2.5 * len(methods)),
                             squeeze=False)
    for r, (name, df) in enumerate(methods.items()):
        for c, k in enumerate(PARAMS):
            ax = axes[r, c]
            lo, hi = RANGES[k]
            ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
            if f"{k}_q025" in df:
                ax.vlines(df[f"{k}_true"], df[f"{k}_q025"], df[f"{k}_q975"], color=COLORS[name],
                          alpha=0.35, lw=1, zorder=2)
            ax.scatter(df[f"{k}_true"], df[f"{k}_est"], s=14, marker=MARKERS[name], facecolor=FACES[name],
                       edgecolor=COLORS[name], linewidth=0.7, zorder=3)
            err = df[f"{k}_est"] - df[f"{k}_true"]
            ax.text(0.04, 0.96, f"r = {np.corrcoef(df[f'{k}_true'], df[f'{k}_est'])[0, 1]:.3f}\n"
                                f"RMSE = {np.sqrt((err ** 2).mean()):.3f}",
                    transform=ax.transAxes, va="top", fontsize=8, color=INK)
            if r == 0:
                ax.set_title(LABELS[k], color=INK)
            if c == 0:
                ax.set_ylabel(f"{name}\nestimate", color=INK)
            if r == len(methods) - 1:
                ax.set_xlabel("true value")
    fig.tight_layout()
    save(fig, out, "fig1_recovery")


def fig_agreement(methods, out, a="TINN", b="PyDDM"):
    if a not in methods or b not in methods:
        return
    ids = methods[a].index.intersection(methods[b].index)
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(2.7 * len(PARAMS), 2.6))
    for ax, k in zip(axes, PARAMS):
        x, y = methods[b].loc[ids, f"{k}_est"], methods[a].loc[ids, f"{k}_est"]
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        ax.scatter(x, y, s=14, marker=MARKERS[a], facecolor=FACES[a], edgecolor=COLORS[a], linewidth=0.7)
        ax.text(0.04, 0.96, f"max |diff| = {np.abs(x - y).max():.3f}", transform=ax.transAxes,
                va="top", fontsize=8, color=INK)
        ax.set_title(LABELS[k], color=INK)
        ax.set_xlabel(f"{b} estimate")
    axes[0].set_ylabel(f"{a} estimate")
    fig.tight_layout()
    save(fig, out, "fig2_agreement")


def fig_identifiability(methods, d, out):
    perr = 1 - d["choice"].mean(1)       # share of lower-bound (error) responses
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(2.7 * len(PARAMS), 2.6), sharex=True)
    for ax, k in zip(axes, PARAMS):
        for name, df in methods.items():
            ax.scatter(np.maximum(perr[df.index], 1e-3), np.abs(df[f"{k}_est"] - df[f"{k}_true"]), s=14,
                       marker=MARKERS[name], facecolor=FACES[name], edgecolor=COLORS[name], linewidth=0.7,
                       label=name)
        ax.set_xscale("log")
        ax.set_title(LABELS[k], color=INK)
        ax.set_xlabel("observed error rate")
    axes[0].set_ylabel("|estimate - true|")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, out, "fig3_identifiability")


def fig_cost(methods, out):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    n = np.arange(0, 1001)
    for name, df in methods.items():
        one_off = float(df["train_sim_s"].iloc[0] + df["train_fit_s"].iloc[0]) if "train_fit_s" in df else 0.0
        total = (one_off + n * df["time_s"].median()) / 60
        ax.plot(n, total, color=COLORS[name], ls=LINES[name], lw=2)
        ax.text(n[-1], total[-1], f" {name}", color=INK, va="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("number of datasets fitted")
    ax.set_ylabel("total compute time (min)")
    fig.tight_layout()
    save(fig, out, "fig4_cost")


def fig_example_fits(methods, d, out):
    """three datasets: easiest, median, hardest by the PyDDM (or first method) error in a.
    One row per condition (DMC: congruent / incongruent)."""
    ref = methods.get("PyDDM", next(iter(methods.values())))
    a = "a" if "a_est" in ref else "a1"
    err = (ref[f"{a}_est"] - ref[f"{a}_true"]).abs().sort_values()
    picks = list(dict.fromkeys([err.index[0], err.index[len(err) // 2], err.index[-1]]))
    n_cond = len(MODELS[MODEL]["conditions"])
    fig, axes = plt.subplots(n_cond, len(picks), figsize=(3.5 * len(picks), 2.9 * n_cond), squeeze=False)
    for col, i in enumerate(picks):
        for row in range(n_cond):
            ax = axes[row, col]
            sel = d["cond"][i] == row
            rt, ch = d["rt"][i][sel], d["choice"][i][sel]
            bins = np.linspace(0, np.quantile(rt, 0.995), 40)
            w = np.full(len(rt), 1.0 / (len(rt) * (bins[1] - bins[0])))
            ax.hist(rt[ch == 1], bins=bins, weights=w[ch == 1], color="#e0e0e0", label="data")
            ax.hist(rt[ch == 0], bins=bins, weights=-w[ch == 0], color="#e0e0e0")
            for name, df in methods.items():
                values = {k: float(df.loc[i, f"{k}_est"]) for k in PARAMS}
                tg, up, lo = densities(MODEL, values, bins[-1], row)
                ax.plot(tg, up, color=COLORS[name], ls=LINES[name], lw=1.5, label=name)
                ax.plot(tg, -lo, color=COLORS[name], ls=LINES[name], lw=1.5)
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_xlim(0, bins[-1])
            if row == 0:
                ax.set_title(", ".join(f"{k}={d[k + '_true'][i]:.2f}" for k in PARAMS), color=INK, fontsize=8)
            if row == n_cond - 1:
                ax.set_xlabel("RT (s)")
            if col == 0:
                cname = COND_NAMES.get(MODEL, [""] * n_cond)[row]
                ax.set_ylabel((cname + "\n" if cname else "") + "density (upper +, lower -)")
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig5_example_fits")


def main():
    global MODEL, PARAMS, RANGES
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/ddm", help="folder with test_datasets.npz and the CSVs")
    ap.add_argument("--data", default=None, help="default <results>/test_datasets.npz")
    ap.add_argument("--ref-dx", type=float, default=1e-3, help="PyDDM referee grid (non-DDM models)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel processes for the likelihood referee")
    args = ap.parse_args()
    out = os.path.join(args.results, "paper")
    os.makedirs(out, exist_ok=True)

    d = load_test_datasets(args.data or os.path.join(args.results, "test_datasets.npz"))
    MODEL, PARAMS, RANGES = d["model"], params_of(d["model"]), ranges_of(d["model"])
    methods = {}
    for name, files in (("TINN", ("tinn.csv", "pinn.csv")), ("PyDDM", ("pyddm.csv",)),
                        ("BayesFlow", ("bayesflow.csv",))):
        for fn in files:
            p = os.path.join(args.results, fn)
            if os.path.exists(p):
                methods[name] = pd.read_csv(p).set_index("dataset").sort_index()
                break
    ids = sorted(set.intersection(*(set(df.index) for df in methods.values())))
    methods = {k: v.loc[ids] for k, v in methods.items()}
    print(f"model: {MODEL};  methods: {list(methods)};  datasets fitted by all: {len(ids)}")

    jobs = {("true", i): {k: float(d[f"{k}_true"][i]) for k in PARAMS} for i in ids}
    jobs.update({(name, i): {k: float(df.loc[i, f"{k}_est"]) for k in PARAMS}
                 for name, df in methods.items() for i in ids})
    ref = referee_all(jobs, d, args.ref_dx, args.workers, os.path.join(out, "referee_nll.csv"))
    dnll = {name: [ref[(name, i)] - ref[("true", i)] for i in ids] for name in methods}

    for name, tab, cap in (("table1_recovery", table_recovery(methods), f"Parameter recovery ({MODEL})"),
                           ("table2_cost", table_cost(methods), f"Computational cost ({MODEL})"),
                           ("table3_likelihood", table_likelihood(dnll),
                            f"NLL at the estimate minus NLL at the true parameters ({MODEL})")):
        tab.to_csv(os.path.join(out, f"{name}.csv"), index=False)
        to_latex(tab, os.path.join(out, f"{name}.tex"), cap)
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(f"\n{name}\n" + tab.round(3).to_string(index=False))

    fig_recovery(methods, out)
    fig_agreement(methods, out)
    fig_identifiability(methods, d, out)
    fig_cost(methods, out)
    fig_example_fits(methods, d, out)
    print("\nsaved tables and figures in", out)


if __name__ == "__main__":
    main()
