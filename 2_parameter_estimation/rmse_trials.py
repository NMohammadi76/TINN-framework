"""
DDM parameter recovery against the number of trials, for TINN, PyDDM and BayesFlow.

    python rmse_trials.py make                      # test datasets, one set per number of trials
    python rmse_trials.py fit --method pyddm --workers 4
    python rmse_trials.py fit --method tinn  --workers 4
    python rmse_trials.py fit --method bayesflow    # retrains for every number of trials (GPU helps)
    python rmse_trials.py plot                      # results/ddm_ntrials/rmse_trials.pdf

Every number of trials uses the same true parameters (same seed), so only the amount of data
changes.  The figure shows the RMSE of each parameter (--nrmse: RMSE / width of its range).
Resumable: datasets already fitted are skipped; BayesFlow is skipped where its .csv exists.
"""
import argparse
import os
import subprocess
import sys
import threading

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import make_test_datasets, ranges_of  # noqa: E402

MODEL = "ddm"
ROOT = os.path.join(HERE, "results", "ddm_ntrials")
TRIALS = [50, 100, 500, 1000, 2000]
METHODS = ["tinn", "pyddm", "bayesflow"]
LABEL = {"tinn": "TINN", "pyddm": "PyDDM", "bayesflow": "BayesFlow"}
STYLE = {"tinn": dict(ls="-", marker="o"), "pyddm": dict(ls="--", marker="s"),
         "bayesflow": dict(ls="-.", marker="^")}
TITLE = {"v": "Drift rate $v$", "a": "Boundary $a$", "t0": "Non-decision time $t_0$"}


def folder(n):
    return os.path.join(ROOT, f"N{n}")


def _relay(proc, log, n):
    """copies a worker's output to its log file and shows the per-dataset lines"""
    with open(log, "a") as f:
        for line in proc.stdout:
            f.write(line)
            if line.startswith(("[tinn] dataset", "[pyddm] dataset")):
                print(f"N = {n}  " + line, end="", flush=True)


def make(args):
    for n in args.trials:
        os.makedirs(folder(n), exist_ok=True)
        path = os.path.join(folder(n), "test_datasets.npz")
        if not os.path.exists(path):
            make_test_datasets(MODEL, args.n_datasets, n, seed=args.seed, path=path)
        print(f"N = {n}: {path}", flush=True)


def fit(args):
    script = {"tinn": "fit_pinn.py", "pyddm": "fit_pyddm.py", "bayesflow": "fit_bayesflow.py"}[args.method]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1") if args.method == "pyddm" else None
    for n in args.trials:
        data = os.path.join(folder(n), "test_datasets.npz")
        out = os.path.join(folder(n), f"{args.method}.csv")
        log = os.path.join(folder(n), f"log_{args.method}")
        print(f"[{args.method}] N = {n}", flush=True)
        if args.method == "bayesflow":
            if not os.path.exists(out):
                cmd = [sys.executable, script, "--data", data, "--out", out,
                       "--n-sim", str(args.n_sim), "--epochs", str(args.epochs)]
                with open(log + ".txt", "w") as f:
                    subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, check=True)
            continue
        n_data = len(np.load(data)["rt"])
        step = -(-n_data // args.workers)
        extra = (["--warm", str(args.warm), "--iters", str(args.iters), "--device", args.device]
                 if args.method == "tinn" else [])
        procs = []
        for w in range(args.workers):
            first, last = w * step, min((w + 1) * step, n_data)
            if first >= last:
                continue
            cmd = [sys.executable, "-u", script, "--data", data, "--out", f"{out[:-4]}_part{w}.csv",
                   "--first", str(first), "--last", str(last)] + extra
            procs.append(subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True))
        threads = [threading.Thread(target=_relay, args=(p, f"{log}_{w}.txt", n)) for w, p in enumerate(procs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if any(p.wait() for p in procs):
            sys.exit(f"[{args.method}] N = {n}: a worker failed, see {log}_*.txt")
        parts = [pd.read_csv(f"{out[:-4]}_part{w}.csv") for w in range(args.workers)
                 if os.path.exists(f"{out[:-4]}_part{w}.csv")]
        pd.concat(parts).sort_values("dataset").to_csv(out, index=False)


def nrmse_table():
    rows = []
    for n in sorted(int(d[1:]) for d in os.listdir(ROOT) if d.startswith("N")):
        for m in METHODS:
            path = os.path.join(folder(n), f"{m}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            for k, (lo, hi) in ranges_of(MODEL).items():
                err = df[f"{k}_est"] - df[f"{k}_true"]
                rows.append({"n_trials": n, "method": LABEL[m], "parameter": k, "n_datasets": len(df),
                             "bias": err.mean(), "rmse": np.sqrt((err ** 2).mean()),
                             "nrmse": np.sqrt((err ** 2).mean()) / (hi - lo)})
    return pd.DataFrame(rows)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})

    tab = nrmse_table()
    if tab.empty:
        sys.exit(f"no results under {ROOT}")
    tab.to_csv(os.path.join(ROOT, "rmse_trials.csv"), index=False)
    y = "nrmse" if args.nrmse else "rmse"
    params = list(ranges_of(MODEL))
    fig, axes = plt.subplots(1, len(params), figsize=(4.2 * len(params), 3.6), sharey=args.sharey)
    for ax, k in zip(axes, params):
        for m in METHODS:
            s = tab[(tab.method == LABEL[m]) & (tab.parameter == k)].sort_values("n_trials")
            if len(s):
                ax.plot(s.n_trials, s[y], color="black", lw=1.3, ms=5, mfc="white", label=LABEL[m], **STYLE[m])
        ax.set_title(TITLE.get(k, k))
        ax.set_xlabel("Number of trials")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(y.upper())
    axes[0].legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(ROOT, f"{y}_trials.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out[:-4] + ".png", dpi=150, bbox_inches="tight")
    print(tab.pivot_table(index=["parameter", "n_trials"], columns="method", values=y).round(4))
    print(f"saved {out}")


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["make", "fit", "plot"])
    ap.add_argument("--trials", type=int, nargs="+", default=TRIALS)
    ap.add_argument("--root", default=ROOT, help="results folder")
    ap.add_argument("--n-datasets", type=int, default=20, help="datasets per number of trials")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--workers", type=int, default=4, help="parallel processes (TINN / PyDDM)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="TINN")
    ap.add_argument("--warm", type=int, default=2000, help="TINN")
    ap.add_argument("--iters", type=int, default=6000, help="TINN")
    ap.add_argument("--n-sim", type=int, default=20000, help="BayesFlow training datasets")
    ap.add_argument("--epochs", type=int, default=50, help="BayesFlow")
    ap.add_argument("--nrmse", action="store_true", help="plot RMSE / range width instead of RMSE")
    ap.add_argument("--sharey", action="store_true", help="same y axis in all panels")
    args = ap.parse_args()
    ROOT = os.path.abspath(args.root)
    if args.step == "fit" and not args.method:
        ap.error("fit needs --method")
    {"make": make, "fit": fit, "plot": plot}[args.step](args)


if __name__ == "__main__":
    main()
