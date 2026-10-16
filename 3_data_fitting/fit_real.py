"""
TINN or PyDDM fits of real data with three conditions that share every parameter except the
bound (e.g. speed / neutral / accuracy instructions): models ddm3, urgency3, ou3, collapse3.

    python fit_real.py --data real.csv --model ddm3 --method tinn  --out tinn.csv
    python fit_real.py --data real.csv --model ddm3 --method pyddm --out pyddm.csv --first 0 --last 5

real.csv has one row per trial: subj, rt (s), choice (1 = upper bound = correct, 0 = error),
cond (0, 1, 2).  Each subject is fitted separately (use subj = "all" for one pooled fit).
Resumable: subjects already in --out are skipped.

Every row also holds the NLL of the estimate computed by the same independent referee as in the
simulation study (closed form for the DDM, Crank-Nicolson otherwise, in coordinates scaled by the
bound for collapsing bounds), so the two methods are compared on the same likelihood.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "2_parameter_estimation"))   # common, fit_pyddm, fit_pinn, make_figures
sys.path.insert(0, os.path.join(HERE, "..", "tinn"))                     # TINN core
from common import params_of, base_of  # noqa: E402

# search box for real data, used by both PyDDM and TINN: wider than the simulation ranges (real
# subjects can be outside them).  beta may be negative (self-excitation).
REAL_BOX = {"v": (0.0, 8.0), "v0": (-2.0, 6.0), "v1": (0.0, 10.0), "tau": (0.01, 3.0), "beta": (-6.0, 6.0),
            "a1": (0.1, 3.0), "a2": (0.1, 3.0), "a3": (0.1, 3.0), "T_inf": (0.1, 5.0)}
# urgency: v1 and tau trade off along a ridge, and on the Forstmann data the best fits sit beyond
# v1 = 10 / tau = 3 (e.g. as1t: v1 12.5, tau 1.4), so the box is wider there
MODEL_BOX = {"urgency3": {"v0": (-10.0, 6.0), "v1": (0.0, 30.0), "tau": (0.01, 10.0)}}


def box_of(model):
    return {**REAL_BOX, **MODEL_BOX.get(model, {})}


def fit_pyddm(model, rt, ch, cond, dx, dt):
    import pyddm
    from fit_pyddm import build, sample_of, t_dur_of, RENAME
    box = {k: box_of(model)[k] for k in params_of(model) if k != "t0"}
    box["t0"] = (0.0, float(rt.min()) - 1e-3)
    if "T_inf" in box:      # bounds must not meet before the slowest response (as in TINN's T_inf transform)
        box["T_inf"] = (float(rt.max()) / 2 + 0.01, max(REAL_BOX["T_inf"][1], float(rt.max())))
    m = build(model, box, t_dur_of(rt, dt), dx, dt)
    m.fit(sample_of(model, rt, ch, cond), lossfunction=pyddm.LossRobustLikelihood, verbose=False)
    names = [RENAME.get(n, n) for n in m.get_model_parameter_names()]
    est = dict(zip(names, map(float, m.get_model_parameters())))
    return {k: est[k] for k in params_of(model)}


# extra starting points for TINN (best by TINN's own likelihood is kept).  On the Forstmann data
# the OU likelihood has two basins: beta near -2.5 with small bounds, and a better one with a
# strongly negative beta and wider bounds (PyDDM: beta at or near the -6 edge).  Gradient descent
# from the default start (beta = 0.5, a = 0.5) ends in the first one.
STARTS = {"ou3": [{}, {"beta": -5.0, "a1": 0.8, "a2": 0.8, "a3": 0.8}]}


def fit_tinn(model, rt, ch, cond, warm, iters, init=None, save=None, tol=None, normalize=False, box=True,
             starts=None):
    """init: state saved by an earlier fit (e.g. the pooled fit) to start network and parameters
    from; save: where to store this fit's state; starts: list of {param: initial value} overrides,
    one fit per entry, the one with the highest TINN likelihood is returned."""
    import torch
    from fit import Estimator
    from fit_pinn import tinn_model
    best = None
    for over in (starts or STARTS.get(model, [{}])):
        torch.manual_seed(0)
        m = tinn_model(model)
        m.param_specs = dict(m.param_specs)
        for k, v in over.items():
            m.param_specs[k] = (m.param_specs[k][0], m.param_specs[k][1], v, *m.param_specs[k][3:])
        if box:  # same search box as PyDDM: keeps TINN out of extreme regions where the network is inaccurate
            for k, spec in list(m.param_specs.items()):
                if k in box_of(model) and spec[0] in ("real", "pos"):
                    m.param_specs[k] = ("box", spec[1], spec[2], *box_of(model)[k])
        est = Estimator(m, torch.tensor(rt, dtype=torch.float32),
                        torch.tensor(ch, dtype=torch.float32), torch.tensor(cond, dtype=torch.long),
                        normalize=normalize)
        if init:
            est.load_state(torch.load(init, map_location=torch.get_default_device()))
        fit = est.run(n_warm=warm, n_joint=iters, verbose=False, tol=tol)
        with torch.no_grad():
            p, t0 = est.params()
            loss = float(est.nll(p, t0)[0])
        if len(starts or STARTS.get(model, [{}])) > 1:
            print(f"  start {over or 'default'}: TINN NLL {loss * len(rt):.1f}", flush=True)
        if best is None or loss < best[0]:
            best = (loss, {k: float(fit[k]) for k in params_of(model)}, est)
    if save:
        torch.save(best[2].state(), save)
    return best[1], best[2].n_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, choices=["ddm3", "urgency3", "ou3", "collapse3"])
    ap.add_argument("--method", required=True, choices=["tinn", "pyddm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--warm", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--init", default=None, help="TINN: start from this saved state (warm start)")
    ap.add_argument("--save", default=None, help="TINN: save the fitted state here (one subject)")
    ap.add_argument("--tol", type=float, default=None,
                    help="TINN: stop once no parameter moved more than tol in 500 iterations")
    ap.add_argument("--truncated", action="store_true",
                    help="TINN: likelihood conditional on RT <= max RT (old behaviour); default: ordinary likelihood")
    ap.add_argument("--no-box", action="store_true",
                    help="TINN: unconstrained parameters (default: the same search box as PyDDM, REAL_BOX)")
    ap.add_argument("--dx", type=float, default=None, help="PyDDM grid; default 0.001 (ddm3) / 0.005")
    ap.add_argument("--dt", type=float, default=None)
    args = ap.parse_args()

    if args.method == "tinn":
        import torch
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        torch.set_default_device(dev)
        print(f"[tinn] device: {dev}", flush=True)
    else:
        import pyddm
        pyddm.set_N_cpus(1)
    dx = args.dx or (0.001 if base_of(args.model) == "ddm" else 0.005)
    dt = args.dt or dx
    from make_figures import referee_nll

    df = pd.read_csv(args.data)
    subjects = list(dict.fromkeys(df["subj"].astype(str)))
    last = args.last if args.last is not None else len(subjects)
    done = set(pd.read_csv(args.out)["subj"].astype(str)) if os.path.exists(args.out) else set()
    for s in subjects[args.first:last]:
        if s in done:
            continue
        d = df[df["subj"].astype(str) == s]
        rt, ch, cond = d["rt"].to_numpy(float), d["choice"].to_numpy(float), d["cond"].to_numpy(int)
        t = time.time()
        if args.method == "tinn":
            est, n_it = fit_tinn(args.model, rt, ch, cond, args.warm, args.iters, args.init, args.save, args.tol,
                                   normalize=args.truncated, box=not args.no_box)
        else:
            est, n_it = fit_pyddm(args.model, rt, ch, cond, dx, dt), None
        el = time.time() - t
        if args.method == "tinn":
            torch.set_default_device("cpu")         # the referee works in numpy on the CPU
        nll = referee_nll(args.model, rt, ch, cond, est) * len(rt)     # total NLL, independent referee
        if args.method == "tinn":
            torch.set_default_device(dev)
        row = {"subj": s, "n": len(rt), **{f"n{k}": int((cond == k).sum()) for k in range(3)},
               "time_s": el, **est, "nll": nll, "iterations": n_it}
        pd.DataFrame([row]).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        print(f"[{args.method}] {s}: " + "  ".join(f"{k} {v:.3f}" for k, v in est.items())
              + f"  NLL {nll:.1f}  ({el:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
