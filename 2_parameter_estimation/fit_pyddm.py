"""
PyDDM maximum-likelihood estimates on the shared test datasets (all five models).

    python fit_pyddm.py --data results/ddm/test_datasets.npz --out results/ddm/pyddm.csv

PyDDM (Shinn et al. 2020, eLife) solves the same Fokker-Planck equation numerically
(analytically for the plain DDM) and maximises the likelihood with differential
evolution.  Its bound B is the half separation (+/- B) and noise = 1, exactly the
convention of models.py.  The default 2% uniform-contaminant mixture of `gddm` is
switched off, because the test data contain no contaminants.
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import pyddm

from common import load_test_datasets, params_of, ranges_of, base_of, MODELS

# lower limits that keep each parameter physically meaningful in the optimiser's search box
FLOOR = {"a": 0.1, "tau": 0.01, "A": 0.0, "T_inf": 0.1, "t0": 0.0}


def dmc_drift(t, cong, mu0, A, tau):
    """mu0 + c * mu1(t) with shape alpha = 2 (same as models.DMC)."""
    t = max(t, 1e-6)
    return mu0 + cong * A * math.exp(-t / tau) * (math.e / tau) * (1.0 - t / tau)


def search_box(model, rt_min):
    box = {}
    for k, (lo, hi) in ranges_of(model).items():
        w = hi - lo
        floor = FLOOR["a"] if k in ("a1", "a2", "a3") else FLOOR.get(k, -np.inf)
        box[k] = (max(lo - 0.25 * w, floor), hi + 0.25 * w)
    box["t0"] = (box["t0"][0], min(box["t0"][1], rt_min - 1e-3))
    return box


def build(model, box, t_dur, dx, dt, values=None):
    """PyDDM model; parameters are fitted within `box`, or fixed to `values` if given."""
    P = (lambda k: float(values[k])) if values is not None else (lambda k: k)
    common = dict(nondecision=P("t0"), noise=1.0, mixture_coef=0.0, T_dur=t_dur, dx=dx, dt=dt,
                  parameters={} if values is not None else box)
    if "base" in MODELS[model]:
        # urgency3 / ou3: bound a_k in condition k (PyDDM condition "bc" = k), drift shared
        if values is not None:
            av = [float(values[k]) for k in ("a1", "a2", "a3")]
            bound = lambda bc: av[int(round(bc))]                       # noqa: E731
            if base_of(model) == "collapse":
                T = float(values["T_inf"])
                bound = lambda t, bc: max(av[int(round(bc))] * (1 - t / (2 * T)), 1e-3)  # noqa: E731
        else:
            bound = lambda bc, a1, a2, a3: (a1, a2, a3)[int(round(bc))]  # noqa: E731
            if base_of(model) == "collapse":
                bound = lambda t, bc, a1, a2, a3, T_inf: max((a1, a2, a3)[int(round(bc))]   # noqa: E731
                                                            * (1 - t / (2 * T_inf)), 1e-3)
        common.update(conditions=["bc"])
        model = base_of(model)
        if model == "collapse":
            return pyddm.gddm(drift=P("v"), bound=bound, **common)
    else:
        bound = P("a")
    if model == "ddm":
        return pyddm.gddm(drift=P("v"), bound=bound, **common)
    if model == "urgency":
        if values is not None:
            v0, v1, tau = (float(values[k]) for k in ("v0", "v1", "tau"))
            return pyddm.gddm(drift=lambda t: v0 + v1 * t / (t + tau), bound=bound, **common)
        return pyddm.gddm(drift=lambda t, v0, v1, tau: v0 + v1 * t / (t + tau), bound=bound, **common)
    if model == "ou":
        if values is not None:
            v0, beta = float(values["v0"]), float(values["beta"])
            return pyddm.gddm(drift=lambda x: v0 - beta * x, bound=bound, **common)
        return pyddm.gddm(drift=lambda x, v0, beta: v0 - beta * x, bound=bound, **common)
    if model == "dmc":
        if values is not None:
            mu0, A, tau = (float(values[k]) for k in ("mu0", "A", "tau"))
            return pyddm.gddm(drift=lambda t, cong: dmc_drift(t, cong, mu0, A, tau), bound=P("a"),
                              conditions=["cong"], **common)
        return pyddm.gddm(drift=dmc_drift, bound="a", conditions=["cong"], **common)
    if model == "collapse":
        if values is not None:
            a, T = float(values["a"]), float(values["T_inf"])
            return pyddm.gddm(drift=P("v"), bound=lambda t: max(a - a * t / (2 * T), 1e-3), **common)
        return pyddm.gddm(drift="v", bound=lambda t, a, T_inf: max(a - a * t / (2 * T_inf), 1e-3), **common)
    raise ValueError(model)


def sample_of(model, rt, ch, cond):
    if model == "dmc":
        cong = np.asarray(MODELS[model]["conditions"])[np.asarray(cond).astype(int)]
        return pyddm.Sample.from_numpy_array(np.c_[rt, ch, cong], ["cong"])
    if "base" in MODELS[model]:
        return pyddm.Sample.from_numpy_array(np.c_[rt, ch, np.asarray(cond, float)], ["bc"])
    return pyddm.Sample.from_numpy_array(np.c_[rt, ch], [])


def t_dur_of(rt, dt):
    return float(np.ceil((rt.max() + 0.1) / dt) * dt)


# PyDDM's internal parameter names -> ours
RENAME = {"drift": "v", "B": "a", "nondectime": "t0"}


def fit_one(model, rt, ch, cond, dx, dt):
    m = build(model, search_box(model, float(rt.min())), t_dur_of(rt, dt), dx, dt)
    # PyDDM stops its numerical solver once less than 1e-4 of the probability is still undecided,
    # so the density of a very slow trial is exactly 0 and the plain likelihood is infinite for
    # every parameter value.  The robust likelihood floors the density instead (PyDDM's own
    # recommendation).  The DDM uses the analytic solution, which has no such cut-off.
    loss = pyddm.LossLikelihood if model == "ddm" else pyddm.LossRobustLikelihood
    m.fit(sample_of(model, rt, ch, cond), lossfunction=loss, verbose=False)
    names = [RENAME.get(n, n) for n in m.get_model_parameter_names()]
    est = dict(zip(names, map(float, m.get_model_parameters())))
    return {k: est[k] for k in params_of(model)}, float(m.get_fit_result().value()) / len(rt)


def nll_at(model, values, rt, ch, cond, dx, dt):
    """NLL per trial of fixed parameter values on PyDDM's grid (used as referee for the
    models without a closed-form density)."""
    if values["t0"] >= rt.min():
        return np.inf
    m = build(model, None, t_dur_of(rt, dt), dx, dt, values=values)
    return pyddm.get_model_loss(model=m, sample=sample_of(model, rt, ch, cond),
                                lossfunction=pyddm.LossLikelihood) / len(rt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/ddm/test_datasets.npz")
    ap.add_argument("--out", default="results/ddm/pyddm.csv")
    ap.add_argument("--dx", type=float, default=None, help="default 0.001 (ddm) / 0.005 (others)")
    ap.add_argument("--dt", type=float, default=None, help="default 0.001 (ddm) / 0.005 (others)")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=None)
    args = ap.parse_args()
    pyddm.set_N_cpus(1)

    d = load_test_datasets(args.data)
    model = d["model"]
    dx = args.dx or (0.001 if model == "ddm" else 0.005)
    dt = args.dt or (0.001 if model == "ddm" else 0.005)
    params = params_of(model)
    done = set(pd.read_csv(args.out)["dataset"]) if os.path.exists(args.out) else set()
    last = args.last if args.last is not None else len(d["rt"])
    for i in range(args.first, last):
        if i in done:
            continue
        t = time.time()
        est, nll = fit_one(model, d["rt"][i], d["choice"][i], d["cond"][i], dx, dt)
        row = {"dataset": i, "time_s": time.time() - t, "nll_pyddm": nll}
        row.update({f"{k}_est": est[k] for k in params})
        row.update({f"{k}_true": float(d[f"{k}_true"][i]) for k in params})
        pd.DataFrame([row]).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        print(f"[pyddm] dataset {i}: " + "  ".join(f"{k} {row[k + '_true']:.3f}->{row[k + '_est']:.3f}"
                                                  for k in params) + f"  ({row['time_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
