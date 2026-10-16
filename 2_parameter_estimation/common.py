"""
Shared pieces of the TINN / PyDDM / BayesFlow comparison, for all five models.

Convention used by every method (and by models.py):
    dX = mu(X, t) dt + sigma dW,  sigma = 1,  X(0) = 0,  bounds l(t), u(t),  RT = T + t0
    choice 1 = upper bound, 0 = lower bound.
PyDDM's `bound` B and HDDM/ssm-simulators' `a` use the same +/- a convention.

    ddm       mu = v                                   bounds +/- a
    urgency   mu = v0 + v1 t / (t + tau)               bounds +/- a
    ou        mu = v0 - beta x                         bounds +/- a
    dmc       mu = mu0 + c mu1(t), alpha = 2,          bounds +/- a,  c = +1 congruent / -1 incongruent
              mu1 = A e^{-t/tau} (e / tau) (1 - t / tau)
    collapse  mu = v,  bounds +/- (a - a t / (2 T_inf))
    ddm3, urgency3, ou3, collapse3: the base model in three conditions that differ only in the bound
              (a1, a2, a3; e.g. speed / neutral / accuracy), all other parameters shared;
              same design as ../multi_condition.py.  Trials are split evenly over the conditions.
"""
import math
import re
import numpy as np
import numba

# Parameter ranges of the test datasets and of the BayesFlow prior.  Last entry is always t0.
# ddm: Matzke & Wagenmakers (2009, Table 3) rescaled to sigma = 1 and half bound separation;
#      Fengler et al. (2021) / HDDM bounds.  The other models keep the same a / t0 scale.
# ou:  a strong leak with a wide bound and a weak input almost never reaches a bound (mean RT
#      > 10 s, trials still running at 30 s), so beta <= 1 and a <= 1.5.
# dmc: Ulrich et al. (2015) values (sigma = 4, ms) rescaled to sigma = 1, seconds.
MODELS = {
    "ddm":      {"id": 0, "conditions": [1.0],
                 "ranges": {"v": (0.2, 4.0), "a": (0.3, 2.0), "t0": (0.2, 0.9)}},
    "urgency":  {"id": 1, "conditions": [1.0],
                 "ranges": {"v0": (0.0, 2.0), "v1": (0.5, 3.0), "tau": (0.05, 1.0), "a": (0.3, 2.0),
                            "t0": (0.2, 0.9)}},
    "ou":       {"id": 2, "conditions": [1.0],
                 "ranges": {"v0": (0.5, 4.0), "beta": (0.0, 1.0), "a": (0.3, 1.5), "t0": (0.2, 0.9)}},
    "dmc":      {"id": 3, "conditions": [1.0, -1.0],
                 "ranges": {"mu0": (1.0, 5.0), "A": (0.05, 0.3), "tau": (0.02, 0.12), "a": (0.4, 1.2),
                            "t0": (0.2, 0.5)}},
    "collapse": {"id": 4, "conditions": [1.0],
                 "ranges": {"v": (0.2, 4.0), "a": (0.5, 2.0), "T_inf": (0.5, 3.0), "t0": (0.2, 0.9)}},
    # three bound conditions; each a_k is drawn independently from the base model's a range,
    # so no method is told how the bounds relate to each other
    "urgency3": {"id": 1, "base": "urgency", "conditions": [0.0, 1.0, 2.0],
                 "ranges": {"v0": (0.0, 2.0), "v1": (0.5, 3.0), "tau": (0.05, 1.0), "a1": (0.3, 2.0),
                            "a2": (0.3, 2.0), "a3": (0.3, 2.0), "t0": (0.2, 0.9)}},
    "ddm3":     {"id": 0, "base": "ddm", "conditions": [0.0, 1.0, 2.0],
                 "ranges": {"v": (0.2, 4.0), "a1": (0.3, 2.0), "a2": (0.3, 2.0), "a3": (0.3, 2.0),
                            "t0": (0.2, 0.9)}},
    "collapse3": {"id": 4, "base": "collapse", "conditions": [0.0, 1.0, 2.0],
                  "ranges": {"v": (0.2, 4.0), "a1": (0.5, 2.0), "a2": (0.5, 2.0), "a3": (0.5, 2.0),
                             "T_inf": (0.5, 3.0), "t0": (0.2, 0.9)}},
    "ou3":      {"id": 2, "base": "ou", "conditions": [0.0, 1.0, 2.0],
                 "ranges": {"v0": (0.5, 4.0), "beta": (0.0, 1.0), "a1": (0.3, 1.5), "a2": (0.3, 1.5),
                            "a3": (0.3, 1.5), "t0": (0.2, 0.9)}},
}


def base_of(model):
    """the single-condition model a multi-bound model is built on (itself otherwise)"""
    return MODELS[model].get("base", model)


def bound_values(model, values, k):
    """parameters of condition k of a multi-bound model as base-model parameters (a = a_k)"""
    out = {p: v for p, v in values.items() if not re.fullmatch(r"a\d+", p)}
    out["a"] = values[f"a{k + 1}"]
    return out

DT = 1e-3          # simulator step (s); the Brownian-bridge correction removes the overshoot bias
T_MAX = 30.0       # trials still undecided at T_MAX are reported as rt = nan


def params_of(model):
    return list(MODELS[model]["ranges"])


def ranges_of(model):
    return MODELS[model]["ranges"]


# ------------------------------------------------------------------ simulator
@numba.njit(cache=True)
def _drift(mid, th, x, t, c):
    if mid == 0 or mid == 4:
        return th[0]
    if mid == 1:
        return th[0] + th[1] * t / (t + th[2])
    if mid == 2:
        return th[0] - th[1] * x
    # dmc, alpha = 2
    tau = th[2]
    return th[0] + c * th[1] * math.exp(-t / tau) * (math.e / tau) * (1.0 - t / tau)


@numba.njit(cache=True)
def _bound(mid, th, t):
    if mid == 4:
        return th[1] - th[1] * t / (2.0 * th[2])
    return th[-2]


@numba.njit(cache=True)
def _sim(mid, th, c, n, dt, t_max, seed):
    np.random.seed(seed)
    rt = np.empty(n)
    ch = np.empty(n)
    sq = math.sqrt(dt)
    t0 = th[-1]
    for i in range(n):
        x = 0.0
        t = 0.0
        rt[i] = np.nan
        ch[i] = -1.0
        u = _bound(mid, th, 0.0)
        while t < t_max:
            xn = x + _drift(mid, th, x, t + 0.5 * dt, c) * dt + sq * np.random.randn()
            t += dt
            un = _bound(mid, th, t)
            if un <= 0.0:                       # collapsing bounds have met
                rt[i] = t + t0
                ch[i] = 1.0 if xn > 0 else 0.0
                break
            if xn >= un:
                rt[i] = t - 0.5 * dt + t0; ch[i] = 1.0; break
            if xn <= -un:
                rt[i] = t - 0.5 * dt + t0; ch[i] = 0.0; break
            # Brownian bridge: probability that the path crossed a (linear) bound inside the step
            if np.random.rand() < math.exp(-2.0 * (u - x) * (un - xn) / dt):
                rt[i] = t - 0.5 * dt + t0; ch[i] = 1.0; break
            if np.random.rand() < math.exp(-2.0 * (x + u) * (xn + un) / dt):
                rt[i] = t - 0.5 * dt + t0; ch[i] = 0.0; break
            x = xn
            u = un
    return rt, ch


@numba.njit(parallel=True, cache=True)
def _sim_batch(mid, theta, conds, n, dt, t_max, seed):
    m = theta.shape[0]
    nc = conds.shape[0]
    per = n // nc
    rt = np.empty((m, per * nc))
    ch = np.empty((m, per * nc))
    ci = np.empty((m, per * nc))
    for j in numba.prange(m):
        for k in range(nc):
            r, c = _sim(mid, theta[j], conds[k], per, dt, t_max, seed + j * nc + k)
            rt[j, k * per:(k + 1) * per] = r
            ch[j, k * per:(k + 1) * per] = c
            ci[j, k * per:(k + 1) * per] = k
    return rt, ch, ci


def simulate_batch(model, p, n, seed, dt=DT, t_max=T_MAX):
    """One dataset of n trials (split evenly over the model's conditions) per row of the
    parameter dict p.  Returns rt, choice, cond index, each (n_datasets, n)."""
    if "base" in MODELS[model]:
        K = len(MODELS[model]["conditions"])
        parts = [simulate_batch(base_of(model), bound_values(model, p, k), int(n) // K,
                                seed + 10_000_000 * (k + 1), dt, t_max) for k in range(K)]
        return (np.concatenate([r for r, _, _ in parts], 1), np.concatenate([c for _, c, _ in parts], 1),
                np.concatenate([np.full_like(r, k) for k, (r, _, _) in enumerate(parts)], 1))
    theta = np.stack([np.asarray(p[k], float) for k in params_of(model)], 1)
    conds = np.asarray(MODELS[model]["conditions"], float)
    return _sim_batch(MODELS[model]["id"], theta, conds, int(n), dt, t_max, int(seed))


def sample_params(model, rng, n):
    return {k: rng.uniform(lo, hi, n) for k, (lo, hi) in ranges_of(model).items()}


def make_test_datasets(model, n_datasets, n_trials, seed=2026, path=None):
    """Test datasets used by every method."""
    p = sample_params(model, np.random.default_rng(seed), n_datasets)
    rt, ch, cond = simulate_batch(model, p, n_trials, seed=seed * 1000)
    if np.isnan(rt).any():
        raise RuntimeError("some trials did not finish before T_MAX; raise T_MAX")
    data = dict(model=model, rt=rt, choice=ch, cond=cond.astype(int),
                **{f"{k}_true": p[k] for k in params_of(model)})
    if path:
        np.savez(path, **data)
    return data


def load_test_datasets(path):
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    out["model"] = str(out.get("model", "ddm"))
    if "cond" not in out:
        out["cond"] = np.zeros_like(out["rt"], dtype=int)
    return out
