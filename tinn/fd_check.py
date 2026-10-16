"""
Independent check with a Crank-Nicolson finite-difference Fokker-Planck solver
(fixed bounds +/- a: ddm, urgency, ou, dmc; collapsing bounds in y = x / u(t), fd_fpt_moving).  It evaluates the exact negative
log-likelihood of the same simulated data at the true parameters and at the
PINN estimate, so you can tell whether a parameter that was missed is a
property of the data (likelihood really is better/flat there) or a PINN error.

    python fd_check.py --model ou --estimate results/estimate_ou.json
"""
import argparse
import json
import math
import numpy as np
import torch
from scipy.linalg import solve_banded

from models import MODELS

D = 0.5


def fd_fpt(model, p, cond_val, t_end, nx=401, dt=2e-4):
    a = float(p["a"])
    x = np.linspace(-a, a, nx)
    dx = x[1] - x[0]
    xi = x[1:-1]
    n = len(xi)
    pr = np.zeros(n)
    pr[np.argmin(np.abs(xi - float(model.start(p))))] = 1.0 / dx   # delta at z
    ts = np.arange(1, int(t_end / dt) + 1) * dt
    gu, gl = np.zeros(len(ts)), np.zeros(len(ts))
    pt = {k: v for k, v in p.items()}

    def op(t):
        # L p = -d/dx(mu p) + D p_xx  (central differences), tridiagonal
        mu = model.drift(torch.tensor(x), torch.tensor(t), pt, cond_val).numpy()
        lo = D / dx ** 2 + mu[:-2] / (2 * dx)        # coefficient of p_{i-1}
        di = -2 * D / dx ** 2 * np.ones(n)
        up = D / dx ** 2 - mu[2:] / (2 * dx)         # coefficient of p_{i+1}
        return lo[1:], di, up[:-1]

    for k, t in enumerate(ts):
        l1, d1, u1 = op(t - dt)
        l2, d2, u2 = op(t)
        rhs = pr + 0.5 * dt * (d1 * pr + np.r_[0, l1 * pr[:-1]] + np.r_[u1 * pr[1:], 0])
        ab = np.zeros((3, n))
        ab[0, 1:] = -0.5 * dt * u2
        ab[1] = 1 - 0.5 * dt * d2
        ab[2, :-1] = -0.5 * dt * l2
        pr = solve_banded((1, 1), ab, rhs)
        gu[k] = D * (4 * pr[-1] - pr[-2]) / (2 * dx)
        gl[k] = D * (4 * pr[0] - pr[1]) / (2 * dx)
    return ts, gu, gl


def fd_fpt_moving(model, p, cond_val, t_end, ny=401, dt=2e-4):
    """Same Crank-Nicolson scheme for symmetric bounds +/- u(t) that move (collapsing bounds):
    in y = x / u(t) the bounds are fixed at +/- 1, with drift mu(u y, t) / u - y u'(t) / u and
    diffusion D / u(t)^2."""
    y = np.linspace(-1, 1, ny)
    dy = y[1] - y[0]
    n = ny - 2
    pr = np.zeros(n)
    pr[np.argmin(np.abs(y[1:-1]))] = 1.0 / dy
    ts = np.arange(1, int(t_end / dt) + 1) * dt
    gu, gl = np.zeros(len(ts)), np.zeros(len(ts))

    def u_of(t):
        return float(model.bounds(torch.tensor(t), p)[1])

    def op(t):
        u = u_of(t)
        du = (u_of(t + 1e-5) - u_of(max(t - 1e-5, 0.0))) / (t + 1e-5 - max(t - 1e-5, 0.0))
        b = model.drift(torch.tensor(u * y), torch.tensor(t), p, cond_val).numpy() / u - y * du / u
        Dt = D / u ** 2
        lo = Dt / dy ** 2 + b[:-2] / (2 * dy)
        di = -2 * Dt / dy ** 2 * np.ones(n)
        up = Dt / dy ** 2 - b[2:] / (2 * dy)
        return lo[1:], di, up[:-1], Dt

    for k, t in enumerate(ts):
        l1, d1, u1, _ = op(t - dt)
        l2, d2, u2, Dt = op(t)
        rhs = pr + 0.5 * dt * (d1 * pr + np.r_[0, l1 * pr[:-1]] + np.r_[u1 * pr[1:], 0])
        ab = np.zeros((3, n))
        ab[0, 1:] = -0.5 * dt * u2
        ab[1] = 1 - 0.5 * dt * d2
        ab[2, :-1] = -0.5 * dt * l2
        pr = solve_banded((1, 1), ab, rhs)
        gu[k] = Dt * (4 * pr[-1] - pr[-2]) / (2 * dy)
        gl[k] = Dt * (4 * pr[0] - pr[1]) / (2 * dy)
    return ts, gu, gl


def nll(model, p, t0, rt, ch, cond):
    t_end = float(rt.max()) - t0 + 0.01
    tot = 0.0
    for ci, cv in enumerate(model.conditions):
        solver = fd_fpt_moving if model.name == "collapse" else fd_fpt
        ts, gu, gl = solver(model, p, cv, t_end)
        sel = cond == ci
        T = rt[sel] - t0
        g = np.where(ch[sel] == 1, np.interp(T, ts, gu), np.interp(T, ts, gl))
        tot += -np.log(np.maximum(g, 1e-300)).sum()
    return tot / len(rt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ou")
    ap.add_argument("--estimate", required=True)
    ap.add_argument("--n-trials", type=int, default=2000)
    ap.add_argument("--t0", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--free-shape", action="store_true", help="DMC fitted with free alpha")
    ap.add_argument("--free-z", action="store_true", help="fitted with free starting point z")
    args = ap.parse_args()
    model = MODELS[args.model](free_shape=True) if args.free_shape else MODELS[args.model]()
    if args.free_z:
        model.with_free_start()
    rt, ch, cond = model.simulate(args.n_trials, model.true_params(), args.t0, seed=args.seed)
    rt, ch, cond = rt.numpy(), ch.numpy(), cond.numpy()
    est = json.load(open(args.estimate))["estimate"]

    def pdict(d):
        p = {k: torch.tensor(float(d[k])) for k in model.param_specs}
        for k, v in model.fixed.items():
            p[k] = torch.tensor(float(v))
        return p

    truth = {k: float(v) for k, v in model.true_params().items()}
    print(f"exact NLL per trial at TRUE parameters    : {nll(model, pdict(truth), args.t0, rt, ch, cond):.5f}")
    print(f"exact NLL per trial at the PINN estimate  : {nll(model, pdict(est), est['t0'], rt, ch, cond):.5f}")
    print("(lower is better; if the estimate is lower, the data themselves favour it)")


if __name__ == "__main__":
    main()
