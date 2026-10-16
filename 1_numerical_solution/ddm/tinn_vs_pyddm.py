"""
Theory-Informed Neural Network (TINN / PINN) for the standard Drift-Diffusion
Model (DDM): Fokker-Planck equation with two absorbing boundaries and the
first-passage-time (FPT) densities -- PyTorch version.

    u_t + v u_x - (sigma^2 / 2) u_xx = 0,      x in (XMIN, XMAX), t in (0, T]
    u(0, x)     = delta(x - X0)                 (true Dirac delta, no Gaussian)
    u(t, XMIN)  = u(t, XMAX) = 0                (absorbing boundaries)

    g_upper(t) = -(sigma^2/2) u_x(t, XMAX),   g_lower(t) = +(sigma^2/2) u_x(t, XMIN)

Main fix (initial condition)
---------------------------
A network cannot represent a Dirac delta, and replacing it by a wide Gaussian
(variance 2*DELTA = 0.156) changes the problem itself, so the result can never
match the exact DDM solution. Here the singular part is subtracted analytically:

    u(t,x) = K(t,x) + w(t,x)
    K(t,x) = N(x; X0 + v t, sigma^2 t)   free-space kernel: satisfies the PDE
                                         and the delta initial condition exactly
    w      : satisfies the same PDE, w(0,x) = 0, w(t,XMIN/XMAX) = -K(t,XMIN/XMAX)

w is smooth, so the network only has to learn w. The initial and boundary
conditions on w are imposed exactly (hard constraints) through the ansatz

    w = B(t,x) + (t/T) * bubble(x) * NN(t,x)

where B linearly interpolates the boundary data -K(t,XMIN), -K(t,XMAX) in x and
bubble(x) vanishes at both walls. The loss is then only the PDE residual.

This script: TINN vs PyDDM (Shinn, Lam & Murray, eLife 2020) for the standard DDM
    v = 0.3, sigma = 1, boundaries -2 and +2, x0 = 0, T = 5
Both are compared with the exact series solution for the FPT densities:
    L2 = sqrt( integral e(t)^2 dt ),  Linf = max |e(t)|   (t >= 0.01)

Install PyDDM first:  pip install pyddm
"""

import logging
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_banded

_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 1.x / 2.x

# =============================================================================
# Configuration
# =============================================================================

torch.set_default_dtype(torch.float64)  # float64: much more accurate for PINNs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# DDM parameters
V = 0.3          # drift
SIGMA = 1.0      # noise
D = 0.5 * SIGMA**2
X0 = 0.0         # starting point
XMIN, XMAX = -2.0, 2.0   # absorbing boundaries
TMIN, TMAX = 0.0, 5.0

# Training
N_R = 4000            # collocation points (resampled every Adam step)
ADAM_ITERS = 10000
ADAM_LR = 1e-3
LBFGS_ITERS = 2000
HIDDEN_LAYERS = 5
WIDTH = 50
EARLY_FRACTION = 0.25  # share of points placed at early times t < 1.5

# Exact solution (eigenfunction series)
N_MODES = 400


# =============================================================================
# Free-space kernel (analytic singular part)
# =============================================================================

def free_kernel(t, x):
    """Gaussian transition density of dX = v dt + sigma dW started at X0."""
    t = t.clamp_min(1e-10)
    var = SIGMA**2 * t
    return torch.exp(-(x - X0 - V * t) ** 2 / (2 * var)) / torch.sqrt(2 * math.pi * var)


def free_kernel_x(t, x):
    t = t.clamp_min(1e-10)
    return -(x - X0 - V * t) / (SIGMA**2 * t) * free_kernel(t, x)


def free_kernel_np(t, x):
    t = np.maximum(t, 1e-10)
    return np.exp(-(x - X0 - V * t) ** 2 / (2 * SIGMA**2 * t)) / np.sqrt(2 * np.pi * SIGMA**2 * t)


def free_kernel_x_np(t, x):
    t = np.maximum(t, 1e-10)
    return -(x - X0 - V * t) / (SIGMA**2 * t) * free_kernel_np(t, x)


# =============================================================================
# Network with hard initial / boundary conditions
# =============================================================================

class TINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, width=WIDTH):
        super().__init__()
        layers, d_in = [], 2
        for _ in range(hidden_layers):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _nn(self, t, x):
        # real input normalisation to [-1, 1] (the original Lambda layer was the identity)
        t_hat = 2.0 * (t - TMIN) / (TMAX - TMIN) - 1.0
        x_hat = 2.0 * (x - XMIN) / (XMAX - XMIN) - 1.0
        return self.net(torch.cat([t_hat, x_hat], dim=1))

    def w(self, t, x):
        """Smooth correction: w(0,x)=0 and w=-K on the walls, exactly."""
        k_right = free_kernel(t, torch.full_like(x, XMAX))
        k_left = free_kernel(t, torch.full_like(x, XMIN))
        B = -(k_right * (x - XMIN) + k_left * (XMAX - x)) / (XMAX - XMIN)
        bubble = 4.0 * (x - XMIN) * (XMAX - x) / (XMAX - XMIN) ** 2
        return B + ((t - TMIN) / (TMAX - TMIN)) * bubble * self._nn(t, x)

    def forward(self, t, x):
        """Full density u = K + w (use for t > 0)."""
        return free_kernel(t, x) + self.w(t, x)


def pde_residual(model, t, x):
    """Residual of the Fokker-Planck operator applied to w (K satisfies it exactly)."""
    t.requires_grad_(True)
    x.requires_grad_(True)
    w = model.w(t, x)
    ones = torch.ones_like(w)
    w_t, w_x = torch.autograd.grad(w, (t, x), ones, create_graph=True)
    w_xx = torch.autograd.grad(w_x, x, torch.ones_like(w_x), create_graph=True)[0]
    return w_t + V * w_x - D * w_xx


# =============================================================================
# Training
# =============================================================================

def sample_collocation(n):
    t = TMIN + (TMAX - TMIN) * torch.rand(n, 1)
    n_early = int(EARLY_FRACTION * n)
    t[:n_early] = TMIN + min(1.5, TMAX - TMIN) * torch.rand(n_early, 1)
    t = t.clamp_min(1e-4)
    x = XMIN + (XMAX - XMIN) * torch.rand(n, 1)
    return t.to(DEVICE), x.to(DEVICE)


def train_model(model):
    history = []
    t0 = time.time()

    # ---- Stage 1: Adam with fresh collocation points every step
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-2) ** (1.0 / ADAM_ITERS))
    print("Adam ...")
    for it in range(ADAM_ITERS):
        t, x = sample_collocation(N_R)
        opt.zero_grad(set_to_none=True)
        loss = pde_residual(model, t, x).pow(2).mean()
        loss.backward()
        opt.step()
        sched.step()
        history.append(loss.item())
        if it % 500 == 0:
            print(f"  iter {it:6d}  loss = {loss.item():.3e}")

    # ---- Stage 2: L-BFGS on a fixed, larger set of points
    print("L-BFGS ...")
    t, x = sample_collocation(2 * N_R)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_ITERS,
                              history_size=50, tolerance_grad=1e-12,
                              tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        loss = pde_residual(model, t, x).pow(2).mean()
        loss.backward()
        history.append(loss.item())
        return loss

    lbfgs.step(closure)
    print(f"  final loss = {history[-1]:.3e}")
    print(f"Training time: {time.time() - t0:.1f} s")
    return history


# =============================================================================
# TINN predictions
# =============================================================================

def tinn_density(model, t, x):
    tt = torch.full((len(x), 1), float(t), device=DEVICE)
    xx = torch.as_tensor(x, device=DEVICE).reshape(-1, 1)
    with torch.no_grad():
        return model(tt, xx).cpu().numpy().ravel()


def tinn_fpt(model, t):
    """FPT densities from the probability flux at the walls (u = 0 there)."""
    tt = torch.as_tensor(t, device=DEVICE).reshape(-1, 1)
    out = {}
    for name, xb, sign in (("upper", XMAX, -1.0), ("lower", XMIN, +1.0)):
        x = torch.full_like(tt, xb).requires_grad_(True)
        w = model.w(tt, x)
        w_x = torch.autograd.grad(w, x, torch.ones_like(w))[0]
        u_x = free_kernel_x(tt, x) + w_x
        out[name] = (sign * D * u_x).detach().cpu().numpy().ravel()
    return out


# =============================================================================
# Exact solution (eigenfunction expansion)
# =============================================================================

L = XMAX - XMIN
Z = X0 - XMIN


def exact_density(t, x, n_modes=N_MODES):
    k = np.arange(1, n_modes + 1)
    xs = np.asarray(x) - XMIN
    coef = np.sin(k * np.pi * Z / L) * np.exp(-k**2 * np.pi**2 * SIGMA**2 * t / (2 * L**2))
    s = (coef[None, :] * np.sin(np.outer(xs, k) * np.pi / L)).sum(1)
    return np.exp(V * (xs - Z) / SIGMA**2 - V**2 * t / (2 * SIGMA**2)) * (2 / L) * s


def exact_fpt(t, n_modes=N_MODES):
    t = np.asarray(t, float)[:, None]
    k = np.arange(1, n_modes + 1)[None, :]
    decay = np.exp(-k**2 * np.pi**2 * SIGMA**2 * t / (2 * L**2))
    sz = np.sin(k * np.pi * Z / L)
    common = np.pi * SIGMA**2 / L**2 * np.exp(-V**2 * t[:, 0] / (2 * SIGMA**2))
    upper = common * np.exp(V * (L - Z) / SIGMA**2) * (k * (-1.0) ** (k + 1) * sz * decay).sum(1)
    lower = common * np.exp(-V * Z / SIGMA**2) * (k * sz * decay).sum(1)
    return {"upper": upper, "lower": lower}


# =============================================================================
# PyDDM
# =============================================================================

try:                                    # PyDDM >= 0.6
    import pyddm
    from pyddm.models import (DriftConstant, NoiseConstant, BoundConstant,
                              ICPointSourceCenter, OverlayNone)
except ImportError:                     # very old releases were called "ddm"
    import ddm as pyddm
    from ddm.models import (DriftConstant, NoiseConstant, BoundConstant,
                            ICPointSourceCenter, OverlayNone)
logging.getLogger("pyddm").setLevel(logging.ERROR)
logging.getLogger("ddm").setLevel(logging.ERROR)

PYDDM_GRIDS = [(0.05, 0.05), (0.02, 0.02), (0.01, 0.01), (0.005, 0.005)]
PYDDM_METHODS = ["cn", "implicit"]      # Crank-Nicolson and backward Euler
N_REPEAT = 3                            # timing: best of N_REPEAT solves
T_MIN_ERR = 0.01

MODEL_FILE = "ddm_tinn.pt"
TINN_FILE = "tinn_fpt_simple.npz"
LOAD_PRETRAINED = True                  # reuse ddm_tinn.pt / tinn_fpt_simple.npz if they exist


def pyddm_model(dx, dt):
    return pyddm.Model(name=f"DDM dx={dx} dt={dt}",
                       drift=DriftConstant(drift=V),
                       noise=NoiseConstant(noise=SIGMA),
                       bound=BoundConstant(B=XMAX),        # boundaries at -B and +B
                       IC=ICPointSourceCenter(),            # start at x = 0
                       overlay=OverlayNone(),               # no mixture, no non-decision time
                       dx=dx, dt=dt, T_dur=TMAX)


def pyddm_pdfs(sol):
    try:
        return sol.pdf("correct"), sol.pdf("error")
    except TypeError:
        return sol.pdf_corr(), sol.pdf_err()


def run_pyddm(dx, dt, method):
    model = pyddm_model(dx, dt)
    solver = {"cn": model.solve_numerical_cn, "implicit": model.solve_numerical_implicit}[method]
    best = np.inf
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        sol = solver()
        best = min(best, time.perf_counter() - t0)
    up, lo = pyddm_pdfs(sol)
    return np.asarray(model.t_domain()), {"upper": np.asarray(up), "lower": np.asarray(lo)}, best


# =============================================================================
# Errors against the exact solution
# =============================================================================

def exact_fpt_safe(t):
    g = exact_fpt(t)
    tiny = np.asarray(t) < 5e-3         # series not converged there; true value < 1e-80
    for s in g:
        g[s][tiny] = 0.0
    return g


def error_norms(t, g, side):
    mask = t >= T_MIN_ERR
    tt = t[mask]
    e = g[side][mask] - exact_fpt_safe(tt)[side]
    return np.sqrt(_trapz(e**2, tt)), np.abs(e).max()


# =============================================================================
# TINN: train (or load) and save the FPT densities
# =============================================================================

def get_tinn_results():
    if LOAD_PRETRAINED and os.path.exists(TINN_FILE):
        d = np.load(TINN_FILE)
        print(f"Loaded TINN results from {TINN_FILE} (set LOAD_PRETRAINED = False to retrain)")
        return d["t"], {"upper": d["upper"], "lower": d["lower"]}, float(d["train_time"])

    model = TINN().to(DEVICE)
    if LOAD_PRETRAINED and os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
        train_time = np.nan
        print(f"Loaded trained TINN from {MODEL_FILE} (training time unknown)")
    else:
        t0 = time.time()
        train_model(model)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - t0
        torch.save(model.state_dict(), MODEL_FILE)

    t = np.linspace(TMIN, TMAX, 1001)
    g = tinn_fpt(model, t)
    np.savez(TINN_FILE, t=t, upper=g["upper"], lower=g["lower"], train_time=train_time)
    print(f"Saved {MODEL_FILE} and {TINN_FILE}")
    return t, g, train_time


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    rows = []

    t, g, train_time = get_tinn_results()
    rows.append(("TINN", f"{HIDDEN_LAYERS}x{WIDTH}, N_r={N_R}",
                 [error_norms(t, g, s) for s in ("upper", "lower")], train_time))

    for method in PYDDM_METHODS:
        label = "PyDDM-CN" if method == "cn" else "PyDDM-BE"
        for dx, dt in PYDDM_GRIDS:
            tp, gp, cpu = run_pyddm(dx, dt, method)
            rows.append((label, f"dx={dx}, dt={dt}", [error_norms(tp, gp, s) for s in ("upper", "lower")], cpu))

    hdr = (f"{'Method':<10}{'Discretisation':<22}"
           f"{'L2 upper':>11}{'Linf upper':>12}{'L2 lower':>11}{'Linf lower':>12}{'time [s]':>10}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for meth, disc, ((l2u, liu), (l2l, lil)), tm in rows:
        print(f"{meth:<10}{disc:<22}{l2u:>11.2e}{liu:>12.2e}{l2l:>11.2e}{lil:>12.2e}{tm:>10.3f}")

    def sci(v):
        m, e = f"{v:.1e}".split("e")
        return f"${m}\\times 10^{{{int(e)}}}$"
    print("\nLaTeX rows:")
    for meth, disc, ((l2u, liu), (l2l, lil)), tm in rows:
        print(f"{meth} & {disc} & {sci(l2u)} & {sci(liu)} & {sci(l2l)} & {sci(lil)} & {tm:.3f} \\\\")
    print("\nNote: TINN time = training time; PyDDM time = one solve (best of "
          f"{N_REPEAT}). They measure different things.")


if __name__ == "__main__":
    main()
