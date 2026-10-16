"""
PINN (PyTorch) for the Diffusion Model for Conflict (DMC; Ulrich et al., 2015):

    u_t + v(t) u_x - (sigma^2/2) u_xx = 0,     x in (XMIN, XMAX), t in (0, T]
    v(t) = mu_c + d/dt X_a(t),   X_a(t) = A (e t / tau) exp(-t / tau)      (shape a = 2)
         = mu_c + (A e / tau) exp(-t / tau) (1 - t / tau)
    u(0, x)    = delta(x - X0)               (true Dirac delta)
    u(t, XMIN) = u(t, XMAX) = 0              (absorbing boundaries)

    FPT densities: g_upper(t) = -(sigma^2/2) u_x(t, XMAX),  g_lower(t) = +(sigma^2/2) u_x(t, XMIN)
    (A > 0: congruent trials, A < 0: incongruent trials)

Initial-condition fix (singularity subtraction)
-----------------------------------------------
    u = K + w,  K(t,x) = N(x; m(t), sigma^2 t),  m(t) = X0 + mu_c t + X_a(t)
K is the free-space solution with the DMC drift (m' = v): it satisfies the PDE and
the delta initial condition exactly. The network learns only the smooth part w
(w(0,x) = 0, w = -K on the walls), imposed exactly by
    w = B(t,x) + (t/T) * bubble(x) * NN(t,x).
The loss is only the PDE residual of w.

Reference: there is no closed-form solution with two absorbing boundaries, so the
reference is a converged Crank-Nicolson solution of the same split problem.

UNITS: TAU must be in the same time unit as t. In the DMC literature tau ~ 50-150
is in milliseconds; with t in [0, 3] and TAU = 150 the drift is almost constant
(0.862 -> 0.848) and the model is practically a standard DDM.

This script: one PINN per value of tau in TAU_LIST (mu_c, A fixed); FPT densities
at the upper and lower boundary for all tau, PINN (markers) with the FD reference
(lines), black-and-white, 12 x 8.
"""

import math
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

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# DMC parameters (TAU in the same time unit as t -- see the note above)
MUC = 0.5        # controlled-process drift mu_c
A_AUTO = 20.0    # amplitude A of the automatic process (A < 0: incongruent)
TAU = 150.0      # time constant tau of the automatic process (overwritten in the study)
SIGMA = 1.0
D = 0.5 * SIGMA**2
X0 = 0.0
XMIN, XMAX = -2.0, 2.0
TMIN, TMAX = 0.0, 3.0

# Training
N_R = 4000             # collocation points (resampled every Adam step)
ADAM_ITERS = 10000
ADAM_LR = 1e-3
LBFGS_ITERS = 2000
HIDDEN_LAYERS = 4
WIDTH = 50

# FD reference
FD_NX, FD_DT = 801, 1e-3
SNAP_TIMES = (0.5, 1.0, 3.0)


def drift(t):
    """v(t) = mu_c + (A e / tau) exp(-t/tau) (1 - t/tau)   (no 1/t: regular at t = 0)."""
    exp = torch.exp if torch.is_tensor(t) else np.exp
    return MUC + A_AUTO * math.e / TAU * exp(-t / TAU) * (1 - t / TAU)


def mean_path(t):
    """m(t) = X0 + mu_c t + A (e t / tau) exp(-t/tau)   (so that m'(t) = v(t))."""
    exp = torch.exp if torch.is_tensor(t) else np.exp
    return X0 + MUC * t + A_AUTO * (math.e * t / TAU) * exp(-t / TAU)


# =============================================================================
# Free-space kernel with time-dependent drift
# =============================================================================

def free_kernel(t, x):
    t = t.clamp_min(1e-10)
    var = SIGMA**2 * t
    return torch.exp(-(x - mean_path(t)) ** 2 / (2 * var)) / torch.sqrt(2 * math.pi * var)


def free_kernel_x(t, x):
    t = t.clamp_min(1e-10)
    return -(x - mean_path(t)) / (SIGMA**2 * t) * free_kernel(t, x)


def free_kernel_np(t, x):
    t = max(t, 1e-10)
    return np.exp(-(x - mean_path(t)) ** 2 / (2 * SIGMA**2 * t)) / np.sqrt(2 * np.pi * SIGMA**2 * t)


def free_kernel_x_np(t, x):
    t = max(t, 1e-10)
    return -(x - mean_path(t)) / (SIGMA**2 * t) * free_kernel_np(t, x)


# =============================================================================
# Network (hard initial and boundary conditions)
# =============================================================================

class PINN(nn.Module):
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
        t_hat = 2.0 * (t - TMIN) / (TMAX - TMIN) - 1.0
        x_hat = 2.0 * (x - XMIN) / (XMAX - XMIN) - 1.0
        return self.net(torch.cat([t_hat, x_hat], dim=1))

    def w(self, t, x):
        k_r = free_kernel(t, torch.full_like(x, XMAX))
        k_l = free_kernel(t, torch.full_like(x, XMIN))
        B = -(k_r * (x - XMIN) + k_l * (XMAX - x)) / (XMAX - XMIN)
        bubble = 4.0 * (x - XMIN) * (XMAX - x) / (XMAX - XMIN) ** 2
        return B + ((t - TMIN) / (TMAX - TMIN)) * bubble * self._nn(t, x)

    def forward(self, t, x):
        return free_kernel(t, x) + self.w(t, x)


def pde_residual(model, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)
    w = model.w(t, x)
    w_t, w_x = torch.autograd.grad(w, (t, x), torch.ones_like(w), create_graph=True)
    w_xx = torch.autograd.grad(w_x, x, torch.ones_like(w_x), create_graph=True)[0]
    return w_t + drift(t) * w_x - D * w_xx


# =============================================================================
# Training
# =============================================================================

def sample_collocation(n):
    t = TMIN + (TMAX - TMIN) * torch.rand(n, 1)
    n_early = n // 4                                 # extra points at early times
    t[:n_early] = TMIN + min(1.5, TMAX - TMIN) * torch.rand(n_early, 1)
    t = t.clamp_min(1e-4)
    x = XMIN + (XMAX - XMIN) * torch.rand(n, 1)
    return t.to(DEVICE), x.to(DEVICE)


def train_model(model):
    history = []
    t0 = time.time()
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
# PINN predictions
# =============================================================================

def pinn_density(model, t, x):
    tt = torch.full((len(x), 1), float(t), device=DEVICE)
    xx = torch.as_tensor(x, device=DEVICE).reshape(-1, 1)
    with torch.no_grad():
        return model(tt, xx).cpu().numpy().ravel()


def pinn_fpt(model, t):
    tt = torch.as_tensor(t, device=DEVICE).reshape(-1, 1)
    out = {}
    for side, xb, sign in (("upper", XMAX, -1.0), ("lower", XMIN, +1.0)):
        x = torch.full_like(tt, xb).requires_grad_(True)
        w = model.w(tt, x)
        w_x = torch.autograd.grad(w, x, torch.ones_like(w))[0]
        out[side] = (sign * D * (free_kernel_x(tt, x) + w_x)).detach().cpu().numpy().ravel()
    return out


# =============================================================================
# Reference: Crank-Nicolson on the smooth part w (same splitting)
# =============================================================================

def finite_difference_solver(nx=FD_NX, dt=FD_DT, snapshot_times=SNAP_TIMES):
    x = np.linspace(XMIN, XMAX, nx)
    dx = x[1] - x[0]
    r = D * dt / dx**2
    n = nx - 2
    steps = int(round((TMAX - TMIN) / dt))
    snap_idx = {int(round(s / dt)): s for s in snapshot_times}

    w = np.zeros(nx)
    ts, g_up, g_lo = np.zeros(steps + 1), np.zeros(steps + 1), np.zeros(steps + 1)
    snaps = {}
    for m in range(steps):
        t1 = TMIN + (m + 1) * dt
        c = drift(t1 - 0.5 * dt) * dt / (2 * dx)        # drift at the half step (2nd order)
        lo, di, up = -r / 2 - c / 2, 1 + r, -r / 2 + c / 2
        ab = np.zeros((3, n))
        ab[0, 1:], ab[1, :], ab[2, :-1] = up, di, lo

        bl, br = -free_kernel_np(t1, XMIN), -free_kernel_np(t1, XMAX)
        rhs = (1 - r) * w[1:-1] + (r / 2 + c / 2) * w[:-2] + (r / 2 - c / 2) * w[2:]
        rhs[0] -= lo * bl
        rhs[-1] -= up * br
        w[1:-1] = solve_banded((1, 1), ab, rhs)
        w[0], w[-1] = bl, br

        ts[m + 1] = t1
        g_up[m + 1] = -D * (free_kernel_x_np(t1, XMAX) + (3 * w[-1] - 4 * w[-2] + w[-3]) / (2 * dx))
        g_lo[m + 1] = +D * (free_kernel_x_np(t1, XMIN) + (-3 * w[0] + 4 * w[1] - w[2]) / (2 * dx))
        if (m + 1) in snap_idx:
            snaps[snap_idx[m + 1]] = free_kernel_np(t1, x) + w

    return {"t": ts, "upper": g_up, "lower": g_lo, "x": x, "snapshots": snaps}


# =============================================================================
# Errors
# =============================================================================

def error_norms(pred, ref, grid):
    e = pred - ref
    return np.sqrt(_trapz(e**2, grid)), np.abs(e).max()


# =============================================================================
# Parameter studies
# =============================================================================

STUDIES = {
    # name: (global variable that is changed, values, LaTeX symbol)
    "tau": ("TAU", [20, 40, 80, 160], "\\tau"),
}
RUN = ["tau"]
N_T = 100                        # PINN markers

FIGSIZE, DPI = (12, 8), 90
FONT = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16}
TICK_SIZE, LEGEND_SIZE = 14, 16
LINESTYLES = ['-', '--', ':', '-.']
MARKERS = ['o', 's', '^', 'd']


def run_study(name):
    """Train one PINN per value. Results (densities and errors) are saved after
    every run in dmc_study_<name>.npz, so a later error never loses finished runs."""
    var, values, _ = STUDIES[name]
    base = globals()[var]
    tspace = np.linspace(TMIN, TMAX, N_T + 1)
    res = {"upper": [], "lower": [], "fd_upper": [], "fd_lower": [], "errors": []}
    fd_t = None

    for i, val in enumerate(values):
        globals()[var] = val                  # drift() and mean_path() read the globals
        torch.manual_seed(SEED)
        print(f"\n===== {name} = {val}   (mu_c = {MUC}, A = {A_AUTO}, tau = {TAU}) =====")
        model = PINN().to(DEVICE)
        train_model(model)
        fd = finite_difference_solver(snapshot_times=())
        fd_t = fd["t"]

        g = pinn_fpt(model, tspace)
        for side in ("upper", "lower"):
            res[side].append(g[side])
            res["fd_" + side].append(fd[side])

        mask = fd["t"] >= 0.01
        gp = pinn_fpt(model, fd["t"][mask])
        errs = []
        for side in ("upper", "lower"):
            l2, linf = error_norms(gp[side], fd[side][mask], fd["t"][mask])
            errs += [l2, linf]
            print(f"{name} = {val}: FPT {side:5s}  L2 = {l2:.3e}, Linf = {linf:.3e}")
        res["errors"].append(errs)            # [L2 up, Linf up, L2 low, Linf low]

        try:
            np.savez(f"dmc_study_{name}.npz", values=np.array(values[:i + 1]), tspace=tspace,
                     fd_t=fd_t, **{k: np.array(a) for k, a in res.items()})
        except OSError as e:
            print(f"[warning] results not saved: {e}")

    globals()[var] = base                     # restore the base value
    return tspace, fd_t, res


def plot_study(name, values, tspace, fd_t, res, side, filename,
               legend_loc='upper right', headroom=1.75):
    """Filled markers = presented method, lines = FD reference."""
    sym = STUDIES[name][2]
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    for i, val in enumerate(values):
        plt.plot(tspace, res[side][i], linestyle='none', marker=MARKERS[i % 4],
                 markersize=6, color='black', label=f'Presented method ${sym} = {val}$')
        plt.plot(fd_t, res["fd_" + side][i], color='black', linestyle=LINESTYLES[i % 4],
                 linewidth=1.3, label=f'FD method ${sym} = {val}$')

    if headroom > 1.0:                        # empty space above the curves for the legend
        plt.ylim(top=headroom * max(np.max(a) for a in res["fd_" + side]))

    plt.xlabel('Time', fontdict=FONT)
    plt.ylabel('First passage time', fontdict=FONT)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.legend(fontsize=LEGEND_SIZE, loc=legend_loc)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


def print_error_table(name, values, res):
    sym = STUDIES[name][2].replace("\\", "")
    print(f"\n{sym:<8}{'L2 upper':>12}{'Linf upper':>12}{'L2 lower':>12}{'Linf lower':>12}")
    for val, e in zip(values, res["errors"]):
        print(f"{val:<8}" + "".join(f"{x:>12.2e}" for x in e))


def main():
    print(f"Device: {DEVICE}")
    for name in RUN:
        tspace, fd_t, res = run_study(name)
        values = STUDIES[name][1]
        print_error_table(name, values, res)
        plot_study(name, values, tspace, fd_t, res, "upper", f"dmc_fpt_upper_{name}.pdf",
                   headroom=1.15)
        plot_study(name, values, tspace, fd_t, res, "lower", f"dmc_fpt_lower_{name}.pdf")


if __name__ == "__main__":
    main()
