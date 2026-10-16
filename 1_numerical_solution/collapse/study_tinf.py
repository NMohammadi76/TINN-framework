"""
PINN (PyTorch) for the DDM with linearly collapsing boundaries:

    dX = v dt + sigma dW,   absorbing boundaries at x = -b(t) and x = +b(t),
    b(t) = B0 (1 - t / (2 T_INF)),   X(0) = X0

Fixed domain via y = x / b(t) in [-1, 1].  For q(t, y) = u(t, b(t) y)   (u = density in x)

    q_t + mu(y, t) q_y - D(t) q_yy = 0,
    mu(y, t) = v / b - y b' / b,     D(t) = sigma^2 / (2 b^2),     b' = -B0 / (2 T_INF)
    q(0, y)  = u(0, B0 y) = delta(B0 y - X0),     q(t, -1) = q(t, 1) = 0

FPT densities (u = 0 on the moving walls, so only the diffusive flux remains):
    g_upper(t) = -(sigma^2/2) u_x(t,  b) = -(sigma^2/2) q_y(t,  1) / b(t)
    g_lower(t) = +(sigma^2/2) u_x(t, -b) = +(sigma^2/2) q_y(t, -1) / b(t)

Initial-condition fix (singularity subtraction)
-----------------------------------------------
    q = K~ + w,   K~(t, y) = K(t, b(t) y),   K(t, x) = N(x; X0 + v t, sigma^2 t)
The free-space kernel K solves the x-equation, so K~ solves the y-equation exactly
and carries the delta initial condition. The network learns only the smooth part w
(w(0,y) = 0, w(t,+-1) = -K(t, +-b(t))), imposed exactly by
    w = B(t, y) + (t/T) (1 - y^2) NN(t, y).
The loss is only the PDE residual of w.

Reference: converged Crank-Nicolson solution of the same split problem in y
(no closed form with collapsing boundaries). Checks: no-collapse limit = exact DDM
series (1e-5), grid refinement (2e-5), mass conservation, Monte Carlo.

This script: one PINN per value of T_INF in the study list (v, B0 fixed); FPT
densities at the upper and lower boundary for all T_INF, PINN (markers) with the
FD reference (lines), black-and-white, 12 x 8.
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

# Model
V = 0.7          # drift
SIGMA = 1.0
X0 = 0.0         # start point
B0 = 2.0         # initial half-width of the boundaries (b(0))
T_INF = 2.0      # b(t) = B0 (1 - t / (2 T_INF)); collapses at t = 2 T_INF (overwritten in the study)
TMIN, TMAX = 0.0, 3.0            # TMAX must be < 2 T_INF

# Training
N_R = 4000
ADAM_ITERS = 10000
ADAM_LR = 1e-3
LBFGS_ITERS = 2000
HIDDEN_LAYERS = 4
WIDTH = 50

# FD reference
FD_NY, FD_DT = 801, 1e-3
SNAP_TIMES = (0.5, 1.0, 3.0)

assert TMAX < 2 * T_INF, "the boundaries collapse at t = 2 T_INF"


def bound(t):
    return B0 * (1 - t / (2 * T_INF))


B_DOT = -B0 / (2 * T_INF)


def drift_y(y, t):
    """mu(y, t) = v / b - y b' / b."""
    b = bound(t)
    return V / b - y * B_DOT / b


def diff_y(t):
    """D(t) = sigma^2 / (2 b^2)."""
    return SIGMA**2 / (2 * bound(t) ** 2)


# =============================================================================
# Free-space kernel (in x) and its y-version
# =============================================================================

def free_kernel(t, x):
    t = t.clamp_min(1e-10)
    var = SIGMA**2 * t
    return torch.exp(-(x - X0 - V * t) ** 2 / (2 * var)) / torch.sqrt(2 * math.pi * var)


def free_kernel_x(t, x):
    t = t.clamp_min(1e-10)
    return -(x - X0 - V * t) / (SIGMA**2 * t) * free_kernel(t, x)


def free_kernel_np(t, x):
    t = max(t, 1e-10)
    return np.exp(-(x - X0 - V * t) ** 2 / (2 * SIGMA**2 * t)) / np.sqrt(2 * np.pi * SIGMA**2 * t)


def free_kernel_x_np(t, x):
    t = max(t, 1e-10)
    return -(x - X0 - V * t) / (SIGMA**2 * t) * free_kernel_np(t, x)


# =============================================================================
# Network (hard initial and boundary conditions, in y)
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

    def _nn(self, t, y):
        t_hat = 2.0 * (t - TMIN) / (TMAX - TMIN) - 1.0
        return self.net(torch.cat([t_hat, y], dim=1))      # y is already in [-1, 1]

    def w(self, t, y):
        b = bound(t)
        k_r = free_kernel(t, b)                              # K at x = +b(t)
        k_l = free_kernel(t, -b)                             # K at x = -b(t)
        B = -(k_r * (1 + y) + k_l * (1 - y)) / 2
        return B + ((t - TMIN) / (TMAX - TMIN)) * (1 - y**2) * self._nn(t, y)

    def forward(self, t, y):
        """q(t, y) = u(t, b(t) y), the density in x at x = b(t) y."""
        return free_kernel(t, bound(t) * y) + self.w(t, y)


def pde_residual(model, t, y):
    t.requires_grad_(True)
    y.requires_grad_(True)
    w = model.w(t, y)
    w_t, w_y = torch.autograd.grad(w, (t, y), torch.ones_like(w), create_graph=True)
    w_yy = torch.autograd.grad(w_y, y, torch.ones_like(w_y), create_graph=True)[0]
    return w_t + drift_y(y, t) * w_y - diff_y(t) * w_yy


# =============================================================================
# Training
# =============================================================================

def sample_collocation(n):
    t = TMIN + (TMAX - TMIN) * torch.rand(n, 1)
    n_early = n // 4
    t[:n_early] = TMIN + min(1.5, TMAX - TMIN) * torch.rand(n_early, 1)
    t = t.clamp_min(1e-4)
    y = -1.0 + 2.0 * torch.rand(n, 1)
    return t.to(DEVICE), y.to(DEVICE)


def train_model(model):
    history = []
    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-2) ** (1.0 / ADAM_ITERS))
    print("Adam ...")
    for it in range(ADAM_ITERS):
        t, y = sample_collocation(N_R)
        opt.zero_grad(set_to_none=True)
        loss = pde_residual(model, t, y).pow(2).mean()
        loss.backward()
        opt.step()
        sched.step()
        history.append(loss.item())
        if it % 500 == 0:
            print(f"  iter {it:6d}  loss = {loss.item():.3e}")

    print("L-BFGS ...")
    t, y = sample_collocation(2 * N_R)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_ITERS,
                              history_size=50, tolerance_grad=1e-12,
                              tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        loss = pde_residual(model, t, y).pow(2).mean()
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

def pinn_density(model, t, y):
    """Density u(t, x) at x = b(t) y."""
    tt = torch.full((len(y), 1), float(t), device=DEVICE)
    yy = torch.as_tensor(y, device=DEVICE).reshape(-1, 1)
    with torch.no_grad():
        return model(tt, yy).cpu().numpy().ravel()


def pinn_fpt(model, t):
    tt = torch.as_tensor(t, device=DEVICE).reshape(-1, 1)
    b = bound(tt)
    out = {}
    for side, yb, sign in (("upper", 1.0, -1.0), ("lower", -1.0, +1.0)):
        y = torch.full_like(tt, yb).requires_grad_(True)
        w = model.w(tt, y)
        w_y = torch.autograd.grad(w, y, torch.ones_like(w))[0]
        u_x = free_kernel_x(tt, yb * b) + w_y / b              # u_x = q_y / b
        out[side] = (sign * 0.5 * SIGMA**2 * u_x).detach().cpu().numpy().ravel()
    return out


# =============================================================================
# Reference: Crank-Nicolson on w in y (coefficients at the half step)
# =============================================================================

def finite_difference_solver(ny=FD_NY, dt=FD_DT, snapshot_times=SNAP_TIMES):
    y = np.linspace(-1.0, 1.0, ny)
    dy = y[1] - y[0]
    n = ny - 2
    steps = int(round((TMAX - TMIN) / dt))
    snap_idx = {int(round(s / dt)): s for s in snapshot_times}

    w = np.zeros(ny)
    ts, g_up, g_lo = np.zeros(steps + 1), np.zeros(steps + 1), np.zeros(steps + 1)
    snaps = {}
    for m in range(steps):
        t1 = TMIN + (m + 1) * dt
        th = t1 - 0.5 * dt
        r = diff_y(th) * dt / dy**2
        c = drift_y(y, th) * dt / (2 * dy)
        lo, di, up = -r / 2 - c / 2, 1 + r, -r / 2 + c / 2
        ab = np.zeros((3, n))
        ab[0, 1:] = up[1:-2]
        ab[1, :] = di
        ab[2, :-1] = lo[2:-1]

        b1 = bound(t1)
        bl, br = -free_kernel_np(t1, -b1), -free_kernel_np(t1, b1)
        ci = c[1:-1]
        rhs = (1 - r) * w[1:-1] + (r / 2 + ci / 2) * w[:-2] + (r / 2 - ci / 2) * w[2:]
        rhs[0] -= lo[1] * bl
        rhs[-1] -= up[-2] * br
        w[1:-1] = solve_banded((1, 1), ab, rhs)
        w[0], w[-1] = bl, br

        ts[m + 1] = t1
        wy_up = (3 * w[-1] - 4 * w[-2] + w[-3]) / (2 * dy)
        wy_lo = (-3 * w[0] + 4 * w[1] - w[2]) / (2 * dy)
        g_up[m + 1] = -0.5 * SIGMA**2 * (free_kernel_x_np(t1, b1) + wy_up / b1)
        g_lo[m + 1] = +0.5 * SIGMA**2 * (free_kernel_x_np(t1, -b1) + wy_lo / b1)
        if (m + 1) in snap_idx:
            snaps[snap_idx[m + 1]] = free_kernel_np(t1, b1 * y) + w

    return {"t": ts, "upper": g_up, "lower": g_lo, "y": y, "snapshots": snaps}


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
    "tinf": ("T_INF", [2, 3, 6, 8], "T_{\\infty}"),
}
RUN = ["tinf"]


def update_derived():
    """Quantities computed from T_INF must follow it."""
    global B_DOT
    B_DOT = -B0 / (2 * T_INF)
    assert TMAX < 2 * T_INF, "the boundaries collapse at t = 2 T_INF"
N_T = 100                        # PINN markers

FIGSIZE, DPI = (12, 8), 90
FONT = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16}
TICK_SIZE, LEGEND_SIZE = 14, 16
LINESTYLES = ['-', '--', ':', '-.']
MARKERS = ['o', 's', '^', 'd']


def run_study(name):
    """Train one PINN per value. Results (densities and errors) are saved after
    every run in collapsing_study_<name>.npz, so a later error never loses finished runs."""
    var, values, _ = STUDIES[name]
    base = globals()[var]
    tspace = np.linspace(TMIN, TMAX, N_T + 1)
    res = {"upper": [], "lower": [], "fd_upper": [], "fd_lower": [], "errors": []}
    fd_t = None

    for i, val in enumerate(values):
        globals()[var] = val                  # bound(), drift_y(), diff_y() read the globals
        update_derived()
        torch.manual_seed(SEED)
        print(f"\n===== {name} = {val}   (v = {V}, B0 = {B0}, collapse at t = {2 * T_INF}) =====")
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
            np.savez(f"collapsing_study_{name}.npz", values=np.array(values[:i + 1]), tspace=tspace,
                     fd_t=fd_t, **{k: np.array(a) for k, a in res.items()})
        except OSError as e:
            print(f"[warning] results not saved: {e}")

    globals()[var] = base                     # restore the base value
    update_derived()
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
    sym = STUDIES[name][0]                    # plain variable name for the text table
    print(f"\n{sym:<8}{'L2 upper':>12}{'Linf upper':>12}{'L2 lower':>12}{'Linf lower':>12}")
    for val, e in zip(values, res["errors"]):
        print(f"{val:<8}" + "".join(f"{x:>12.2e}" for x in e))


def main():
    print(f"Device: {DEVICE}")
    for name in RUN:
        tspace, fd_t, res = run_study(name)
        values = STUDIES[name][1]
        print_error_table(name, values, res)
        plot_study(name, values, tspace, fd_t, res, "upper", f"collapsing_fpt_upper_{name}.pdf")
        plot_study(name, values, tspace, fd_t, res, "lower", f"collapsing_fpt_lower_{name}.pdf")


if __name__ == "__main__":
    main()
