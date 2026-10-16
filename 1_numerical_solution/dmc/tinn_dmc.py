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
TAU = 150.0      # time constant tau of the automatic process
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


def print_errors(model, fd):
    print(f"\n{'Quantity':<14}{'L2 error':>12}{'Linf error':>12}   (PINN vs FD reference)")
    for t in SNAP_TIMES:
        l2, linf = error_norms(pinn_density(model, t, fd["x"]), fd["snapshots"][t], fd["x"])
        print(f"{f'u(x, t={t})':<14}{l2:>12.3e}{linf:>12.3e}")
    mask = fd["t"] >= 0.01
    tt = fd["t"][mask]
    g = pinn_fpt(model, tt)
    for side in ("upper", "lower"):
        l2, linf = error_norms(g[side], fd[side][mask], tt)
        print(f"{f'FPT {side}':<14}{l2:>12.3e}{linf:>12.3e}")


# =============================================================================
# Plots (format of the original figure)
# =============================================================================

FIGSIZE, DPI = (12, 8), 90
FONT = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 17}
LEGEND_SIZE, TICK_SIZE = 18, 16


def plot_density(model, fd, filename="ex4-3.pdf"):
    plt.figure(figsize=FIGSIZE, dpi=DPI)

    # FD lines
    for t, ls in zip(SNAP_TIMES, ['-', '--', '-.']):
        plt.plot(fd["x"], fd["snapshots"][t], label=f't = {t} (FD method)',
                 color='black', linestyle=ls)

    # PINN points
    xspace = np.linspace(XMIN, XMAX, 101)
    for t, marker in zip(SNAP_TIMES, ['o', 's', '^']):
        plt.scatter(xspace, pinn_density(model, t, xspace),
                    label=f't = {t} (Presented method)', color='black', marker=marker)

    plt.xlabel('x', fontdict=FONT)
    plt.ylabel('Probability', fontdict=FONT)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


def plot_fpt(model, fd, side, filename):
    tspace = np.linspace(TMIN, TMAX, 101)
    g_pinn = pinn_fpt(model, tspace)[side]

    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.plot(fd["t"], fd[side], label='FD method', color='black', linestyle='-')
    plt.scatter(tspace, g_pinn, label='Presented method', color='black', marker='o')

    plt.xlabel('Time', fontdict=FONT)
    plt.ylabel('First Passage Time Probability', fontdict=FONT)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


def plot_loss(history, filename="ex4-7.pdf"):
    """IC and BC are hard constraints (their losses are exactly zero), so the
    total loss equals the physical (PDE) loss and is drawn as one curve."""
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.plot(history[100:], label='Total loss (= Physical loss)',
             color='black', linestyle='-')

    plt.xlabel('Number of iterations', fontdict=FONT)
    plt.ylabel('MSE', fontdict=FONT)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


def plot_surface(model, filename="ex4-2.pdf"):
    """3D surface of the PINN density. t starts slightly above 0 because
    u(0, x) is a Dirac delta."""
    N = 500
    tspace = np.linspace(0.05, TMAX, N + 1)
    xspace = np.linspace(XMIN, XMAX, N + 1)
    T, X = np.meshgrid(tspace, xspace)
    U = np.stack([pinn_density(model, t, xspace) for t in tspace], axis=1)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, U, cmap='gray')

    ax.set_xlabel('$t$', fontdict=FONT, labelpad=14)
    ax.set_ylabel('$x$', fontdict=FONT, labelpad=14)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    model = PINN().to(DEVICE)
    history = train_model(model)
    try:
        torch.save(model.state_dict(), "ddm_dmc_pinn.pt")   # saved right after training
        np.save("ddm_dmc_loss.npy", np.array(history))
    except OSError as e:
        print(f"[warning] model not saved: {e}")

    print("FD reference ...")
    fd = finite_difference_solver()

    print_errors(model, fd)

    plot_loss(history, "ex4-7.pdf")
    plot_surface(model, "ex4-2.pdf")
    plot_density(model, fd, "ex4-3.pdf")
    plot_fpt(model, fd, "upper", "ex4-1.pdf")
    plot_fpt(model, fd, "lower", "ex4-1-lower.pdf")
    return model


if __name__ == "__main__":
    model = main()
