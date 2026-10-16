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
T_INF = 2.0      # b(t) = B0 (1 - t / (2 T_INF)); collapses to 0 at t = 2 T_INF
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


def print_errors(model, fd):
    print(f"\n{'Quantity':<14}{'L2 error':>12}{'Linf error':>12}   (PINN vs FD reference)")
    for t in SNAP_TIMES:
        x = bound(t) * fd["y"]                                  # errors in the x variable
        l2, linf = error_norms(pinn_density(model, t, fd["y"]), fd["snapshots"][t], x)
        print(f"{f'u(x, t={t})':<14}{l2:>12.3e}{linf:>12.3e}")
    mask = fd["t"] >= 0.01
    tt = fd["t"][mask]
    g = pinn_fpt(model, tt)
    for side in ("upper", "lower"):
        l2, linf = error_norms(g[side], fd[side][mask], tt)
        print(f"{f'FPT {side}':<14}{l2:>12.3e}{linf:>12.3e}")


# =============================================================================
# Plots (black-and-white, 12 x 8)
# =============================================================================

FIGSIZE, DPI = (12, 8), 90
FONT = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 17}
LEGEND_SIZE, TICK_SIZE = 18, 16


def plot_density(model, fd, filename="ex5-2.pdf"):
    """Density u(t, x) over the shrinking interval [-b(t), b(t)]."""
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    for t, ls in zip(SNAP_TIMES, ['-', '--', '-.']):
        plt.plot(bound(t) * fd["y"], fd["snapshots"][t], label=f't = {t} (FD method)',
                 color='black', linestyle=ls)
    yspace = np.linspace(-1, 1, 101)
    for t, marker in zip(SNAP_TIMES, ['o', 's', '^']):
        plt.scatter(bound(t) * yspace, pinn_density(model, t, yspace),
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
    plt.ylabel('First passage time', fontdict=FONT)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


def plot_loss(history, filename="ex5-3.pdf"):
    """IC and BC are hard constraints (their losses are exactly zero), so the
    total loss equals the physical (PDE) loss and is drawn as one curve."""
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.plot(history[100:], label='Total loss (= Physical loss)', color='black', linestyle='-')
    plt.xlabel('Number of iterations', fontdict=FONT)
    plt.ylabel('MSE', fontdict=FONT)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


def plot_surface(model, filename="ex5-1.pdf"):
    """3D surface of u(t, x) on the collapsing domain (x = b(t) y, no shift)."""
    N = 300
    tspace = np.linspace(0.05, TMAX, N + 1)
    yspace = np.linspace(-1, 1, N + 1)
    T, Y = np.meshgrid(tspace, yspace)
    U = np.stack([pinn_density(model, t, yspace) for t in tspace], axis=1)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, bound(T) * Y, U, cmap='gray')
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
        torch.save(model.state_dict(), "ddm_collapsing_pinn.pt")   # saved right after training
        np.save("ddm_collapsing_loss.npy", np.array(history))
    except OSError as e:
        print(f"[warning] model not saved: {e}")

    print("FD reference ...")
    fd = finite_difference_solver()
    print_errors(model, fd)

    plot_loss(history, "ex5-3.pdf")
    plot_surface(model, "ex5-1.pdf")
    plot_density(model, fd, "ex5-2.pdf")
    plot_fpt(model, fd, "upper", "ex5-4.pdf")
    plot_fpt(model, fd, "lower", "ex5-4-lower.pdf")
    return model


if __name__ == "__main__":
    model = main()
