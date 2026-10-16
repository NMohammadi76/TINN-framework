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
# Finite-difference reference (Crank-Nicolson on the smooth part w)
# =============================================================================

def finite_difference_solver(nx=801, dt=1e-3, snapshot_times=(0.5, 1.0, 3.0)):
    """
    Same singularity subtraction as the TINN: CN is applied to w (smooth,
    w(0)=0, w = -K on the walls) and u = K + w. Banded solve, BCs inside the
    linear system, one-sided O(dx^2) derivatives with the true grid spacing.
    """
    x = np.linspace(XMIN, XMAX, nx)
    dx = x[1] - x[0]
    r = D * dt / dx**2
    c = V * dt / (2 * dx)
    lo, di, up = -r / 2 - c / 2, 1 + r, -r / 2 + c / 2   # LHS coefficients of w_{i-1}, w_i, w_{i+1}

    n = nx - 2
    ab = np.zeros((3, n))
    ab[0, 1:] = up
    ab[1, :] = di
    ab[2, :-1] = lo

    steps = int(round((TMAX - TMIN) / dt))
    w = np.zeros(nx)
    ts = np.zeros(steps + 1)
    g_up = np.zeros(steps + 1)
    g_lo = np.zeros(steps + 1)
    snaps = {}
    snap_idx = {int(round(s / dt)): s for s in snapshot_times}

    for m in range(steps):
        t1 = TMIN + (m + 1) * dt
        bl, br = -free_kernel_np(t1, XMIN), -free_kernel_np(t1, XMAX)
        rhs = (1 - r) * w[1:-1] + (r / 2 + c / 2) * w[:-2] + (r / 2 - c / 2) * w[2:]
        rhs[0] -= lo * bl
        rhs[-1] -= up * br
        w[1:-1] = solve_banded((1, 1), ab, rhs)
        w[0], w[-1] = bl, br

        wx_up = (3 * w[-1] - 4 * w[-2] + w[-3]) / (2 * dx)
        wx_lo = (-3 * w[0] + 4 * w[1] - w[2]) / (2 * dx)
        ts[m + 1] = t1
        g_up[m + 1] = -D * (free_kernel_x_np(t1, XMAX) + wx_up)
        g_lo[m + 1] = +D * (free_kernel_x_np(t1, XMIN) + wx_lo)
        if (m + 1) in snap_idx:
            snaps[snap_idx[m + 1]] = free_kernel_np(t1, x) + w

    return {"t": ts, "upper": g_up, "lower": g_lo, "x": x, "snapshots": snaps}


# =============================================================================
# Plots and error metrics
# =============================================================================

def rel_l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def error_norms(pred, ref, grid):
    """||e||_2 = sqrt( integral e^2 ), ||e||_inf = max|e|, and the relative L2 error."""
    e = pred - ref
    l2 = np.sqrt(_trapz(e**2, grid))
    linf = np.abs(e).max()
    rel = l2 / np.sqrt(_trapz(ref**2, grid))
    return l2, linf, rel


def compute_error_table(model, fd):
    """L2 / L-infinity errors of PINN vs FD, PINN vs exact and FD vs exact for
    the densities at t = 0.5, 1, 3 and for both FPT densities."""
    rows = []
    x = fd["x"]
    for t in (0.5, 1.0, 3.0):
        u_pinn = tinn_density(model, t, x)
        u_fd = fd["snapshots"][t]
        u_ex = exact_density(t, x)
        rows.append((f"u(x, t={t})", "PINN vs FD", *error_norms(u_pinn, u_fd, x)))
        rows.append((f"u(x, t={t})", "PINN vs Exact", *error_norms(u_pinn, u_ex, x)))
        rows.append((f"u(x, t={t})", "FD vs Exact", *error_norms(u_fd, u_ex, x)))

    # FPT on the FD time grid (t >= 0.01; below that the densities are < 1e-80)
    mask = fd["t"] >= 0.01
    tt = fd["t"][mask]
    g_pinn = tinn_fpt(model, tt)
    g_ex = exact_fpt(tt)
    for side in ("upper", "lower"):
        g_fd = fd[side][mask]
        name = f"FPT {side}"
        rows.append((name, "PINN vs FD", *error_norms(g_pinn[side], g_fd, tt)))
        rows.append((name, "PINN vs Exact", *error_norms(g_pinn[side], g_ex[side], tt)))
        rows.append((name, "FD vs Exact", *error_norms(g_fd, g_ex[side], tt)))
    return rows


def report_error_table(rows, csv_file="error_norms.csv", tex_file="error_norms.tex"):
    header = f"{'Quantity':<16}{'Comparison':<16}{'L2 norm':>12}{'Linf norm':>12}{'rel. L2':>12}"
    print("\n" + header)
    print("-" * len(header))
    for q, c, l2, linf, rel in rows:
        print(f"{q:<16}{c:<16}{l2:>12.3e}{linf:>12.3e}{rel:>12.3e}")

    with open(csv_file, "w") as f:
        f.write("quantity,comparison,L2,Linf,rel_L2\n")
        for q, c, l2, linf, rel in rows:
            f.write(f"{q},{c},{l2:.6e},{linf:.6e},{rel:.6e}\n")

    def sci(v):
        m, e = f"{v:.2e}".split("e")
        return f"${m}\\times 10^{{{int(e)}}}$"

    with open(tex_file, "w") as f:
        f.write("\\begin{tabular}{llccc}\n\\hline\n")
        f.write("Quantity & Comparison & $\\|e\\|_2$ & $\\|e\\|_\\infty$ & Rel. $L_2$ \\\\\n\\hline\n")
        for q, c, l2, linf, rel in rows:
            f.write(f"{q} & {c} & {sci(l2)} & {sci(linf)} & {sci(rel)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print(f"Saved {csv_file} and {tex_file}")


FONT = {
    'family': 'serif',
    'color': 'black',
    'weight': 'normal',
    'size': 16,
}


def plot_training_history(history):
    """Same style as the original report figure.
    IC and BC are hard constraints now (exactly zero), so only the
    theoretical (PDE) loss -- which is also the total loss -- is drawn."""
    plt.figure(figsize=(10, 8))

    plt.plot(history[100:], label='Total loss (= Theoretical loss)',
             color='black', linestyle='-')

    plt.xlabel('Number of iterations', fontsize=16)
    plt.ylabel('MSE', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)
    plt.savefig("loss_plot.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_solution_surface(model):
    """3D surface, same style as the original (t starts slightly above 0,
    because u(0,x) is a Dirac delta)."""
    N = 500
    tspace = np.linspace(0.05, TMAX, N + 1)
    xspace = np.linspace(XMIN, XMAX, N + 1)
    T, X = np.meshgrid(tspace, xspace)
    U = np.stack([tinn_density(model, t, xspace) for t in tspace], axis=1)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, U, cmap='gray')

    ax.set_xlabel('$t$', fontsize=16)
    ax.set_ylabel('$x$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.savefig("surface_plot_bw.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_solution_slices(model, fd):
    """Density at t = 0.5, 1, 3: TINN (lines) and FD (hollow markers) together,
    same black-and-white style as the original."""
    xspace = fd["x"]
    time_points = [0.5, 1.0, 3.0]
    linestyles = ['-', '--', ':']
    markers = ['o', 's', '^']
    every = 25                       # draw every 25th FD grid point as a marker

    plt.figure(figsize=(12, 8), dpi=90)

    for t, linestyle, marker in zip(time_points, linestyles, markers):
        upred = tinn_density(model, t, xspace)
        plt.plot(xspace, upred, color='black', linestyle=linestyle,
                 label=f'TINN, t = {t}')
        plt.plot(xspace[::every], fd["snapshots"][t][::every], linestyle='none',
                 marker=marker, markersize=7, markerfacecolor='none',
                 markeredgecolor='black', label=f'FD, t = {t}')

    plt.xlabel('x', fontdict=FONT)
    plt.ylabel('Probability', fontdict=FONT)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig("solution_slices.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_first_passage_time_comparison(J1_tinn, J1_fd, J1_exact, tspace,
                                       filename="first_passage_time_comparison.pdf"):
    """FPT density at one boundary, same style as the original figure
    (the exact solution is added as a thin solid black line)."""
    plt.figure(figsize=(12, 8), dpi=90)

    plt.scatter(tspace, J1_tinn, label='TINN Approximation',
                color='black', marker='o', s=50)
    plt.plot(tspace, J1_fd, label='Finite Difference Solution',
             color='black', linestyle='--', linewidth=2)
    plt.plot(tspace, J1_exact, label='Exact Solution',
             color='black', linestyle='-', linewidth=1)

    plt.xlabel('Time', fontdict=FONT)
    plt.ylabel('First Passage Time Probability', fontdict=FONT)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    model = TINN().to(DEVICE)
    history = train_model(model)

    print("Finite-difference reference ...")
    fd = finite_difference_solver()

    # error norms (printed + saved as CSV and LaTeX table)
    rows = compute_error_table(model, fd)
    report_error_table(rows)

    # mass balance: survival + absorbed = 1
    t_eval = np.linspace(0.01, TMAX, 500)
    g_tinn = tinn_fpt(model, t_eval)
    xx = np.linspace(XMIN, XMAX, 2001)
    survival = _trapz(tinn_density(model, TMAX, xx), xx)
    absorbed = _trapz(g_tinn["upper"] + g_tinn["lower"], t_eval)
    print(f"Mass check at T: survival {survival:.4f} + absorbed {absorbed:.4f} = {survival + absorbed:.4f}")

    # report figures (same layout as the original code)
    plot_training_history(history)
    plot_solution_surface(model)
    plot_solution_slices(model, fd)

    tspace = np.linspace(TMIN, TMAX, 51)          # 51 points as in the original
    g_pinn = tinn_fpt(model, tspace)
    g_exact = exact_fpt(tspace[1:])
    for side, fname in (("upper", "first_passage_time_comparison.pdf"),
                        ("lower", "first_passage_time_comparison_lower.pdf")):
        J_fd = np.interp(tspace, fd["t"], fd[side])
        J_exact = np.r_[0.0, g_exact[side]]        # g(0) = 0
        plot_first_passage_time_comparison(g_pinn[side], J_fd, J_exact, tspace, fname)

    torch.save(model.state_dict(), "ddm_tinn.pt")
    print("Done.")


if __name__ == "__main__":
    main()
