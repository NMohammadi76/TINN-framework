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

This script: fixed drift V, symmetric boundaries x = -a and x = +a for every a in
A_LIST; one TINN per boundary; FPT densities at the upper and lower boundary for
all a, together with the exact solution (black-and-white, 12 x 8).
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
V = 0.3          # drift (fixed)
SIGMA = 1.0      # noise
D = 0.5 * SIGMA**2
X0 = 0.0         # starting point
XMIN, XMAX = -2.0, 2.0   # absorbing boundaries (overwritten for every a in A_LIST)
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
# Boundary study (fixed drift)
# =============================================================================

A_LIST = [1.0, 1.5, 2.0, 2.5]    # boundaries at x = -a and x = +a (start point X0 = 0)
N_T = 50                         # number of time points for the TINN markers

FIGSIZE, DPI = (12, 8), 90
FONT = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16}
TICK_SIZE, LEGEND_SIZE = 14, 14
LINESTYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D']


def set_boundaries(a):
    """Set the walls to -a, +a (all functions read these globals)."""
    global XMIN, XMAX, L, Z
    XMIN, XMAX = -a, a
    L = XMAX - XMIN
    Z = X0 - XMIN


def error_norms(pred, ref, grid):
    e = pred - ref
    return np.sqrt(_trapz(e**2, grid)), np.abs(e).max()


def run_boundary_study(a_list=A_LIST):
    """Train one TINN per boundary value. Results are saved after every run
    (boundary_study.npz), so a later error never loses finished runs."""
    tspace = np.linspace(TMIN, TMAX, N_T + 1)
    t_fine = np.linspace(TMIN, TMAX, 1001)
    res = {"upper": [], "lower": [], "exact_upper": [], "exact_lower": []}

    for i, a in enumerate(a_list):
        set_boundaries(a)
        torch.manual_seed(SEED)
        print(f"\n===== boundaries x = -{a}, +{a}  (v = {V}) =====")
        model = TINN().to(DEVICE)
        train_model(model)

        g_pinn = tinn_fpt(model, tspace)
        g_ex = exact_fpt(t_fine[1:])
        for side in ("upper", "lower"):
            res[side].append(g_pinn[side])
            res["exact_" + side].append(np.r_[0.0, g_ex[side]])     # g(0) = 0

        tt = np.linspace(0.01, TMAX, 1000)
        g_p, g_e = tinn_fpt(model, tt), exact_fpt(tt)
        for side in ("upper", "lower"):
            l2, linf = error_norms(g_p[side], g_e[side], tt)
            print(f"a = {a}: FPT {side:5s}  L2 = {l2:.3e}, Linf = {linf:.3e}")

        try:
            np.savez("boundary_study.npz", a=np.array(a_list[:i + 1]), tspace=tspace,
                     t_fine=t_fine, **{k: np.array(v) for k, v in res.items()})
        except OSError as e:
            print(f"[warning] results not saved: {e}")

    return tspace, t_fine, res


def plot_boundary_study(a_list, tspace, t_fine, res, side, filename):
    """Exact solution (lines) and TINN (hollow markers) for every boundary."""
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    for i, a in enumerate(a_list):
        plt.plot(t_fine, res["exact_" + side][i], color='black',
                 linestyle=LINESTYLES[i % 4], linewidth=1.3,
                 label=f'Exact solution, $a = {a}$')
        plt.plot(tspace, res[side][i], linestyle='none', marker=MARKERS[i % 4],
                 markersize=6, markerfacecolor='none', markeredgecolor='black',
                 label=f'TINN approximation, $a = {a}$')

    ymax = max(np.max(res["exact_" + side][i]) for i in range(len(a_list)))
    plt.ylim(top=1.4 * ymax)                    # room for the legend above the curves

    plt.xlabel('Time', fontdict=FONT)
    plt.ylabel('First Passage Time Probability', fontdict=FONT)
    plt.legend(fontsize=LEGEND_SIZE, ncol=2, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


def main():
    print(f"Device: {DEVICE}")
    tspace, t_fine, res = run_boundary_study()
    plot_boundary_study(A_LIST, tspace, t_fine, res, "upper", "fpt_upper_boundaries.pdf")
    plot_boundary_study(A_LIST, tspace, t_fine, res, "lower", "fpt_lower_boundaries.pdf")


if __name__ == "__main__":
    main()
