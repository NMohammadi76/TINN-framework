"""
Standard DDM (Fokker-Planck, two absorbing boundaries) -- PINN in PyTorch.
Error of the PINN vs the exact solution for different numbers of
collocation points in space (L_inf and L_2 on twin axes).

u = K + w : K = free-space Gaussian kernel (exact delta initial condition),
            w = smooth part learned by the network; IC and BC are hard constraints.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 1.x / 2.x

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1234

# ---- DDM parameters
V, SIGMA = 0.3, 1.0
D = 0.5 * SIGMA**2
X0 = 0.0
XMIN, XMAX = -2.0, 2.0
TMIN, TMAX = 0.0, 5.0

# ---- Network / training
HIDDEN_LAYERS, WIDTH = 5, 50
ADAM_ITERS, ADAM_LR = 10000, 1e-3
LBFGS_ITERS = 1000

# ---- Study
NX_LIST = [5,10, 20, 40, 80]   # collocation points in space
NT = 50                      # collocation points in time (fixed)
ERROR_TARGET = "density"     # "density", "fpt_upper" or "fpt_lower"
N_MODES = 400                # terms of the exact series


# =============================================================================
# Model
# =============================================================================

def free_kernel(t, x):
    t = t.clamp_min(1e-10)
    var = SIGMA**2 * t
    return torch.exp(-(x - X0 - V * t) ** 2 / (2 * var)) / torch.sqrt(2 * math.pi * var)


def free_kernel_x(t, x):
    t = t.clamp_min(1e-10)
    return -(x - X0 - V * t) / (SIGMA**2 * t) * free_kernel(t, x)


class TINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers, d_in = [], 2
        for _ in range(HIDDEN_LAYERS):
            layers += [nn.Linear(d_in, WIDTH), nn.Tanh()]
            d_in = WIDTH
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def w(self, t, x):
        t_hat = 2.0 * (t - TMIN) / (TMAX - TMIN) - 1.0
        x_hat = 2.0 * (x - XMIN) / (XMAX - XMIN) - 1.0
        n = self.net(torch.cat([t_hat, x_hat], dim=1))
        k_r = free_kernel(t, torch.full_like(x, XMAX))
        k_l = free_kernel(t, torch.full_like(x, XMIN))
        B = -(k_r * (x - XMIN) + k_l * (XMAX - x)) / (XMAX - XMIN)
        bubble = 4.0 * (x - XMIN) * (XMAX - x) / (XMAX - XMIN) ** 2
        return B + ((t - TMIN) / (TMAX - TMIN)) * bubble * n

    def forward(self, t, x):
        return free_kernel(t, x) + self.w(t, x)


def pde_residual(model, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)
    w = model.w(t, x)
    w_t, w_x = torch.autograd.grad(w, (t, x), torch.ones_like(w), create_graph=True)
    w_xx = torch.autograd.grad(w_x, x, torch.ones_like(w_x), create_graph=True)[0]
    return w_t + V * w_x - D * w_xx


def grid_collocation(nx, nt):
    """nx interior points in x  x  nt points in (0, T]."""
    x = np.linspace(XMIN, XMAX, nx + 2)[1:-1]
    t = np.linspace(TMIN, TMAX, nt + 1)[1:]
    T, X = np.meshgrid(t, x, indexing="ij")
    to_t = lambda a: torch.tensor(a.reshape(-1, 1), device=DEVICE)
    return to_t(T), to_t(X)


def train(model, t, x):
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-2) ** (1.0 / ADAM_ITERS))
    for _ in range(ADAM_ITERS):
        opt.zero_grad(set_to_none=True)
        loss = pde_residual(model, t, x).pow(2).mean()
        loss.backward()
        opt.step()
        sched.step()

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_ITERS,
                              history_size=50, tolerance_grad=1e-12,
                              tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        loss = pde_residual(model, t, x).pow(2).mean()
        loss.backward()
        return loss

    lbfgs.step(closure)


# =============================================================================
# Exact solution and errors
# =============================================================================

L, Z = XMAX - XMIN, X0 - XMIN


def exact_density(t, x):
    k = np.arange(1, N_MODES + 1)
    xs = np.asarray(x) - XMIN
    coef = np.sin(k * np.pi * Z / L) * np.exp(-k**2 * np.pi**2 * SIGMA**2 * t / (2 * L**2))
    s = (coef[None, :] * np.sin(np.outer(xs, k) * np.pi / L)).sum(1)
    return np.exp(V * (xs - Z) / SIGMA**2 - V**2 * t / (2 * SIGMA**2)) * (2 / L) * s


def exact_fpt(t, side):
    t = np.asarray(t, float)[:, None]
    k = np.arange(1, N_MODES + 1)[None, :]
    decay = np.exp(-k**2 * np.pi**2 * SIGMA**2 * t / (2 * L**2))
    sz = np.sin(k * np.pi * Z / L)
    common = np.pi * SIGMA**2 / L**2 * np.exp(-V**2 * t[:, 0] / (2 * SIGMA**2))
    if side == "upper":
        return common * np.exp(V * (L - Z) / SIGMA**2) * (k * (-1.0) ** (k + 1) * sz * decay).sum(1)
    return common * np.exp(-V * Z / SIGMA**2) * (k * sz * decay).sum(1)


def pinn_density(model, t, x):
    tt = torch.full((len(x), 1), float(t), device=DEVICE)
    xx = torch.as_tensor(x, device=DEVICE).reshape(-1, 1)
    with torch.no_grad():
        return model(tt, xx).cpu().numpy().ravel()


def pinn_fpt(model, t, side):
    tt = torch.as_tensor(t, device=DEVICE).reshape(-1, 1)
    xb, sign = (XMAX, -1.0) if side == "upper" else (XMIN, +1.0)
    x = torch.full_like(tt, xb).requires_grad_(True)
    w = model.w(tt, x)
    w_x = torch.autograd.grad(w, x, torch.ones_like(w))[0]
    return (sign * D * (free_kernel_x(tt, x) + w_x)).detach().cpu().numpy().ravel()


def errors(model):
    """L2 = sqrt(integral e^2), Linf = max|e|, against the exact solution."""
    if ERROR_TARGET == "density":
        tt = np.linspace(0.1, TMAX, 50)
        xx = np.linspace(XMIN, XMAX, 201)
        E = np.stack([pinn_density(model, t, xx) - exact_density(t, xx) for t in tt])
        return np.sqrt(_trapz(_trapz(E**2, xx, axis=1), tt)), np.abs(E).max()
    side = ERROR_TARGET.split("_")[1]
    tt = np.linspace(0.01, TMAX, 500)
    e = pinn_fpt(model, tt, side) - exact_fpt(tt, side)
    return np.sqrt(_trapz(e**2, tt)), np.abs(e).max()


# =============================================================================
# Plot
# =============================================================================

def plot_errors(nx_list, l2_list, linf_list, filename="collocation_study.pdf"):
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=90)

    ax1.plot(nx_list, linf_list, color='black', linestyle='-', marker='o',
             markersize=6, label=r'$L_\infty$ error')
    ax1.set_xlabel('Number of collocation points on space', fontsize=14)
    ax1.set_ylabel(r'$L_\infty$ error', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax1.legend(loc='upper left', fontsize=14)

    ax2 = ax1.twinx()
    ax2.plot(nx_list, l2_list, color='black', linestyle='--', marker='s',
             markersize=6, label=r'$L_2$ error')
    ax2.set_ylabel(r'$L_2$ error', fontsize=14)
    ax2.tick_params(axis='y', which='major', labelsize=13)
    ax2.legend(loc='upper right', fontsize=14)

    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    l2_list, linf_list = [], []
    for nx in NX_LIST:
        torch.manual_seed(SEED)              # same initial weights in every run
        model = TINN().to(DEVICE)
        train(model, *grid_collocation(nx, NT))
        l2, linf = errors(model)
        l2_list.append(l2)
        linf_list.append(linf)
        print(f"Nx = {nx:3d}:  L2 = {l2:.3e},  Linf = {linf:.3e}")

    plot_errors(NX_LIST, l2_list, linf_list)


if __name__ == "__main__":
    main()
