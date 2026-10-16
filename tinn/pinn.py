"""
Shared PINN core: solves the Fokker-Planck (forward Kolmogorov) equation

    dp/dt = -d/dx [ mu(x,t) p ] + D d2p/dx2,      D = sigma^2 / 2
    p(x, 0) = delta(x - x0),   p(l(t), t) = p(u(t), t) = 0   (absorbing)

for any model in models.py, and turns the solution into the first-passage-time
(FPT) densities, i.e. the RT/choice likelihood.

Ansatz (removes the delta initial condition and enforces the boundaries exactly):

    p(x,t) = G(x,t) * h(x,t),     G = N(x; m(t), s2(t))  (free solution)
    h(x,t) = B(x,t) * exp(w_NN(xi, t, c)),          xi = (x - l) / (u - l)
    B(x,t) = [1 - exp(-2 (u-x)(u0-x0) / (sigma^2 t))] [1 - exp(-2 (x-l)(x0-l0) / (sigma^2 t))]
    (u0, l0 = bounds at t = 0; exact near each bound for fixed or linearly moving bounds)

B is the method-of-images factor: it vanishes on both bounds, tends to 1 inside
as t -> 0, and already contains the sharp boundary layer of small times, so the
network only learns the smooth correction w (w -> 0 as t -> 0).

Because G solves the free Fokker-Planck equation, h obeys

    h_t + [mu(x,t) + 2 D (x - m) / s2] h_x - D h_xx = 0,     h(x, 0+) = 1,

which is what the network is trained on.  The FPT densities are the
probability fluxes through the bounds:

    g_u(t) = -D p_x(u,t) = -D G(u,t) B_x(u,t) exp(w(1,t))
    g_l(t) =  D p_x(l,t) =  D G(l,t) B_x(l,t) exp(w(0,t))
"""
import math
import torch
import torch.nn as nn

from models import SIGMA

D = 0.5 * SIGMA ** 2


class MLP(nn.Module):
    def __init__(self, n_in=4, width=64, depth=4):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.f = nn.Sequential(*layers)

    def forward(self, z):
        return self.f(z).squeeze(-1)


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def log_gauss(x, m, s2):
    return -0.5 * torch.log(2 * math.pi * s2) - (x - m) ** 2 / (2 * s2)


def image_factor(x, t, l, u, l0, u0, x0=0.0):
    """Product of the two single-boundary image solutions.  For a bound moving
    at constant speed the exact single-boundary factor uses the bound's
    *initial* distance to x0 (u0 - x0) and its *current* distance to x."""
    s2 = SIGMA ** 2 * t
    return (-torch.expm1(-2 * (u - x) * (u0 - x0) / s2)) * (-torch.expm1(-2 * (x - l) * (x0 - l0) / s2))


def log_image_slope(t, l, u, l0, u0, upper, x0=0.0):
    """log |dB/dx| at the upper (upper=True) or lower bound."""
    s2 = SIGMA ** 2 * t
    d_near = torch.where(upper, u0 - x0, x0 - l0)
    d_far = torch.where(upper, x0 - l0, u0 - x0)
    return torch.log(2 * d_near / s2) + torch.log(-torch.expm1(-2 * (u - l) * d_far / s2))


class FPTSolver(nn.Module):
    """PINN for p(x,t) of one decision model (all its conditions at once)."""

    def __init__(self, model, t_scale, width=64, depth=4, t_min=5e-3):
        super().__init__()
        self.model = model
        self.t_scale = t_scale
        self.t_min = t_min
        # the model parameters are network inputs, so the PINN learns the
        # solution in a neighbourhood of theta and d(solution)/d(theta) is exact
        self.pkeys = list(model.param_specs)
        self.pscale = {k: max(abs(model.param_specs[k][2]), 0.1) for k in self.pkeys}
        self.net = MLP(4 + len(self.pkeys), width, depth)

    # -- network ------------------------------------------------------------
    def w(self, xi, t, c, p):
        tn = t / self.t_scale
        feats = [2 * xi - 1, tn, torch.sqrt(tn), c]
        feats += [p[k] / self.pscale[k] + 0 * t for k in self.pkeys]
        return self.net(torch.stack(feats, -1))

    # -- physics loss -------------------------------------------------------
    def pde_residual(self, p, xi, t, c):
        model = self.model
        t = t.clone().requires_grad_(True)
        l, u = model.bounds(t, p, c)
        x = (l + xi * (u - l)).detach().requires_grad_(True)
        xi_g = (x - l) / (u - l)
        l0, u0 = model.bounds(torch.zeros_like(t), p, c)
        B = image_factor(x, t, l, u, l0, u0, model.start(p))
        w = self.w(xi_g, t, c, p)
        m, s2 = self._moments(t, p, c)
        drift = self._drift(x, t, p, c)

        w_x, w_t = grad(w, x), grad(w, t)
        w_xx = grad(w_x, x)
        B_x, B_t = grad(B, x), grad(B, t)
        B_xx = grad(B_x, x)
        cvel = drift + 2 * D * (x - m) / s2
        # residual of the h-equation divided by exp(w)  (keeps it O(1))
        R = (B_t + B * w_t) + cvel * (B_x + B * w_x) \
            - D * (B_xx + 2 * B_x * w_x + B * (w_xx + w_x ** 2))
        return R * torch.sqrt(t / (t + 0.05))   # tame the 1/t term near t = 0

    def ic_loss(self, p, n):
        """At t_min the image factor is already the solution: w(x, t_min) = 0."""
        conds = torch.tensor(self.model.conditions)
        c = conds[torch.randint(len(conds), (n,))]
        t = torch.full((n,), self.t_min)
        xi = torch.rand(n)
        return (self.w(xi, t, c, p) ** 2).mean()

    # -- helpers that broadcast condition values ----------------------------
    def _per_cond(self, fn, c):
        out = None
        for cv in self.model.conditions:
            r = fn(cv)
            mask = (c == cv)
            if out is None:
                out = [torch.zeros_like(c) for _ in r]
            out = [torch.where(mask, ri + 0 * c, oi) for ri, oi in zip(r, out)]
        return out

    def _moments(self, t, p, c):
        return self._per_cond(lambda cv: self.model.free_moments(t, p, cv), c)

    def _drift(self, x, t, p, c):
        return self._per_cond(lambda cv: (self.model.drift(x, t, p, cv),), c)[0]

    # -- first-passage-time densities ---------------------------------------
    def log_fpt(self, p, t, c, upper):
        """log g_upper(t) or log g_lower(t); `upper` is a bool tensor."""
        l, u = self.model.bounds(t, p, c)
        l0, u0 = self.model.bounds(torch.zeros_like(t), p, c)
        m, s2 = self._moments(t, p, c)
        xb = torch.where(upper, u, l)
        xi = upper.to(t.dtype)
        return (math.log(D) + log_gauss(xb, m, s2) + log_image_slope(t, l, u, l0, u0, upper, self.model.start(p))
                + self.w(xi, t, c, p))

    def fpt_curves(self, p, t_end, n_grid=256):
        """Returns t grid and per-condition (g_upper, g_lower)."""
        t = torch.linspace(self.t_min, float(t_end), n_grid)
        out = []
        for cv in self.model.conditions:
            c = torch.full_like(t, cv)
            gu = self.log_fpt(p, t, c, torch.ones_like(t, dtype=torch.bool)).exp()
            gl = self.log_fpt(p, t, c, torch.zeros_like(t, dtype=torch.bool)).exp()
            out.append((gu, gl))
        return t, out
