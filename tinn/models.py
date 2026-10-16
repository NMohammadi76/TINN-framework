"""
Decision-model definitions for the PINN Fokker-Planck solver.

Every model is a 1-D diffusion of the decision variable X(t):

    dX = mu(X, t) dt + sigma dW,      X(0) = z  (z = 0 unless estimated, see with_free_start)

absorbed at a lower bound l(t) and an upper bound u(t).  Reaction time is
RT = T + t0, where T is the first-passage time and t0 the non-decision time.
sigma is fixed to 1 (scaling parameter), so all parameters are in "sigma = 1,
time in seconds" units.  Upper bound = choice 1, lower bound = choice 0.

Each model provides, besides drift and bounds, the mean m(t) and variance
s2(t) of the *free* (unbounded) process.  For drifts that are linear in X these
are available in closed form and the free density G(x,t) = N(x; m, s2) solves
the Fokker-Planck equation exactly.  The PINN then only has to learn the smooth
ratio h = p / G, which removes the Dirac-delta initial condition.

Model equations (as in the project's LaTeX text):

1. Standard DDM          mu = v,                                bounds +/- a
2. Urgency gating        v(t) = v0 + v1 * t / (t + tau),        bounds +/- a
3. OU (leaky) model      mu(x) = -beta * x + v0  (= beta (mu - x)),  bounds +/- a
4. DMC (Ulrich et al. 2015; Richter et al. 2023)
      v(t) = mu0 + c * mu1(t),     c = +1 congruent, -1 incongruent
      mu1(t) = A e^{-t/tau} (t e / ((alpha-1) tau))^{alpha-1} ((alpha-1)/t - 1/tau)
      (alpha is the shape called "a" in the text; renamed so it does not clash
       with the bound a.  alpha = 2 gives A e^{-t/tau} (t e / tau)(1/t - 1/tau).)
      mu1 is the time derivative of the gamma-shaped activation
      x1(t) = A e^{-t/tau} (t e / ((alpha-1) tau))^{alpha-1}.
      bounds +/- a
5. Collapsing bounds     mu = v,  l(t) = -a + a t / (2 T_inf),
                                   u(t) =  a - a t / (2 T_inf)
                         (bounds meet at 0 when t = 2 T_inf)
"""
import math
import torch

SIGMA = 1.0


def softplus_inv(y):
    return math.log(math.expm1(y))


class DecisionModel:
    """Base class.  Subclasses set `name`, `param_specs`, `conditions`."""

    name = "base"
    # name -> (transform, true value used for simulation, initial guess for fitting)
    # transform: "pos" (softplus, > 0), "real" (identity), "gt1" (1 + softplus),
    #            "zrel" (a * tanh, a starting point strictly between the bounds),
    #            "tinf" (T_dmax / 2 + softplus, so the bounds meet after the longest decision),
    #            "box" (lo + (hi - lo) * sigmoid; spec = ("box", true, init, lo, hi))
    param_specs = {}
    conditions = [1.0]          # experimental conditions sharing the parameters
    fixed = {}                  # parameters held fixed (not estimated)

    # ---- parameter handling -------------------------------------------------
    def raw_init(self, which="init"):
        idx = 1 if which == "true" else 2
        raw = {}
        for k, spec in self.param_specs.items():
            val = spec[idx]
            if spec[0] == "pos":
                raw[k] = softplus_inv(val)
            elif spec[0] == "gt1":
                raw[k] = softplus_inv(val - 1)
            elif spec[0] == "tinf":
                raw[k] = None   # resolved in transform (needs data context)
            elif spec[0] == "zrel":
                raw[k] = math.atanh(val / self.param_specs["a"][idx])
            elif spec[0] == "box":
                lo, hi = spec[3], spec[4]
                f = min(max((val - lo) / (hi - lo), 0.02), 0.98)
                raw[k] = math.log(f / (1 - f))
            else:
                raw[k] = val
        return raw

    def true_params(self):
        """Constrained parameter dict with the true (simulation) values."""
        p = {k: torch.tensor(float(spec[1])) for k, spec in self.param_specs.items()}
        for k, v in self.fixed.items():
            p[k] = torch.tensor(float(v))
        return p

    def transform(self, raw, ctx=None):
        """raw tensors -> constrained parameter tensors."""
        p = {}
        for k, spec in self.param_specs.items():
            r = raw[k]
            if spec[0] == "pos":
                p[k] = torch.nn.functional.softplus(r)
            elif spec[0] == "gt1":
                p[k] = 1 + torch.nn.functional.softplus(r)
            elif spec[0] == "tinf":
                p[k] = 0.5 * ctx["t_dmax"] + torch.nn.functional.softplus(r)
            elif spec[0] == "zrel":
                continue
            elif spec[0] == "box":
                p[k] = spec[3] + (spec[4] - spec[3]) * torch.sigmoid(r)
            else:
                p[k] = r
        for k, spec in self.param_specs.items():
            if spec[0] == "zrel":
                p[k] = p["a"] * torch.tanh(raw[k])
        for k, v in self.fixed.items():
            p[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
        return p

    def with_free_start(self, z_true=0.1, z_init=0.0):
        """Also estimate the starting point z of p(x, 0) = delta(x - z)."""
        self.param_specs = dict(self.param_specs, z=("zrel", z_true, z_init))
        return self

    @staticmethod
    def start(p):
        """starting point z (0 unless z is a parameter)."""
        return p["z"] if "z" in p else 0.0

    # ---- dynamics (override) ------------------------------------------------
    def drift(self, x, t, p, c):
        raise NotImplementedError

    def free_moments(self, t, p, c):
        """mean and variance of the unbounded process started at x0 = 0."""
        raise NotImplementedError

    def bounds(self, t, p, c=None):
        return -p["a"] + 0 * t, p["a"] + 0 * t

    # ---- Monte-Carlo simulator (Euler-Maruyama) -----------------------------
    @torch.no_grad()
    def simulate(self, n, p, t0, dt=2e-4, t_max=5.0, seed=0):
        """Returns rt (s), choice (1 = upper, 0 = lower), condition index."""
        g = torch.Generator().manual_seed(seed)
        out_rt, out_ch, out_cond = [], [], []
        n_per = n // len(self.conditions)
        for ci, c in enumerate(self.conditions):
            x = torch.zeros(n_per) + self.start(p)
            T = torch.full((n_per,), float("nan"))
            ch = torch.full((n_per,), -1.0)
            alive = torch.ones(n_per, dtype=torch.bool)
            t = 0.0
            sq = SIGMA * math.sqrt(dt)
            while alive.any() and t < t_max:
                tm = torch.tensor(t + 0.5 * dt)
                x = x + self.drift(x, tm, p, c) * dt + sq * torch.randn(n_per, generator=g)
                t += dt
                l, u = self.bounds(torch.tensor(t), p, c)
                up = alive & (x >= u)
                lo = alive & (x <= l)
                T[up | lo] = t
                ch[up] = 1.0
                ch[lo] = 0.0
                alive = alive & ~(up | lo)
            keep = ~alive
            out_rt.append(T[keep] + t0)
            out_ch.append(ch[keep])
            out_cond.append(torch.full((int(keep.sum()),), ci, dtype=torch.long))
        return torch.cat(out_rt), torch.cat(out_ch), torch.cat(out_cond)


class StandardDDM(DecisionModel):
    name = "ddm"
    param_specs = {"v": ("real", 1.0, 0.3), "a": ("pos", 0.6, 0.5)}

    def drift(self, x, t, p, c):
        return p["v"] + 0 * x

    def free_moments(self, t, p, c):
        return self.start(p) + p["v"] * t, SIGMA ** 2 * t


class UrgencyDDM(DecisionModel):
    name = "urgency"
    param_specs = {"v0": ("real", 0.5, 0.2), "v1": ("real", 1.5, 0.8),
                   "tau": ("pos", 0.3, 0.5), "a": ("pos", 0.6, 0.5)}

    def drift(self, x, t, p, c):
        return p["v0"] + p["v1"] * t / (t + p["tau"]) + 0 * x

    def free_moments(self, t, p, c):
        tau = p["tau"]
        # integral of v1 * s / (s + tau) ds from 0 to t
        m = self.start(p) + p["v0"] * t + p["v1"] * (t - tau * torch.log1p(t / tau))
        return m, SIGMA ** 2 * t


class OUModel(DecisionModel):
    name = "ou"
    param_specs = {"v0": ("real", 1.0, 0.3), "beta": ("real", 1.5, 0.5),
                   "a": ("pos", 0.6, 0.5)}

    def drift(self, x, t, p, c):
        return -p["beta"] * x + p["v0"]

    def free_moments(self, t, p, c):
        b = p["beta"]
        bs = torch.where(b.abs() < 1e-4, torch.full_like(b, 1e-4), b)
        # m(t) = v0 (1 - e^{-bt}) / b,   s2(t) = sigma^2 (1 - e^{-2bt}) / (2b)
        m = self.start(p) * torch.exp(-bs * t) + p["v0"] * (-torch.expm1(-bs * t)) / bs
        s2 = SIGMA ** 2 * (-torch.expm1(-2 * bs * t)) / (2 * bs)
        return m, s2


class DMC(DecisionModel):
    """Diffusion Model for Conflict tasks; shape alpha fixed to 2 by default
    (DMC(free_shape=True) also estimates it)."""
    name = "dmc"
    param_specs = {"mu0": ("real", 2.5, 1.5), "A": ("pos", 0.15, 0.08),
                   "tau": ("pos", 0.06, 0.1), "a": ("pos", 0.7, 0.5)}
    conditions = [1.0, -1.0]      # congruent, incongruent
    fixed = {"alpha": 2.0}

    def __init__(self, free_shape=False):
        if free_shape:
            self.param_specs = dict(self.param_specs, alpha=("gt1", 2.0, 2.5))
            self.fixed = {}

    def x_auto(self, t, p):
        s, tau = p["alpha"], p["tau"]
        return p["A"] * torch.exp(-t / tau) * (math.e * t / ((s - 1) * tau)) ** (s - 1)

    def drift(self, x, t, p, c):
        s, tau = p["alpha"], p["tau"]
        t = t.clamp(min=1e-6)
        # mu1 = x1 * ((s-1)/t - 1/tau), written so it stays finite at t = 0 for s >= 2
        k = p["A"] * torch.exp(-t / tau) * (math.e / ((s - 1) * tau)) ** (s - 1)
        mu_a = k * ((s - 1) * t ** (s - 2) - t ** (s - 1) / tau)
        return p["mu0"] + c * mu_a + 0 * x

    def free_moments(self, t, p, c):
        return self.start(p) + p["mu0"] * t + c * self.x_auto(t, p), SIGMA ** 2 * t


class CollapsingBound(DecisionModel):
    name = "collapse"
    param_specs = {"v": ("real", 1.0, 0.3), "a": ("pos", 0.8, 0.6),
                   "T_inf": ("tinf", 1.25, 1.0)}   # 2 T_inf > max decision time

    def raw_init(self, which="init"):
        raw = super().raw_init(which)
        raw["T_inf"] = 0.0  # T_inf = T_dmax / 2 + 0.69 s
        return raw

    def drift(self, x, t, p, c):
        return p["v"] + 0 * x

    def free_moments(self, t, p, c):
        return self.start(p) + p["v"] * t, SIGMA ** 2 * t

    def bounds(self, t, p, c=None):
        shrink = p["a"] * t / (2 * p["T_inf"])
        return -p["a"] + shrink, p["a"] - shrink


MODELS = {m.name: m for m in [StandardDDM, UrgencyDDM, OUModel, DMC, CollapsingBound]}
