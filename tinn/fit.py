"""
Inverse-PINN parameter estimation.

Two coupled updates per iteration:

  network  <-  lam_pde * mean(R^2) + lam_ic * L_ic
               R is the Fokker-Planck residual (pinn.py), evaluated at parameter
               values jittered around the current estimate, so the network
               learns the solution w(x, t; theta) in a neighbourhood of theta.

  theta    <-  NLL(data | theta) using the PINN fluxes through the bounds as the
               joint RT/choice density, normalised over [0, RT_max - t0].
               theta enters the network as an input, so the gradient includes
               how the PDE solution itself changes with theta.

Keeping the physics loss out of the theta update matters: otherwise the
parameters drift toward values where the PDE is merely *easier to solve*
instead of where the likelihood is highest.  A warm-up phase first solves the
PDE around the initial theta.
"""
import math
import time
import torch

from pinn import FPTSolver


class Estimator:
    def __init__(self, model, rt, choice, cond, width=64, depth=4, t_min=5e-3,
                 init="init", t0_init=0.2, normalize=True):
        self.model = model
        self.rt, self.choice, self.cond = rt, choice.bool(), cond
        self.cvals = torch.tensor(model.conditions)[cond]
        self.rt_min, self.rt_max = float(rt.min()), float(rt.max())
        self.t_min = t_min
        # normalize=True: likelihood conditional on RT <= RT_max (densities renormalised over the
        # observed range); False: the ordinary likelihood, as in PyDDM and the referee.  The
        # conditional form ignores mass beyond RT_max, which matters when a model (e.g. OU with a
        # strong leak) can put many decisions there.
        self.normalize = normalize
        self.solver = FPTSolver(model, t_scale=self.rt_max, width=width, depth=depth, t_min=t_min)

        raw = model.raw_init(init)
        self.raw = {k: torch.nn.Parameter(torch.tensor(float(v))) for k, v in raw.items()}
        frac = min(max(t0_init / (self.rt_min - t_min), 0.02), 0.98)
        self.raw["t0"] = torch.nn.Parameter(torch.tensor(math.log(frac / (1 - frac))))

    # ---------------------------------------------------------------------
    def params(self, raw=None):
        raw = self.raw if raw is None else raw
        t0 = (self.rt_min - self.t_min) * torch.sigmoid(raw["t0"])
        ctx = {"t_dmax": self.rt_max - t0}
        p = self.model.transform(raw, ctx)
        return p, t0

    def jittered_params(self, n, jitter):
        """n parameter sets: current estimate + Gaussian noise in raw space."""
        raw = {k: v.detach() + (jitter * torch.randn(n) if k != "t0" else 0.0)
               for k, v in self.raw.items()}
        return self.params(raw)

    def physics_loss(self, n_col, jitter):
        p, t0 = self.jittered_params(n_col, jitter)
        t_end = self.rt_max - t0
        conds = torch.tensor(self.model.conditions)
        c = conds[torch.randint(len(conds), (n_col,))]
        xi = 0.002 + 0.996 * torch.rand(n_col)
        u = torch.rand(n_col)
        u = torch.where(torch.rand(n_col) < 0.5, u, u ** 2)   # extra points at early t
        t = self.t_min + (t_end - self.t_min) * u
        R = self.solver.pde_residual(p, xi, t, c)
        p_ic, _ = self.jittered_params(512, jitter)
        return (R ** 2).mean(), self.solver.ic_loss(p_ic, 512)

    def nll(self, p, t0, n_grid=200):
        T = self.rt - t0
        logg = self.solver.log_fpt(p, T, self.cvals, self.choice)
        # normalisation per condition over [t_min, RT_max - t0]
        s = torch.linspace(0, 1, n_grid)
        tg = self.t_min + (self.rt_max - t0 - self.t_min) * s
        up, lo = torch.ones_like(tg, dtype=torch.bool), torch.zeros_like(tg, dtype=torch.bool)
        logZ = []
        for cv in self.model.conditions:
            c = torch.full_like(tg, cv)
            g = self.solver.log_fpt(p, tg, c, up).exp() + self.solver.log_fpt(p, tg, c, lo).exp()
            logZ.append(torch.log(torch.trapezoid(g, tg)))
        logZ = torch.stack(logZ)
        if not self.normalize:
            return -logg.mean(), logZ.exp().detach()
        return -(logg - logZ[self.cond]).mean(), logZ.exp().detach()

    # ---------------------------------------------------------------------
    def run(self, n_warm=2000, n_joint=6000, n_col=2000, lam_pde=10.0, lam_ic=10.0,
            lr_net=1e-3, lr_theta=5e-3, jitter=0.1, log_every=500, fix_theta=False,
            verbose=True, tol=None, patience=500):
        """tol: stop the joint phase early once no parameter has moved more than tol in the
        last `patience` iterations (None = always run all iterations)."""
        net_params = list(self.solver.parameters())
        theta = list(self.raw.values())
        opt_net = torch.optim.Adam(net_params, lr=lr_net)
        opt_th = torch.optim.Adam(theta, lr=lr_theta)
        n_tot = n_warm + n_joint
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_net, n_tot, eta_min=lr_net * 0.05)
        sched_th = torch.optim.lr_scheduler.CosineAnnealingLR(opt_th, max(n_joint, 1), eta_min=lr_theta * 0.05)
        t_start = time.time()
        for it in range(n_tot):
            joint = it >= n_warm and not fix_theta
            # 1) network: solve the PDE around the current theta
            L_pde, L_ic = self.physics_loss(n_col, 0.0 if fix_theta else jitter)
            opt_net.zero_grad()
            (lam_pde * L_pde + lam_ic * L_ic).backward()
            torch.nn.utils.clip_grad_norm_(net_params, 10.0)
            opt_net.step(); sched.step()
            # 2) theta: maximum likelihood through the current PINN solution
            if joint:
                p, t0 = self.params()
                L_nll, Z = self.nll(p, t0)
                grads = torch.autograd.grad(L_nll, theta)
                for th, g in zip(theta, grads):
                    th.grad = g
                opt_th.step(); sched_th.step()
            if tol is not None and joint and (it - n_warm) % 100 == 0:
                cur = self.estimates()
                hist = getattr(self, "_hist", [])
                hist.append(cur)
                self._hist = hist
                back = patience // 100
                if len(hist) > back and max(abs(cur[k] - hist[-1 - back][k]) for k in cur) < tol:
                    if verbose:
                        print(f"[{it:5d}] converged (change < {tol} over {patience} iterations)", flush=True)
                    self.n_iter = it + 1
                    return cur
            if verbose and (it % log_every == 0 or it == n_tot - 1):
                msg = f"[{it:5d}] pde={L_pde.item():.2e} ic={L_ic.item():.2e}"
                if joint:
                    msg += f" nll={L_nll.item():.4f} Z={[round(z, 3) for z in Z.tolist()]}"
                msg += " | " + self.format_params() + f" ({time.time() - t_start:.0f}s)"
                print(msg, flush=True)
        self.n_iter = n_tot
        return self.estimates()

    # ---- warm start: reuse a network (and parameters) from an earlier fit -------------------
    def state(self):
        return {"net": self.solver.state_dict(), "t_scale": self.solver.t_scale,
                "raw": {k: v.detach().cpu() for k, v in self.raw.items()}}

    def load_state(self, st, params=True):
        self.solver.load_state_dict(st["net"])
        self.solver.t_scale = st["t_scale"]        # the network's time input keeps its meaning
        if params:
            with torch.no_grad():
                for k, v in st["raw"].items():
                    if k in self.raw:
                        self.raw[k].copy_(v)

    def estimates(self):
        with torch.no_grad():
            p, t0 = self.params()
            est = {k: float(v) for k, v in p.items() if k not in self.model.fixed}
            est["t0"] = float(t0)
        return est

    def format_params(self):
        return " ".join(f"{k}={v:.3f}" for k, v in self.estimates().items())
