# Part 1: Numerical approximation

TINN solutions of the Kolmogorov forward (Fokker-Planck) equation with two absorbing bounds and the
resulting first-passage-time (FPT) densities, for fixed model parameters, compared with an exact
eigenfunction series (standard DDM) or a Crank-Nicolson finite-difference solution (other models).

Every script is standalone (PyTorch, float64, fixed seed `1234`) and writes its figures (black-and-white
PDF) and, where noted, its numbers to the folder it is run from:

```
cd ddm
python tinn_ddm.py
```

## Method used in these scripts

The Dirac initial condition is removed analytically: `u = K + w`, where `K` is the free-space transition
density of the model (it solves the equation and carries the delta initial condition exactly) and `w`
is smooth. The initial and boundary conditions on `w` are hard constraints of the form
`w = B(t, x) + (t / T) * bubble(x) * NN(t, x)`, so the loss is the PDE residual alone. Training: Adam
(10,000 iterations, learning rate 1e-3) followed by L-BFGS, with 4,000 collocation points resampled at
every Adam step. For the collapsing bounds the domain is mapped to a fixed interval by `y = x / b(t)`.
Each script's docstring gives its equations, and its configuration block gives every setting.

## Scripts

| Model | Script | What it does | Output |
|---|---|---|---|
| Standard DDM | `ddm/tinn_ddm.py` | TINN solution for `v = 0.3`, bounds `+/-2`; error norms against the exact series | `loss_plot.pdf`, `surface_plot_bw.pdf`, `solution_slices.pdf`, `first_passage_time_comparison*.pdf`, `error_norms.csv`, `ddm_tinn.pt` |
| | `ddm/study_collocation.py` | error (L2, L-inf) against the number of collocation points in space | `collocation_study.pdf` |
| | `ddm/study_network_size.py` | error against neurons per layer and number of hidden layers | `error_vs_neurons.pdf`, `error_vs_layers.pdf` |
| | `ddm/study_drift.py` | FPT densities for drifts 0.2 to 0.8 | `fpt_upper_drifts.pdf`, `fpt_lower_drifts.pdf`, `drift_study.npz` |
| | `ddm/study_boundary.py` | FPT densities for bounds 1 to 2.5 | `fpt_upper_boundaries.pdf`, `fpt_lower_boundaries.pdf`, `boundary_study.npz` |
| | `ddm/tinn_vs_pyddm.py` | error norms and time of TINN against PyDDM (Crank-Nicolson and backward Euler, four grids), all against the exact series | printed table with LaTeX rows, `tinn_fpt_simple.npz` |
| Urgency gating | `urgency/tinn_urgency.py` | TINN solution for `v0 = 0.3`, `v1 = 0.4`, `tau = 0.8` | `ex2-*.pdf`, `ddm_urgency_pinn.pt` |
| | `urgency/study_tau.py` | FPT densities for several `tau` | `ex2-tau.pdf`, `tau_study.npz` |
| | `urgency/study_v0.py`, `urgency/study_v1.py` | FPT densities for several `v0` / `v1` | `urgency_fpt_*_v0.pdf`, `urgency_fpt_*_v1.pdf`, `.npz` |
| Ornstein-Uhlenbeck | `ou/tinn_ou.py` | TINN solution for `v0 = 0.3`, `beta = 0.4` | `ex3-*.pdf`, `ddm_ou_pinn.pt` |
| | `ou/study_v0_beta.py` | FPT densities for several `v0` and `beta` (`RUN` selects the studies) | `ou_fpt_*_{v0,beta}.pdf`, `ou_study_*.npz` |
| DMC | `dmc/tinn_dmc.py` | TINN solution for `mu_c = 0.5`, `A = 20`, `tau = 150` (congruent; `A < 0` gives incongruent trials) | `ex4-*.pdf`, `ddm_dmc_pinn.pt` |
| | `dmc/study_tau.py` | FPT densities for several `tau` of the automatic process | `dmc_fpt_*_tau.pdf`, `dmc_study_tau.npz` |
| | `dmc/replot_tau_study.py` | redraws the `tau` study from `dmc_study_tau.npz` without `tau = 20` (as in the paper) | `dmc_fpt_*_tau.pdf` |
| Collapsing bounds | `collapse/tinn_collapse.py` | TINN solution for `v = 0.7`, `b(0) = 2`, `T_inf = 2` | `ex5-*.pdf`, `ddm_collapsing_pinn.pt` |
| | `collapse/study_tinf.py` | FPT densities for several `T_inf` | `collapsing_fpt_*_tinf.pdf`, `collapsing_study_tinf.npz` |

The parameter studies save their results after every value (`.npz`), so a stopped run can be redrawn.
`tinn_vs_pyddm.py` needs `pyddm`; the other scripts need only `numpy`, `scipy`, `matplotlib` and `torch`.
