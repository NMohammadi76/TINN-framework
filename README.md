# TINN: Theory-Informed Neural Networks for diffusion decision models

Code for the paper

> Mohammadi, N., Abbaszadeh, M., Dehghan, M., Hadian Rasanan, A. H., & Amani Rad, J.
> *From Physics to Cognition: A Theory-Informed Neural Network for Efficient Inference in
> Time-dependent Diffusion Decision Models* (journal, year and DOI to be added on publication).

A TINN solves the Kolmogorov forward (Fokker-Planck) equation of a diffusion decision model with a
neural network and uses the probability flux through the bounds, the first-passage-time (FPT) density,
as the likelihood of response times and choices. The repository reproduces the three parts of the paper:

| Part | Folder | What it does |
|---|---|---|
| 1. Numerical approximation | [`1_numerical_solution/`](1_numerical_solution) | TINN solution of the forward equation and the FPT densities for five models, compared with exact or finite-difference solutions; studies of collocation points, network size and model parameters |
| 2. Parameter estimation | [`2_parameter_estimation/`](2_parameter_estimation) | Simulation benchmark of TINN against PyDDM and BayesFlow on the same 200 datasets per model; parameter recovery against the number of trials |
| 3. Data fitting | [`3_data_fitting/`](3_data_fitting) | Fits of the speed / neutral / accuracy data of Forstmann et al. (2008) with four models, by TINN and PyDDM |

The shared TINN code used by parts 2 and 3 is in [`tinn/`](tinn).

## Models

All models are one-dimensional diffusions `dX = mu(X, t) dt + sigma dW`, `sigma = 1`, `X(0) = 0`,
absorbed at a lower bound `l(t)` and an upper bound `u(t)`; `RT = T + t0`.

| Name | Drift `mu` | Bounds |
|---|---|---|
| `ddm` (standard DDM) | `v` | `+/- a` |
| `urgency` (urgency gating) | `v0 + v1 t / (t + tau)` | `+/- a` |
| `ou` (Ornstein-Uhlenbeck, leaky) | `v0 - beta x` | `+/- a` |
| `dmc` (diffusion model for conflict) | `mu0 + c A e^{-t/tau} (e / tau)(1 - t / tau)`, `c = +1` congruent / `-1` incongruent | `+/- a` |
| `collapse` (linearly collapsing bounds) | `v` | `+/- a (1 - t / (2 T_inf))` |

`ddm3`, `urgency3`, `ou3` and `collapse3` are the same models with three conditions that differ only in
the bound (`a1`, `a2`, `a3`), as in the speed / neutral / accuracy design of part 3.

## Repository layout

```
TINN/
├── tinn/                      shared TINN code (parts 2 and 3)
│   ├── models.py              model definitions: drift, bounds, free-process mean and variance
│   ├── pinn.py                TINN solver: ansatz, network, PDE residual, FPT densities
│   ├── fit.py                 parameter estimation (Estimator)
│   ├── multi_condition.py     conditions that differ only in the bound
│   └── fd_check.py            independent Crank-Nicolson solver (likelihood referee)
├── 1_numerical_solution/      part 1: one folder per model, standalone scripts
├── 2_parameter_estimation/    part 2: benchmark against PyDDM and BayesFlow
├── 3_data_fitting/            part 3: Forstmann et al. (2008) data
│   └── data/FDBNCRW2008.RData
├── requirements.txt
└── LICENSE
```

Each folder has its own README with the scripts, their inputs and outputs, and how to run them.

## Installation

Python 3.10 or newer.

```
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch uses the GPU when it finds one (install the CUDA build of `torch` from pytorch.org for that);
every script also runs on a CPU. The BayesFlow part uses Keras 3 with the torch backend
(`KERAS_BACKEND=torch`, set inside the scripts).

If you see *"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"*, a package in the
environment was built against NumPy 1. Upgrade it in the same environment
(`pip install -U pandas numba pyreadr`) or install `numpy<2`.

## Quick start

```
# Part 1: TINN solution of the standard DDM and its FPT densities
cd 1_numerical_solution/ddm && python tinn_ddm.py

# Part 2: benchmark for one model (see 2_parameter_estimation/README.md)
cd 2_parameter_estimation && jupyter notebook comparison_local.ipynb

# Part 3: real data (see 3_data_fitting/README.md)
cd 3_data_fitting && jupyter notebook real_data.ipynb
```

Seeds are fixed in every script, so reruns on the same hardware and library versions give the same
numbers; results can differ slightly across GPUs, CPUs and library versions.

## Data

`3_data_fitting/data/FDBNCRW2008.RData` contains the behavioural data of

> Forstmann, B. U., Dutilh, G., Brown, S., Neumann, J., von Cramon, D. Y., Ridderinkhof, K. R., &
> Wagenmakers, E.-J. (2008). Striatum and pre-SMA facilitate decision-making under time pressure.
> *Proceedings of the National Academy of Sciences*, 105(45), 17538–17542.
> https://doi.org/10.1073/pnas.0805903105

(19 participants, 15,818 trials, speed / neutral / accuracy instructions). Please cite the original
study when you use these data.

## Citation

If you use this code, please cite the paper above (BibTeX to be added on publication).

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The data in
`3_data_fitting/data/` belong to their original authors.
