# Part 2: Parameter estimation (simulation benchmark)

TINN, PyDDM and BayesFlow estimate the parameters of the same simulated datasets: 200 datasets of
1,000 trials for each of the five models (`ddm`, `urgency`, `ou`, `dmc`, `collapse`). Every estimate is
then scored by the same independent likelihood referee, so that the methods are compared on parameter
recovery, on the likelihood they reach and on computing time.

## How the results in the paper were produced

1. **`comparison_local.ipynb`** (local machine, one model at a time; set `MODEL`): test datasets, PyDDM
   and TINN, then the tables and figures. Results go to `results/<MODEL>/`.
2. **`bayesflow_colab.ipynb`** (Google Colab, GPU): BayesFlow on the same `test_datasets.npz` of every
   model, then the tables and figures again with all three methods.
3. **`make_summary_table.py`**: one table over all models.
4. **`rmse_trials.ipynb`** (local, runs `rmse_trials.py`): recovery of the DDM parameters against the
   number of trials (50 to 2,000; 20 datasets each), for TINN and PyDDM.

## Scripts

| Script | Role | Main output |
|---|---|---|
| `common.py` | parameter ranges (also the BayesFlow prior), Euler simulator (numba), test datasets | `results/<model>/test_datasets.npz` |
| `fit_pinn.py` | TINN estimates (2,000 warm-up + 6,000 joint iterations; `../tinn/fit.py`) | `tinn.csv` |
| `fit_pyddm.py` | PyDDM maximum likelihood, differential evolution, mixture switched off; robust likelihood for the models other than the DDM | `pyddm.csv` |
| `fit_bayesflow.py` | optional command-line BayesFlow (DeepSet + coupling flow), not used for the paper's results | `bayesflow.csv` |
| `make_figures.py` | tables (recovery, cost, likelihood relative to the truth) and black-and-white figures for one model | `results/<model>/paper/` |
| `make_summary_table.py` | summary over all models | `results/summary_table.{tex,csv}` |
| `rmse_trials.py` | RMSE against the number of trials, DDM (TINN and PyDDM in the paper; BayesFlow optional) | `results/ddm_ntrials/rmse_trials.pdf` |

`fit_pinn.py` and `fit_pyddm.py` take `--first` / `--last` to split the datasets over processes; the
notebook does this and merges the parts. All fits are resumable: datasets already in the output file
are skipped.

## Command-line use (one model)

```
python -c "from common import make_test_datasets; make_test_datasets('ou', 200, 1000, path='results/ou/test_datasets.npz')"
python fit_pyddm.py --data results/ou/test_datasets.npz --out results/ou/pyddm.csv
python fit_pinn.py  --data results/ou/test_datasets.npz --out results/ou/tinn.csv --device auto
python fit_bayesflow.py --data results/ou/test_datasets.npz --out results/ou/bayesflow.csv
python make_figures.py --results results/ou
python make_summary_table.py --results results
```

For `rmse_trials.py`: `make`, then `fit --method pyddm|tinn|bayesflow`, then `plot` (see its docstring).

## Settings

* **Test datasets**: parameters uniform in the ranges of `common.MODELS`; seed fixed in `make_test_datasets`.
* **TINN**: 4 x 64 tanh network; network updated on the PDE and initial-condition losses, parameters on the
  negative log-likelihood only; 2,000 warm-up and 6,000 joint iterations (`--warm`, `--iters`).
* **PyDDM**: `dx = dt = 0.001` for the DDM and `0.005` for the other models (`--dx`, `--dt`).
* **BayesFlow (`bayesflow_colab.ipynb`)**: DeepSet summary network (64 summaries) and a diffusion-model
  posterior network, trained once per model on 20,000 simulated datasets (300 for validation) for 100
  epochs with batch size 128; 1,000 posterior draws per dataset, posterior mean as the point estimate.
* **Likelihood referee**: Navarro-Fuss series for the DDM and the Crank-Nicolson solver of
  `../tinn/fd_check.py` for the other models.

TINN runs on a GPU when PyTorch finds one; on a CPU one dataset takes a few minutes on one core.
BayesFlow is run in its own process because, with the torch backend, it switches autograd off for the
whole process.
