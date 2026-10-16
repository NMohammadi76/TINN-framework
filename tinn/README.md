# tinn: shared TINN code

Used by parts 2 and 3 (the scripts there add this folder to `sys.path`).

| Module | Contents |
|---|---|
| `models.py` | the five models (`ddm`, `urgency`, `ou`, `dmc`, `collapse`): drift, bounds, mean and variance of the free process, parameter transforms |
| `pinn.py` | `FPTSolver`: ansatz `p = G * B * exp(w_NN)` (free-process density `G` with the exact Dirac initial condition, method-of-images factor `B` that vanishes on both bounds), PDE residual, first-passage densities at the bounds |
| `fit.py` | `Estimator`: network updated on the PDE and initial-condition losses, parameters on the negative log-likelihood only; warm-up phase, optional early stopping and warm start |
| `multi_condition.py` | `MultiBound`: conditions that differ only in the bound (`a1 ... aK`) |
| `fd_check.py` | independent Crank-Nicolson solver of the forward equation and its NLL (likelihood referee) |

Each module's docstring gives the equations.

Example (one dataset):

```python
import torch
from models import MODELS
from fit import Estimator

model = MODELS["ou"]()
est = Estimator(model, rt, choice, cond)          # torch tensors: rt in s, choice 0/1, condition index
fit = est.run(n_warm=2000, n_joint=6000)
print(fit)                                         # {parameter: estimate}
```
