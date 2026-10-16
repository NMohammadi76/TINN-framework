# Part 3: Data fitting (Forstmann et al., 2008)

Each of the 19 participants of Forstmann et al. (2008) (15,818 trials) is fitted separately by TINN and
by PyDDM with four models in which only the bound differs between the speed, neutral and accuracy
instructions (`a1`, `a2`, `a3`) and every other parameter is shared:

| Model | Parameters |
|---|---|
| `ddm3` | `v`, `a1`, `a2`, `a3`, `t0` |
| `urgency3` | `v0`, `v1`, `tau`, `a1`, `a2`, `a3`, `t0` |
| `ou3` | `v0`, `beta`, `a1`, `a2`, `a3`, `t0` |
| `collapse3` | `v`, `a1`, `a2`, `a3`, `T_inf`, `t0` |

A correct response is the upper bound (`choice = 1`). Both methods are scored by the same independent
likelihood referee as in part 2, and the report gives G2, AIC and BIC.

## Run

Open **`real_data.ipynb`** from this folder, set `MODEL` and run all cells. It

1. reads `data/FDBNCRW2008.RData` (columns `subj`, `RT`, `correct`, `instruction`) and writes
   `results/<MODEL>_subject/data.csv` (`subj`, `rt` in s, `choice`, `cond`: 0 speed, 1 neutral, 2 accuracy);
2. fits PyDDM and TINN in parallel processes (`fit_real.py`), resumable per participant;
3. writes the tables and the density figure (`real_report.py`) to `results/<MODEL>_subject/paper/`.

Command line, one model:

```
python fit_real.py --data results/ddm3_subject/data.csv --model ddm3 --method pyddm --out results/ddm3_subject/pyddm.csv
python fit_real.py --data results/ddm3_subject/data.csv --model ddm3 --method tinn  --out results/ddm3_subject/tinn.csv
python real_report.py --model ddm3 --results results/ddm3_subject --data results/ddm3_subject/data.csv
```

## Scripts

| Script | Role | Output |
|---|---|---|
| `fit_real.py` | TINN (2,000 warm-up + 6,000 joint iterations) or PyDDM fit of every participant, with the referee NLL of each estimate; search box `REAL_BOX` (wider than the simulation ranges); extra TINN starting points for `ou3` | `tinn.csv`, `pyddm.csv` (one row per participant) |
| `real_report.py` | parameters per instruction (mean over participants), G2, AIC, BIC, and the densities against the data | `paper/table_tinn.tex`, `paper/table_pyddm.tex`, `paper/table_fit.{tex,csv}`, `paper/per_subject_*.csv`, `paper/densities.{pdf,png}` |

`fit_real.py` uses the shared TINN code in `../tinn/` and the PyDDM model builder and referee in
`../2_parameter_estimation/`.

In the notebook, `TINN_WARM_START = False` (used for the paper) fits every participant from the default
start; `True` first fits the pooled data and starts every participant from that fit, with early stopping.

## Data

`data/FDBNCRW2008.RData`: Forstmann, B. U., Dutilh, G., Brown, S., Neumann, J., von Cramon, D. Y.,
Ridderinkhof, K. R., & Wagenmakers, E.-J. (2008). Striatum and pre-SMA facilitate decision-making under
time pressure. *PNAS*, 105(45), 17538–17542. https://doi.org/10.1073/pnas.0805903105
