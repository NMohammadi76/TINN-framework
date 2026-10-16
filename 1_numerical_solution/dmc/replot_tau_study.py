"""
Redraws the DMC tau study from dmc_study_tau.npz (written by study_tau.py) without tau = 20,
as in the paper:  python replot_tau_study.py
"""
import numpy as np

from study_tau import plot_study


def main():
    r = dict(np.load("dmc_study_tau.npz"))
    keep = r["values"] != 20                       # drop tau = 20
    r2 = {k: (v[keep] if k in ("values", "upper", "lower", "fd_upper", "fd_lower", "errors") else v)
          for k, v in r.items()}
    vals = list(r2["values"])

    plot_study("tau", vals, r2["tspace"], r2["fd_t"], r2, "upper", "dmc_fpt_upper_tau.pdf", headroom=1.3)
    plot_study("tau", vals, r2["tspace"], r2["fd_t"], r2, "lower", "dmc_fpt_lower_tau.pdf")


if __name__ == "__main__":
    main()
