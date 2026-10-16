"""
TINN estimates on the shared test datasets (any of the five models; the model is stored
in the .npz file).

    python fit_pinn.py --data results/ddm/test_datasets.npz --out results/ddm/tinn.csv

Resumable: datasets already in --out are skipped.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinn"))
from models import MODELS  # noqa: E402
from fit import Estimator       # noqa: E402
from multi_condition import MultiBound  # noqa: E402
from common import load_test_datasets, params_of, base_of, MODELS as CMP_MODELS  # noqa: E402


def tinn_model(model):
    """models.py model; for urgency3 / ou3 the base model with one bound per condition (a1..a3)"""
    m = MODELS[base_of(model)]()
    return MultiBound(m, (1.0,) * len(CMP_MODELS[model]["conditions"])) if "base" in CMP_MODELS[model] else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/ddm/test_datasets.npz")
    ap.add_argument("--out", default="results/ddm/tinn.csv")
    ap.add_argument("--warm", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto = the GPU (cuda) if PyTorch can see one, otherwise the CPU")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    # every tensor the TINN code creates (network, collocation points, data) goes to this device
    torch.set_default_device(device)
    print(f"[tinn] device: {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""),
          flush=True)

    d = load_test_datasets(args.data)
    PARAMS = params_of(d["model"])
    done = set(pd.read_csv(args.out)["dataset"]) if os.path.exists(args.out) else set()
    last = args.last if args.last is not None else len(d["rt"])
    for i in range(args.first, last):
        if i in done:
            continue
        torch.manual_seed(i)
        rt = torch.tensor(d["rt"][i], dtype=torch.float32)
        ch = torch.tensor(d["choice"][i], dtype=torch.float32)
        cond = torch.tensor(d["cond"][i], dtype=torch.long)
        est = Estimator(tinn_model(d["model"]), rt, ch, cond)
        t = time.time()
        fit = est.run(n_warm=args.warm, n_joint=args.iters, verbose=False)
        row = {"dataset": i, "time_s": time.time() - t}
        row.update({f"{k}_est": fit[k] for k in PARAMS})
        row.update({f"{k}_true": float(d[f"{k}_true"][i]) for k in PARAMS})
        pd.DataFrame([row]).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        print(f"[tinn] dataset {i}: " + "  ".join(f"{k} {row[k + '_true']:.3f}->{row[k + '_est']:.3f}"
                                                 for k in PARAMS) + f"  ({row['time_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
