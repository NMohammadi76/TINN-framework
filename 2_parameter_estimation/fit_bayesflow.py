"""
BayesFlow (amortized neural posterior estimation) on the shared test datasets.

    python fit_bayesflow.py --data results/ddm/test_datasets.npz --out results/ddm/bayesflow.csv
    python fit_bayesflow.py ... --n-sim 50000 --epochs 100      # bigger training (GPU)
(any of the five models; the model is stored in the .npz file)

1. simulates --n-sim training datasets (parameters from the prior = RANGES, same
   simulator and same number of trials as the test datasets),
2. trains a DeepSet summary network + coupling-flow posterior once,
3. samples the posterior of every test dataset; the estimate is the posterior mean
   (median and 95% interval are saved too).

Run it in its own process: with the torch backend BayesFlow switches autograd off
globally, which would break the PINN if both ran in the same Python session.
"""
import argparse
import os
import time

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402
import bayesflow as bf  # noqa: E402
import keras          # noqa: E402

from common import load_test_datasets, sample_params, simulate_batch, params_of, ranges_of, MODELS, T_MAX  # noqa: E402,E501


def to_features(model, rt, ch, cond):
    """(n_datasets, n_trials) rt/choice/cond -> (n_datasets, n_trials, n_features) set input:
    log RT, choice (+1 / -1) and, for models with several conditions, the condition value."""
    rt = np.where(np.isnan(rt), T_MAX, rt)
    feats = [np.log(rt), 2.0 * ch - 1.0]
    if "base" in MODELS[model]:          # urgency3 / ou3: one-hot bound condition
        feats += [(cond == k).astype(float) for k in range(len(MODELS[model]["conditions"]))]
    elif len(MODELS[model]["conditions"]) > 1:
        feats.append(np.asarray(MODELS[model]["conditions"])[cond.astype(int)])
    return np.stack(feats, -1).astype("float32")


def training_data(model, n_sim, n_trials, seed):
    p = sample_params(model, np.random.default_rng(seed), n_sim)
    rt, ch, cond = simulate_batch(model, p, n_trials, seed=seed)
    ch = np.where(np.isnan(rt), 0.5, ch)
    out = {k: p[k][:, None].astype("float32") for k in params_of(model)}
    out["x"] = to_features(model, rt, ch, cond)
    return out


def make_workflow(model):
    params, ranges = params_of(model), ranges_of(model)
    adapter = bf.Adapter().convert_dtype("float64", "float32")
    for k in params:
        adapter = adapter.constrain(k, lower=ranges[k][0], upper=ranges[k][1])
    adapter = adapter.concatenate(params, into="inference_variables").rename("x", "summary_variables")
    return bf.BasicWorkflow(adapter=adapter,
                            summary_network=bf.networks.DeepSet(summary_dim=16),
                            inference_network=bf.networks.CouplingFlow(),
                            standardize="inference_variables")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/ddm/test_datasets.npz")
    ap.add_argument("--out", default="results/ddm/bayesflow.csv")
    ap.add_argument("--save", default=None, help="where to save the trained network (.keras)")
    ap.add_argument("--n-sim", type=int, default=20000, help="training datasets")
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=2000, help="posterior draws per dataset")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--diag-dir", default=None, help="save BayesFlow's recovery / calibration plots (.pdf) here")
    args = ap.parse_args()

    d = load_test_datasets(args.data)
    model = d["model"]
    PARAMS = params_of(model)
    n_trials = d["rt"].shape[1]
    workflow = make_workflow(model)

    # ---- training (once; amortized over all test datasets) ----
    t = time.time()
    train = training_data(model, args.n_sim, n_trials, args.seed)
    val = training_data(model, args.n_val, n_trials, args.seed + 1)
    t_sim = time.time() - t
    print(f"simulated {args.n_sim} training datasets in {t_sim:.0f}s", flush=True)
    t = time.time()
    workflow.fit_offline(train, epochs=args.epochs, batch_size=args.batch_size, validation_data=val, verbose=2)
    t_train = time.time() - t
    print(f"training took {t_train:.0f}s", flush=True)
    if args.save:
        workflow.approximator.save(args.save)
    if args.diag_dir:
        os.makedirs(args.diag_dir, exist_ok=True)
        figs = workflow.plot_default_diagnostics(test_data=val, num_samples=500)
        for name, fig in figs.items():
            fig.savefig(os.path.join(args.diag_dir, f"bayesflow_{name}.pdf"), bbox_inches="tight")

    # ---- inference on the test datasets ----
    x = to_features(model, d["rt"], d["choice"], d["cond"])
    t = time.time()
    post = workflow.sample(conditions={"x": x}, num_samples=args.n_samples)
    t_inf = time.time() - t
    rows = []
    for i in range(len(x)):
        row = {"dataset": i, "time_s": t_inf / len(x), "train_sim_s": t_sim, "train_fit_s": t_train}
        for k in PARAMS:
            s = np.asarray(post[k][i]).ravel()
            row[f"{k}_est"] = float(s.mean())
            row[f"{k}_median"] = float(np.median(s))
            row[f"{k}_q025"], row[f"{k}_q975"] = map(float, np.quantile(s, [0.025, 0.975]))
            row[f"{k}_true"] = float(d[f"{k}_true"][i])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    cover = {k: float(((df[f"{k}_q025"] <= df[f"{k}_true"]) & (df[f"{k}_true"] <= df[f"{k}_q975"])).mean())
             for k in PARAMS}
    print(f"inference: {t_inf:.1f}s for {len(x)} datasets; 95% interval coverage {cover}")


if __name__ == "__main__":
    main()
