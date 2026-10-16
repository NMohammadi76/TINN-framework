"""
Several experimental conditions that differ ONLY in the bound a (e.g. speed /
neutral / accuracy instructions), with drift and all other parameters shared.

    python multi_condition.py --model ou                  # 3 conditions, 3 x 667 trials
    python multi_condition.py --model ou --per-cond 2000  # 3 x 2000 trials

MultiBound wraps any model in models.py: parameter `a` is replaced by a_1..a_K,
and each condition code says which bound to use (for DMC it is combined with the
congruent / incongruent sign, giving 2K conditions).
"""
import argparse
import json
import math
import os
import torch

from models import MODELS, DecisionModel
from fit import Estimator


class MultiBound(DecisionModel):
    def __init__(self, base, factors=(0.75, 1.0, 1.33)):
        self.base = base
        self.name = base.name + f"_{len(factors)}bounds"
        self.K = len(factors)
        self.nb = len(base.conditions)
        self.fixed = dict(base.fixed)
        kind, a_true, a_init = base.param_specs["a"]
        specs = {}
        for k, spec in base.param_specs.items():
            if k == "a":
                for i, f in enumerate(factors):
                    specs[f"a{i + 1}"] = (kind, round(a_true * f, 4), a_init)
            else:
                specs[k] = spec
        self.param_specs = specs
        # condition code = bound index + (base condition index) / nb
        self.conditions = [i + j / self.nb for i in range(self.K) for j in range(self.nb)]

    def raw_init(self, which="init"):
        raw = super().raw_init(which)
        if "T_inf" in raw:
            raw["T_inf"] = 0.0
        return raw

    def _decode(self, cv):
        i = int(math.floor(cv + 1e-6))
        j = int(round((cv - i) * self.nb))
        return i, self.base.conditions[j]

    def _with_a(self, p, a):
        q = dict(p)
        q["a"] = a
        return q

    def _a_of(self, p, c):
        if not torch.is_tensor(c):
            return p[f"a{self._decode(c)[0] + 1}"]
        idx = torch.floor(c + 1e-6).long()
        a = torch.zeros_like(c)
        for i in range(self.K):
            a = torch.where(idx == i, p[f"a{i + 1}"] + 0 * c, a)
        return a

    def drift(self, x, t, p, c):
        i, cb = self._decode(c)
        return self.base.drift(x, t, self._with_a(p, p[f"a{i + 1}"]), cb)

    def free_moments(self, t, p, c):
        i, cb = self._decode(c)
        return self.base.free_moments(t, self._with_a(p, p[f"a{i + 1}"]), cb)

    def bounds(self, t, p, c=None):
        return self.base.bounds(t, self._with_a(p, self._a_of(p, c)), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ddm", choices=list(MODELS))
    ap.add_argument("--per-cond", type=int, default=667, help="trials per bound condition")
    ap.add_argument("--t0", type=float, default=0.3)
    ap.add_argument("--warm", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/three_conditions")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model = MultiBound(MODELS[args.model]())
    rt, ch, cond = model.simulate(args.per_cond * model.K, model.true_params(), args.t0, seed=args.seed)
    truth = {k: float(v) for k, v in model.true_params().items() if k not in model.fixed}
    truth["t0"] = args.t0
    print(f"model={model.name} trials={len(rt)} P(upper)={ch.mean():.3f} "
          f"RT range=[{rt.min():.3f}, {rt.max():.3f}]")

    est = Estimator(model, rt, ch, cond)
    init = est.estimates()
    fit = est.run(n_warm=args.warm, n_joint=args.iters, log_every=(args.warm + args.iters) // 10)
    print("\nparameter     true      init  estimate")
    for k in fit:
        print(f"{k:9s} {truth[k]:8.3f}  {init[k]:8.3f}  {fit[k]:8.3f}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"estimate_{args.model}_{len(rt)}.json"), "w") as f:
        json.dump({"model": model.name, "n_trials": len(rt), "true": truth, "init": init,
                   "estimate": fit}, f, indent=2)


if __name__ == "__main__":
    main()
