"""
run_cross_dataset.py — the generalization experiment.

Setup (external validation / domain shift on real security data):
  * Train the uncertainty-weighted online ensemble on NSL-KDD KDDTrain+ (streamed).
  * Evaluate IN-DOMAIN on a held-out slice of Train+ (same distribution).
  * Evaluate CROSS-DOMAIN on KDDTest+ (shifted: novel attack types).
  * Report the GENERALIZATION GAP (in-domain F1 minus cross-domain F1).

Also compares the ensemble against each individual base learner, to show what
the minimum-uncertainty fusion buys under distribution shift.

Usage:
  PYTHONPATH=. python experiments/run_cross_dataset.py [--limit N] [--seed S]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.data import nsl_kdd
from src.eval.metrics import evaluate, format_row, generalization_gap
from src.models.online_learners import default_pool
from src.models.uncertainty_ensemble import UncertaintyEnsemble


def _predict_scores(model, X):
    preds = np.empty(X.shape[0], dtype=np.int64)
    scores = np.empty(X.shape[0], dtype=np.float32)
    for i in range(X.shape[0]):
        s = model.score(X[i])
        scores[i] = s
        preds[i] = 1 if s >= 0 else -1
    return preds, scores


def _stream_train(model, X, y):
    for i in range(X.shape[0]):
        model.partial_fit(X[i], int(y[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="cap training stream length (0 = all) for a quick run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/nsl_kdd_generalization.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train, test = nsl_kdd.load()
    dim = train.X.shape[1]

    # Hold out 20% of Train+ as the in-domain test (same distribution).
    n = train.X.shape[0]
    idx = np.random.permutation(n)
    n_holdout = n // 5
    hold_idx, stream_idx = idx[:n_holdout], idx[n_holdout:]
    if args.limit:
        stream_idx = stream_idx[: args.limit]

    Xtr = torch.tensor(train.X[stream_idx]); ytr = train.y[stream_idx]
    Xin = torch.tensor(train.X[hold_idx]);   yin = train.y[hold_idx]
    Xcr = torch.tensor(test.X);              ycr = test.y

    print(f"dim={dim}  stream={len(stream_idx)}  in-domain={len(hold_idx)}  "
          f"cross-domain(Test+)={len(ycr)}\n")

    results = {"config": {"dim": dim, "stream": int(len(stream_idx)),
                          "seed": args.seed}}

    # --- The uncertainty ensemble ---------------------------------------
    ens = UncertaintyEnsemble(default_pool(dim))
    _stream_train(ens, Xtr, ytr)

    p_in, s_in = _predict_scores(ens, Xin)
    p_cr, s_cr = _predict_scores(ens, Xcr)
    m_in = evaluate(yin, p_in, s_in)
    m_cr = evaluate(ycr, p_cr, s_cr)
    results["ensemble"] = {"in_domain": m_in, "cross_domain": m_cr,
                           "gap_f1": generalization_gap(m_in, m_cr, "f1"),
                           "final_weights": ens.weights().tolist()}

    print("=== Uncertainty-weighted ensemble (the method) ===")
    print(format_row("  in-domain (Train+)", m_in))
    print(format_row("  cross-domain(Test+)", m_cr))
    print(f"  generalization gap (F1): {results['ensemble']['gap_f1']:+.4f}\n")

    # --- Each base learner alone, for comparison ------------------------
    print("=== Individual base learners (cross-domain Test+) ===")
    names = ["Perceptron", "PA-I", "PA-II", "ConfidenceWeighted"]
    results["baselines"] = {}
    for name in names:
        pool = default_pool(dim)
        single = UncertaintyEnsemble([pool[names.index(name)]])
        _stream_train(single, Xtr, ytr)
        p, s = _predict_scores(single, Xcr)
        m = evaluate(ycr, p, s)
        results["baselines"][name] = m
        print(format_row("  " + name, m))

    # Ensemble vs best single learner on cross-domain F1.
    best_single = max(results["baselines"].values(), key=lambda m: m["f1"])["f1"]
    print(f"\ncross-domain F1 — ensemble {m_cr['f1']:.4f} vs "
          f"best single {best_single:.4f} "
          f"({m_cr['f1'] - best_single:+.4f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
