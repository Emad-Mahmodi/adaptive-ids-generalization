"""
synthetic.py — offline two-domain generator (no network needed).

Produces a source domain A and a shifted domain B: same task (rare-anomaly
detection) but B has covariate shift (shifted/scaled features) and a different
anomaly rate. Used for unit tests and for running the pipeline when the real
datasets are unavailable. The real study uses NSL-KDD (see nsl_kdd.py).
"""
from __future__ import annotations

import numpy as np


def make_domain(n: int, dim: int, anomaly_rate: float,
                shift: float, scale: float, rng: np.random.Generator):
    n_anom = int(n * anomaly_rate)
    n_norm = n - n_anom
    normal = rng.normal(0.0, 1.0, size=(n_norm, dim))
    # Anomalies live off the normal manifold; the offset is domain-shifted.
    anom = rng.normal(shift, scale, size=(n_anom, dim)) + 2.5
    X = np.vstack([normal, anom]).astype(np.float32)
    y = np.concatenate([-np.ones(n_norm), np.ones(n_anom)]).astype(np.int64)
    order = rng.permutation(n)
    return X[order], y[order]


def make_two_domains(dim: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Domain A (source) and B (shifted target).
    Xa, ya = make_domain(6000, dim, anomaly_rate=0.15, shift=0.0, scale=1.0, rng=rng)
    Xb, yb = make_domain(3000, dim, anomaly_rate=0.30, shift=0.8, scale=1.6, rng=rng)
    return (Xa, ya), (Xb, yb)
