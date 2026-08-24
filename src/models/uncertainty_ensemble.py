"""
uncertainty_ensemble.py — the minimum-uncertainty fusion.

This is the PyTorch port of the original MATLAB contribution (the weight logic in
run_experiment_bc.m + the Dr_Sadoghi.m combiner): an ensemble of online learners
whose predictions are fused with weights INVERSELY PROPORTIONAL to each learner's
prediction-error variance. A learner that has been making stable, low-variance
errors is trusted more; a learner whose error is erratic is down-weighted.

This is what makes the method drift-aware: when the stream shifts and a learner's
error variance spikes, its weight falls automatically, without any explicit
drift detector.
"""
from __future__ import annotations

from typing import List

import torch

from .online_learners import OnlineLinearLearner


class UncertaintyEnsemble:
    def __init__(self, learners: List[OnlineLinearLearner],
                 ema: float = 0.01, eps: float = 1e-6):
        """
        learners : the base online learners to fuse.
        ema      : EMA rate for tracking each learner's running error mean/variance.
        eps      : floor added to variance so a perfect learner doesn't get
                   infinite weight.
        """
        self.learners = learners
        self.ema = ema
        self.eps = eps
        k = len(learners)
        # Running mean and variance of each learner's per-sample error.
        self.err_mean = torch.zeros(k)
        self.err_var = torch.ones(k)      # start neutral (equal weights)

    def weights(self) -> torch.Tensor:
        """Fusion weights: w_k proportional to 1 / (error variance + eps)."""
        inv = 1.0 / (self.err_var + self.eps)
        return inv / inv.sum()

    def score(self, x: torch.Tensor) -> float:
        """Weighted combination of the base learners' signed margins."""
        w = self.weights()
        s = 0.0
        for wk, lrn in zip(w.tolist(), self.learners):
            s += wk * lrn.score(x).item()
        return s

    def predict(self, x: torch.Tensor) -> int:
        return 1 if self.score(x) >= 0 else -1

    def partial_fit(self, x: torch.Tensor, y: int) -> None:
        """Observe one labeled sample: update error stats, then each learner."""
        for i, lrn in enumerate(self.learners):
            # Per-sample error = hinge loss of this learner (>=0).
            margin = lrn.score(x).item()
            err = max(0.0, 1.0 - y * margin)
            # Welford-style EMA update of mean and variance.
            prev_mean = self.err_mean[i].item()
            new_mean = (1 - self.ema) * prev_mean + self.ema * err
            self.err_mean[i] = new_mean
            self.err_var[i] = (1 - self.ema) * self.err_var[i].item() + \
                self.ema * (err - prev_mean) * (err - new_mean)
            lrn.update(x, y)

    # -- Convenience for batch streaming -----------------------------------
    def stream_fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        for i in range(X.shape[0]):
            self.partial_fit(X[i], int(y[i].item()))

    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.predict(X[i]) for i in range(X.shape[0])])
