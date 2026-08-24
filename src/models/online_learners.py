"""
online_learners.py — linear online-learning base learners, in PyTorch.

These are faithful re-implementations (in the spirit of the LIBOL toolbox the
original MATLAB code benchmarked against) of the classic online linear
classifiers. Each processes one sample at a time: predict, observe the true
label, update. Labels are +1 (attack/anomaly) / -1 (normal).

They are deliberately simple and identical in interface so the
UncertaintyEnsemble can combine any set of them.
"""
from __future__ import annotations

import torch


class OnlineLinearLearner:
    """Base class. Keeps a weight vector w; margin = w . x."""

    def __init__(self, dim: int, device: str = "cpu"):
        self.w = torch.zeros(dim, dtype=torch.float32, device=device)
        self.b = 0.0                      # bias / intercept
        self.device = device

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Signed margin (float). >0 predicts +1."""
        return torch.dot(self.w, x) + self.b

    def predict(self, x: torch.Tensor) -> int:
        return 1 if self.score(x).item() >= 0 else -1

    def update(self, x: torch.Tensor, y: int) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class Perceptron(OnlineLinearLearner):
    """Rosenblatt perceptron: update only on mistakes."""

    def update(self, x: torch.Tensor, y: int) -> None:
        if y * self.score(x).item() <= 0:
            self.w += y * x
            self.b += y


class PassiveAggressive(OnlineLinearLearner):
    """
    Passive-Aggressive (Crammer et al. 2006).
    variant: 'pa' (hard), 'pa1' (slack C), 'pa2' (squared slack).
    """

    def __init__(self, dim: int, C: float = 1.0, variant: str = "pa1", device: str = "cpu"):
        super().__init__(dim, device)
        self.C = C
        self.variant = variant

    def update(self, x: torch.Tensor, y: int) -> None:
        margin = self.score(x).item()
        loss = max(0.0, 1.0 - y * margin)          # hinge
        if loss == 0.0:
            return
        sq = torch.dot(x, x).item()
        if sq == 0.0:
            return
        if self.variant == "pa":
            tau = loss / sq
        elif self.variant == "pa1":
            tau = min(self.C, loss / sq)
        else:  # pa2
            tau = loss / (sq + 1.0 / (2.0 * self.C))
        self.w += tau * y * x
        self.b += tau * y


class ConfidenceWeighted(OnlineLinearLearner):
    """
    A light AROW-style confidence-weighted learner: keeps a per-feature variance
    Sigma (diagonal) and takes larger steps on less-confident features. Captures
    the 'second-order' family from LIBOL without the full matrix machinery.
    """

    def __init__(self, dim: int, r: float = 1.0, device: str = "cpu"):
        super().__init__(dim, device)
        self.sigma = torch.ones(dim, dtype=torch.float32, device=device)
        self.r = r

    def update(self, x: torch.Tensor, y: int) -> None:
        margin = self.score(x).item()
        loss = max(0.0, 1.0 - y * margin)
        if loss == 0.0:
            return
        v = torch.dot(x * self.sigma, x).item()     # confidence of the margin
        beta = 1.0 / (v + self.r)
        alpha = loss * beta
        self.w += alpha * y * (self.sigma * x)
        self.b += alpha * y
        # shrink variance in the directions we just used
        self.sigma = self.sigma - beta * (self.sigma * x) ** 2
        self.sigma.clamp_(min=1e-4)


def default_pool(dim: int, device: str = "cpu"):
    """The representative ensemble pool used in the experiments."""
    return [
        Perceptron(dim, device),
        PassiveAggressive(dim, C=1.0, variant="pa1", device=device),
        PassiveAggressive(dim, C=1.0, variant="pa2", device=device),
        ConfidenceWeighted(dim, r=1.0, device=device),
    ]
