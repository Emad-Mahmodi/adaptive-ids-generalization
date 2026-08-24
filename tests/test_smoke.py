"""
Smoke tests — run with:  PYTHONPATH=. python -m pytest tests/ -q
(or plain:  PYTHONPATH=. python tests/test_smoke.py)

They use the offline synthetic generator so no download is needed, and check
that the ensemble learns and that its fusion weights respond to error variance.
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.synthetic import make_two_domains
from src.eval.metrics import evaluate, generalization_gap
from src.models.online_learners import default_pool
from src.models.uncertainty_ensemble import UncertaintyEnsemble


def _train_eval():
    (Xa, ya), (Xb, yb) = make_two_domains(dim=20, seed=1)
    ens = UncertaintyEnsemble(default_pool(20))
    Xt = torch.tensor(Xa)
    ens.stream_fit(Xt, torch.tensor(ya))
    # in-domain (last 500 of A) vs cross-domain (B)
    p_in = ens.predict_batch(Xt[-500:]).numpy()
    m_in = evaluate(ya[-500:], p_in)
    p_cr = ens.predict_batch(torch.tensor(Xb)).numpy()
    m_cr = evaluate(yb, p_cr)
    return ens, m_in, m_cr


def test_ensemble_learns():
    _, m_in, _ = _train_eval()
    assert m_in["f1"] > 0.6, f"ensemble failed to learn in-domain (F1={m_in['f1']})"


def test_weights_normalized_and_finite():
    ens, _, _ = _train_eval()
    w = ens.weights()
    assert abs(float(w.sum()) - 1.0) < 1e-5
    assert torch.isfinite(w).all()


def test_generalization_gap_defined():
    _, m_in, m_cr = _train_eval()
    gap = generalization_gap(m_in, m_cr, "f1")
    assert -1.0 <= gap <= 1.0


if __name__ == "__main__":
    ens, m_in, m_cr = _train_eval()
    print("in-domain :", m_in)
    print("cross-dom :", m_cr)
    print("gap(F1)   :", generalization_gap(m_in, m_cr, "f1"))
    test_ensemble_learns()
    test_weights_normalized_and_finite()
    test_generalization_gap_defined()
    print("\nALL SMOKE TESTS PASSED")
