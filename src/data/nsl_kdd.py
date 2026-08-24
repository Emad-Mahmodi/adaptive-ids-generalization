"""
nsl_kdd.py — load the NSL-KDD network-intrusion dataset.

Why NSL-KDD for a generalization study: its two official splits, KDDTrain+ and
KDDTest+, are a built-in DISTRIBUTION SHIFT — the test set deliberately contains
attack types under-represented or absent in training. Training on Train+ and
testing on Test+ is therefore a real external-validation / domain-shift setting
on genuine security data, exactly the structure the Karolinska project studies
across patient populations (here: across attack distributions).

Small (~22 MB total), canonical, and license-open, so the whole study runs in
minutes on a laptop.

The engine never needs the raw files committed: `download()` fetches them into
data/ (git-ignored), and the loader fits its encoder/scaler on Train+ and applies
the SAME transform to Test+ (no leakage).
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 41 features + class label + difficulty level (last col), per the NSL-KDD spec.
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

_MIRROR = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master"
_FILES = {"train": "KDDTrain%2B.txt", "test": "KDDTest%2B.txt"}


def download(data_dir: str = "data/nsl_kdd") -> None:
    """Fetch KDDTrain+ / KDDTest+ into data_dir if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    for split, fname in _FILES.items():
        dest = os.path.join(data_dir, f"KDD{split.capitalize()}+.txt")
        if os.path.exists(dest):
            continue
        url = f"{_MIRROR}/{fname}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        print(f"downloading {split} -> {dest}")
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())


@dataclass
class Dataset:
    X: np.ndarray            # float32 feature matrix (standardized)
    y: np.ndarray            # +1 attack, -1 normal
    feature_names: list


def _read_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    return df


def load(data_dir: str = "data/nsl_kdd"):
    """
    Return (train, test) as Dataset objects with aligned, standardized features.
    Encoder + scaler are fit on TRAIN ONLY and applied to TEST — no leakage, so
    the Train+ -> Test+ evaluation is a clean external validation.
    """
    download(data_dir)
    tr = _read_raw(os.path.join(data_dir, "KDDTrain+.txt"))
    te = _read_raw(os.path.join(data_dir, "KDDTest+.txt"))

    # Binary label: normal -> -1, any attack -> +1.
    def to_y(df):
        return np.where(df["label"].values == "normal", -1, 1).astype(np.int64)
    ytr, yte = to_y(tr), to_y(te)
    tr, te = tr.drop(columns=["label"]), te.drop(columns=["label"])

    # One-hot the 3 categoricals, aligning test columns to train's.
    tr_enc = pd.get_dummies(tr, columns=CATEGORICAL)
    te_enc = pd.get_dummies(te, columns=CATEGORICAL)
    te_enc = te_enc.reindex(columns=tr_enc.columns, fill_value=0)

    feat = list(tr_enc.columns)
    Xtr = tr_enc.values.astype(np.float32)
    Xte = te_enc.values.astype(np.float32)

    # Standardize on train stats only.
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    return Dataset(Xtr, ytr, feat), Dataset(Xte, yte, feat)


if __name__ == "__main__":
    train, test = load()
    print(f"train: X{train.X.shape}  attacks={int((train.y==1).sum())}  "
          f"normal={int((train.y==-1).sum())}")
    print(f"test:  X{test.X.shape}  attacks={int((test.y==1).sum())}  "
          f"normal={int((test.y==-1).sum())}")
    print(f"features: {len(train.feature_names)}")
