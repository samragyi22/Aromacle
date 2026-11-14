"""Baseline model definitions (classical, scikit-learn / XGBoost).

Multi-label setting: each molecule has a binary vector of odor labels.
RandomForest natively supports multi-output classification; XGBoost is
applied one-vs-rest with safe handling of labels that lack both classes
in the training set.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

SEED = 42
RF_N_ESTIMATORS = 300
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1


def build_random_forest(seed: int = SEED) -> RandomForestClassifier:
    """Multi-label RandomForest (native multi-output)."""
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


class OneVsRestXGB:
    """One-vs-rest XGBoost for multi-label data.

    Labels that have no positives (or no negatives) in the training set
    cannot form a binary problem; those columns are skipped at fit time
    and predicted as 0.0, which is reported by :attr:`skipped_labels_`.
    """

    def __init__(
        self,
        seed: int = SEED,
        n_estimators: int = XGB_N_ESTIMATORS,
        max_depth: int = XGB_MAX_DEPTH,
        learning_rate: float = XGB_LEARNING_RATE,
    ) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.estimators_: dict[int, XGBClassifier] = {}
        self.skipped_labels_: list[int] = []
        self.n_labels_: int = 0

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "OneVsRestXGB":
        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n_samples, n_labels), got shape {Y.shape}")
        self.n_labels_ = Y.shape[1]
        self.estimators_.clear()
        self.skipped_labels_.clear()
        for j in range(self.n_labels_):
            positives = int(Y[:, j].sum())
            if positives == 0 or positives == len(Y):
                # Not a learnable binary problem in this split.
                self.skipped_labels_.append(j)
                continue
            clf = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=self.seed + j,
            )
            clf.fit(X, Y[:, j])
            self.estimators_[j] = clf
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probability per label; skipped labels get 0.0."""
        out = np.zeros((X.shape[0], self.n_labels_), dtype=np.float64)
        for j, est in self.estimators_.items():
            out[:, j] = est.predict_proba(X)[:, 1]
        return out


def build_model(name: str, seed: int = SEED):
    """Build a baseline model by name: ``'rf'`` or ``'xgb'``."""
    if name == "rf":
        return build_random_forest(seed=seed)
    if name == "xgb":
        return OneVsRestXGB(seed=seed)
    raise ValueError(f"Unknown model {name!r}; expected 'rf' or 'xgb'")


def predict_positive_proba(model, X: np.ndarray, n_labels: int) -> np.ndarray:
    """Return positive-class probabilities with shape (n_samples, n_labels).

    Handles both multi-output RandomForest (which returns a 3-D array or a
    list of per-output arrays) and the one-vs-rest XGBoost wrapper.
    """
    proba = model.predict_proba(X)
    if isinstance(proba, list):  # outputs with differing class counts
        out = np.zeros((X.shape[0], n_labels), dtype=np.float64)
        for j, arr in enumerate(proba):
            out[:, j] = arr[:, 1] if arr.shape[1] > 1 else 0.0
        return out
    proba = np.asarray(proba)
    if proba.ndim == 3:  # (n_samples, n_outputs, n_classes)
        return proba[:, :, 1]
    if proba.ndim == 2 and proba.shape[1] == n_labels:
        return proba
    raise ValueError(f"Unexpected predict_proba shape {proba.shape} for {n_labels} labels")
