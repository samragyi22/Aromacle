"""Tests for :mod:`aromacle.evaluate_preds`.

Verifies that a saved predictions CSV is evaluated through the *same* logic
as the classical baselines: identical split, identical skipped labels, and a
prevalence-style constant prediction reproducing the prevalence baseline
(ROC-AUC 0.5).

Requires the dataset at data/raw/curated_GS_LF_merged_4983.csv (gitignored);
the tests skip gracefully if it is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aromacle import evaluate_preds
from aromacle.data import load_dataset, load_split_json, split_indices_from_json

DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "curated_GS_LF_merged_4983.csv"
)
SPLIT_PATH = Path(__file__).resolve().parents[1] / "results" / "split_seed42.json"


@pytest.fixture(scope="module")
def dataset():
    if not DATA_PATH.exists():
        pytest.skip(f"dataset not found: {DATA_PATH}")
    smiles, Y, label_names = load_dataset(DATA_PATH)
    return smiles, Y, label_names


def test_evaluate_preds_matches_prevalence_baseline(dataset, tmp_path, monkeypatch) -> None:
    _, Y, label_names = dataset
    _, _, test_idx = split_indices_from_json(load_split_json(SPLIT_PATH))
    Y_test = Y[np.asarray(test_idx, dtype=int)]

    # Constant per-label prevalence -> ROC-AUC 0.5 for every label.
    pred = np.broadcast_to(Y_test.mean(axis=0), Y_test.shape).copy()
    preds_path = tmp_path / "preds.csv"
    pd.DataFrame(pred, columns=label_names).to_csv(preds_path, index=False)

    # Keep the bootstrap cheap; the CI value itself is not under test here.
    monkeypatch.setattr(
        evaluate_preds,
        "bootstrap_ci",
        lambda *a, **k: {
            "mean_roc_auc_ci_lower": 0.5,
            "mean_roc_auc_ci_upper": 0.5,
            "mean_pr_auc_ci_lower": float(Y_test.mean()),
            "mean_pr_auc_ci_upper": float(Y_test.mean()),
        },
    )

    rc = evaluate_preds.main(
        [
            "--preds",
            str(preds_path),
            "--data",
            str(DATA_PATH),
            "--split",
            str(SPLIT_PATH),
            "--results-dir",
            str(tmp_path),
            "--model",
            "openpom",
            "--features",
            "mpnn",
        ]
    )
    assert rc == 0

    table = pd.read_csv(tmp_path / "baseline_metrics.csv")
    row = table[(table["model"] == "openpom") & (table["features"] == "mpnn")]
    assert len(row) == 1
    assert row["mean_roc_auc"].iloc[0] == pytest.approx(0.5)
    # Same two labels (brandy, cherry) are skipped as in the baselines.
    assert int(row["n_labels_skipped"].iloc[0]) == 2
    assert (tmp_path / "openpom_mpnn_per_label.csv").exists()


def test_load_predictions_rejects_missing_columns(dataset, tmp_path) -> None:
    _, _, label_names = dataset
    bad = tmp_path / "bad.csv"
    pd.DataFrame({name: [0.5] for name in label_names[:-1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        evaluate_preds.load_predictions(bad, label_names)
