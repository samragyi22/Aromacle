"""Multi-label evaluation: per-label and mean ROC-AUC / PR-AUC.

Labels with a single class in the evaluated split have an undefined
ROC-AUC and are skipped (and counted) rather than crashing or silently
biasing the mean.  PR-AUC (average precision) is reported alongside
ROC-AUC because odor labels are heavily imbalanced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "baseline_metrics.csv"
_SUMMARY_COLUMNS = [
    "model",
    "features",
    "seed",
    "n_labels_evaluated",
    "n_labels_skipped",
    "mean_roc_auc",
    "mean_pr_auc",
]


def per_label_metrics(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    label_names: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Compute per-label ROC-AUC and PR-AUC.

    Returns
    -------
    metrics
        DataFrame with columns ``label``, ``support`` (positives in this
        split), ``roc_auc``, ``pr_auc`` — one row per *evaluated* label.
    skipped
        Names of labels excluded because only one class is present.
    """
    Y_true = np.asarray(Y_true)
    rows: list[dict] = []
    skipped: list[str] = []
    for j, name in enumerate(label_names):
        y = Y_true[:, j]
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == len(y):
            skipped.append(name)
            continue
        rows.append(
            {
                "label": name,
                "support": n_pos,
                "roc_auc": roc_auc_score(y, Y_pred[:, j]),
                "pr_auc": average_precision_score(y, Y_pred[:, j]),
            }
        )
    metrics = pd.DataFrame(rows, columns=["label", "support", "roc_auc", "pr_auc"])
    return metrics, skipped


def save_metrics(
    metrics: pd.DataFrame,
    skipped: list[str],
    model: str,
    features: str,
    seed: int,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """Write per-label CSV and upsert a summary row into baseline_metrics.csv.

    The summary row for the same (model, features) pair is replaced, so
    re-running an experiment keeps the file idempotent.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    per_label_path = results_dir / f"{model}_{features}_per_label.csv"
    metrics.to_csv(per_label_path, index=False)
    summary_path = results_dir / SUMMARY_PATH.name

    summary = {
        "model": model,
        "features": features,
        "seed": seed,
        "n_labels_evaluated": len(metrics),
        "n_labels_skipped": len(skipped),
        "mean_roc_auc": float(metrics["roc_auc"].mean()),
        "mean_pr_auc": float(metrics["pr_auc"].mean()),
    }
    if summary_path.exists():
        table = pd.read_csv(summary_path)
        table = table[~((table["model"] == model) & (table["features"] == features))]
        table = pd.concat([table, pd.DataFrame([summary])], ignore_index=True)
    else:
        table = pd.DataFrame([summary])
    table = table.sort_values(["model", "features"]).reset_index(drop=True)
    table.to_csv(summary_path, index=False)
    print(f"[eval] wrote {per_label_path}")
    print(f"[eval] wrote {summary_path}")
    return summary


def print_baseline_table(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Print the accumulated baseline table; returns it as a DataFrame."""
    summary_path = results_dir / SUMMARY_PATH.name
    if not summary_path.exists():
        print("[eval] no baseline_metrics.csv yet")
        return pd.DataFrame()
    table = pd.read_csv(summary_path)
    display = table.copy()
    display["mean_roc_auc"] = display["mean_roc_auc"].map(lambda v: f"{v:.4f}")
    display["mean_pr_auc"] = display["mean_pr_auc"].map(lambda v: f"{v:.4f}")
    print("\n===== Baseline results (test set) =====")
    print(
        display[["model", "features", "mean_roc_auc", "mean_pr_auc", "n_labels_skipped"]]
        .to_string(index=False)
    )
    return table
