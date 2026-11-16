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
    "mean_roc_auc_ci_lower",
    "mean_roc_auc_ci_upper",
    "mean_pr_auc_ci_lower",
    "mean_pr_auc_ci_upper",
]

BOOTSTRAP_N_RESAMPLES = 500
BOOTSTRAP_SEED = 42


def per_label_metrics(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    label_names: list[str],
) -> tuple[pd.DataFrame, list[str]]:
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


def bootstrap_ci(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    label_names: list[str],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Bootstrap 95% CI for mean ROC-AUC and mean PR-AUC.

    Resamples test molecules with replacement (``n_resamples`` times, fixed
    ``seed``).  Labels with a single class in a resample are skipped for
    that resample.  Returns a dict with lower/upper bounds.
    """
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)
    n = Y_true.shape[0]
    rng = np.random.default_rng(seed)
    roc_means: list[float] = []
    pr_means: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = Y_true[idx]
        yp = Y_pred[idx]
        rocs: list[float] = []
        prs: list[float] = []
        for j in range(len(label_names)):
            y = yt[:, j]
            n_pos = int(y.sum())
            if n_pos == 0 or n_pos == len(y):
                continue
            rocs.append(roc_auc_score(y, yp[:, j]))
            prs.append(average_precision_score(y, yp[:, j]))
        if rocs:
            roc_means.append(float(np.mean(rocs)))
            pr_means.append(float(np.mean(prs)))
    roc_lo, roc_hi = np.percentile(roc_means, [2.5, 97.5])
    pr_lo, pr_hi = np.percentile(pr_means, [2.5, 97.5])
    return {
        "mean_roc_auc_ci_lower": float(roc_lo),
        "mean_roc_auc_ci_upper": float(roc_hi),
        "mean_pr_auc_ci_lower": float(pr_lo),
        "mean_pr_auc_ci_upper": float(pr_hi),
    }

def save_metrics(
    metrics: pd.DataFrame,
    skipped: list[str],
    model: str,
    features: str,
    seed: int,
    results_dir: Path = RESULTS_DIR,
    Y_true: np.ndarray | None = None,
    Y_pred: np.ndarray | None = None,
    label_names: list[str] | None = None,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    n_bootstrap: int = BOOTSTRAP_N_RESAMPLES,
) -> dict:
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
        "mean_roc_auc_ci_lower": np.nan,
        "mean_roc_auc_ci_upper": np.nan,
        "mean_pr_auc_ci_lower": np.nan,
        "mean_pr_auc_ci_upper": np.nan,
    }
    if Y_true is not None and Y_pred is not None and label_names is not None:
        ci = bootstrap_ci(Y_true, Y_pred, label_names, n_resamples=n_bootstrap, seed=bootstrap_seed)
        summary.update(ci)

    if summary_path.exists():
        table = pd.read_csv(summary_path)
        table = table[~((table["model"] == model) & (table["features"] == features))]
        table = pd.concat([table, pd.DataFrame([summary])], ignore_index=True)
    else:
        table = pd.DataFrame([summary])
    # Ensure all summary columns exist.
    for col in _SUMMARY_COLUMNS:
        if col not in table.columns:
            table[col] = np.nan
    table = table[_SUMMARY_COLUMNS]
    table = table.sort_values(["model", "features"]).reset_index(drop=True)
    table.to_csv(summary_path, index=False)
    print(f"[eval] wrote {per_label_path}")
    print(f"[eval] wrote {summary_path}")
    return summary


def print_baseline_table(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    summary_path = results_dir / SUMMARY_PATH.name
    if not summary_path.exists():
        print("[eval] no baseline_metrics.csv yet")
        return pd.DataFrame()
    table = pd.read_csv(summary_path)
    display = table.copy()
    for col in ("mean_roc_auc", "mean_pr_auc"):
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    for lo, hi, base in (
        ("mean_roc_auc_ci_lower", "mean_roc_auc_ci_upper", "mean_roc_auc"),
        ("mean_pr_auc_ci_lower", "mean_pr_auc_ci_upper", "mean_pr_auc"),
    ):
        if lo in display.columns:
            display[f"{base}_ci"] = display.apply(
                lambda r: f"({r[lo]:.4f}, {r[hi]:.4f})", axis=1
            )
    print("\n===== Baseline results (test set) =====")
    cols = ["model", "features", "mean_roc_auc", "mean_pr_auc", "n_labels_skipped"]
    if "mean_roc_auc_ci" in display.columns:
        cols = ["model", "features", "mean_roc_auc", "mean_roc_auc_ci", "mean_pr_auc", "mean_pr_auc_ci", "n_labels_skipped"]
    print(display[cols].to_string(index=False))
    return table


def prevalence_baseline_predictions(Y_true: np.ndarray) -> np.ndarray:
    """Constant prediction equal to per-label prevalence (random-classifier baseline)."""
    Y_true = np.asarray(Y_true)
    prev = Y_true.mean(axis=0)
    return np.broadcast_to(prev, Y_true.shape).copy()


def upsert_prevalence_baseline(
    results_dir: Path,
    Y_true: np.ndarray,
    label_names: list[str],
    seed: int,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    Y_true = np.asarray(Y_true)
    pred = prevalence_baseline_predictions(Y_true)
    metrics, skipped = per_label_metrics(Y_true, pred, label_names)
    return save_metrics(
        metrics,
        skipped,
        model="prevalence",
        features="none",
        seed=seed,
        results_dir=results_dir,
        Y_true=Y_true,
        Y_pred=pred,
        label_names=label_names,
        bootstrap_seed=bootstrap_seed,
    )


def _cli(argv=None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Aromacle evaluation helpers")
    parser.add_argument("--prevalence-baseline", action="store_true",
                        help="Add a prevalence-only baseline row to baseline_metrics.csv")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.prevalence_baseline:
        parser.print_help()
        return 0

    from .data import default_data_path, load_dataset, scaffold_split

    data_path = args.data or default_data_path()
    smiles, Y, label_names = load_dataset(data_path)
    train_idx, val_idx, test_idx = scaffold_split(smiles, seed=args.seed)
    upsert_prevalence_baseline(
        args.results_dir, Y[test_idx], label_names, seed=args.seed
    )
    print_baseline_table(args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
