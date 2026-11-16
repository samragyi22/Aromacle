"""Evaluate a saved predictions CSV (e.g. OpenPOM) on the Aromacle split.

Usage::

    python -m aromacle.evaluate_preds --preds results/openpom_test_preds.csv
    python -m aromacle.evaluate_preds --preds results/openpom_test_preds.csv \\
        --model openpom --features mpnn

The predictions CSV must contain one row per test molecule (in split order)
and one column per odor label (the 138 label names).  The script reuses the
exact evaluation logic in :mod:`aromacle.evaluate` (same skipped labels,
ROC-AUC, PR-AUC and bootstrap CIs), so GNN and classical baselines are
compared on identical terms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import default_data_path, load_dataset, load_split_json, split_indices_from_json
from .evaluate import (
    BOOTSTRAP_SEED,
    bootstrap_ci,
    per_label_metrics,
    print_baseline_table,
    save_metrics,
)


def default_split_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "results" / "split_seed42.json"
    if candidate.exists():
        return candidate
    return Path("results") / "split_seed42.json"


def load_predictions(preds_path: Path, label_names: list[str]) -> np.ndarray:
    df = pd.read_csv(preds_path)
    missing = [name for name in label_names if name not in df.columns]
    if missing:
        raise ValueError(
            f"Predictions CSV is missing {len(missing)} label columns: {missing[:10]}"
        )
    pred = df[label_names].to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError("Predictions contain non-finite values")
    return pred


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a predictions CSV")
    parser.add_argument("--preds", type=Path, required=True, help="Predictions CSV")
    parser.add_argument("--model", default="openpom", help="Model name for the summary row")
    parser.add_argument("--features", default="mpnn", help="Feature/model variant name")
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed")
    parser.add_argument("--data", type=Path, default=None, help="Path to the dataset CSV")
    parser.add_argument("--split", type=Path, default=None, help="Path to split JSON")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Where to write metrics"
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=BOOTSTRAP_SEED, help="Bootstrap seed"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = args.data or default_data_path()
    split_path = args.split or default_split_path()

    smiles, Y, label_names = load_dataset(data_path)
    payload = load_split_json(split_path)
    train_idx, val_idx, test_idx = split_indices_from_json(payload)
    test_idx = np.asarray(test_idx, dtype=int)

    Y_test = Y[test_idx]
    pred = load_predictions(args.preds, label_names)
    if pred.shape[0] != len(test_idx):
        raise ValueError(
            f"Predictions have {pred.shape[0]} rows but test split has {len(test_idx)}"
        )

    metrics, skipped = per_label_metrics(Y_test, pred, label_names)
    ci = bootstrap_ci(Y_test, pred, label_names, seed=args.bootstrap_seed)
    print(
        f"[eval] {args.model}/{args.features}: mean ROC-AUC={metrics['roc_auc'].mean():.4f} "
        f"({ci['mean_roc_auc_ci_lower']:.4f}, {ci['mean_roc_auc_ci_upper']:.4f}), "
        f"mean PR-AUC={metrics['pr_auc'].mean():.4f} "
        f"({ci['mean_pr_auc_ci_lower']:.4f}, {ci['mean_pr_auc_ci_upper']:.4f}) "
        f"({len(metrics)} labels, {len(skipped)} skipped)"
    )

    save_metrics(
        metrics,
        skipped,
        model=args.model,
        features=args.features,
        seed=args.seed,
        results_dir=args.results_dir,
        Y_true=Y_test,
        Y_pred=pred,
        label_names=label_names,
        bootstrap_seed=args.bootstrap_seed,
        ci=ci,
    )
    # Keep a small sidecar with the CI values (already in baseline_metrics.csv).
    sidecar = {
        "model": args.model,
        "features": args.features,
        "seed": args.seed,
        "n_test": int(len(test_idx)),
        "n_labels_evaluated": int(len(metrics)),
        "n_labels_skipped": int(len(skipped)),
        "skipped_labels": skipped,
        "mean_roc_auc": float(metrics["roc_auc"].mean()),
        "mean_pr_auc": float(metrics["pr_auc"].mean()),
    }
    sidecar.update(ci)
    sidecar_path = args.results_dir / f"{args.model}_{args.features}_eval.json"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"[eval] wrote {sidecar_path}")

    print_baseline_table(args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
