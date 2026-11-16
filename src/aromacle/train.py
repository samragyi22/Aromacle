"""Training CLI for the classical baselines.

Usage::

    python -m aromacle.train --model rf --features morgan
    python -m aromacle.train --model xgb --features descriptors

The model is fit on the training split only; the validation split is
reported for reference (and reserved for future hyperparameter choices),
and the test split is evaluated once at the end.  Seeds are fixed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .data import DEFAULT_SEED, load_dataset, scaffold_split
from .evaluate import per_label_metrics, print_baseline_table, save_metrics
from .features import featurize
from .models import build_model, predict_positive_proba


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aromacle baseline training")
    parser.add_argument("--model", choices=("rf", "xgb"), required=True)
    parser.add_argument("--features", choices=("morgan", "descriptors"), required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data", type=Path, default=None, help="Path to the dataset CSV")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Where to write metrics"
    )
    return parser.parse_args(argv)


def _mean_auc_line(tag: str, metrics) -> str:
    return (
        f"[eval] {tag} mean ROC-AUC={metrics['roc_auc'].mean():.4f}, "
        f"mean PR-AUC={metrics['pr_auc'].mean():.4f} "
        f"({len(metrics)} labels)"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[train] model={args.model} features={args.features} seed={args.seed}")

    from .data import default_data_path  # local import to keep CLI help fast

    data_path = args.data or default_data_path()
    smiles, Y, label_names = load_dataset(data_path)

    train_idx, val_idx, test_idx = scaffold_split(smiles, seed=args.seed)

    t0 = time.time()
    X = featurize(smiles, args.features, train_indices=train_idx)
    print(f"[featurize] {args.features}: shape {X.shape} in {time.time() - t0:.1f}s")

    model = build_model(args.model, seed=args.seed)
    model.fit(X[train_idx], Y[train_idx])
    skipped_fit = getattr(model, "skipped_labels_", [])
    if skipped_fit:
        print(f"[train] labels skipped during fit (no positive/negative): {len(skipped_fit)}")

    # Validation metrics: reference only, no tuning is done on the test set.
    val_pred = predict_positive_proba(model, X[val_idx], Y.shape[1])
    val_metrics, _ = per_label_metrics(Y[val_idx], val_pred, label_names)
    print(_mean_auc_line("val (reference)", val_metrics))

    test_pred = predict_positive_proba(model, X[test_idx], Y.shape[1])
    test_metrics, skipped = per_label_metrics(Y[test_idx], test_pred, label_names)
    print(_mean_auc_line("test", test_metrics))
    print(f"[eval] skipped labels (single class in test): {len(skipped)}")
    if skipped:
        print(f"[eval]   {', '.join(skipped)}")

    save_metrics(
        test_metrics,
        skipped,
        model=args.model,
        features=args.features,
        seed=args.seed,
        results_dir=args.results_dir,
        Y_true=Y[test_idx],
        Y_pred=test_pred,
        label_names=label_names,
    )
    print_baseline_table(args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
