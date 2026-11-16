"""Train OpenPOM's MPNN-POM model on the Aromacle scaffold split.

This module is deliberately isolated: every DGL / PyTorch / DeepChem /
OpenPOM import happens *inside* a function, so ``python -m aromacle.train``
and the rest of the classical baseline pipeline keep working on a machine
without those dependencies (e.g. this macOS Apple-Silicon box, which has no
CUDA build of DGL).  The model itself is trained in Google Colab on a GPU.

Run (Colab GPU runtime, from the repository root)::

    python -m aromacle.gnn.train_openpom \
        --csv curated_GS_LF_merged_4983.csv \
        --split results/split_seed42.json \
        --results-dir results

Inputs:
  * the curated dataset CSV (``nonStereoSMILES`` column + 138 binary labels);
  * ``results/split_seed42.json`` (row indices into the CSV, see
    :func:`aromacle.data.save_split_json`).

Outputs:
  * ``results/openpom_test_preds.csv`` — one row per test molecule *in split
    order*, one column per label (138 columns), values are probabilities;
  * ``results/openpom_config.json`` — training config, seed, split sizes and
    the per-epoch history.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

SMILES_COLUMN = "nonStereoSMILES"
DESCRIPTORS_COLUMN = "descriptors"  # readable label text; never a feature/label
N_LABELS = 138


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OpenPOM MPNN-POM on the Aromacle scaffold split"
    )
    parser.add_argument(
        "--csv", default="curated_GS_LF_merged_4983.csv", help="Dataset CSV path"
    )
    parser.add_argument(
        "--split", default="results/split_seed42.json", help="Split JSON path"
    )
    parser.add_argument(
        "--results-dir", default="results", help="Directory for output files"
    )
    parser.add_argument("--epochs", type=int, default=40, help="Max training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience on validation ROC-AUC (epochs)",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


def label_columns(csv_path: str | Path) -> list[str]:
    """Return the label column names of the dataset CSV.

    The ``nonStereoSMILES`` (input) and ``descriptors`` (readable label text,
    a leakage vector) columns are excluded; the remaining columns are the
    binary odor labels, in the exact order used everywhere else in Aromacle.
    """
    import pandas as pd

    columns = list(pd.read_csv(csv_path, nrows=0).columns)
    labels = [c for c in columns if c not in (SMILES_COLUMN, DESCRIPTORS_COLUMN)]
    if len(labels) != N_LABELS:
        raise ValueError(
            f"expected {N_LABELS} label columns, found {len(labels)} in {csv_path}"
        )
    return labels


def model_hyperparameters(
    *,
    batch_size: int,
    number_atom_features: int,
    number_bond_features: int,
    model_dir: str,
    device_name: str,
) -> dict:
    """Canonical MPNN-POM hyperparameters from the OpenPOM example notebook.

    Kept in one place so the trained model and the recorded config can never
    drift apart.
    """
    return dict(
        batch_size=batch_size,
        loss_aggr_type="sum",
        node_out_feats=100,
        edge_hidden_feats=75,
        edge_out_feats=100,
        num_step_message_passing=5,
        mpnn_residual=True,
        message_aggregator_type="sum",
        mode="classification",
        number_atom_features=number_atom_features,
        number_bond_features=number_bond_features,
        n_classes=1,
        readout_type="set2set",
        num_step_set2set=3,
        num_layer_set2set=2,
        ffn_hidden_list=[392, 392],
        ffn_embeddings=256,
        ffn_activation="relu",
        ffn_dropout_p=0.12,
        ffn_dropout_at_input_no_act=False,
        weight_decay=1e-5,
        self_loop=False,
        optimizer_name="adam",
        log_frequency=32,
        model_dir=model_dir,
        device_name=device_name,
    )


def build_model(n_tasks: int, train_ratios: list, args: argparse.Namespace, device_name: str):
    """Construct an ``MPNNPOMModel`` with OpenPOM's canonical settings."""
    import deepchem as dc
    from openpom.feat.graph_featurizer import GraphConvConstants
    from openpom.models.mpnn_pom import MPNNPOMModel

    learning_rate = dc.models.optimizers.ExponentialDecay(
        initial_rate=0.001, decay_rate=0.5, decay_steps=32 * 15, staircase=True
    )
    return MPNNPOMModel(
        n_tasks=n_tasks,
        class_imbalance_ratio=train_ratios,
        learning_rate=learning_rate,
        **model_hyperparameters(
            batch_size=args.batch_size,
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,
            model_dir=str(Path(args.results_dir) / "openpom_experiments"),
            device_name=device_name,
        ),
    )


def mean_roc_auc(model, dataset, n_tasks: int) -> float:
    """Mean ROC-AUC over labels that have both classes in ``dataset``.

    Uses the same single-class-skip rule as :mod:`aromacle.evaluate`, so the
    early-stopping signal is computed on identical terms to the final report.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(dataset.y)
    y_pred = np.asarray(model.predict(dataset))
    scores = [
        roc_auc_score(y_true[:, j], y_pred[:, j])
        for j in range(n_tasks)
        if 0 < y_true[:, j].sum() < len(y_true)
    ]
    return float(np.mean(scores))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import numpy as np
    import pandas as pd
    import torch
    import deepchem as dc
    from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
    from openpom.utils.data_utils import get_class_imbalance_ratio

    from ..data import load_split_json, split_indices_from_json

    # --- Reproducibility ---------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[openpom] device={device_name} torch={torch.__version__} "
        f"cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}"
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset -----------------------------------------------------------
    labels = label_columns(args.csv)
    featurizer = GraphFeaturizer()
    loader = dc.data.CSVLoader(
        tasks=labels, feature_field=SMILES_COLUMN, featurizer=featurizer
    )
    dataset = loader.create_dataset(inputs=[str(args.csv)])
    n_tasks = len(dataset.tasks)
    print(f"[openpom] featurized {len(dataset)} molecules, {n_tasks} tasks")

    payload = load_split_json(args.split)
    train_idx, val_idx, test_idx = split_indices_from_json(payload)
    if len(train_idx) + len(val_idx) + len(test_idx) != len(dataset):
        raise ValueError(
            f"split covers {len(train_idx) + len(val_idx) + len(test_idx)} rows "
            f"but the dataset has {len(dataset)}"
        )
    print(
        f"[openpom] split: train={len(train_idx)} val={len(val_idx)} "
        f"test={len(test_idx)}"
    )

    train_dataset = dataset.select(train_idx)
    valid_dataset = dataset.select(val_idx)
    test_dataset = dataset.select(test_idx)

    # Class-imbalance ratios must come from the TRAIN split only.
    train_ratios = get_class_imbalance_ratio(train_dataset)
    assert len(train_ratios) == n_tasks

    model = build_model(n_tasks, train_ratios, args, device_name)

    # --- Training with early stopping on validation ROC-AUC ----------------
    history: list[dict] = []
    best_val = float("-inf")
    best_epoch = 0
    best_state = None
    epochs_run = 0
    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        loss = model.fit(
            train_dataset,
            nb_epoch=1,
            max_checkpoints_to_keep=1,
            deterministic=False,
            restore=epoch > 1,
        )
        train_auc = mean_roc_auc(model, train_dataset, n_tasks)
        val_auc = mean_roc_auc(model, valid_dataset, n_tasks)
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss),
                "train_roc_auc": train_auc,
                "val_roc_auc": val_auc,
            }
        )
        print(
            f"[openpom] epoch {epoch}/{args.epochs} loss={loss:.4f} "
            f"train_roc_auc={train_auc:.4f} val_roc_auc={val_auc:.4f}"
        )
        if val_auc > best_val:
            best_val = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.model.state_dict())
        elif epoch - best_epoch >= args.patience:
            print(
                f"[openpom] early stopping at epoch {epoch} "
                f"(best epoch {best_epoch}, val ROC-AUC {best_val:.4f})"
            )
            break

    if best_state is not None:
        model.model.load_state_dict(best_state)
    print(
        f"[openpom] restoring best model from epoch {best_epoch} "
        f"(val ROC-AUC {best_val:.4f})"
    )

    # --- Test predictions --------------------------------------------------
    test_pred = np.asarray(model.predict(test_dataset))
    if test_pred.ndim == 3 and test_pred.shape[-1] == 1:
        test_pred = test_pred[..., 0]
    if test_pred.shape != (len(test_idx), n_tasks):
        raise ValueError(
            f"unexpected prediction shape {test_pred.shape}, "
            f"expected {(len(test_idx), n_tasks)}"
        )

    preds_path = results_dir / "openpom_test_preds.csv"
    pd.DataFrame(test_pred, columns=labels).to_csv(preds_path, index=False)
    print(f"[openpom] wrote {preds_path} ({test_pred.shape[0]} x {test_pred.shape[1]})")

    # --- Config ------------------------------------------------------------
    config = {
        "model": "openpom_mpnn_pom",
        "seed": args.seed,
        "device": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "csv": str(args.csv),
        "split": str(args.split),
        "n_tasks": n_tasks,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "max_epochs": args.epochs,
        "patience": args.patience,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_val,
        "learning_rate": {
            "optimizer": "adam",
            "schedule": "ExponentialDecay",
            "initial_rate": 0.001,
            "decay_rate": 0.5,
            "decay_steps": 32 * 15,
            "staircase": True,
        },
        "model_hyperparameters": model_hyperparameters(
            batch_size=args.batch_size,
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,
            model_dir=str(results_dir / "openpom_experiments"),
            device_name=device_name,
        ),
        "history": history,
    }
    config_path = results_dir / "openpom_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"[openpom] wrote {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
