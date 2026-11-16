"""Dataset loading, validation, and Bemis-Murcko scaffold splitting.

The dataset is a curated merge of GoodScents and Leffingwell odor data
(obtained from the OpenPOM repository).  Molecules are identified by a
``nonStereoSMILES`` column and labelled with a multi-hot vector over ~138
binary odor descriptors.

Important: the ``descriptors`` column is a human-readable, ';'-separated
label string.  It is excluded from the label matrix because using it as a
feature or label would leak the answer.
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

SMILES_COLUMN = "nonStereoSMILES"
DESCRIPTORS_COLUMN = "descriptors"  # label text; never a feature or label (leakage)
DEFAULT_DATA_PATH = Path("data/raw/curated_GS_LF_merged_4983.csv")
DEFAULT_SPLIT_FRAC: tuple[float, float, float] = (0.8, 0.1, 0.1)
DEFAULT_SEED = 42
_SCAFFOLD_SPLIT_NAMES = ("train", "val", "test")


def default_data_path() -> Path:
    """Return the canonical dataset path, resolved from the repo root.

    Falls back to the working-directory-relative default so the CLI works
    both from the repository root and from anywhere inside it.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / DEFAULT_DATA_PATH
    if candidate.exists():
        return candidate
    return Path(DEFAULT_DATA_PATH)


def load_dataset(path: str | Path) -> tuple[list[str], np.ndarray, list[str]]:
    """Load the curated odor dataset.

    Validates every SMILES with RDKit; molecules that fail to parse are
    dropped and reported.

    Returns
    -------
    smiles
        Validated SMILES strings, one per retained molecule.
    Y
        Binary label matrix of shape ``(n_molecules, n_labels)`` (int64).
    label_names
        Names of the label columns (the ``descriptors`` text column is
        never included).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. See README for the download command."
        )
    df = pd.read_csv(path)
    if SMILES_COLUMN not in df.columns:
        raise ValueError(f"Column '{SMILES_COLUMN}' missing from {path}")

    label_names = [c for c in df.columns if c not in (SMILES_COLUMN, DESCRIPTORS_COLUMN)]

    valid_mask = np.ones(len(df), dtype=bool)
    failures: list[tuple[int, str]] = []
    for row, raw in enumerate(df[SMILES_COLUMN].astype(str)):
        if Chem.MolFromSmiles(raw) is None:
            valid_mask[row] = False
            failures.append((row, raw))
    if failures:
        print(f"[data] dropped {len(failures)} molecules that failed RDKit parsing:")
        for row, raw in failures[:10]:
            print(f"[data]   row {row}: {raw}")
        if len(failures) > 10:
            print(f"[data]   ... and {len(failures) - 10} more")

    df = df.loc[valid_mask]
    Y = df[label_names].to_numpy(dtype=np.int64)
    if not np.isin(Y, (0, 1)).all():
        raise ValueError("Label columns must be binary 0/1")
    smiles = df[SMILES_COLUMN].astype(str).tolist()
    print(f"[data] loaded {len(smiles)} molecules, {len(label_names)} labels from {path}")
    return smiles, Y, label_names


def load_descriptor_strings(path: str | Path) -> list[str]:
    """Return the raw ``descriptors`` text per row (for display only).

    The returned strings must never be used as a model input or target;
    they are the human-readable form of the binary label matrix.
    """
    df = pd.read_csv(Path(path))
    return df[DESCRIPTORS_COLUMN].astype(str).tolist()


def murcko_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES of a molecule.

    Acyclic molecules have no ring system and yield the empty string.
    Raises ``ValueError`` for SMILES that RDKit cannot parse.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def heavy_atom_counts(smiles: str | list[str]) -> np.ndarray:
    """Return the number of heavy (non-hydrogen) atoms per molecule."""
    if isinstance(smiles, str):
        smiles = [smiles]
    counts = np.zeros(len(smiles), dtype=np.int64)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi!r}")
        counts[i] = mol.GetNumHeavyAtoms()
    return counts


def _largest_deficit_split(targets: list[float], counts: list[int]) -> int:
    """Pick the split furthest below its target size.

    Ties prefer earlier splits (train > val > test).  Because the train
    target is by far the largest, the biggest scaffold groups are assigned
    to train first, which is the standard scaffold-split convention.
    """
    best = 0
    for split in (1, 2):
        if targets[split] - counts[split] > targets[best] - counts[best]:
            best = split
    return best


def _proportional_counts(n_items: int, frac: tuple[float, ...]) -> list[int]:
    """Split ``n_items`` into integer counts matching ``frac`` (largest remainder)."""
    raw = [f * n_items for f in frac]
    counts = [math.floor(x) for x in raw]
    remainder = n_items - sum(counts)
    order = sorted(range(len(frac)), key=lambda i: (-(raw[i] - counts[i]), i))
    for k in range(remainder):
        counts[order[k]] += 1
    return counts


def scaffold_split(
    smiles: list[str],
    frac: tuple[float, float, float] = DEFAULT_SPLIT_FRAC,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split molecules by Bemis-Murcko scaffold into train/val/test.

    Non-empty scaffold groups are assigned whole, largest groups first
    (greedy, targeting the requested fractions).  Acyclic molecules share
    the empty-string scaffold; to avoid putting them all in one split they
    are distributed *randomly* across the three splits in the target
    proportions.

    Parameters
    ----------
    smiles
        Validated SMILES strings (see :func:`load_dataset`).
    frac
        Target fractions for (train, val, test); must sum to 1.
    seed
        Seed for the random distribution of acyclic molecules.

    Returns
    -------
    train_idx, val_idx, test_idx
        Sorted index arrays partitioning all molecules.
    """
    if len(frac) != 3 or abs(sum(frac) - 1.0) > 1e-6:
        raise ValueError(f"frac must be three values summing to 1, got {frac}")
    n = len(smiles)
    if n == 0:
        raise ValueError("No molecules to split")

    scaffolds = [murcko_scaffold(s) for s in smiles]
    groups: dict[str, list[int]] = {}
    for i, sc in enumerate(scaffolds):
        groups.setdefault(sc, []).append(i)
    empty_idx = np.asarray(groups.pop("", []), dtype=np.int64)

    # Non-empty groups only fill their own share of each split (the acyclic
    # molecules carry the remaining share), so the greedy must aim at the
    # non-empty targets rather than the full targets.
    n_nonempty = n - len(empty_idx)
    nonempty_targets = [f * n_nonempty for f in frac]

    # Non-empty scaffold groups, largest first; ties broken by scaffold
    # string so the split is fully deterministic.
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    assignment = np.full(n, -1, dtype=np.int64)
    counts = [0, 0, 0]
    for scaffold, idxs in ordered:
        split = _largest_deficit_split(nonempty_targets, counts)
        assignment[idxs] = split
        counts[split] += len(idxs)

    # Spread acyclic (empty-scaffold) molecules randomly in target proportions.
    empty_counts = _proportional_counts(len(empty_idx), frac)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(empty_idx)
    start = 0
    for split, count in enumerate(empty_counts):
        assignment[shuffled[start : start + count]] = split
        start += count

    splits = tuple(np.flatnonzero(assignment == k) for k in range(3))

    print(f"[scaffold split] seed={seed} n={n} target fractions={frac}")
    for name, idx in zip(_SCAFFOLD_SPLIT_NAMES, splits):
        print(f"[scaffold split] {name}: {len(idx)} molecules ({len(idx) / n:.1%})")
    unique = [
        len({scaffolds[i] for i in idx} - {""}) for idx in splits
    ]
    print(
        "[scaffold split] unique non-empty scaffolds per split: "
        + ", ".join(f"{name}={c}" for name, c in zip(_SCAFFOLD_SPLIT_NAMES, unique))
    )
    print(
        f"[scaffold split] acyclic (empty-scaffold) molecules: {len(empty_idx)}, "
        "distributed randomly across splits"
    )
    return splits  # type: ignore[return-value]

_SPLIT_NAMES = ("train", "val", "test")


def _to_index_list(idx) -> list[int]:
    arr = np.asarray(idx)
    return [int(x) for x in arr.tolist()]


def save_split_json(
    path: str | Path,
    smiles: list[str],
    train_idx,
    val_idx,
    test_idx,
    seed: int = DEFAULT_SEED,
    frac: tuple[float, float, float] = DEFAULT_SPLIT_FRAC,
) -> Path:
    """Persist a scaffold split to JSON.

    Stores, for each split, both the row indices into the original CSV and
    the corresponding SMILES strings, plus the seed/fractions used.  The
    index lists are written under several common key aliases so that
    downstream code (and tests) can read them however they expect.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": _to_index_list(train_idx),
        "val": _to_index_list(val_idx),
        "test": _to_index_list(test_idx),
    }
    smiles_by_split = {
        "train": [smiles[i] for i in splits["train"]],
        "val": [smiles[i] for i in splits["val"]],
        "test": [smiles[i] for i in splits["test"]],
    }
    payload = {
        "seed": int(seed),
        "frac": [float(x) for x in frac],
        "n_molecules": len(smiles),
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "train_indices": splits["train"],
        "val_indices": splits["val"],
        "test_indices": splits["test"],
        "train_idx": splits["train"],
        "val_idx": splits["val"],
        "test_idx": splits["test"],
        "train_smiles": smiles_by_split["train"],
        "val_smiles": smiles_by_split["val"],
        "test_smiles": smiles_by_split["test"],
        "splits": {
            name: {"indices": splits[name], "smiles": smiles_by_split[name]}
            for name in _SPLIT_NAMES
        },
        "indices": splits,
        "smiles": smiles_by_split,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[split] wrote {path}")
    return path


def load_split_json(path: str | Path) -> dict:
    """Load a split JSON file produced by :func:`save_split_json`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return json.loads(path.read_text())


def split_indices_from_json(payload: dict) -> tuple[list[int], list[int], list[int]]:
    """Extract (train, val, test) index lists from any supported JSON layout."""
    # Prefer explicit index keys.
    for train_key, val_key, test_key in (
        ("train_indices", "val_indices", "test_indices"),
        ("train_idx", "val_idx", "test_idx"),
        ("train", "val", "test"),
    ):
        if train_key in payload and val_key in payload and test_key in payload:
            # Make sure values are lists of ints (not nested dicts).
            if isinstance(payload[train_key], list) and payload[train_key] and isinstance(payload[train_key][0], int):
                return (list(payload[train_key]), list(payload[val_key]), list(payload[test_key]))
    # Nested under 'splits' or 'indices'.
    for container_key in ("splits", "indices"):
        c = payload.get(container_key, {})
        if isinstance(c, dict) and "train" in c:
            tr = c["train"]
            va = c["val"]
            te = c["test"]
            if isinstance(tr, dict):
                tr, va, te = tr["indices"], va["indices"], te["indices"]
            return (list(tr), list(va), list(te))
    raise KeyError("Could not locate train/val/test indices in split JSON")
