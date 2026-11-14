"""Tests for the Bemis-Murcko scaffold split.

Run from the repository root:  pytest -v
Requires the dataset at data/raw/curated_GS_LF_merged_4983.csv (gitignored);
tests skip gracefully if it is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aromacle.data import DEFAULT_SPLIT_FRAC, load_dataset, murcko_scaffold, scaffold_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "curated_GS_LF_merged_4983.csv"


@pytest.fixture(scope="module")
def smiles() -> list[str]:
    if not DATA_PATH.exists():
        pytest.skip(f"dataset not found: {DATA_PATH}")
    validated, _, _ = load_dataset(DATA_PATH)
    return validated


@pytest.fixture(scope="module")
def splits(smiles: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return scaffold_split(smiles)


def test_splits_are_disjoint_and_cover_all(smiles: list[str], splits) -> None:
    train, val, test = splits
    combined = np.concatenate(splits)
    assert len(combined) == len(smiles), "split sizes must sum to the dataset size"
    assert len(np.unique(combined)) == len(smiles), "splits must not overlap"
    assert set(combined.tolist()) == set(range(len(smiles))), "splits must cover all molecules"


def test_no_scaffold_in_more_than_one_split(smiles: list[str], splits) -> None:
    scaffolds = [murcko_scaffold(s) for s in smiles]
    split_of: dict[str, set[int]] = {}
    for split_id, idx in enumerate(splits):
        for i in idx:
            sc = scaffolds[i]
            if sc == "":  # acyclic molecules legitimately share the empty scaffold
                continue
            split_of.setdefault(sc, set()).add(split_id)
    leaks = {sc: ids for sc, ids in split_of.items() if len(ids) > 1}
    assert not leaks, f"non-empty scaffolds leaked across splits: {sorted(leaks)[:10]}"


def test_split_sizes_within_3_percent(smiles: list[str], splits) -> None:
    n = len(smiles)
    for name, frac, idx in zip(("train", "val", "test"), DEFAULT_SPLIT_FRAC, splits):
        deviation = abs(len(idx) / n - frac)
        assert deviation <= 0.03, f"{name} fraction {len(idx) / n:.3f} deviates {deviation:.3f} from {frac}"


def test_acyclic_molecules_spread_in_target_proportions(smiles: list[str], splits) -> None:
    empty_mask = np.array([murcko_scaffold(s) == "" for s in smiles])
    n_empty = int(empty_mask.sum())
    assert n_empty > 0, "expected acyclic molecules in this dataset"
    for name, frac, idx in zip(("train", "val", "test"), DEFAULT_SPLIT_FRAC, splits):
        share = int(empty_mask[idx].sum()) / n_empty
        assert abs(share - frac) <= 0.02, f"{name} holds {share:.3f} of acyclic molecules, target {frac}"


def test_split_is_deterministic(smiles: list[str]) -> None:
    first = scaffold_split(smiles, seed=42)
    second = scaffold_split(smiles, seed=42)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
