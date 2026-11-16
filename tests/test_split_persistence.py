"""Tests for the persisted scaffold split JSON."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aromacle.data import (
    default_data_path,
    load_dataset,
    load_split_json,
    murcko_scaffold,
    scaffold_split,
    split_indices_from_json,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "curated_GS_LF_merged_4983.csv"
SPLIT_PATH = Path(__file__).resolve().parents[1] / "results" / "split_seed42.json"


@pytest.fixture(scope="module")
def smiles() -> list[str]:
    if not DATA_PATH.exists():
        pytest.skip(f"dataset not found: {DATA_PATH}")
    validated, _, _ = load_dataset(DATA_PATH)
    return validated


def test_split_json_exists() -> None:
    assert SPLIT_PATH.exists(), f"missing split file: {SPLIT_PATH}"


def test_split_json_reproduces_scaffold_split(smiles: list[str]) -> None:
    train, val, test = scaffold_split(smiles)
    payload = load_split_json(SPLIT_PATH)
    tr, va, te = split_indices_from_json(payload)
    np.testing.assert_array_equal(np.asarray(tr), train)
    np.testing.assert_array_equal(np.asarray(va), val)
    np.testing.assert_array_equal(np.asarray(te), test)
    # SMILES round-trip.
    assert payload["train_smiles"] == [smiles[i] for i in train]
    assert payload["val_smiles"] == [smiles[i] for i in val]
    assert payload["test_smiles"] == [smiles[i] for i in test]
    # Sizes.
    assert len(tr) == 3986 and len(va) == 499 and len(te) == 498
