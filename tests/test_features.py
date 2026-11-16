"""Tests for feature computation, including train-only imputation."""

from __future__ import annotations

import numpy as np
import pytest

from aromacle import features


def test_rdkit_descriptors_no_nan_with_train_indices() -> None:
    smiles = ["C", "CC", "CCC", "CCCC"]
    X = features.rdkit_descriptors(smiles, train_indices=[0, 1])
    assert X.shape[0] == 4
    assert np.isfinite(X).all()


def test_descriptor_imputation_uses_train_stats(monkeypatch) -> None:
    smiles = ["C", "CC", "CCC", "CCCC"]
    data = [1.0, 2.0, np.nan, 100.0]

    counter = {"i": 0}

    def fake_calc(mol):
        i = counter["i"]
        counter["i"] += 1
        return {"d": data[i]}

    monkeypatch.setattr("aromacle.features.Descriptors.CalcMolDescriptors", fake_calc)

    # Reset counter before call.
    counter["i"] = 0
    X = features.rdkit_descriptors(smiles, train_indices=[0, 1])
    # Row 2 had NaN; imputed with train median of [1,2] = 1.5.
    assert X[2, 0] == pytest.approx(1.5)

    # Global (no train split) would use 0.0 fallback for the NaN.
    counter["i"] = 0
    X0 = features.rdkit_descriptors(smiles)
    assert X0[2, 0] == pytest.approx(0.0)
