"""Featurization: Morgan fingerprints and RDKit physicochemical descriptors."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

DEFAULT_RADIUS = 2
DEFAULT_N_BITS = 2048

# Morgan generators are expensive to construct; cache one per (radius, n_bits).
_FINGERPRINT_GENERATORS: dict[tuple[int, int], object] = {}


def _morgan_generator(radius: int, n_bits: int):
    """Return a cached RDKit Morgan fingerprint generator."""
    key = (radius, n_bits)
    if key not in _FINGERPRINT_GENERATORS:
        _FINGERPRINT_GENERATORS[key] = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
    return _FINGERPRINT_GENERATORS[key]


def morgan_fingerprints(
    smiles: list[str],
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
) -> np.ndarray:
    """Compute Morgan (circular) fingerprints as a uint8 bit array.

    Parameters
    ----------
    smiles
        SMILES strings; should already be validated with
        :func:`aromacle.data.load_dataset` (unparseable input yields an
        all-zero row rather than an error).
    radius
        Morgan fingerprint radius (ECFP4 uses radius=2).
    n_bits
        Fingerprint length in bits.

    Returns
    -------
    ndarray of shape (n_molecules, n_bits), dtype uint8
        Bit vector, 0/1 per bit.
    """
    generator = _morgan_generator(radius, n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # validated upstream; keep an all-zero row as a safe fallback
        fp = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, out[i])
    return out


def _resolve_train_indices(
    n: int, train_indices=None, train_idx=None, train_mask=None
):
    if train_mask is not None:
        mask = np.asarray(train_mask, dtype=bool)
        return np.flatnonzero(mask)
    for candidate in (train_indices, train_idx):
        if candidate is not None:
            return np.asarray(candidate, dtype=int)
    return None


def rdkit_descriptors(
    smiles: list[str],
    train_indices=None,
    train_idx=None,
    train_mask=None,
    impute_strategy: str = "median",
):
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(None)
            continue
        desc = Descriptors.CalcMolDescriptors(mol)
        rows.append(np.fromiter(desc.values(), dtype=np.float64, count=len(desc)))

    n_desc = next(r.shape[0] for r in rows if r is not None)
    out = np.zeros((len(smiles), n_desc), dtype=np.float64)
    for i, row in enumerate(rows):
        if row is not None:
            out[i] = row

    train_idx_arr = _resolve_train_indices(len(smiles), train_indices, train_idx, train_mask)

    nonfinite = ~np.isfinite(out)
    if not nonfinite.any():
        return out

    if train_idx_arr is None:
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    train_rows = out[train_idx_arr]
    if impute_strategy == "mean":
        stat = np.nanmean(np.where(np.isfinite(train_rows), train_rows, np.nan), axis=0)
    else:
        stat = np.nanmedian(np.where(np.isfinite(train_rows), train_rows, np.nan), axis=0)
    stat = np.where(np.isfinite(stat), stat, 0.0)

    for j in range(n_desc):
        col_mask = nonfinite[:, j]
        if col_mask.any():
            out[col_mask, j] = stat[j]
    return out

def featurize(
    smiles: list[str],
    feature_set: str,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    train_indices=None,
    train_idx=None,
    train_mask=None,
    impute_strategy: str = "median",
):
    if feature_set == "morgan":
        return morgan_fingerprints(smiles, radius=radius, n_bits=n_bits)
    if feature_set == "descriptors":
        return rdkit_descriptors(
            smiles,
            train_indices=train_indices,
            train_idx=train_idx,
            train_mask=train_mask,
            impute_strategy=impute_strategy,
        )
    raise ValueError(f"Unknown feature set {feature_set!r}; expected 'morgan' or 'descriptors'")

