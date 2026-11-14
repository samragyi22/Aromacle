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


def rdkit_descriptors(smiles: list[str]) -> np.ndarray:
    """Compute the full RDKit 2D descriptor set (~217 values per molecule).

    Non-finite values (NaN/inf, which some descriptors produce for edge
    cases) are replaced with 0.0 so downstream estimators never see them.

    Returns
    -------
    ndarray of shape (n_molecules, n_descriptors), dtype float64
    """
    rows: list[np.ndarray] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(None)  # type: ignore[arg-type]
            continue
        desc = Descriptors.CalcMolDescriptors(mol)
        rows.append(np.fromiter(desc.values(), dtype=np.float64, count=len(desc)))

    # Determine the descriptor dimensionality from the first valid molecule.
    n_desc = next(r.shape[0] for r in rows if r is not None)
    out = np.zeros((len(smiles), n_desc), dtype=np.float64)
    for i, row in enumerate(rows):
        if row is not None:
            out[i] = row
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def featurize(
    smiles: list[str],
    feature_set: str,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
) -> np.ndarray:
    """Dispatch to a feature set by name: ``'morgan'`` or ``'descriptors'``."""
    if feature_set == "morgan":
        return morgan_fingerprints(smiles, radius=radius, n_bits=n_bits)
    if feature_set == "descriptors":
        return rdkit_descriptors(smiles)
    raise ValueError(f"Unknown feature set {feature_set!r}; expected 'morgan' or 'descriptors'")
