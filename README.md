# Aromacle

**An oracle for molecular scent.** Aromacle predicts the odor descriptors of a
molecule (e.g. "fruity", "woody", "musky") from its chemical structure given as
a SMILES string.

> **Note:** Aromacle does not physically smell anything. Odor perception is
> predicted from chemical structure — a multi-label classification problem on
> molecular graphs/fingerprints.

## Project layout

```
aromacle/
  README.md, LICENSE, requirements.txt, .gitignore
  data/            # raw & processed data (gitignored if large)
  notebooks/       # EDA, error analysis
  src/aromacle/    # data.py, features.py, models.py, train.py, evaluate.py, predict.py
  app/             # Streamlit/Gradio app
  results/         # tables, plots, saved metrics
  tests/
```

## Environment

```bash
conda create -n aromacle python=3.10 -y
conda activate aromacle
pip install -r requirements.txt
pip install -e .          # makes `python -m aromacle.*` work anywhere
```

## Running the baselines

```bash
python -m aromacle.train --model rf    --features morgan
python -m aromacle.train --model rf    --features descriptors
python -m aromacle.train --model xgb   --features morgan
python -m aromacle.train --model xgb   --features descriptors
pytest -v                                 # scaffold-split tests
```

Every run writes `results/<model>_<features>_per_label.csv` and upserts a row
into `results/baseline_metrics.csv`.

## Roadmap

1. **Environment** — Python 3.10, dependencies, repo scaffolding. ✅
2. **Classical baseline** — Morgan fingerprints + RandomForest/XGBoost. ✅
3. **GNN comparison** — OpenPOM's MPNNPOMModel (DeepChem) on the same scaffold split.
4. **Similarity search** — embedding-based "molecules that smell like X".
5. **App** — Streamlit/Gradio: SMILES → molecule image, top predicted descriptors with confidence, similar molecules.
6. **Deploy** — Hugging Face Spaces / Streamlit Cloud.

## Data

- **Source:** curated GoodScents + Leffingwell merged dataset from the
  [OpenPOM](https://github.com/ARY2260/openpom) repository
  (`openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv`).
- **Size:** 4,983 molecules x 138 binary odor labels.
- **Download** (from the repository root):

  ```bash
  mkdir -p data/raw
  curl -L -o data/raw/curated_GS_LF_merged_4983.csv \
    https://raw.githubusercontent.com/ARY2260/openpom/main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv
  ```

- **Note:** `data/` is gitignored — the CSV is not committed; run the command above after cloning.

**Format.** Columns: `nonStereoSMILES` (structure), `descriptors`
(';'-separated human-readable label text), then 138 binary 0/1 label columns.
`descriptors` is *never* used as a feature or label — it is the readable form
of the label matrix and would leak the answer.

**Splits.** All experiments use a Bemis-Murcko scaffold split (80/10/10,
seed 42), never a random split. Acyclic molecules share the empty scaffold
(44.9% of the dataset) and are distributed randomly across splits instead of
being lumped together; non-empty scaffold groups are assigned whole, largest
first.

## Baseline results

Phase 1 classical baselines on the test split (mean over 136 evaluated labels;
`brandy` and `cherry` have a single class in the test split and are skipped).
No hyperparameter tuning; validation split is reported by the CLI for reference.
Full source: [`results/baseline_metrics.csv`](results/baseline_metrics.csv).

| Model | Features | Mean ROC-AUC | Mean PR-AUC | Labels skipped |
|-------|----------|--------------|-------------|----------------|
| RandomForest | Morgan (r=2, 2048 bits) | 0.7979 | 0.2283 | 2 |
| RandomForest | RDKit descriptors (217) | **0.8387** | 0.2910 | 2 |
| XGBoost (OvR) | Morgan (r=2, 2048 bits) | 0.8106 | 0.2521 | 2 |
| XGBoost (OvR) | RDKit descriptors (217) | 0.8380 | **0.3049** | 2 |

## License

MIT
