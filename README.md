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
  src/aromacle/    # data.py, features.py, models.py, train.py, evaluate.py, evaluate_preds.py
  src/aromacle/gnn/# OpenPOM MPNN-POM training (runs on Colab GPU; imports kept isolated)
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
3. **GNN comparison** — OpenPOM's MPNNPOMModel (DeepChem) on the same scaffold split. 🔧 code ready, runs on Colab GPU.
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
Bootstrap 95% CIs (500 resamples of test molecules, seed 42) in parentheses.
Full source: [`results/baseline_metrics.csv`](results/baseline_metrics.csv).

| Model | Features | Mean ROC-AUC (95% CI) | Mean PR-AUC (95% CI) | Labels skipped |
|-------|----------|----------------------|----------------------|----------------|
| Prevalence (random) | label frequencies | 0.5000 (0.5000, 0.5000) | 0.0325 (0.0316, 0.0355) | 2 |
| RandomForest | Morgan (r=2, 2048 bits) | 0.7979 (0.7796, 0.8129) | 0.2283 (0.2231, 0.2751) | 2 |
| RandomForest | RDKit descriptors (217) | **0.8415** (0.8233, 0.8554) | 0.2941 (0.2922, 0.3430) | 2 |
| XGBoost (OvR) | Morgan (r=2, 2048 bits) | 0.8106 (0.7931, 0.8249) | 0.2521 (0.2488, 0.2990) | 2 |
| XGBoost (OvR) | RDKit descriptors (217) | 0.8384 (0.8228, 0.8524) | **0.3048** (0.2987, 0.3553) | 2 |

## GNN comparison (Phase 2 — OpenPOM MPNN-POM)

OpenPOM's MPNN-POM needs a CUDA build of DGL and specific torch versions, so it
runs in a **Google Colab GPU notebook**, not on this Mac. It is trained on the
*same* scaffold split (`results/split_seed42.json`) and evaluated with the
*same* code as the classical baselines, so the comparison is apples-to-apples.

```text
notebooks/02_openpom_colab.ipynb   # installs deps, trains, predicts the test split
src/aromacle/gnn/train_openpom.py  # training logic (DGL/PyTorch imports isolated)
src/aromacle/evaluate_preds.py     # scores a saved predictions CSV locally
```

Workflow:

1. Run `notebooks/02_openpom_colab.ipynb` on Colab (T4). It writes
   `results/openpom_test_preds.csv` (rows = test molecules in split order,
   138 label columns) and `results/openpom_config.json`.
2. Copy the predictions CSV into the local `results/`, then score it on
   identical terms and append the row to `results/baseline_metrics.csv`:

   ```bash
   python -m aromacle.evaluate_preds --preds results/openpom_test_preds.csv \
       --model openpom --features mpnn
   ```

The OpenPOM row is intentionally absent from the table above until that Colab
run completes. Only claim OpenPOM "beats" a baseline if its bootstrap CI
supports it.

## License

MIT
