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
```

## Roadmap

1. **Environment** — Python 3.10, dependencies, repo scaffolding. ✅
2. **Classical baseline** — Morgan fingerprints + RandomForest/XGBoost.
3. **GNN comparison** — OpenPOM's MPNNPOMModel (DeepChem) on the same scaffold split.
4. **Similarity search** — embedding-based "molecules that smell like X".
5. **App** — Streamlit/Gradio: SMILES → molecule image, top predicted descriptors with confidence, similar molecules.
6. **Deploy** — Hugging Face Spaces / Streamlit Cloud.

## Data

Curated GoodScents + Leffingwell merged dataset from the
[OpenPOM](https://github.com/AakashGupta18/OpenPOM) repo
(`curated_GS_LF_merged_4983.csv`): ~5k molecules, ~138 binary odor labels.

## License

MIT
