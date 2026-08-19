"""One-off script: append the 19 linguistic features to the feat dataset.

Reads data_raw.csv and writes data.csv with the feature columns appended.
Run once before the main experiment pipeline.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # features.py lives here.
from features import compute_features_batch, FEATURE_NAMES

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "01_raw_datasets" / "feat"
df = pd.read_csv(DATA_DIR / "data_raw.csv")
texts = df['clean_essay'].fillna('').astype(str).tolist()
print(f"computing 19 features for {len(texts)} essays ...")

feats = compute_features_batch(texts)
for i, c in enumerate(FEATURE_NAMES):
    df[c] = feats[:, i]

df.to_csv(DATA_DIR / "data.csv", index=False)
print(f'done: {df.shape[0]} rows x {df.shape[1]} cols, {len(FEATURE_NAMES)} features')
