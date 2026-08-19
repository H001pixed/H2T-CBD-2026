"""One-off script: normalise ASAP-1 scores per essay set to a common 0-10 range.

Each essay_set originally had its own score range; this maps each set's
min/max to 0/10 so all 8 prompts share the same scale.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "01_raw_datasets" / "feat" / "data.csv"
df = pd.read_csv(DATA_PATH)

for es in sorted(df['essay_set'].unique()):
    m = df['essay_set'] == es
    lo = float(df.loc[m, 'final_score'].min())
    hi = float(df.loc[m, 'final_score'].max())
    df.loc[m, 'final_score'] = np.round((df.loc[m, 'final_score'] - lo) / (hi - lo) * 10).astype(int)
    print(f'set {es}: raw [{lo}, {hi}] -> new [{df.loc[m, "final_score"].min()}, {df.loc[m, "final_score"].max()}]')

df.to_csv(DATA_PATH, index=False)
print(f'saved, max score: {df["final_score"].max()}')
