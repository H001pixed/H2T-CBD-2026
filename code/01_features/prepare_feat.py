"""Convert ASAP-1 training_set_rel3.tsv into data_raw.csv.

The Kaggle ASAP-AES file ships essays plus per-rater scores and a resolved
domain1_score. Downstream only needs three columns: clean_essay (the raw
essay text), final_score (the resolved integer score), and essay_set.
The TSV is decoded as cp1252 (Windows-1252) because it contains curly quotes,
em-dashes, and other punctuation that latin-1 cannot represent correctly.
errors="replace" guards the few Windows-1252 undefined bytes (0x81/0x8D/0x8F/
0x90/0x9D), which are replaced instead of aborting the read.
"""
import io
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "01_raw_datasets" / "feat"
src = DATA_DIR / "training_set_rel3.tsv"
dst = DATA_DIR / "data_raw.csv"

raw = src.read_bytes()
# Windows-1252 with replacement for undefined bytes (0x81/0x8D/0x8F/0x90/0x9D).
text = raw.decode("cp1252", errors="replace")
df = pd.read_csv(io.StringIO(text), sep="\t")
out = pd.DataFrame({
    "clean_essay": df["essay"].fillna("").astype(str),
    "final_score": df["domain1_score"].astype(int),
    "essay_set": df["essay_set"].astype(int),
})
out.to_csv(dst, index=False)
print(f"saved {dst} ({len(out)} essays)")
