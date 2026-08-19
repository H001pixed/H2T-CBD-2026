"""Recompute the six ARS values directly from data.csv.

Background: earlier notes contained ARS values that could not be reproduced
from the only correlation file in the repository
(feature_score_correlation.csv, verified value-by-value against data.csv).
This script recomputes the six ARS values from data.csv using the same
Spearman-based procedure as exp5; the output is fully reproducible.

Formula (identical to the exp8 notes):
    rho_bar_k(p) = mean_{f in group_k} |Spearman(f, y; p)|   per-prompt mean
    mu_k         = mean_p rho_bar_k(p)                       signal strength
    sigma_k      = std_p  rho_bar_k(p) (ddof=0)              cross-prompt spread
    ARS_k        = mu_k - sigma_k

Usage: python recompute_ars.py
Dependencies: pandas, numpy, scipy (same as exp5).
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "01_raw_datasets" / "feat" / "data.csv"

FEATURES = [
    "char_count", "word_count", "sent_count", "avg_word_len", "spell_err_count",
    "noun_count", "adj_count", "verb_count", "adv_count", "readability_score",
    "punctuation_score", "vocabulary_richness", "complex_sentence_ratio",
    "clause_density", "semantic_coherence", "sentiment_subjectivity",
    "transitional_phrase_use", "figurative_language_use", "question_usage",
]

ANCHOR_GROUPS = {
    "length_fluency": ["char_count", "word_count", "sent_count"],
    "lexical_sophistication": ["avg_word_len", "vocabulary_richness", "adj_count", "adv_count"],
    "syntactic_complexity": ["complex_sentence_ratio", "clause_density", "verb_count", "noun_count"],
    "mechanics": ["spell_err_count", "punctuation_score"],
    "coherence_readability": ["readability_score", "semantic_coherence"],
    "rhetoric_engagement": ["sentiment_subjectivity", "transitional_phrase_use",
                            "figurative_language_use", "question_usage"],
}

df = pd.read_csv(DATA_PATH)
print(f"loaded {len(df)} essays, {df['essay_set'].nunique()} prompts")

# ---- 1. Per-prompt per-feature Spearman |r| (identical to exp5) ----
per_feat = {}   # fold -> {feat: |r|}
for fold in sorted(df["essay_set"].unique()):
    sub = df[df["essay_set"] == fold]
    scores = sub["final_score"].values
    d = {}
    for feat in FEATURES:
        vals = sub[feat].values
        mask = np.isfinite(vals)
        if mask.sum() < 10 or np.std(vals[mask]) == 0:
            d[feat] = np.nan
            continue
        r, _ = stats.spearmanr(vals[mask], scores[mask])
        d[feat] = abs(r)
    per_feat[fold] = d

prompts = sorted(df["essay_set"].unique())

# ---- 2. Per-prompt group means -> mu/sigma/ARS ----
THETA = 0.1  # Threshold: Cohen's "negligible/unrelated" |rho| < 0.10.
print(f"\n{'group':>24s} {'mu':>8s} {'sigma':>8s} {'ARS':>8s}  {'<=0.1?':>7s}")
print("-" * 62)
rows = []
for gname, feats in ANCHOR_GROUPS.items():
    rho_bar = []
    for p in prompts:
        rs = [per_feat[p][f] for f in feats if not np.isnan(per_feat[p][f])]
        rho_bar.append(float(np.mean(rs)))
    mu = float(np.mean(rho_bar))
    sigma = float(np.std(rho_bar))          # ddof=0, as in the exp8 notes.
    ars = mu - sigma
    rows.append((gname, mu, sigma, ars, rho_bar))
    print(f"{gname:>24s} {mu:>8.3f} {sigma:>8.3f} {ars:>8.3f}  {ars <= THETA!s:>7s}")

print("\nPer-prompt rho_bar_k(p) (p = 1..8):")
for gname, mu, sigma, ars, rho_bar in rows:
    print(f"  {gname:>24s}: " + "  ".join(f"{v:.3f}" for v in rho_bar))

print(f"\nTheta=0.1 decision: deactivate {sum(1 for _,_,_,a,_ in rows if a <= THETA)} group(s), retain the rest")
