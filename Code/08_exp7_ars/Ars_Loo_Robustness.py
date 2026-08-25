"""Leave-one-out robustness check for the ARS values.

Reviewer concern: Recompute_Ars.py computes mu/sigma/ARS on all 8 prompts,
while P-LOPO only trains on 7 prompts per fold; paper Section 2.6 says the
statistics are computed "on the training prompts". Read literally, the
configuration choice could leak the held-out prompt's feature-score relation.

This script recomputes ARS per fold using only the 7 training prompts and
checks whether the screening decision matches the full 8-prompt computation.

Expected result: in every fold only the rhetoric-engagement group falls below
theta = 0.1 (its ARS ranges 0.065-0.078); the other five groups stay above,
so the decision is identical to the full-data computation (no real leakage).
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
THETA = 0.1

df = pd.read_csv(DATA_PATH)
prompts = sorted(df["essay_set"].unique())
print(f"loaded {len(df)} essays, {len(prompts)} prompts")

# Per-prompt per-feature Spearman |r| (identical to Recompute_Ars.py).
per_feat = {}
for fold in prompts:
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

# Per-prompt group means rho_bar_k(p).
rho_bar = {}   # group -> {prompt: value}
for gname, feats in ANCHOR_GROUPS.items():
    rho_bar[gname] = {}
    for p in prompts:
        rs = [per_feat[p][f] for f in feats if not np.isnan(per_feat[p][f])]
        rho_bar[gname][p] = float(np.mean(rs))


def ars_on(subset):
    """Recompute mu/sigma/ARS for all six groups on a prompt subset."""
    out = {}
    for gname in ANCHOR_GROUPS:
        vals = [rho_bar[gname][p] for p in subset]
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))  # ddof=0, as in Recompute_Ars.py.
        out[gname] = mu - sigma
    return out


# Full 8-prompt baseline.
full = ars_on(prompts)
print("\n=== Full 8 prompts (baseline) ===")
for g, a in full.items():
    print(f"  {g:>24s}: ARS={a:.3f}  {'deactivate' if a <= THETA else 'retain'}")

# Leave-one-out over 7 training prompts (8 folds).
print("\n=== Leave-one-out (7 training prompts, 8 folds) ===")
rhet_lo = []
all_below_only_rhet = True
for held in prompts:
    train = [p for p in prompts if p != held]
    loo = ars_on(train)
    below = [g for g, a in loo.items() if a <= THETA]
    rhet_lo.append(loo["rhetoric_engagement"])
    print(f"  held={held}  rhetoric ARS={loo['rhetoric_engagement']:.3f}  below-threshold={below}")
    if below != ["rhetoric_engagement"]:
        all_below_only_rhet = False

print(f"\nRhetoric leave-one-out ARS range: {min(rhet_lo):.3f} - {max(rhet_lo):.3f}")
print(f"Only rhetoric below theta=0.1 in all 8 folds: {all_below_only_rhet}")

# Decision consistency per group: LOO min/max vs the full-data decision.
print("\n=== Decision consistency (per group) ===")
for g in ANCHOR_GROUPS:
    loo_vals = [ars_on([p for p in prompts if p != h])[g] for h in prompts]
    lo_min, lo_max = min(loo_vals), max(loo_vals)
    same = (lo_min > THETA) == (full[g] > THETA)
    print(f"  {g:>24s}: LOO range [{lo_min:.3f},{lo_max:.3f}]  full={full[g]:.3f}  consistent={same}")
