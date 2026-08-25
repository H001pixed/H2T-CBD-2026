"""Feature-score correlation: 8 prompts x 19 features x score (Spearman).

Checks whether hurt folds (3/4/5/6) have weaker feature-score correlations than
benefit folds (1/2/8), and identifies globally weak anchor groups.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Data file: fixed path relative to the packaged project root.
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "01_raw_datasets" / "feat" / "data.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "results" / "feature_score_correlation.csv"

df = pd.read_csv(DATA_PATH)
print(f"loaded {len(df)} essays, {df['essay_set'].nunique()} prompts\n")

# 19 feature names.
FEATURES = [
    "char_count", "word_count", "sent_count", "avg_word_len", "spell_err_count",
    "noun_count", "adj_count", "verb_count", "adv_count", "readability_score",
    "punctuation_score", "vocabulary_richness", "complex_sentence_ratio",
    "clause_density", "semantic_coherence", "sentiment_subjectivity",
    "transitional_phrase_use", "figurative_language_use", "question_usage",
]

# The 6 anchor groups.
ANCHOR_GROUPS = {
    "length_fluency": ["char_count", "word_count", "sent_count"],
    "lexical_sophistication": ["avg_word_len", "vocabulary_richness", "adj_count", "adv_count"],
    "syntactic_complexity": ["complex_sentence_ratio", "clause_density", "verb_count", "noun_count"],
    "mechanics": ["spell_err_count", "punctuation_score"],
    "coherence_readability": ["readability_score", "semantic_coherence"],
    "rhetoric_engagement": ["sentiment_subjectivity", "transitional_phrase_use",
                            "figurative_language_use", "question_usage"],
}

# LOPO fold groups: benefit / hurt / neutral.
BENEFIT_FOLDS = [1, 2, 8]
HURT_FOLDS = [3, 4, 5, 6]
NEUTRAL_FOLDS = [7]


print("19 features vs score (Spearman r) per prompt")
print("=" * 60)

all_fold_stats = {}

for fold in sorted(df["essay_set"].unique()):
    sub = df[df["essay_set"] == fold]
    scores = sub["final_score"].values

    print(f"\n{'─' * 60}")
    print(f"  Fold {fold}  (n={len(sub)})")
    print(f"{'─' * 60}")
    print(f"  {'feature':28s} {'Spearman r':>10s} {'p-value':>10s}   strength")
    print(f"  {'─' * 50}")

    fold_corrs = {}
    for feat in FEATURES:
        vals = sub[feat].values
        # Skip NaN-heavy or constant columns.
        mask = np.isfinite(vals)
        if mask.sum() < 10 or np.std(vals[mask]) == 0:
            continue
        r, p = stats.spearmanr(vals[mask], scores[mask])
        fold_corrs[feat] = {"r": r, "p": p}

        # Strength label.
        if abs(r) >= 0.3:
            strength = "strong"
        elif abs(r) >= 0.15:
            strength = "medium"
        elif abs(r) >= 0.08:
            strength = "weak"
        else:
            strength = "v.weak"

        print(f"  {feat:28s} {r:>+10.4f} {p:>10.4f}  {strength}")

    all_fold_stats[fold] = fold_corrs


print("\n6 anchor groups: mean |r| per prompt")
print("=" * 80)

header = f"  {'group':>16s}"
for fold in sorted(all_fold_stats.keys()):
    header += f" {'fold'+str(fold):>8s}"
header += f" {'benefit':>10s} {'hurt':>10s} {'diff':>8s}"
print(header)
print(f"  {'-' * 90}")

for group_name, group_feats in ANCHOR_GROUPS.items():
    row = f"  {group_name:>16s}"
    benefit_rs = []
    hurt_rs = []
    for fold in sorted(all_fold_stats.keys()):
        rs = [all_fold_stats[fold][f]["r"] for f in group_feats if f in all_fold_stats[fold]]
        if rs:
            avg_r = np.mean(np.abs(rs))
            row += f" {avg_r:>+8.4f}"
            if fold in BENEFIT_FOLDS:
                benefit_rs.extend(rs)
            elif fold in HURT_FOLDS:
                hurt_rs.extend(rs)
        else:
            row += f" {'?':>8s}"

    b_mean = np.mean(np.abs(benefit_rs)) if benefit_rs else float("nan")
    h_mean = np.mean(np.abs(hurt_rs)) if hurt_rs else float("nan")
    diff = b_mean - h_mean if (benefit_rs and hurt_rs) else float("nan")
    row += f" {b_mean:>+10.4f} {h_mean:>+10.4f} {diff:>+8.4f}"
    print(row)


print("\nper-fold anchor group diagnosis")
print("=" * 60)

WEAK_THRESHOLD = 0.10

for fold in HURT_FOLDS:
    print(f"\n  fold {fold}:")
    should_disable = []
    should_keep = []
    for group_name, group_feats in ANCHOR_GROUPS.items():
        rs = [all_fold_stats[fold][f]["r"] for f in group_feats if f in all_fold_stats[fold]]
        if not rs:
            continue
        avg_abs_r = np.mean(np.abs(rs))
        if avg_abs_r < WEAK_THRESHOLD:
            should_disable.append((group_name, avg_abs_r))
        else:
            should_keep.append((group_name, avg_abs_r))

    for name, r in sorted(should_disable, key=lambda x: x[1]):
        print(f"    DISABLE {name}: mean |r| = {r:.4f}")
    for name, r in sorted(should_keep, key=lambda x: -x[1]):
        print(f"    keep   {name}: mean |r| = {r:.4f}")


print("\nbenefit vs hurt: overall feature-score correlation")
print("=" * 60)

for label, fold_list in [("benefit (1,2,8)", BENEFIT_FOLDS),
                          ("hurt (3,4,5,6)", HURT_FOLDS),
                          ("neutral (7)", NEUTRAL_FOLDS)]:
    all_rs = []
    for fold in fold_list:
        for feat in FEATURES:
            if feat in all_fold_stats[fold]:
                all_rs.append(abs(all_fold_stats[fold][feat]["r"]))
    if all_rs:
        print(f"  {label:25s}: mean |r| = {np.mean(all_rs):.4f}, "
              f"median |r| = {np.median(all_rs):.4f}, "
              f"weak (|r|<0.1) = {sum(1 for r in all_rs if r<0.1)/len(all_rs):.1%}")


benefit_all_r = []
for fold in BENEFIT_FOLDS:
    for feat in FEATURES:
        if feat in all_fold_stats[fold]:
            benefit_all_r.append(abs(all_fold_stats[fold][feat]["r"]))
hurt_all_r = []
for fold in HURT_FOLDS:
    for feat in FEATURES:
        if feat in all_fold_stats[fold]:
            hurt_all_r.append(abs(all_fold_stats[fold][feat]["r"]))

b_mean = np.mean(benefit_all_r)
h_mean = np.mean(hurt_all_r)
print(f"\nbenefit mean |r| = {b_mean:.4f}")
print(f"hurt mean |r|    = {h_mean:.4f}")
print(f"diff = {b_mean - h_mean:.4f}")

# Save the 8x19 correlation matrix (rows = features, cols = prompts).
heatmap = pd.DataFrame(index=FEATURES, columns=sorted(all_fold_stats.keys()))
for fold in sorted(all_fold_stats.keys()):
    for feat in FEATURES:
        if feat in all_fold_stats[fold]:
            heatmap.loc[feat, fold] = round(all_fold_stats[fold][feat]["r"], 4)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
heatmap.to_csv(OUT_PATH)
print(f"\nsaved: {OUT_PATH}")
