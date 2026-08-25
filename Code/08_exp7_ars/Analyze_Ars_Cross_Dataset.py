"""Cross-dataset ARS diagnostic: anchor-group feature-score stability beyond ASAP-1.

Reviewer point 1.2: the ARS screening decision was only validated on ASAP-1.
The official ASAP-2.0 (LLM-AES 2.0) and Feedback Prize ELL files contain no
prompt labels, so a per-prompt ARS (with its cross-prompt spread term sigma)
cannot be computed on those datasets. Instead, each dataset is treated as a
single pseudo-prompt: we compute the same per-feature |Spearman(feature,
score)| statistics and average within the six anchor groups, producing
rho_bar_k(dataset). We then compare these values with the ASAP-1 per-prompt
distribution and check whether (i) the relative ordering of the six groups is
preserved and (ii) rhetorical engagement remains the weakest group in every
dataset.

No model is retrained; this script only recomputes the 19 features and
correlations. Outputs are written under results/runs/ars_cross_dataset/.

Usage: python Analyze_Ars_Cross_Dataset.py
Dependencies: pandas, numpy, scipy, and the NLTK resources used by Features.py.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

CODE = Path(__file__).resolve().parent.parent
for core in ("00_core",):
    if (CODE / core).is_dir():
        sys.path.insert(0, str(CODE / core))
        break
for feat in ("01_features",):
    if (CODE / feat).is_dir():
        sys.path.insert(0, str(CODE / feat))
        break

import Common as C  # noqa: E402
from Features import compute_features_batch, FEATURE_NAMES  # noqa: E402

OUT_DIR = C.RUNS / "ars_cross_dataset"
THETA = 0.1


def pooled_feature_rhos(feats, scores):
    """Pooled per-feature |Spearman(feature, score)| over one dataset."""
    out = {}
    for f in C.FEAT_COLS:
        i = FEATURE_NAMES.index(f)
        vals = feats[:, i]
        mask = np.isfinite(vals)
        if mask.sum() < 10 or np.std(vals[mask]) == 0:
            out[f] = np.nan
            continue
        r, _ = stats.spearmanr(vals[mask], scores[mask])
        out[f] = abs(r)
    return out


def group_means(rhos):
    """Average |r| within each anchor group."""
    out = {}
    for g, fs in C.ANCHOR_GROUPS.items():
        vals = [rhos[f] for f in fs if not np.isnan(rhos[f])]
        out[g] = float(np.mean(vals)) if vals else np.nan
    return out


def per_prompt_group_means(df):
    """ASAP-1 style: per-prompt per-feature |r| -> per-prompt group means."""
    prompts = sorted(df["essay_set"].unique())
    per_feat = {}
    for p in prompts:
        sub = df[df["essay_set"] == p]
        scores = sub["score"].values
        d = {}
        for f in C.FEAT_COLS:
            vals = sub[f].values
            mask = np.isfinite(vals)
            if mask.sum() < 10 or np.std(vals[mask]) == 0:
                d[f] = np.nan
                continue
            r, _ = stats.spearmanr(vals[mask], scores[mask])
            d[f] = abs(r)
        per_feat[p] = d
    out = {}
    for g in C.ANCHOR_GROUPS:
        out[g] = {}
        for p in prompts:
            rs = [per_feat[p][f] for f in C.ANCHOR_GROUPS[g]
                  if not np.isnan(per_feat[p][f])]
            out[g][p] = float(np.mean(rs)) if rs else np.nan
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- ASAP-1 reference: per-prompt distribution (features already in data.csv).
    df = C.load_dataset("feat")
    per_prompt = per_prompt_group_means(df)
    asap1 = {}
    for g in C.ANCHOR_GROUPS:
        vals = np.array([v for v in per_prompt[g].values() if not np.isnan(v)])
        mu = float(vals.mean())
        sd = float(vals.std(ddof=0))
        asap1[g] = {
            "mu": round(mu, 4),
            "sigma": round(sd, 4),
            "ars": round(mu - sd, 4),
            "per_prompt_min": round(float(vals.min()), 4),
            "per_prompt_max": round(float(vals.max()), 4),
            "per_prompt_values": {int(k): round(float(v), 4)
                                  for k, v in per_prompt[g].items()},
        }

    # ---- Pooled correlations: ASAP-1 uses data.csv features; ASAP-2.0 and
    # Feedback recompute the 19 features with the same pipeline (cached).
    pooled = {}
    feats_df = df[C.FEAT_COLS].values.astype(np.float32)
    pooled["feat"] = pooled_feature_rhos(feats_df, df["score"].values)
    for name in ("asap2", "dysf"):
        d = C.load_dataset(name)
        print(f"computing 19 features for {name} ({len(d)} essays) ...", flush=True)
        cache = OUT_DIR / f"{name}_features.npy"
        feats = compute_features_batch(d["text"].tolist(), cache_path=str(cache))
        pooled[name] = pooled_feature_rhos(feats, d["score"].values)

    group_pooled = {name: group_means(pooled[name]) for name in ("feat", "asap2", "dysf")}

    # ---- Ranks within each dataset (1 = strongest association).
    ranks = {}
    for name in ("feat", "asap2", "dysf"):
        order = sorted(group_pooled[name], key=lambda g: group_pooled[name][g],
                       reverse=True)
        ranks[name] = {g: order.index(g) + 1 for g in C.ANCHOR_GROUPS}

    # ---- Comparison table.
    print(f"\n{'group':>24s} {'ASAP1_ARS':>9s} {'ASAP1_mu':>9s} "
          f"{'ASAP1_range':>18s} {'asap2':>7s} {'dysf':>7s}")
    print("-" * 82)
    rows = []
    for g in C.ANCHOR_GROUPS:
        a = asap1[g]
        lo, hi = a["per_prompt_min"], a["per_prompt_max"]
        v2, vd = group_pooled["asap2"][g], group_pooled["dysf"][g]
        in2 = lo <= v2 <= hi if not np.isnan(v2) else False
        ind = lo <= vd <= hi if not np.isnan(vd) else False
        rows.append({
            "group": g,
            "asap1_ars": a["ars"],
            "asap1_mu": a["mu"],
            "asap1_sigma": a["sigma"],
            "asap1_range_min": lo,
            "asap1_range_max": hi,
            "asap2_rho_bar": round(float(v2), 4),
            "dysf_rho_bar": round(float(vd), 4),
            "asap2_in_range": in2,
            "dysf_in_range": ind,
            "rank_feat": ranks["feat"][g],
            "rank_asap2": ranks["asap2"][g],
            "rank_dysf": ranks["dysf"][g],
        })
        print(f"{g:>24s} {a['ars']:>9.3f} {a['mu']:>9.3f} "
              f"[{lo:.3f},{hi:.3f}]{'':>5s} {v2:>7.3f} {vd:>7.3f}")

    # Rhetoric-engagement decision observation.
    rhet = {name: group_pooled[name]["rhetoric_engagement"]
            for name in ("feat", "asap2", "dysf")}
    weakest = {name: ranks[name]["rhetoric_engagement"] == 6
               for name in ("feat", "asap2", "dysf")}
    print(f"\nrhetoric rho_bar: feat={rhet['feat']:.3f} "
          f"asap2={rhet['asap2']:.3f} dysf={rhet['dysf']:.3f} "
          f"(theta reference {THETA})")
    print(f"rhetoric ranked weakest (rank 6): feat={weakest['feat']} "
          f"asap2={weakest['asap2']} dysf={weakest['dysf']}")

    # ---- Save outputs.
    result = {
        "note": ("Pseudo-prompt diagnostic: ASAP-2.0 and Feedback have no prompt "
                 "labels, so each dataset is treated as one prompt; pooled "
                 "|Spearman| per anchor group is compared with the ASAP-1 "
                 "per-prompt distribution. This validates feature-pipeline and "
                 "group-ordering transferability, not a full ARS (no sigma)."),
        "theta": THETA,
        "asap1": asap1,
        "pooled_group_rho_bar": group_pooled,
        "ranks_within_dataset": ranks,
        "rhetoric_weakest_all": all(weakest.values()),
        "rhetoric_rho_bar": rhet,
        "table": rows,
    }
    (OUT_DIR / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Per-feature CSV (pooled |r| per dataset).
    feat_rows = []
    for f in C.FEAT_COLS:
        g = next(gr for gr, fs in C.ANCHOR_GROUPS.items() if f in fs)
        feat_rows.append([f, g, round(pooled["feat"][f], 4),
                          round(pooled["asap2"][f], 4),
                          round(pooled["dysf"][f], 4)])
    pd.DataFrame(feat_rows, columns=["feature", "group", "feat_rho",
                                     "asap2_rho", "dysf_rho"]).to_csv(
        OUT_DIR / "feature_score_corr_cross_dataset.csv", index=False)

    print(f"\nsaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
