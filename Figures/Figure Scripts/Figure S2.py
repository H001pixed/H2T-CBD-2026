"""Figure S2 - interpretability evidence: single radar of per-group decoding.

Structural interpretability of H2T-CBD means each bottleneck dimension can be
decoded back to its linguistic feature group by the anchor head. On the full
test set of the seed-42 P-in split (1,300 essays), we quantify per-group
decoding quality as the merged Pearson correlation between the anchor head's
outputs and the true z-scored features (all group features pooled): length
0.93, lexical 0.75, syntactic 0.64, mechanics 0.86, coherence 0.63, rhetoric
0.28 with the current data. The weakest-decoded group (rhetoric) is exactly
the one ARS drops.

The figure is a 6-axis radar chart of these decoding correlations (the
rhetoric_engagement vertex is marked red). The example essay's full record
(idx 12244, prompt 7, human 5, predicted 5, 174 words) is written to
result.json and reported in the paper text, but is not plotted.

Usage: python generate_figure_s2.py
Outputs: results/runs/interpretability_example/result.json
         <figures>/Figure S2.png
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path(__file__).resolve().parents[2]
for core in ("代码/00_核心模型", "code/00_core"):
    p = ROOT / core
    if p.is_dir():
        sys.path.insert(0, str(p))
        break

import Common as C  # noqa: E402
from Engine import load_emb, FeatNorm, ANCHOR_DIMS  # noqa: E402
from Models import H2TCBD  # noqa: E402

SEED = 42
GROUP_NAMES = list(C.ANCHOR_GROUPS.keys())
MODEL_ID = f"B1_Pin_feat_h2tcbd_s{SEED}"
OUT_DIR = C.RUNS / "interpretability_example"
BLUE = '#1A6DB5'
RED = '#D64545'
GRAY = '#5A5A5A'


def _fmt2(v):
    """Round to two decimals with round-half-up (matches the paper text)."""
    return format(Decimal(str(v)).quantize(Decimal('0.01'),
                                           rounding=ROUND_HALF_UP), 'f')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = load_emb("feat")

    # Reproduce the seed-42 P-in split and ScaleMap exactly as in the main runs.
    dfp = pd.DataFrame({"score": d["score"]})
    itr, iva, ite = C.stratified_split(dfp, seed=SEED)
    smap = C.ScaleMap(C.DATASETS["feat"]["smin"], C.DATASETS["feat"]["smax"]).fit(
        d["score"][itr])
    fn = FeatNorm().fit(d["feats"][itr])

    model = H2TCBD(in_dim=d["emb"].shape[1], K=6, anchor_dims=ANCHOR_DIMS,
                   use_anchor=True)
    model.load_state_dict(torch.load(
        C.RUNS / MODEL_ID / "model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        z_test = model.bottleneck(model.encoder(
            torch.tensor(d["emb"][ite], dtype=torch.float32))).numpy()
        pred_u = torch.sigmoid(model.readout(
            torch.tensor(z_test, dtype=torch.float32))).numpy().squeeze(-1)
    pred_int = smap.from_unit(pred_u)
    qwk_check = C.qwk(d["score"][ite], pred_int, smap.smin, smap.smax)
    ref = json.loads((C.RUNS / MODEL_ID / "result.json").read_text(
        encoding="utf-8"))
    assert abs(qwk_check - ref["test_qwk"]) < 5e-3, "split/model mismatch"

    feat_z = fn.transform(d["feats"][ite])
    n_test = len(ite)

    # ---- Per-group anchor-head decoding quality (merged Pearson r) ----
    decoding = {}
    per_feature = {}
    for k, g in enumerate(GROUP_NAMES):
        dims = ANCHOR_DIMS[k]
        true = feat_z[:, dims]
        with torch.no_grad():
            pred = model.anchor_heads[k](
                torch.tensor(z_test[:, k:k + 1], dtype=torch.float32)).numpy()
        r = float(np.corrcoef(pred.ravel(), true.ravel())[0, 1])
        decoding[g] = {"merged_r": round(r, 4), "r2": round(r * r, 4)}
        per_feature[g] = {
            C.FEAT_COLS[j]: round(float(np.corrcoef(pred[:, m], true[:, m])[0, 1]), 4)
            for m, j in enumerate(dims)
        }

    # ---- z dim vs feature-group mean correlation matrix (for the record) ----
    group_means = {g: feat_z[:, [C.FEAT_COLS.index(f) for f in fs]].mean(1)
                   for g, fs in C.ANCHOR_GROUPS.items()}
    corr_matrix = {}
    for i, gi in enumerate(GROUP_NAMES):
        corr_matrix[gi] = {
            gj: round(float(np.corrcoef(z_test[:, i], group_means[gj])[0, 1]), 4)
            for gj in GROUP_NAMES
        }

    # ---- Representative essay ----
    score5 = ite[(d["score"][ite] == 5) & (pred_int == 5)]
    wc = d["feats"][:, C.FEAT_COLS.index("word_count")]
    med_wc = float(np.median(wc[ite]))
    idx = int(score5[np.argmin(np.abs(wc[score5] - med_wc))])
    zi = int(np.where(ite == idx)[0][0])
    pid = int(d["essay_set"][idx])
    z = z_test[zi]
    z_mu, z_sd = z_test.mean(0), z_test.std(0) + 1e-8
    z_std = ((z - z_mu) / z_sd).tolist()
    feat_ex = fn.transform(d["feats"][idx:idx + 1])[0]
    g_ex = {g: round(float(np.mean(
        [feat_ex[C.FEAT_COLS.index(f)] for f in C.ANCHOR_GROUPS[g]])), 4)
        for g in GROUP_NAMES}
    k_rhet = GROUP_NAMES.index("rhetoric_engagement")
    with torch.no_grad():
        pred_rhet = model.anchor_heads[k_rhet](
            torch.tensor(z[k_rhet:k_rhet + 1], dtype=torch.float32)).numpy()[0]
    feat_raw = {f: round(float(d["feats"][idx, i]), 4)
                for i, f in enumerate(C.FEAT_COLS)}

    record = {
        "model": MODEL_ID,
        "seed": SEED,
        "n_test": n_test,
        "qwk_verification": {"recomputed": round(float(qwk_check), 4),
                             "result_json": ref["test_qwk"]},
        "decoding_merged_r": {g: decoding[g] for g in GROUP_NAMES},
        "decoding_per_feature": per_feature,
        "z_feature_group_corr_matrix": corr_matrix,
        "example": {
            "essay_index": idx,
            "essay_set": pid,
            "human_score": int(d["score"][idx]),
            "predicted_score": int(pred_int[zi]),
            "word_count": int(wc[idx]),
            "bottleneck_z": [round(float(v), 4) for v in z],
            "bottleneck_z_std": [round(float(v), 4) for v in z_std],
            "feature_group_z": g_ex,
            "rhetoric_decoded_mean": round(float(pred_rhet.mean()), 4),
            "features_raw": feat_raw,
        },
    }
    (OUT_DIR / "result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {OUT_DIR / 'result.json'}")
    print("decoding merged r:",
          {g: decoding[g]["merged_r"] for g in GROUP_NAMES})
    print("example:", idx, "prompt", pid, "human",
          int(d["score"][idx]), "pred", int(pred_int[zi]),
          "wc", int(wc[idx]))

    # ---- Figure: single 6-axis radar of per-group decoding quality ----
    labels = ['Length /\nFluency', 'Lexical\nSophistication',
              'Syntactic\nComplexity', 'Mechanics',
              'Coherence /\nReadability', 'Rhetorical\nEngagement']
    vals = [decoding[g]["merged_r"] for g in GROUP_NAMES]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    series = vals + vals[:1]

    fig, ax = plt.subplots(figsize=(6.6, 6.0), subplot_kw=dict(polar=True))
    ax.set_rlim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.plot(angles, series, color=BLUE, linewidth=2.0,
            label='Retained groups')
    ax.fill(angles, series, color=BLUE, alpha=0.12)
    k_r = GROUP_NAMES.index("rhetoric_engagement")
    for i, v in enumerate(vals):
        ax.annotate(_fmt2(v), (angles[i], v), textcoords='offset points',
                    xytext=(0, 12), ha='center', fontsize=10,
                    fontweight='bold', color=(RED if i == k_r else GRAY))
    ax.scatter([angles[k_r]], [vals[k_r]], color=RED, s=60, zorder=5,
               label='Dropped (ARS ≤ 0.1)')
    ax.scatter([angles[i] for i in range(n) if i != k_r],
               [vals[i] for i in range(n) if i != k_r],
               color=BLUE, s=40, zorder=5)
    tl = ax.get_xticklabels()
    tl[k_r].set_color(RED)
    tl[k_r].set_fontweight('bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.10), fontsize=10)
    ax.grid(True)

    figdir = Path(__file__).resolve().parent.parent
    out_png = figdir / "Figure S2.png"
    fig.savefig(str(out_png), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved -> {out_png}")


if __name__ == "__main__":
    main()
