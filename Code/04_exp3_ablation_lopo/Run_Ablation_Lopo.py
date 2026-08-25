"""LOPO ablation on feat: no_anchor x 3 seeds x 8 folds = 24 units.

Answers: does removing the anchor loss shrink cross-prompt variance,
and do the hurt folds (4/5/6) recover?
"""

from __future__ import annotations

import json, re, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_core"))

import Common as C
from Engine import load_emb, train_eval, FeatNorm
from Metrics_Logger import MetricsLogger

SEEDS = [13, 42, 123]


def main():
    print("=" * 60)
    print("LOPO ablation: no_anchor on feat, 24 folds")
    print("=" * 60)

    d = load_emb("feat")
    smin, smax = C.DATASETS["feat"]["smin"], C.DATASETS["feat"]["smax"]
    sets = sorted(np.unique(d["essay_set"]).tolist())

    all_results = []

    for seed in SEEDS:
        fold_qwks = []
        for hold in sets:
            te_mask = d["essay_set"] == hold
            tr_mask = ~te_mask
            tr_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]

            rng = np.random.default_rng(seed)
            rng.shuffle(tr_idx)
            n_va = max(1, int(0.1 * len(tr_idx)))
            iva, itr = tr_idx[:n_va], tr_idx[n_va:]

            fn = FeatNorm().fit(d["feats"][itr])
            smap = C.ScaleMap(smin, smax).fit(d["score"][itr])

            sub = lambda idx: {
                "emb": d["emb"][idx],
                "score": d["score"][idx],
                "feats": d["feats"][idx],
                "essay_set": d["essay_set"][idx],
            }

            exp_id = f"B1_Abl_LOPO_feat_no_anchor_s{seed}_fold{hold}"
            if C.already_done(exp_id):
                r = json.loads(C.result_path(exp_id).read_text())
                print(f"[skip] {exp_id} qwk={r['test_qwk']}")
            else:
                ml = MetricsLogger(C.RUNS / exp_id)
                metrics, _ = train_eval(
                    "no_anchor", sub(itr), sub(iva), sub(te_idx), smap,
                    seed=seed, ml=ml, feat_norm=fn,
                )
                r = {
                    "exp_id": exp_id, "variant": "no_anchor", "protocol": "P-LOPO",
                    "seed": seed, "fold": int(hold),
                    **metrics,
                }
                C.save_result(exp_id, r)
                print(f"[done] {exp_id} qwk={metrics['test_qwk']:.4f}")

            fold_qwks.append(r["test_qwk"])
            all_results.append(r)

        mean_q = float(np.mean(fold_qwks))
        gexp = f"B1_Abl_LOPO_feat_no_anchor_s{seed}"
        payload = {
            "exp_id": gexp, "variant": "no_anchor", "protocol": "P-LOPO",
            "seed": seed, "test_qwk": round(mean_q, 4),
            "fold_qwks": [round(x, 4) for x in fold_qwks],
        }
        C.save_result(gexp, payload)
        print(f"[aggregate] {gexp} mean_qwk={round(mean_q, 4)}")

    # Summary: compare with the existing blackbox/h2tcbd runs.
    print()
    print("=" * 60)
    print("LOPO ablation: per-fold delta-QWK")
    print("=" * 60)

    ref = {}
    for fname in C.RUNS.iterdir():
        if not fname.name.startswith("B1_LOPO_feat_") or "_fold" not in fname.name:
            continue
        rpath = fname / "result.json"
        if not rpath.exists():
            continue
        data = json.loads(rpath.read_text())
        variant = data.get("variant", "?")
        fold = data.get("fold", "?")
        if fold == "?":
            continue
        m = re.search(r"_s(\d+)_fold", fname.name)
        if not m:
            continue
        seed = int(m.group(1))
        ref[(variant, seed, int(fold))] = data["test_qwk"]

    print(f"{'Fold':>5s} {'Seed':>5s} {'BB':>8s} {'H2T':>8s} {'no_anc':>8s}  {'H2T-BB':>8s} {'no-BB':>8s}")
    print("-" * 58)

    fold_stats = {f: {"h2t": [], "na": []} for f in sets}

    for fold in sets:
        for seed in SEEDS:
            bb = ref.get(("blackbox", seed, fold))
            h2t = ref.get(("h2tcbd", seed, fold))
            na = next(
                (r["test_qwk"] for r in all_results
                 if r.get("variant") == "no_anchor" and r.get("seed") == seed and r.get("fold") == fold),
                None,
            )
            if bb is None:
                continue
            d_h2t = h2t - bb if h2t is not None else None
            d_na = na - bb if na is not None else None
            h2t_s = f"{h2t:8.4f}" if h2t is not None else "       ?"
            na_s = f"{na:8.4f}" if na is not None else "       ?"
            dh_s = f"{d_h2t:+8.4f}" if d_h2t is not None else "       ?"
            dn_s = f"{d_na:+8.4f}" if d_na is not None else "       ?"
            print(f"{fold:>5d} {seed:>5d} {bb:>8.4f} {h2t_s} {na_s}  {dh_s} {dn_s}")
            if d_h2t is not None:
                fold_stats[fold]["h2t"].append(d_h2t)
            if d_na is not None:
                fold_stats[fold]["na"].append(d_na)

    print()
    print(f"{'Fold':>5s} {'H2T delta':>10s} {'no_anc delta':>12s}  H2T std -> no_anc std")
    print("-" * 48)

    all_h2t, all_na = [], []
    for fold in sets:
        h2t_d = fold_stats[fold]["h2t"]
        na_d = fold_stats[fold]["na"]
        if h2t_d and na_d:
            h2t_mean, h2t_std = np.mean(h2t_d), np.std(h2t_d)
            na_mean, na_std = np.mean(na_d), np.std(na_d)
            all_h2t.extend(h2t_d)
            all_na.extend(na_d)
            print(f"{fold:>5d} {h2t_mean:>+10.4f} {na_mean:>+12.4f}  {h2t_std:.4f} -> {na_std:.4f}")

    print("-" * 48)
    print(f"{'ALL':>5s} {np.mean(all_h2t):>+10.4f} {np.mean(all_na):>+12.4f}  "
          f"{np.std(all_h2t):.4f} -> {np.std(all_na):.4f}")

    print()
    h2t_range = np.max(all_h2t) - np.min(all_h2t)
    na_range = np.max(all_na) - np.min(all_na)
    print(f"H2T-CBD   range = {h2t_range:.4f}  std = {np.std(all_h2t):.4f}")
    print(f"no_anchor range = {na_range:.4f}  std = {np.std(all_na):.4f}")
    h2t_std = np.std(all_h2t)
    na_std = np.std(all_na)
    if na_std < h2t_std:
        print(f"std reduction: {h2t_std:.4f} -> {na_std:.4f}  (delta = {na_std - h2t_std:+.4f})")
    else:
        print(f"std unchanged: {h2t_std:.4f} -> {na_std:.4f}  (delta = {na_std - h2t_std:+.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
