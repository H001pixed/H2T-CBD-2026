"""P-in ablation on feat: no_anchor + featconcat vs blackbox + h2tcbd.

Decomposes the three components:
  - Bottleneck cost: no_anchor  vs blackbox
  - Anchor cost:     h2tcbd     vs no_anchor
  - Feature value:   featconcat vs blackbox
"""

from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_core"))

import Common as C
from Engine import load_emb, train_eval, FeatNorm
from Metrics_Logger import MetricsLogger

SEEDS = [13, 42, 123]

NEW_VARIANTS = ["no_anchor", "featconcat"]
REF_VARIANTS = ["blackbox", "h2tcbd"]


def main():
    print("=" * 60)
    print("P-in ablation: bottleneck / anchor / feature alignment")
    print("=" * 60)

    d = load_emb("feat")
    smin, smax = C.DATASETS["feat"]["smin"], C.DATASETS["feat"]["smax"]

    results = []

    for seed in SEEDS:
        df = pd.DataFrame({"score": d["score"]})
        itr, iva, ite = C.stratified_split(df, seed=seed)
        smap = C.ScaleMap(smin, smax).fit(d["score"][itr])
        fn = FeatNorm().fit(d["feats"][itr])

        sub = lambda idx: {
            "emb": d["emb"][idx],
            "score": d["score"][idx],
            "feats": d["feats"][idx],
            "essay_set": d["essay_set"][idx],
        }
        tr, va, te = sub(itr), sub(iva), sub(ite)

        # Run the new ablation variants.
        for variant in NEW_VARIANTS:
            exp_id = f"B1_Abl_Pin_feat_{variant}_s{seed}"
            if C.already_done(exp_id):
                r = json.loads(C.result_path(exp_id).read_text())
                print(f"[skip] {exp_id} qwk={r['test_qwk']}")
            else:
                ml = MetricsLogger(C.RUNS / exp_id)
                metrics, _ = train_eval(variant, tr, va, te, smap, seed=seed,
                                        ml=ml, feat_norm=fn)
                r = {"exp_id": exp_id, "variant": variant, "protocol": "P-in",
                     "seed": seed, "desc": f"P-in feat {variant} seed{seed}", **metrics}
                C.save_result(exp_id, r)
                print(f"[done] {exp_id} qwk={metrics['test_qwk']:.4f}")
            results.append(r)

        # Reuse the already-run reference variants.
        for variant in REF_VARIANTS:
            prev_id = f"B1_Pin_feat_{variant}_s{seed}"
            prev_path = C.RUNS / prev_id / "result.json"
            if prev_path.exists():
                results.append(json.loads(prev_path.read_text()))

    # Summary
    print()
    print("=" * 60)
    print("P-in feat ablation summary")
    print("=" * 60)
    print(f"{'Variant':>14s} {'Seed':>5s} {'QWK':>8s} {'RMSE':>8s} {'Pearson':>8s}")
    print("-" * 47)

    variants_order = ["blackbox", "h2tcbd", "no_anchor", "featconcat"]
    agg = {}
    for variant in variants_order:
        vr = [r for r in results if r.get("variant") == variant and r.get("seed") in SEEDS]
        for r in sorted(vr, key=lambda x: x["seed"]):
            print(f"{variant:>14s} {r['seed']:>5d} {r['test_qwk']:>8.4f} {r['test_rmse']:>8.4f} {r['test_pearson']:>8.4f}")
        if vr:
            qwks = [r["test_qwk"] for r in vr]
            agg[variant] = {"mean_qwk": np.mean(qwks), "std_qwk": np.std(qwks)}

    print("-" * 47)

    bb = agg.get("blackbox", {}).get("mean_qwk", None)
    na = agg.get("no_anchor", {}).get("mean_qwk", None)
    h2t = agg.get("h2tcbd", {}).get("mean_qwk", None)
    fc = agg.get("featconcat", {}).get("mean_qwk", None)

    if bb is not None:
        print(f"\n  blackbox   = {bb:.4f}")
    if na is not None and bb is not None:
        print(f"  no_anchor  = {na:.4f}  delta vs bb = {na - bb:+.4f}")
    if h2t is not None and na is not None:
        print(f"  h2tcbd     = {h2t:.4f}  delta vs na = {h2t - na:+.4f}")
    if fc is not None and bb is not None:
        print(f"  featconcat = {fc:.4f}  delta vs bb = {fc - bb:+.4f}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
