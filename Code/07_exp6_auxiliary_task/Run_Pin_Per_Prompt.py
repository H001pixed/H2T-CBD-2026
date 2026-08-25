"""P-in per-prompt evaluation: verify BB-QWK vs delta-QWK negative correlation.

Trains blackbox + h2tcbd x 3 seeds, evaluates per essay_set independently.
Combines with existing LOPO data (24 folds) to test the regularisation hypothesis.
"""

from __future__ import annotations

import json, sys, re
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_core"))

import Common as C
from Engine import load_emb, train_eval, FeatNorm

SEEDS = [13, 42, 123]


def evaluate_per_prompt(model, X, scores, essay_sets, smap, device):
    """Compute per-prompt QWK by splitting the test set on essay_set.

    Args:
        model: trained PyTorch model.
        X: test embeddings (N, 768).
        scores: true test scores (N,).
        essay_sets: essay_set labels of the test set (N,).
        smap: score mapper.
        device: cpu/cuda.

    Returns:
        per_prompt: {essay_set: {"qwk": ..., "n": ...}}.
    """
    import torch
    model.eval()
    per_prompt = {}
    for es in sorted(np.unique(essay_sets)):
        mask = essay_sets == es
        if mask.sum() == 0:
            continue
        X_es = torch.tensor(X[mask], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_u = model(X_es)["score"].cpu().numpy()
        pred_int = smap.from_unit(pred_u)
        qwk_val = C.qwk(scores[mask], pred_int, smap.smin, smap.smax)
        per_prompt[int(es)] = {
            "qwk": round(qwk_val, 4),
            "n": int(mask.sum()),
        }
    return per_prompt


def main():
    print("P-in per-prompt: BB QWK vs delta-QWK")

    d = load_emb("feat")
    smin, smax = C.DATASETS["feat"]["smin"], C.DATASETS["feat"]["smax"]

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_pairs = []  # Each entry: {fold, seed, bb_qwk, h2t_qwk, delta, protocol}.

    for seed in SEEDS:
        # Stratified split.
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

        # ---- BlackBox ----
        exp_bb = f"B1_PinPerPrompt_feat_blackbox_s{seed}"
        if C.already_done(exp_bb):
            r = json.loads(C.result_path(exp_bb).read_text())
            print(f"[skip] {exp_bb}")
        else:
            metrics, model = train_eval("blackbox", tr, va, te, smap, seed=seed)
            per_prompt = evaluate_per_prompt(
                model,
                te["emb"], te["score"], te["essay_set"], smap, device,
            )
            r = {
                "exp_id": exp_bb, "variant": "blackbox", "seed": seed,
                "overall_qwk": metrics["test_qwk"],
                "per_prompt": per_prompt,
            }
            C.save_result(exp_bb, r)
            print(f"[done] {exp_bb} overall_qwk={metrics['test_qwk']:.4f}")

        bb_per = r["per_prompt"]

        # ---- H2T-CBD ----
        exp_h2t = f"B1_PinPerPrompt_feat_h2tcbd_s{seed}"
        if C.already_done(exp_h2t):
            r = json.loads(C.result_path(exp_h2t).read_text())
            print(f"[skip] {exp_h2t}")
        else:
            metrics, model = train_eval("h2tcbd", tr, va, te, smap, seed=seed,
                                        feat_norm=fn)
            per_prompt = evaluate_per_prompt(
                model,
                te["emb"], te["score"], te["essay_set"], smap, device,
            )
            r = {
                "exp_id": exp_h2t, "variant": "h2tcbd", "seed": seed,
                "overall_qwk": metrics["test_qwk"],
                "per_prompt": per_prompt,
            }
            C.save_result(exp_h2t, r)
            print(f"[done] {exp_h2t} overall_qwk={metrics['test_qwk']:.4f}")

        h2t_per = r["per_prompt"]

        # Merge into one BB-delta pair per prompt (JSON keys become strings).
        bb_per_i = {int(k): v for k, v in bb_per.items()}
        h2t_per_i = {int(k): v for k, v in h2t_per.items()}
        for fold in sorted(bb_per_i):
            if fold in h2t_per_i:
                bb_q = bb_per_i[fold]["qwk"]
                h2t_q = h2t_per_i[fold]["qwk"]
                delta = h2t_q - bb_q
                all_pairs.append({
                    "fold": fold, "seed": seed,
                    "bb_qwk": bb_q, "h2t_qwk": h2t_q,
                    "delta": delta, "protocol": "P-in",
                })

    # ---- Read the existing LOPO runs ----
    for fname in C.RUNS.iterdir():
        if not fname.name.startswith("B1_LOPO_feat_") or "_fold" not in fname.name:
            continue
        rpath = fname / "result.json"
        if not rpath.exists():
            continue
        data = json.loads(rpath.read_text())
        variant = data.get("variant")
        fold = data.get("fold")
        if variant not in ("blackbox", "h2tcbd") or fold is None:
            continue
        fold = int(fold)
        m = re.search(r"_s(\d+)_fold", fname.name)
        if not m:
            continue
        seed = int(m.group(1))
        # Locate the paired h2tcbd run for the same seed/fold.
        if variant == "blackbox":
            h2t_path = C.RUNS / fname.name.replace("blackbox", "h2tcbd") / "result.json"
            if h2t_path.exists():
                h2t_data = json.loads(h2t_path.read_text())
                bb_q = data["test_qwk"]
                h2t_q = h2t_data["test_qwk"]
                delta = h2t_q - bb_q
                all_pairs.append({
                    "fold": fold, "seed": seed,
                    "bb_qwk": bb_q, "h2t_qwk": h2t_q,
                    "delta": delta, "protocol": "LOPO",
                })

    # Summary.
    print()
    print(f"n = {len(all_pairs)} BB-delta pairs")

    bb_qwks = np.array([p["bb_qwk"] for p in all_pairs])
    deltas = np.array([p["delta"] for p in all_pairs])
    from scipy import stats
    all_sr, all_sp = stats.spearmanr(bb_qwks, deltas)
    all_pr, all_pp = stats.pearsonr(bb_qwks, deltas)

    print(f"all ({len(all_pairs)}):")
    print(f"  Pearson r = {all_pr:.4f}  p = {all_pp:.4f}")
    print(f"  Spearman r = {all_sr:.4f}  p = {all_sp:.4f}")

    pin_pairs = [p for p in all_pairs if p["protocol"] == "P-in"]
    lopo_pairs = [p for p in all_pairs if p["protocol"] == "LOPO"]

    for label, pairs in [("P-in", pin_pairs), ("LOPO", lopo_pairs)]:
        if pairs:
            bb = np.array([p["bb_qwk"] for p in pairs])
            dd = np.array([p["delta"] for p in pairs])
            pr, pp = stats.pearsonr(bb, dd)
            sr, sp = stats.spearmanr(bb, dd)
            print(f"\n{label} ({len(pairs)}):")
            print(f"  Pearson r = {pr:.4f}  p = {pp:.4f}")
            print(f"  Spearman r = {sr:.4f}  p = {sp:.4f}")

    print(f"\n{'Protocol':>8s} {'Fold':>5s} {'Seed':>5s} {'BB':>8s} {'H2T':>8s} {'delta':>8s}")
    print("-" * 48)
    for p in sorted(all_pairs, key=lambda x: (x["bb_qwk"])):
        print(f"{p['protocol']:>8s} {p['fold']:>5d} {p['seed']:>5d} "
              f"{p['bb_qwk']:>8.4f} {p['h2t_qwk']:>8.4f} {p['delta']:>+8.4f}")

    out = C.RUNS / "pin_per_prompt_analysis.json"
    out.write_text(json.dumps({
        "n_pairs": len(all_pairs),
        "pearson_r": round(float(all_pr), 4),
        "pearson_p": round(float(all_pp), 4),
        "spearman_r": round(float(all_sr), 4),
        "spearman_p": round(float(all_sp), 4),
        "pairs": all_pairs,
    }, indent=2))
    print(f"\nsaved: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
