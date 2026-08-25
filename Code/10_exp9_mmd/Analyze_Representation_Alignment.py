"""Core Gap-3 verification: does invariant anchoring align representations?

Extract the 6-dimensional bottleneck representations z from LOPO models
trained with 6-group and 5-group anchoring, then measure the alignment of the
representation distributions across prompts:
  - cross-prompt cosine similarity: mean cosine between prompt mean vectors;
  - cross-prompt MMD (Maximum Mean Discrepancy): distance between distributions.

If the 5-group representations are more aligned than the 6-group ones, the
selective-anchoring mechanism is supported at the representation level.

Note: the main experiments save model.pt checkpoints. This script uses them
only when all 48 runs (2 variants x 3 seeds x 8 folds) have checkpoints;
otherwise it retrains all 48 quick models (fixed 30 epochs, no early
stopping). The two variants are therefore always compared under the same
training protocol. The MMD values are used only for relative comparisons
between variants.

Usage: python Analyze_Representation_Alignment.py
"""
from __future__ import annotations
import json, sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_Core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "11_Analysis"))
import Common as C
from Engine import load_emb, make_model
from Rbf_Kernel import mmd2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [13, 42, 123]


def extract_z(model, embs):
    """Extract the bottleneck representations z (6-dimensional vectors)."""
    model.eval()
    zs = []
    X = torch.tensor(embs, dtype=torch.float32).to(DEVICE)
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            out = model(xb)
            zs.append(out["z"].cpu().numpy())
    return np.concatenate(zs, axis=0)


def load_trained_model(variant, seed, fold):
    """Load a trained LOPO model from its checkpoint, if one was saved."""
    # Map variants to run directories.
    if variant == "6groups":
        run_dir = C.RUNS / f"B1_LOPO_feat_h2tcbd_s{seed}_fold{fold}"
    elif variant == "5groups":
        run_dir = C.RUNS / f"B1_5grp_rescaled_LOPO_feat_s{seed}_fold{fold}"
    else:
        raise ValueError(variant)

    # The run result and the saved weights must both exist.
    rpath = run_dir / "result.json"
    ckpt = run_dir / "model.pt"
    if not (rpath.exists() and ckpt.exists()):
        return None, None

    # Rebuild the model and restore the saved weights.
    d = load_emb("feat")
    in_dim = d["emb"].shape[1]
    model = make_model("h2tcbd", in_dim=in_dim, K=6).to(DEVICE)
    if variant == "5groups":
        model.anchor_heads = model.anchor_heads[:5]
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    result = json.loads(rpath.read_text(encoding="utf-8"))
    return model, result


def quick_train_for_z(variant, seed, fold, epochs=30):
    """Quickly train a model solely to extract z representations."""
    d = load_emb("feat")
    smin, smax = C.DATASETS["feat"]["smin"], C.DATASETS["feat"]["smax"]
    te_mask = d["essay_set"] == fold
    tr_idx = np.where(~te_mask)[0]
    te_idx = np.where(te_mask)[0]
    rng = np.random.default_rng(seed); rng.shuffle(tr_idx)
    n_va = max(1, int(0.1 * len(tr_idx)))
    iva, itr = tr_idx[:n_va], tr_idx[n_va:]

    from Engine import FeatNorm
    fn = FeatNorm().fit(d["feats"][itr])
    smap = C.ScaleMap(smin, smax).fit(d["score"][itr])

    C.set_seed(seed)
    in_dim = d["emb"].shape[1]
    import torch.nn as nn
    from Engine import ANCHOR_DIMS as AD_6
    AD_5 = AD_6[:5]
    model = make_model("h2tcbd", in_dim=in_dim, K=6).to(DEVICE)
    if variant == "5groups":
        model.anchor_heads = model.anchor_heads[:5]
        anchor_dims = AD_5
    else:
        anchor_dims = AD_6

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse_fn = nn.MSELoss()
    Xtr = torch.tensor(d["emb"][itr], dtype=torch.float32)
    ytr = torch.tensor(smap.to_unit(d["score"][itr]), dtype=torch.float32)
    Ftr = torch.tensor(fn.transform(d["feats"][itr]), dtype=torch.float32)
    Xva = torch.tensor(d["emb"][iva], dtype=torch.float32).to(DEVICE)
    n = len(Xtr)

    best_qwk, best_ep = -1e9, -1
    z_te = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            xb = Xtr[idx].to(DEVICE); yb = ytr[idx].to(DEVICE)
            fb = Ftr[idx].to(DEVICE)
            out = model(xb)
            loss = mse_fn(out["score"], yb)
            loss_anchor = 0.0
            for k in range(len(anchor_dims)):
                pred = model.anchor_heads[k](out["z"][:, k:k + 1])
                loss_anchor += mse_fn(pred, fb[:, anchor_dims[k]])
            lam = 0.5 * len(anchor_dims) / 6  # rescaled: per-group weight stays 0.5/6
            loss = loss + lam * loss_anchor / len(anchor_dims)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pred_u = model(Xva)["score"].cpu().numpy()
        vq = C.qwk(d["score"][iva], smap.from_unit(pred_u), smap.smin, smap.smax)
        if vq > best_qwk:
            best_qwk, best_ep = vq, ep
            # Extract z for the test set at the best epoch.
            z_te = extract_z(model, d["emb"][te_idx])

    if z_te is None:
        raise RuntimeError("no best epoch found for representation extraction")
    return z_te, d["essay_set"][te_idx], best_qwk  # (N,6), (N,), float


def main():
    print("cross-prompt representation alignment analysis")

    # All 8 folds x 2 variants x 3 seeds = 48 quick trainings (30 epochs each).
    rep_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    variants = ["6groups", "5groups"]

    def _checkpoints_ready() -> bool:
        for variant in variants:
            for seed in SEEDS:
                for fold in rep_folds:
                    if variant == "6groups":
                        run_dir = C.RUNS / f"B1_LOPO_feat_h2tcbd_s{seed}_fold{fold}"
                    else:
                        run_dir = C.RUNS / f"B1_5grp_rescaled_LOPO_feat_s{seed}_fold{fold}"
                    if not ((run_dir / "result.json").exists() and (run_dir / "model.pt").exists()):
                        return False
        return True

    use_checkpoints = _checkpoints_ready()
    print(f"use saved checkpoints: {use_checkpoints}")

    all_data = {}  # (variant, seed, fold) -> (z, essay_sets, qwk)
    summary = {"6groups": {}, "5groups": {}}

    for variant in variants:
        for seed in SEEDS:
            for fold in rep_folds:
                key = (variant, seed, fold)
                print(f"  train {variant} s{seed} fold{fold} ...", end=" ", flush=True)
                if use_checkpoints:
                    model, meta = load_trained_model(variant, seed, fold)
                    if model is None:
                        raise RuntimeError(f"missing checkpoint for {key}")
                    d = load_emb("feat")
                    te_idx = np.where(d["essay_set"] == fold)[0]
                    z = extract_z(model, d["emb"][te_idx])
                    es = d["essay_set"][te_idx]
                    qwk = meta["test_qwk"]
                    src = "checkpoint"
                else:
                    z, es, qwk = quick_train_for_z(variant, seed, fold, epochs=30)
                    src = "retrained"
                all_data[key] = (z, es, qwk)
                print(f"QWK={qwk:.4f} ({src})")

    # ---- Analysis ----
    print()
    print("cross-prompt representation alignment results")
    print("=" * 60)

    for variant in variants:
        print(f"\n{variant}:")
    # For each seed, collect the z representations across folds.
        for seed in SEEDS:
            z_by_fold = {}
            for fold in rep_folds:
                key = (variant, seed, fold)
                if key in all_data:
                    z_by_fold[fold] = all_data[key][0]

            if len(z_by_fold) < 2:
                continue

            folds_sorted = sorted(z_by_fold.keys())
            # Compute cosine similarity and MMD for all fold pairs.
            cosines = []
            mmds = []
            for i in range(len(folds_sorted)):
                for j in range(i + 1, len(folds_sorted)):
                    fi, fj = folds_sorted[i], folds_sorted[j]
                    zi, zj = z_by_fold[fi], z_by_fold[fj]
                    # Cosine: mean vector per prompt, cosine across prompts.
                    cos = float(np.dot(zi.mean(0), zj.mean(0)) /
                                (np.linalg.norm(zi.mean(0)) * np.linalg.norm(zj.mean(0)) + 1e-8))
                    cosines.append(cos)
                    # MMD: distribution distance (sample 500 points).
                    ni = min(500, len(zi))
                    nj = min(500, len(zj))
                    m = mmd2(torch.tensor(zi[:ni], dtype=torch.float32), torch.tensor(zj[:nj], dtype=torch.float32)).item()
                    mmds.append(m)
                    print(f"    s{seed} fold{fi}-fold{fj}: cos={cos:+.4f}  mmd={m:.4f}")

            if cosines:
                print(f"    s{seed} mean: cos={np.mean(cosines):+.4f}  mmd={np.mean(mmds):.4f}")
                summary[variant][str(seed)] = {
                    "cos": round(float(np.mean(cosines)), 4),
                    "mmd": round(float(np.mean(mmds)), 4),
                }

        # Summary over seeds.
        all_cos = []
        all_mmds = []
        for seed in SEEDS:
            z_by_fold = {}
            for fold in rep_folds:
                key = (variant, seed, fold)
                if key in all_data:
                    z_by_fold[fold] = all_data[key][0]
            if len(z_by_fold) < 2: continue
            for i, fi in enumerate(sorted(z_by_fold.keys())):
                for fj in sorted(z_by_fold.keys())[i + 1:]:
                    zi, zj = z_by_fold[fi], z_by_fold[fj]
                    cos = float(np.dot(zi.mean(0), zj.mean(0)) /
                                (np.linalg.norm(zi.mean(0)) * np.linalg.norm(zj.mean(0)) + 1e-8))
                    all_cos.append(cos)
                    ni, nj = min(500, len(zi)), min(500, len(zj))
                    all_mmds.append(mmd2(torch.tensor(zi[:ni], dtype=torch.float32), torch.tensor(zj[:nj], dtype=torch.float32)).item())

        print(f"  {variant} overall: cos={np.mean(all_cos):+.4f}  mmd={np.mean(all_mmds):.4f}")
        summary[variant]["overall"] = {
            "cos": round(float(np.mean(all_cos)), 4),
            "mmd": round(float(np.mean(all_mmds)), 4),
        }

    # Comparison.
    print()
    print("=" * 60)
    print("comparison")
    print("=" * 60)
    for metric, name, _ in [("cos", "cosine", "higher"), ("mmd", "MMD", "lower")]:
        v6_vals = []
        v5_vals = []
        for seed in SEEDS:
            for variant in ["6groups", "5groups"]:
                z_by_fold = {}
                for fold in rep_folds:
                    key = (variant, seed, fold)
                    if key in all_data:
                        z_by_fold[fold] = all_data[key][0]
                if len(z_by_fold) < 2: continue
                vals = []
                for i, fi in enumerate(sorted(z_by_fold.keys())):
                    for fj in sorted(z_by_fold.keys())[i + 1:]:
                        zi, zj = z_by_fold[fi], z_by_fold[fj]
                        if metric == "cos":
                            v = float(np.dot(zi.mean(0), zj.mean(0)) /
                                      (np.linalg.norm(zi.mean(0)) * np.linalg.norm(zj.mean(0)) + 1e-8))
                        else:
                            ni, nj = min(500, len(zi)), min(500, len(zj))
                            v = mmd2(torch.tensor(zi[:ni], dtype=torch.float32), torch.tensor(zj[:nj], dtype=torch.float32)).item()
                        vals.append(v)
                if variant == "6groups": v6_vals.extend(vals)
                else: v5_vals.extend(vals)

        if v6_vals and v5_vals:
            m6, m5 = np.mean(v6_vals), np.mean(v5_vals)
            print(f"  {name}: 6grp={m6:.4f}, 5grp={m5:.4f}, diff={m5-m6:+.4f}")

    # ---- Paired prompt-pair MMD test (6-group vs 5-group) ----
    from scipy import stats as sstats
    pair_vals = {}
    for variant in variants:
        pair_vals[variant] = {}
        for seed in SEEDS:
            z_by_fold = {}
            for fold in rep_folds:
                key = (variant, seed, fold)
                if key in all_data:
                    z_by_fold[fold] = all_data[key][0]
            folds_sorted = sorted(z_by_fold.keys())
            for i in range(len(folds_sorted)):
                for j in range(i + 1, len(folds_sorted)):
                    fi, fj = folds_sorted[i], folds_sorted[j]
                    zi, zj = z_by_fold[fi], z_by_fold[fj]
                    ni, nj = min(500, len(zi)), min(500, len(zj))
                    m = mmd2(torch.tensor(zi[:ni], dtype=torch.float32),
                             torch.tensor(zj[:nj], dtype=torch.float32)).item()
                    pair_vals[variant][(seed, fi, fj)] = m

    keys = sorted(pair_vals["6groups"])
    diffs = np.array([pair_vals["5groups"][k] - pair_vals["6groups"][k]
                      for k in keys])
    obs_mean = float(diffs.mean())
    n_pairs = len(diffs)

    rng = np.random.default_rng(42)
    perm = np.empty(100_000)
    for i in range(100_000):
        signs = rng.choice([-1.0, 1.0], size=n_pairs)
        perm[i] = (signs * diffs).mean()
    p_two = float((np.abs(perm) >= abs(obs_mean)).mean())
    p_one = float((perm <= obs_mean).mean())

    wres = sstats.wilcoxon(diffs)
    wres_less = sstats.wilcoxon(diffs, alternative="less")
    tres = sstats.ttest_1samp(diffs, 0)

    seed_diff = {s: float(np.mean([pair_vals["5groups"][(s, fi, fj)]
                                   - pair_vals["6groups"][(s, fi, fj)]
                                   for (ss, fi, fj) in keys if ss == s]))
                 for s in SEEDS}

    seed_arr = np.array([seed_diff[s] for s in SEEDS])
    tres_seed = sstats.ttest_1samp(seed_arr, 0)
    seed_level = {
        "n_seeds": len(seed_arr),
        "mean_diff": round(float(seed_arr.mean()), 4),
        "sd": round(float(seed_arr.std(ddof=1)), 4),
        "paired_t_p_two_sided": round(float(tres_seed.pvalue), 4),
        "paired_t_p_one_sided": round(float(tres_seed.pvalue / 2), 4),
        "sign_test_p_one_sided": round(0.5 ** len(seed_arr), 4),
    }

    pairwise_test = {
        "n_pairs": n_pairs,
        "mean_diff_mmd5_minus_6": round(obs_mean, 4),
        "n_negative_diffs": int((diffs < 0).sum()),
        "seed_mean_diff": {str(s): round(seed_diff[s], 4) for s in SEEDS},
        "seed_level": seed_level,
        "permutation_p_two_sided": round(p_two, 4),
        "permutation_p_one_sided": round(p_one, 4),
        "wilcoxon_p_two_sided": round(float(wres.pvalue), 4),
        "wilcoxon_p_one_sided": round(float(wres_less.pvalue), 4),
        "ttest_84_units_p_two_sided": round(float(tres.pvalue), 4),
        "ttest_84_units_p_one_sided": round(float(tres.pvalue / 2), 4),
    }
    print("paired prompt-pair MMD test:")
    for k, v in pairwise_test.items():
        print(f"  {k}: {v}")

    # ---- Prompt-label permutation test (exact, 8! = 40,320) ----
    from itertools import permutations
    perms = list(permutations(range(8)))

    def mmd_matrix(variant, seed):
        M = np.zeros((8, 8))
        for (ss, fi, fj), v in pair_vals[variant].items():
            if ss == seed:
                M[fi - 1, fj - 1] = v
                M[fj - 1, fi - 1] = v
        return M

    triu = np.triu_indices(8, k=1)
    drop_perms = np.empty(len(perms))
    for b, pi in enumerate(perms):
        ds = []
        for seed in SEEDS:
            m6 = mmd_matrix("6groups", seed)[np.ix_(pi, pi)][triu].mean()
            m5 = mmd_matrix("5groups", seed)[np.ix_(pi, pi)][triu].mean()
            ds.append(m5 - m6)
        drop_perms[b] = np.mean(ds)

    obs_drop = float(drop_perms[0])  # identity permutation = observed grouping
    p_one = float((drop_perms <= obs_drop).mean())
    p_two = float((np.abs(drop_perms) >= abs(obs_drop)).mean())
    prompt_label_perm = {
        "n_permutations": int(len(perms)),
        "observed_mean_drop": round(obs_drop, 4),
        "permutation_mean": round(float(drop_perms.mean()), 4),
        "permutation_sd": round(float(drop_perms.std(ddof=1)), 4),
        "permutation_min": round(float(drop_perms.min()), 4),
        "permutation_max": round(float(drop_perms.max()), 4),
        "p_one_sided": round(p_one, 4),
        "p_two_sided": round(p_two, 4),
    }
    print("prompt-label permutation test (exact, 8! permutations):")
    for k, v in prompt_label_perm.items():
        print(f"  {k}: {v}")

    C.save_result("Exp10_MMD", {"summary": summary, "pairwise_test": pairwise_test,
                                "prompt_label_permutation": prompt_label_perm})
    print("saved: runs/Exp10_MMD/result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
