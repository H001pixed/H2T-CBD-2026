"""Cross-dataset selective anchoring on ASAP-2.0 and Feedback (P-in protocol,
Exp9 extension).

Reviewer point 1.2 / Exp11: the ARS screening decision was only validated on
ASAP-1. The official ASAP-2.0 and Feedback files contain no prompt labels, so
ARS (mu - sigma) cannot be computed there; each dataset is instead screened by
its pooled |Spearman| group means, rho_bar < theta = 0.1 (see
analyze_ars_cross_dataset.py). With the current data this screens out
coherence_readability on ASAP-2.0 and rhetoric_engagement on Feedback.

For each dataset we train, under the standard 80/10/10 stratified P-in
protocol and three seeds:
  6grp : H2T-CBD with all six anchor heads active;
  5grp : H2T-CBD with the screened-out group's anchor head removed (its
         bottleneck dimension stays free).
The existing unanchored H2T-CBD runs (free bottleneck, no features attached)
are reused as a no-anchor reference.

Anchor weight: lambda = 0.5 divided by the number of kept groups, matching the
main experiments.

Usage: python run_cross_dataset_selective_anchor.py
Outputs: per-run result.json under results/runs/B1_AnchorPin_* and a summary
under results/runs/ars_cross_dataset/selective_anchor_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn

for core in ("00_core",):
    p = Path(__file__).resolve().parents[1] / core
    if p.is_dir():
        sys.path.insert(0, str(p))
        break

import common as C  # noqa: E402
from engine import load_emb, FeatNorm  # noqa: E402
from models import H2TCBD  # noqa: E402
from metrics_logger import MetricsLogger  # noqa: E402

SEEDS = [13, 42, 123]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAM = 0.5
THETA = 0.1
GROUP_NAMES = list(C.ANCHOR_GROUPS.keys())
ANCHOR_DIMS = [[C.FEAT_COLS.index(f) for f in grp] for grp in C.ANCHOR_GROUPS.values()]


def screened_groups(feats, scores):
    """Pooled per-group |Spearman| means; groups below THETA are screened out."""
    rhos = {}
    for g, fs in C.ANCHOR_GROUPS.items():
        vals = []
        for f in fs:
            i = C.FEAT_COLS.index(f)
            vals_i = feats[:, i]
            mask = np.isfinite(vals_i)
            if mask.sum() < 10 or np.std(vals_i[mask]) == 0:
                continue
            r, _ = stats.spearmanr(vals_i[mask], scores[mask])
            vals.append(abs(r))
        rhos[g] = float(np.mean(vals)) if vals else np.nan
    drop = [g for g in GROUP_NAMES if rhos[g] < THETA]
    return rhos, drop


def train_anchored(tr, va, te, scoremap_tr, seed, feat_norm, kept, ml=None):
    """Train H2T-CBD with only the kept anchor heads; dropped dims stay free."""
    C.set_seed(seed)
    in_dim = tr["emb"].shape[1]
    model = H2TCBD(in_dim=in_dim, K=6, anchor_dims=ANCHOR_DIMS,
                   use_anchor=True).to(DEVICE)
    model.anchor_heads = nn.ModuleList([model.anchor_heads[k] for k in kept])
    kept_dims = [ANCHOR_DIMS[k] for k in kept]
    n_kept = len(kept)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse_fn = nn.MSELoss()
    Xtr = torch.tensor(tr["emb"], dtype=torch.float32)
    ytr = torch.tensor(scoremap_tr.to_unit(tr["score"]), dtype=torch.float32)
    Ftr = torch.tensor(feat_norm.transform(tr["feats"]), dtype=torch.float32)
    Xva = torch.tensor(va["emb"], dtype=torch.float32).to(DEVICE)
    Xte = torch.tensor(te["emb"], dtype=torch.float32).to(DEVICE)
    n = len(Xtr)

    def evaluate(X, raw, smap):
        model.eval()
        with torch.no_grad():
            pred_u = model(X)["score"].cpu().numpy()
        pred_i = smap.from_unit(pred_u)
        return C.qwk(raw, pred_i, smap.smin, smap.smax), pred_i

    best_qwk, best_state, best_ep = -1e9, None, -1
    for ep in range(60):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            xb = Xtr[idx].to(DEVICE)
            yb = ytr[idx].to(DEVICE)
            fb = Ftr[idx].to(DEVICE)
            out = model(xb)
            loss = mse_fn(out["score"], yb)
            a_loss = 0.0
            for k, head in enumerate(model.anchor_heads):
                pred = head(out["z"][:, kept[k]:kept[k] + 1])
                a_loss = a_loss + mse_fn(pred, fb[:, kept_dims[k]])
            loss = loss + LAM * a_loss / n_kept
            opt.zero_grad()
            loss.backward()
            opt.step()
        vq, _ = evaluate(Xva, va["score"], scoremap_tr)
        if ml is not None:
            ml.log(epoch=ep, val_qwk=round(vq, 4))
        if vq > best_qwk:
            best_qwk, best_ep = vq, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if ep - best_ep >= 12:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        if ml is not None:
            torch.save(best_state, Path(ml.exp_dir) / "model.pt")
    tq, pred_i = evaluate(Xte, te["score"], scoremap_tr)
    metrics = {
        "test_qwk": round(tq, 4),
        "test_rmse": round(C.rmse(te["score"], pred_i), 4),
        "test_pearson": round(C.pearson(te["score"], pred_i), 4),
        "val_qwk": round(best_qwk, 4),
        "best_epoch": best_ep,
        "n_train": int(n),
        "n_test": int(len(te["score"])),
    }
    return metrics, model


def main():
    print("Cross-dataset selective anchoring (P-in): ASAP-2.0 & Feedback")
    summary = {"theta": THETA, "datasets": {}}

    for name in ("asap2", "dysf"):
        d = load_emb(name)
        feats = np.load(C.RUNS / "ars_cross_dataset" / f"{name}_features.npy")
        df = C.load_dataset(name)
        assert len(feats) == len(d["emb"]) == len(d["score"]) == len(df), (
            name, feats.shape, d["emb"].shape, len(d["score"]), len(df))
        assert np.array_equal(df["score"].values.astype(np.int64),
                              d["score"].astype(np.int64)), (
            f"{name}: feature cache and embedding npz score order mismatch")
        d["feats"] = feats

        smin, smax = C.DATASETS[name]["smin"], C.DATASETS[name]["smax"]
        rhos, drop = screened_groups(feats, d["score"])
        kept = [i for i, g in enumerate(GROUP_NAMES) if g not in drop]
        print(f"\n{name}: pooled rho_bar = "
              f"{ {g: round(rhos[g], 4) for g in GROUP_NAMES} }")
        print(f"{name}: screened out (rho_bar < {THETA}) -> {drop}; "
              f"kept {len(kept)} groups")
        summary["datasets"][name] = {
            "pooled_rho_bar": {g: round(v, 4) for g, v in rhos.items()},
            "screened_out": drop,
            "kept": [GROUP_NAMES[i] for i in kept],
        }

        per_seed = {}
        for seed in SEEDS:
            dfp = pd.DataFrame({"score": d["score"]})
            itr, iva, ite = C.stratified_split(dfp, seed=seed)
            smap = C.ScaleMap(smin, smax).fit(d["score"][itr])
            fn = FeatNorm().fit(d["feats"][itr])
            sub = lambda i: {"emb": d["emb"][i], "score": d["score"][i],
                             "feats": d["feats"][i]}
            tr, va, te = sub(itr), sub(iva), sub(ite)
            row = {"seed": seed}
            for n_grp, kept_list in ((6, list(range(6))), (len(kept), kept)):
                exp_id = f"B1_AnchorPin_{name}_{n_grp}grp_s{seed}"
                if C.already_done(exp_id):
                    r = json.loads(C.result_path(exp_id).read_text())
                    print(f"[skip] {exp_id} qwk={r['test_qwk']}")
                else:
                    ml = MetricsLogger(C.RUNS / exp_id)
                    metrics, _ = train_anchored(tr, va, te, smap, seed=seed,
                                                feat_norm=fn, kept=kept_list,
                                                ml=ml)
                    r = {"exp_id": exp_id, "dataset": name, "n_groups": n_grp,
                         "dropped": [GROUP_NAMES[i] for i in range(6)
                                     if i not in kept_list],
                         "seed": seed, "protocol": "P-in", **metrics}
                    C.save_result(exp_id, r)
                    print(f"[done] {exp_id} qwk={metrics['test_qwk']:.4f}")
                row[f"qwk_{n_grp}grp"] = r["test_qwk"]
                row[f"rmse_{n_grp}grp"] = r["test_rmse"]

            ref_id = f"B1_Pin_{name}_h2tcbd_s{seed}"
            ref_path = C.RUNS / ref_id / "result.json"
            if ref_path.exists():
                ref = json.loads(ref_path.read_text())
                row["qwk_free"] = ref["test_qwk"]
            per_seed[seed] = row
        summary["datasets"][name]["per_seed"] = per_seed

        print(f"\n{name}  P-in QWK (free = unanchored H2T-CBD reference):")
        print(f"{'Seed':>5s} {'free':>8s} {'6grp':>8s} {'5grp':>8s} "
              f"{'5-6':>8s} {'5-free':>8s}")
        for seed in SEEDS:
            r = per_seed[seed]
            qf = r.get("qwk_free")
            q6 = r.get("qwk_6grp")
            q5 = r.get("qwk_5grp")
            sf = f"{qf:>8.4f}" if qf is not None else f"{'n/a':>8s}"
            sd = f"{q5 - qf:>+8.4f}" if qf is not None else f"{'n/a':>8s}"
            print(f"{seed:>5d} {sf} {q6:>8.4f} {q5:>8.4f} "
                  f"{q5 - q6:>+8.4f} {sd}")
        q6s = [per_seed[s]["qwk_6grp"] for s in SEEDS]
        q5s = [per_seed[s]["qwk_5grp"] for s in SEEDS]
        summary["datasets"][name]["mean_6grp"] = round(float(np.mean(q6s)), 4)
        summary["datasets"][name]["mean_5grp"] = round(float(np.mean(q5s)), 4)
        summary["datasets"][name]["mean_delta"] = round(
            float(np.mean(q5s) - np.mean(q6s)), 4)
        print(f"mean: 6grp={np.mean(q6s):.4f}  5grp={np.mean(q5s):.4f}  "
              f"delta={np.mean(q5s) - np.mean(q6s):+.4f}")

    out = C.RUNS / "ars_cross_dataset" / "selective_anchor_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
