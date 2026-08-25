"""Cross-dataset 5-group control with rescaled anchor weight (Exp2.8).

Mirrors the ASAP-1 K/lambda control on ASAP-2.0 and Feedback: the 5-group
selective-anchoring configs (ASAP-2.0 drops coherence/readability, Feedback
drops rhetoric/engagement) are retrained with lambda_rescaled = 0.5 * 5 / 6 =
0.4167 so that the per-group anchor weight equals the 6-group setting
(lambda/6). This separates "removing the screened-out group" from "increasing
the per-group weight".

Usage: python Run_Cross_Dataset_Selective_Anchor_Rescaled.py
Outputs: results/runs/B1_AnchorPin_{name}_5grp_rescaled_s{seed} and
         results/runs/k_lambda_control_cross/result.json
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

for core in ("00_Core",):
    p = Path(__file__).resolve().parents[1] / core
    if p.is_dir():
        sys.path.insert(0, str(p))
        break

import Common as C  # noqa: E402
from Engine import load_emb, FeatNorm  # noqa: E402
from Models import H2TCBD  # noqa: E402
from Metrics_Logger import MetricsLogger  # noqa: E402

SEEDS = [13, 42, 123]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAM_RESCALED = 0.5 * 5 / 6
THETA = 0.1
GROUP_NAMES = list(C.ANCHOR_GROUPS.keys())
ANCHOR_DIMS = [[C.FEAT_COLS.index(f) for f in grp] for grp in C.ANCHOR_GROUPS.values()]


def screened_groups(feats, scores):
    rhos = {}
    for g, fs in C.ANCHOR_GROUPS.items():
        vals = []
        for f in fs:
            i = C.FEAT_COLS.index(f)
            v = feats[:, i]
            mask = np.isfinite(v)
            if mask.sum() < 10 or np.std(v[mask]) == 0:
                continue
            r, _ = stats.spearmanr(v[mask], scores[mask])
            vals.append(abs(r))
        rhos[g] = float(np.mean(vals)) if vals else np.nan
    return [g for g in GROUP_NAMES if rhos[g] < THETA]


def train_anchored(tr, va, te, scoremap_tr, seed, feat_norm, kept, lam, ml=None):
    """Train H2T-CBD with the kept anchor heads and the given anchor weight."""
    C.set_seed(seed)
    model = H2TCBD(in_dim=tr["emb"].shape[1], K=6, anchor_dims=ANCHOR_DIMS,
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
            loss = loss + lam * a_loss / n_kept
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
    return {"test_qwk": round(tq, 4),
            "test_rmse": round(C.rmse(te["score"], pred_i), 4),
            "test_pearson": round(C.pearson(te["score"], pred_i), 4),
            "val_qwk": round(best_qwk, 4),
            "best_epoch": best_ep}, model


def load_qwk(exp_id):
    p = C.RUNS / exp_id / "result.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["test_qwk"]


def main():
    summary = {}
    for name in ("asap2", "dysf"):
        d = load_emb(name)
        feats = np.load(C.RUNS / "ars_cross_dataset" / f"{name}_features.npy")
        df = C.load_dataset(name)
        assert len(feats) == len(d["emb"]) == len(df)
        assert np.array_equal(df["score"].values.astype(np.int64),
                              d["score"].astype(np.int64))
        d["feats"] = feats
        smin, smax = C.DATASETS[name]["smin"], C.DATASETS[name]["smax"]
        drop = screened_groups(feats, d["score"])
        kept = [i for i, g in enumerate(GROUP_NAMES) if g not in drop]
        print(f"{name}: screened out {drop}; kept {len(kept)} groups")

        for seed in SEEDS:
            dfp = pd.DataFrame({"score": d["score"]})
            itr, iva, ite = C.stratified_split(dfp, seed=seed)
            smap = C.ScaleMap(smin, smax).fit(d["score"][itr])
            fn = FeatNorm().fit(d["feats"][itr])
            sub = lambda i: {"emb": d["emb"][i], "score": d["score"][i],
                             "feats": d["feats"][i]}
            exp_id = f"B1_AnchorPin_{name}_5grp_rescaled_s{seed}"
            if C.already_done(exp_id):
                r = json.loads(C.result_path(exp_id).read_text())
                print(f"[skip] {exp_id} qwk={r['test_qwk']}")
            else:
                ml = MetricsLogger(C.RUNS / exp_id)
                metrics, _ = train_anchored(sub(itr), sub(iva), sub(ite), smap,
                                            seed=seed, feat_norm=fn, kept=kept,
                                            lam=LAM_RESCALED, ml=ml)
                r = {"exp_id": exp_id, "dataset": name, "n_groups": len(kept),
                     "dropped": drop, "lambda": round(LAM_RESCALED, 6),
                     "seed": seed, "protocol": "P-in", **metrics}
                C.save_result(exp_id, r)
                print(f"[done] {exp_id} qwk={metrics['test_qwk']:.4f}")

        # Four-condition comparison per dataset.
        rows = []
        print(f"\n{name} (P-in QWK):")
        for cfg, prefix, tag in (
            ("free", f"B1_Pin_{name}_h2tcbd", None),
            ("6grp", f"B1_AnchorPin_{name}_6grp", None),
            ("5grp_rescaled", f"B1_AnchorPin_{name}_5grp_rescaled", None),
        ):
            vals = [load_qwk(f"{prefix}_s{s}") for s in SEEDS]
            if any(v is None for v in vals):
                print(f"  {cfg}: missing")
                continue
            rows.append({"config": cfg, "qwk": [vals]})
            print(f"  {cfg}: " + " ".join(f"{v:.4f}" for v in vals)
                  + f"  mean={np.mean(vals):.4f}")
        summary[name] = {
            "screened_out": drop,
            "lambda_rescaled": round(LAM_RESCALED, 6),
            "rows": rows,
        }

    out_dir = C.RUNS / "k_lambda_control_cross"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
