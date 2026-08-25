"""5-group selective anchoring (drop rhetoric_engagement) on LOPO, 24 folds.

Rhetoric engagement has the lowest mean |r| (0.106) and is the only group whose
ARS (0.070) falls below the threshold theta = 0.1. Expected: hurt folds
(3/4/5/6) recover while benefit folds stay largely unchanged.
"""

from __future__ import annotations
import json, sys, re
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_core"))
import Common as C
from Engine import load_emb, FeatNorm, make_model
import torch, torch.nn as nn
from Metrics_Logger import MetricsLogger

SEEDS = [13, 42, 123]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 6 groups -> 5 groups: drop rhetoric engagement (index 5, the last group).
# ANCHOR_GROUPS order: length_fluency(0), lexical_sophistication(1),
#   syntactic_complexity(2), mechanics(3), coherence_readability(4),
#   rhetoric_engagement(5) <- dropped.
ANCHOR_DIMS_6 = [[C.FEAT_COLS.index(f) for f in grp] for grp in C.ANCHOR_GROUPS.values()]
assert list(C.ANCHOR_GROUPS)[-1] == "rhetoric_engagement", (
    "rhetoric_engagement must remain the last anchor group"
)
ANCHOR_DIMS_5 = ANCHOR_DIMS_6[:5]  # Keep the first five groups.


def train_5groups(tr, va, te, scoremap_tr, seed, feat_norm, te_scoremap=None, ml=None):
    """Train the 5-group anchored H2T-CBD (rhetoric engagement dropped)."""
    C.set_seed(seed)
    in_dim = tr["emb"].shape[1]
    model = make_model("h2tcbd", in_dim=in_dim, K=6).to(DEVICE)
    # Replace anchor heads with the five kept groups.
    model.anchor_heads = model.anchor_heads[:5]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse_fn = nn.MSELoss()
    Xtr = torch.tensor(tr["emb"], dtype=torch.float32)
    ytr = torch.tensor(scoremap_tr.to_unit(tr["score"]), dtype=torch.float32)
    Ftr = torch.tensor(feat_norm.transform(tr["feats"]), dtype=torch.float32)
    Xva = torch.tensor(va["emb"], dtype=torch.float32).to(DEVICE)
    Xte = torch.tensor(te["emb"], dtype=torch.float32).to(DEVICE)
    n = len(Xtr)
    te_map = te_scoremap or scoremap_tr

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
        tot_mse, tot_anchor = 0.0, 0.0
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            xb = Xtr[idx].to(DEVICE); yb = ytr[idx].to(DEVICE)
            fb = Ftr[idx].to(DEVICE)
            out = model(xb)
            loss_mse = mse_fn(out["score"], yb)
            loss = loss_mse
            loss_anchor_val = 0.0
            if fb is not None:
                for k, head in enumerate(model.anchor_heads):
                    pred = head(out["z"][:, k:k+1])
                    loss_anchor_val += mse_fn(pred, fb[:, ANCHOR_DIMS_5[k]])
                loss_anchor_val /= 5
                loss = loss + 0.5 * loss_anchor_val
                tot_anchor += loss_anchor_val.item() * len(idx)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_mse += loss_mse.item() * len(idx)

        vq, _ = evaluate(Xva, va["score"], scoremap_tr)
        if ml is not None:
            ml.log(epoch=ep, train_mse=round(tot_mse/n,5),
                   train_anchor=round(tot_anchor/n,5), val_qwk=round(vq,4))
        if vq > best_qwk:
            best_qwk, best_ep = vq, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= 12: break

    if best_state: model.load_state_dict(best_state)
    tq, pred_i = evaluate(Xte, te["score"], te_map)
    return {"test_qwk": round(tq,4), "test_rmse": round(C.rmse(te["score"],pred_i),4),
            "test_pearson": round(C.pearson(te["score"],pred_i),4),
            "val_qwk": round(best_qwk,4), "best_epoch": best_ep}, model


def main():
    print("5-group LOPO (drop rhetoric_engagement) x 24 folds")
    d = load_emb("feat")
    smin, smax = C.DATASETS["feat"]["smin"], C.DATASETS["feat"]["smax"]
    sets = sorted(np.unique(d["essay_set"]).tolist())
    all_r = []

    for seed in SEEDS:
        for hold in sets:
            te_mask = d["essay_set"] == hold
            tr_idx = np.where(~te_mask)[0]; te_idx = np.where(te_mask)[0]
            rng = np.random.default_rng(seed); rng.shuffle(tr_idx)
            n_va = max(1, int(0.1*len(tr_idx)))
            iva, itr = tr_idx[:n_va], tr_idx[n_va:]
            fn = FeatNorm().fit(d["feats"][itr])
            smap = C.ScaleMap(smin, smax).fit(d["score"][itr])
            sub = lambda i: {"emb":d["emb"][i],"score":d["score"][i],"feats":d["feats"][i]}
            exp_id = f"B1_5grp_LOPO_feat_s{seed}_fold{hold}"
            if C.already_done(exp_id):
                r = json.loads(C.result_path(exp_id).read_text())
                print(f"[skip] {exp_id} qwk={r['test_qwk']}")
            else:
                ml = MetricsLogger(C.RUNS / exp_id)
                metrics, model = train_5groups(sub(itr), sub(iva), sub(te_idx), smap,
                                               seed=seed, feat_norm=fn, ml=ml)
                r = {"exp_id":exp_id,"seed":seed,"fold":int(hold),**metrics}
                C.save_result(exp_id, r)
                # Persist the best weights for later representation analysis.
                torch.save(model.state_dict(), C.RUNS / exp_id / "model.pt")
                print(f"[done] {exp_id} qwk={metrics['test_qwk']:.4f}")
            all_r.append(r)

    # Compare with the 6-group anchoring runs.
    ref = {}
    for fname in C.RUNS.iterdir():
        if not fname.name.startswith("B1_LOPO_feat_h2tcbd") or "_fold" not in fname.name: continue
        rpath = fname / "result.json"
        if not rpath.exists(): continue
        data = json.loads(rpath.read_text())
        fold = data.get("fold")
        if fold is None: continue
        m = re.search(r"_s(\d+)_fold", fname.name)
        if not m: continue
        ref[(int(m.group(1)), int(fold))] = data["test_qwk"]

    print(f"\n{'Fold':>5s} {'Seed':>5s} {'6grp':>8s} {'5grp':>8s} {'diff':>8s}")
    print("-"*40)
    fold_6, fold_5 = {f:[] for f in sets}, {f:[] for f in sets}
    for fold in sets:
        for seed in SEEDS:
            q6 = ref.get((seed, fold))
            r5 = next((r for r in all_r if r["seed"]==seed and r["fold"]==fold), None)
            q5 = r5["test_qwk"] if r5 else None
            if q6 and q5:
                print(f"{fold:>5d} {seed:>5d} {q6:>8.4f} {q5:>8.4f} {q5-q6:>+8.4f}")
                fold_6[fold].append(q6); fold_5[fold].append(q5)

    print(f"\n{'Fold':>5s} {'6grp_mean':>10s} {'5grp_mean':>10s} {'diff':>8s}")
    print("-"*50)
    all_6, all_5 = [], []
    for fold in sets:
        m6, m5 = np.mean(fold_6[fold]), np.mean(fold_5[fold])
        all_6.extend(fold_6[fold]); all_5.extend(fold_5[fold])
        print(f"{fold:>5d} {m6:>10.4f} {m5:>10.4f} {m5-m6:>+8.4f}")
    print("-"*50)
    print(f"{'ALL':>5s} {np.mean(all_6):>10.4f} {np.mean(all_5):>10.4f} {np.mean(all_5)-np.mean(all_6):>+8.4f}")
    print(f"std: 6grp={np.std(all_6):.4f}  5grp={np.std(all_5):.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
