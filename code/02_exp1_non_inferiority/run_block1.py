"""Main experiment: BlackBox + H2T-CBD across P-in, P-LOPO, and P-cross.

Each (variant, protocol, split, seed) is a unit experiment with a unique
exp_id. result.json acts as a completion marker; re-running skips completed
units. Results accumulate into results.tsv. This script is the initial entry
point of the whole experiment pipeline.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_core"))
import common as C
from engine import load_emb, train_eval, FeatNorm
from metrics_logger import MetricsLogger

SEEDS = [13, 42, 123]
VARIANTS_MAIN = ["blackbox", "h2tcbd"]


def append_results_tsv(rows):
    p = C.ROOT / "results.tsv"
    header = "exp_id\tgroup_id\tcall_id\tcommit\tmetric\tstatus\tdescription\n"
    if not p.exists():
        p.write_text(header)
    with p.open("a") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def run_one(exp_id, group_id, variant, protocol, tr, va, te, smap_tr, seed,
            smap_te=None, feat_norm=None, K=6, desc=""):
    if C.already_done(exp_id):
        r = json.loads(C.result_path(exp_id).read_text())
        print(f"[skip] {exp_id} qwk={r.get('test_qwk')}")
        return r
    ml = MetricsLogger(C.RUNS / exp_id)
    t0 = time.time()
    metrics, model = train_eval(variant, tr, va, te, smap_tr, K=K, seed=seed,
                                ml=ml, feat_norm=feat_norm, te_scoremap=smap_te)
    payload = {"exp_id": exp_id, "group_id": group_id, "variant": variant,
               "protocol": protocol, "seed": seed, "commit": C.git_sha(),
               "seconds": round(time.time() - t0, 1), "desc": desc, **metrics}
    C.save_result(exp_id, payload)
    ml.done(**{k: metrics[k] for k in ("test_qwk", "test_rmse", "test_pearson")})
    print(f"[done] {exp_id} qwk={metrics['test_qwk']} ({payload['seconds']}s)")
    return payload


def protocol_pin(data_by_name):
    """P-in: in-dataset 80/10/10 stratified split for each dataset."""
    rows, results = [], []
    for name in ["asap2", "dysf", "feat"]:
        d = data_by_name[name]
        smin, smax = C.DATASETS[name]["smin"], C.DATASETS[name]["smax"]
        import pandas as pd
        df = pd.DataFrame({"score": d["score"]})
        for seed in SEEDS:
            itr, iva, ite = C.stratified_split(df, seed=seed)
            smap = C.ScaleMap(smin, smax).fit(d["score"][itr])
            sub = lambda idx: {k: d[k][idx] for k in d if k in ("emb", "score", "feats", "essay_set")}
            tr, va, te = sub(itr), sub(iva), sub(ite)
            fn = FeatNorm().fit(d["feats"][itr]) if name == "feat" else None
            for variant in VARIANTS_MAIN:
                exp = f"B1_Pin_{name}_{variant}_s{seed}"
                r = run_one(exp, f"Pin_{name}_{variant}", variant, "P-in",
                            tr, va, te, smap, seed, feat_norm=fn,
                            desc=f"P-in {name} {variant} seed{seed}")
                results.append(r)
                rows.append([exp, f"Pin_{name}_{variant}", variant, r["commit"],
                             f"QWK={r['test_qwk']}", "done",
                             f"P-in {name} {variant} seed{seed} QWK={r['test_qwk']} RMSE={r['test_rmse']}"])
    append_results_tsv(rows)
    return results


def protocol_lopo(data_by_name):
    """P-LOPO: leave-one-prompt-out on ASAP-1 (feat), 8 folds."""
    rows, results = [], []
    d = data_by_name["feat"]
    smin, smax = C.DATASETS["feat"]["smin"], C.DATASETS["feat"]["smax"]
    sets = np.unique(d["essay_set"])
    for seed in SEEDS:
        for variant in VARIANTS_MAIN:
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
                sub = lambda idx: {"emb": d["emb"][idx], "score": d["score"][idx],
                                   "feats": d["feats"][idx], "essay_set": d["essay_set"][idx]}
                exp = f"B1_LOPO_feat_{variant}_s{seed}_fold{hold}"
                if C.already_done(exp):
                    r = json.loads(C.result_path(exp).read_text())
                else:
                    ml = MetricsLogger(C.RUNS / exp)
                    from engine import train_eval
                    m, model = train_eval(variant, sub(itr), sub(iva), sub(te_idx), smap,
                                          seed=seed, ml=ml, feat_norm=fn)
                    r = {"exp_id": exp, "variant": variant, "fold": int(hold),
                         "commit": C.git_sha(), **m}
                    C.save_result(exp, r)
                fold_qwks.append(r["test_qwk"])
            mean_q = float(np.mean(fold_qwks))
            gexp = f"B1_LOPO_feat_{variant}_s{seed}"
            payload = {"exp_id": gexp, "group_id": f"LOPO_feat_{variant}", "variant": variant,
                       "protocol": "P-LOPO", "seed": seed, "commit": C.git_sha(),
                       "test_qwk": round(mean_q, 4), "fold_qwks": [round(x, 4) for x in fold_qwks],
                       "desc": f"P-LOPO feat {variant} seed{seed} mean over 8 folds"}
            C.save_result(gexp, payload)
            results.append(payload)
            rows.append([gexp, f"LOPO_feat_{variant}", variant, payload["commit"],
                         f"QWK={round(mean_q,4)}", "done",
                         f"P-LOPO feat {variant} seed{seed} meanQWK={round(mean_q,4)}"])
            print(f"[done] {gexp} meanQWK={round(mean_q,4)}")
    append_results_tsv(rows)
    return results


def protocol_cross(data_by_name):
    """P-cross: train on source, zero-shot on target with separate score scaling."""
    rows, results = [], []
    pairs = [("asap2", "dysf"), ("asap2", "feat"),
             ("dysf", "asap2"), ("dysf", "feat")]
    import pandas as pd
    for src, tgt in pairs:
        ds = data_by_name[src]
        dt = data_by_name[tgt]
        smin_s, smax_s = C.DATASETS[src]["smin"], C.DATASETS[src]["smax"]
        smin_t, smax_t = C.DATASETS[tgt]["smin"], C.DATASETS[tgt]["smax"]
        for seed in SEEDS:
            df = pd.DataFrame({"score": ds["score"]})
            itr, iva, _ = C.stratified_split(df, seed=seed)
            smap_s = C.ScaleMap(smin_s, smax_s).fit(ds["score"][itr])
            smap_t = C.ScaleMap(smin_t, smax_t).fit(dt["score"])
            fn = None
            sub_s = lambda idx: {"emb": ds["emb"][idx], "score": ds["score"][idx],
                                 **({"feats": ds["feats"][idx]} if "feats" in ds else {})}
            te = {"emb": dt["emb"], "score": dt["score"],
                  **({"feats": dt["feats"]} if "feats" in dt else {})}
            for variant in VARIANTS_MAIN:
                exp = f"B1_cross_{src}2{tgt}_{variant}_s{seed}"
                r = run_one(exp, f"cross_{src}2{tgt}_{variant}", variant, "P-cross",
                            sub_s(itr), sub_s(iva), te, smap_s, seed,
                            smap_te=smap_t, feat_norm=fn,
                            desc=f"P-cross {src}->{tgt} {variant} seed{seed}")
                results.append(r)
                rows.append([exp, f"cross_{src}2{tgt}_{variant}", variant, r["commit"],
                             f"QWK={r['test_qwk']}", "done",
                             f"P-cross {src}->{tgt} {variant} seed{seed} QWK={r['test_qwk']}"])
    append_results_tsv(rows)
    return results


def main():
    print("loading cached embeddings ...")
    data = {n: load_emb(n) for n in ["asap2", "dysf", "feat"]}

    print("=== P-in ===")
    protocol_pin(data)
    print("=== P-LOPO ===")
    protocol_lopo(data)
    print("=== P-cross ===")
    protocol_cross(data)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
