"""Sanity checks: QWK correctness, ScaleMap round-trip, data pipeline connectivity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import Common as C


def test_qwk():
    rng = np.random.default_rng(0)
    # Perfect agreement -> 1.0
    y = rng.integers(0, 6, size=500)
    assert abs(C.qwk(y, y, 0, 5) - 1.0) < 1e-9, "perfect agreement should be 1.0"
    # Random predictions -> ~0
    yp = rng.integers(0, 6, size=500)
    k_rand = C.qwk(y, yp, 0, 5)
    assert abs(k_rand) < 0.15, f"random qwk should be ~0, got {k_rand}"
    # Verify against a hand-computed 3x3 case
    yt = np.array([0, 0, 1, 1, 2, 2])
    yp2 = np.array([0, 1, 1, 2, 2, 2])
    k = C.qwk(yt, yp2, 0, 2)
    n = 3
    O = np.zeros((n, n))
    for a, b in zip(yt, yp2):
        O[a, b] += 1
    w = np.array([[(i - j) ** 2 / 4 for j in range(n)] for i in range(n)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    k_ref = 1 - (w * O).sum() / (w * E).sum()
    assert abs(k - k_ref) < 1e-9, (k, k_ref)
    return {"qwk_perfect": 1.0, "qwk_random": round(k_rand, 4), "qwk_known": round(k, 4)}


def test_scalemap():
    out = {}
    for name in C.DATASETS:
        df = C.load_dataset(name)
        sm = C.ScaleMap(C.DATASETS[name]["smin"], C.DATASETS[name]["smax"]).fit(df["score"].values)
        u = sm.to_unit(df["score"].values)
        back = sm.from_unit(u)
        recover = float((back == df["score"].values).mean())
        assert recover > 0.999, f"{name} roundtrip {recover}"
        out[name] = {
            "n": len(df), "smin": sm.smin, "smax": sm.smax,
            "unit_min": round(float(u.min()), 3), "unit_max": round(float(u.max()), 3),
            "roundtrip_acc": round(recover, 4),
        }
    return out


def test_features(n_sample=200, seed=0):
    """Recompute 19 features on a sample and compare them with data.csv."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_features"))
    from Features import FEATURE_NAMES, compute_features_batch

    df = C.load_dataset("feat")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    texts = df["text"].iloc[idx].tolist()
    feats = compute_features_batch(texts, n_jobs=1)
    corrs = {}
    for j, c in enumerate(FEATURE_NAMES):
        a = feats[:, j]
        b = df[c].values[idx]
        if a.std() == 0 or b.std() == 0:
            corrs[c] = 1.0 if (a.std() == 0 and b.std() == 0 and np.allclose(a, b)) else 0.0
        else:
            corrs[c] = float(np.corrcoef(a, b)[0, 1])
    low = {c: v for c, v in corrs.items() if v < 0.95}
    assert not low, f"feature reproducibility failed: {low}"
    return {
        "n_sample": int(len(idx)),
        "min_corr": round(min(corrs.values()), 4),
        "failed": list(low),
    }


def main():
    res = {"qwk": test_qwk(), "scalemap": test_scalemap(), "features": test_features()}
    C.save_result("M0_sanity", {"status": "pass", **res})
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("M0 SANITY PASSED")


if __name__ == "__main__":
    sys.exit(main())
