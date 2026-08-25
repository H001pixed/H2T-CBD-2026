"""Data loading, score scaling, QWK, and anchor-group definitions (H2T-CBD).

This module is the shared foundation for all experiments. Paths are relative to
the packaged project root so the code runs from any machine.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Project root: this file sits two levels below the packaged root.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "01_raw_datasets"
RUNS = ROOT / "results" / "runs"
EMB = RUNS / "_emb"

# Score ranges for each dataset (min/max inclusive).
DATASETS = {
    "asap2": {"path": DATA / "asap2/train.csv", "text": "full_text", "score": "score",
              "smin": 1, "smax": 6},
    "dysf": {"path": DATA / "dysf/train.csv", "text": "full_text", "score": None,
             "smin": 1, "smax": 5},
    "feat": {"path": DATA / "feat/data.csv", "text": "clean_essay", "score": "final_score",
             "smin": 0, "smax": 10},
}

# Official Feedback Prize - English Language Learning analytic columns. The
# Feedback target used in this work is their mean rounded to an integer 1-5.
ELL_ANALYTIC_COLS = [
    "cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions",
]

# 19 linguistically motivated features computed per essay.
FEAT_COLS = [
    "char_count", "word_count", "sent_count", "avg_word_len", "spell_err_count",
    "noun_count", "adj_count", "verb_count", "adv_count", "readability_score",
    "punctuation_score", "vocabulary_richness", "complex_sentence_ratio",
    "clause_density", "semantic_coherence", "sentiment_subjectivity",
    "transitional_phrase_use", "figurative_language_use", "question_usage",
]

# 6 anchor groups: a disjoint partition of FEAT_COLS by linguistic category.
ANCHOR_GROUPS = {
    "length_fluency": ["char_count", "word_count", "sent_count"],
    "lexical_sophistication": ["avg_word_len", "vocabulary_richness", "adj_count", "adv_count"],
    "syntactic_complexity": ["complex_sentence_ratio", "clause_density", "verb_count", "noun_count"],
    "mechanics": ["spell_err_count", "punctuation_score"],
    "coherence_readability": ["readability_score", "semantic_coherence"],
    "rhetoric_engagement": ["sentiment_subjectivity", "transitional_phrase_use",
                            "figurative_language_use", "question_usage"],
}
FACTOR_NAMES = list(ANCHOR_GROUPS.keys())
assert len(FACTOR_NAMES) == 6
_flat = [f for g in ANCHOR_GROUPS.values() for f in g]
assert len(_flat) == len(set(_flat)) == len(FEAT_COLS), (len(_flat), len(set(_flat)))


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(name: str) -> pd.DataFrame:
    """Load one dataset, normalising columns to text/score.

    The feat dataset additionally loads essay_set and the 19 feature columns.
    The dysf dataset builds its single 1-5 target from the mean of the six
    official analytic scores. Rows with a missing score are dropped.
    """
    meta = DATASETS[name]
    df = pd.read_csv(meta["path"])
    if name == "dysf":
        analytic = df[ELL_ANALYTIC_COLS].apply(pd.to_numeric, errors="coerce")
        out = pd.DataFrame({
            "text": df[meta["text"]].fillna("").astype(str),
            "score": analytic.mean(axis=1),
        })
    else:
        out = pd.DataFrame({
            "text": df[meta["text"]].fillna("").astype(str),
            "score": pd.to_numeric(df[meta["score"]], errors="coerce"),
        })
    if name == "feat":
        out["essay_set"] = df["essay_set"].astype(int)
        for c in FEAT_COLS:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    out = out.dropna(subset=["score"]).reset_index(drop=True)
    out["score"] = out["score"].round().astype(int)
    return out


class ScaleMap:
    """Monotone score mapping via empirical CDF midpoints.

    Each integer score level maps to the midpoint of its training-set CDF bin
    in [0, 1]; the inverse mapping returns the nearest integer level.
    """

    def __init__(self, smin: int, smax: int):
        self.smin = int(smin)
        self.smax = int(smax)
        self.levels = list(range(self.smin, self.smax + 1))
        self.cdf = None  # int level -> midpoint in [0,1]

    def fit(self, scores: np.ndarray) -> "ScaleMap":
        scores = np.asarray(scores).astype(int)
        n = len(scores)
        counts = np.array([(scores == lv).sum() for lv in self.levels], dtype=float)
        cum = np.cumsum(counts)
        lower = (cum - counts) / n
        upper = cum / n
        mid = (lower + upper) / 2.0
        self.cdf = {lv: float(m) for lv, m in zip(self.levels, mid)}
        return self

    def to_unit(self, scores: np.ndarray) -> np.ndarray:
        return np.array([self.cdf[int(s)] for s in scores], dtype=np.float32)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """Map [0,1] back to the nearest integer score level."""
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        lv = np.array(self.levels)
        mids = np.array([self.cdf[l] for l in lv])
        idx = np.abs(u[:, None] - mids[None, :]).argmin(axis=1)
        return lv[idx]

    def to_dict(self):
        return {"smin": self.smin, "smax": self.smax, "cdf": self.cdf}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["smin"], d["smax"])
        m.levels = list(range(m.smin, m.smax + 1))
        m.cdf = {int(k): float(v) for k, v in d["cdf"].items()}
        return m


def qwk(y_true, y_pred, min_rating=None, max_rating=None) -> float:
    """Quadratic weighted kappa on integer score scales (pure NumPy)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if min_rating is None:
        min_rating = int(min(y_true.min(), y_pred.min()))
    if max_rating is None:
        max_rating = int(max(y_true.max(), y_pred.max()))
    n = max_rating - min_rating + 1
    if n <= 1:
        return 0.0
    O = np.zeros((n, n), dtype=float)
    for t, p in zip(y_true, y_pred):
        O[t - min_rating, p - min_rating] += 1
    w = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            w[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist) / O.sum()
    num = (w * O).sum()
    den = (w * E).sum()
    if den == 0:
        return 1.0
    return float(1.0 - num / den)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def pearson(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if y_true.std() == 0 or y_pred.std() == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def stratified_split(df: pd.DataFrame, fracs=(0.8, 0.1, 0.1), seed=42):
    """Stratified 80/10/10 split by score level."""
    rng = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for _, g in df.groupby("score"):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * fracs[0]))
        n_va = int(round(n * fracs[1]))
        idx_tr += idx[:n_tr].tolist()
        idx_va += idx[n_tr:n_tr + n_va].tolist()
        idx_te += idx[n_tr + n_va:].tolist()
    return np.array(idx_tr), np.array(idx_va), np.array(idx_te)


def result_path(exp_id: str) -> Path:
    return RUNS / exp_id / "result.json"


def already_done(exp_id: str) -> bool:
    """True if result.json exists and parses as valid JSON (completion marker)."""
    p = result_path(exp_id)
    if not p.exists():
        return False
    try:
        json.loads(p.read_text())
        return True
    except Exception:
        return False


def save_result(exp_id: str, payload: dict) -> None:
    d = RUNS / exp_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def git_sha() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return "nogit"
