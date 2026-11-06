"""Shared canonical CSV contract; no NLP imports or feature computation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("feat", "asap2", "dysf")
REQUIRED = ["id", "text", "score", "essay_set", "partition", "text_hash"]
ORIGINAL19 = [
    "char_count", "word_count", "sent_count", "avg_word_len", "spell_err_count",
    "noun_count", "adj_count", "verb_count", "adv_count", "readability_score",
    "punctuation_score", "vocabulary_richness", "complex_sentence_ratio",
    "clause_density", "semantic_coherence", "sentiment_subjectivity",
    "transitional_phrase_use", "figurative_language_use", "question_usage",
]


def text_hash(text: str) -> str:
    """Hash whitespace-normalized text without changing the stored essay."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def dataset_names(names: list[str]) -> list[str]:
    if names == ["all3"]:
        return list(DATASETS)
    if not names or any(name not in DATASETS for name in names):
        raise ValueError("datasets must be feat/asap2/dysf, or all3 on its own")
    if len(set(names)) != len(names):
        raise ValueError("duplicate dataset selection")
    return list(names)


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate without dropping, sorting, renumbering, or deduplicating rows."""
    missing = [column for column in REQUIRED if column not in frame]
    if missing:
        raise ValueError(f"missing canonical columns: {missing}")
    frame = frame.copy()
    for name in ("id", "text", "essay_set", "partition", "text_hash"):
        if frame[name].isna().any():
            raise ValueError(f"missing canonical {name}")
        frame[name] = frame[name].astype(str)
    if frame["id"].str.strip().eq("").any():
        raise ValueError("empty canonical ID")
    if frame["id"].duplicated().any():
        duplicates = frame.loc[frame["id"].duplicated(keep=False), "id"].tolist()
        raise ValueError(f"duplicate canonical IDs: {duplicates[:10]}")
    if frame["essay_set"].str.strip().eq("").any():
        raise ValueError("empty essay_set/prompt label")
    if not frame["partition"].isin(["train", "test"]).all():
        raise ValueError("partition must be train or test")
    score = pd.to_numeric(frame["score"].replace("", np.nan), errors="raise").astype(float)
    if not np.isfinite(score.to_numpy()).all():
        raise ValueError("scored CSV contains missing/infinite labels; exclude unlabelled source rows")
    frame["score"] = score
    expected_hash = frame["text"].map(text_hash)
    if not frame["text_hash"].eq(expected_hash).all():
        raise ValueError("text_hash does not match the stored text")
    present = [name for name in ORIGINAL19 if name in frame]
    if present and len(present) != len(ORIGINAL19):
        raise ValueError("canonical data must contain either zero or all 19 features")
    for name in present:
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
        if not np.isfinite(frame[name].to_numpy()).all():
            raise ValueError(f"non-finite feature: {name}")
    extras = [column for column in frame if column not in REQUIRED + ORIGINAL19]
    return frame[REQUIRED + present + extras]


def read_canonical(path: str | Path) -> pd.DataFrame:
    # Preserve lexical IDs: 001, NA and 1e3 must not be inferred as numbers/NA.
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return canonicalize(frame)
