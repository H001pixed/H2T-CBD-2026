"""FeatAlign: recomputation of the 19 hand-crafted linguistic features.

Design rule: compute_features() is used only for the ASAP-1 feat dataset,
because ASAP-1 is the only dataset that activates anchor loss. ASAP-2.0 and
Feedback do not carry or consume the 19 linguistic features. The feat dataset
ships 19 feature columns; the M0 sanity gate requires per-column correlation
r > 0.95 between recomputed and provided values.

The implementation is pure data processing (numpy/pandas/nltk/textstat/
textblob/pyspellchecker); no modelling or training happens here.

Feature definitions (reverse-engineered from the feat columns):
  char_count             = len(text)
  word_count             = len(text.split())
  sent_count             = len(sent_tokenize(text))
  avg_word_len           = char_count / word_count
  spell_err_count        = number of unknown words (pyspellchecker)
  noun/adj/verb/adv      = POS counts (noun = NN*, adj_count = JJ*,
                           verb_count = VB*, adv = RB*).
  readability_score      = textstat.flesch_reading_ease
  punctuation_score      = #([.,!?;:...]) / word_count
  vocabulary_richness    = type-token ratio
  complex_sentence_ratio = fraction of sentences containing a subordinator
  clause_density         = (commas/semicolons/colons + coordinators +
                            subordinators + sentence count) / sentence count
  semantic_coherence     = mean TF-cosine between adjacent sentences
  sentiment_subjectivity = TextBlob subjectivity
  transitional_phrase_use= #(transition words/phrases) / word_count
  figurative_language_use= #(figurative cues: like/as...as/metaphor) / word_count
  question_usage         = #('?') / sent_count
"""
from __future__ import annotations

import re
import math

import numpy as np

# Lazy loading of heavy NLP dependencies to keep imports cheap for pure statistics.
_NLTK_READY = False
_SPELL = None


def _ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk  # noqa
    # Trigger one load so missing resources fail early (before M0).
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    word_tokenize("warm up.")
    pos_tag(["warm", "up"])
    _NLTK_READY = True


def _spell():
    global _SPELL
    if _SPELL is None:
        from spellchecker import SpellChecker
        _SPELL = SpellChecker()
    return _SPELL


FEATURE_NAMES = [
    "char_count", "word_count", "sent_count", "avg_word_len", "spell_err_count",
    "noun_count", "adj_count", "verb_count", "adv_count", "readability_score",
    "punctuation_score", "vocabulary_richness", "complex_sentence_ratio",
    "clause_density", "semantic_coherence", "sentiment_subjectivity",
    "transitional_phrase_use", "figurative_language_use", "question_usage",
]

_SUBORD = {
    "because", "although", "though", "since", "while", "whereas", "if", "when",
    "after", "before", "unless", "until", "that", "which", "who", "whom",
    "whose", "where", "as",
}
_COORD = {"and", "but", "or", "nor", "yet", "so", "for"}
_TRANSITIONS = [
    "however", "therefore", "moreover", "furthermore", "consequently", "thus",
    "hence", "nevertheless", "meanwhile", "additionally", "firstly", "secondly",
    "finally", "in conclusion", "for example", "for instance", "in addition",
    "on the other hand", "as a result", "in contrast", "in fact", "to sum up",
    "in summary", "first of all",
]
_FIGURATIVE = [
    " like ", " as if ", " as though ", "metaphor", " symboliz", " resembl",
    " just as ", " similar to ",
]
_PUNCT_RE = re.compile(r"[.,!?;:\"'`()\-]")
_WORD_RE = re.compile(r"[a-zA-Z]+")


def _tf_vector(tokens):
    d = {}
    for t in tokens:
        d[t] = d.get(t, 0) + 1
    return d


def _cosine(a, b):
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def compute_features(text: str) -> dict:
    """Compute the 19 features for one text; keys follow FEATURE_NAMES."""
    _ensure_nltk()
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk import pos_tag
    from textblob import TextBlob

    text = "" if text is None else str(text)
    char_count = len(text)
    sp_words = text.split()
    word_count = len(sp_words)
    sents = sent_tokenize(text) if text.strip() else []
    sent_count = max(len(sents), 0)

    safe_wc = word_count if word_count > 0 else 1
    safe_sc = sent_count if sent_count > 0 else 1

    avg_word_len = char_count / safe_wc

    # POS counts (each feature name matches its Penn Treebank tag).
    toks = word_tokenize(text) if text.strip() else []
    tags = pos_tag(toks) if toks else []
    noun = adjJJ = verbVB = adv = 0
    for _, tg in tags:
        if tg.startswith("NN"):
            noun += 1
        elif tg.startswith("JJ"):
            adjJJ += 1
        elif tg.startswith("VB"):
            verbVB += 1
        elif tg.startswith("RB"):
            adv += 1
    noun_count, adj_count, verb_count, adv_count = noun, adjJJ, verbVB, adv

    # spell error
    lower_words = _WORD_RE.findall(text.lower())
    spell_err_count = len(_spell().unknown(lower_words)) if lower_words else 0

    # readability
    import textstat
    try:
        readability_score = textstat.flesch_reading_ease(text) if text.strip() else 0.0
    except Exception:
        readability_score = 0.0

    punctuation_score = len(_PUNCT_RE.findall(text)) / safe_wc

    low = [w for w in lower_words]
    vocabulary_richness = (len(set(low)) / len(low)) if low else 0.0

    # Complex-sentence ratio: share of sentences containing a subordinator.
    if sents:
        complex_cnt = sum(
            1 for s in sents if any(w in _SUBORD for w in re.findall(r"[a-zA-Z]+", s.lower()))
        )
        complex_sentence_ratio = complex_cnt / len(sents)
    else:
        complex_sentence_ratio = 0.0

    # Clause density: clause markers per sentence.
    n_clause_markers = (
        len(re.findall(r"[,;:]", text))
        + sum(1 for w in lower_words if w in _SUBORD or w in _COORD)
        + sent_count
    )
    clause_density = n_clause_markers / safe_sc

    # Semantic coherence: mean TF-cosine between adjacent sentences.
    if len(sents) >= 2:
        vecs = [_tf_vector(re.findall(r"[a-zA-Z]+", s.lower())) for s in sents]
        sims = [_cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
        semantic_coherence = float(np.mean(sims)) if sims else 0.0
    else:
        semantic_coherence = 0.0

    try:
        sentiment_subjectivity = TextBlob(text).sentiment.subjectivity if text.strip() else 0.0
    except Exception:
        sentiment_subjectivity = 0.0

    tl = " " + text.lower() + " "
    transitional_phrase_use = sum(tl.count(p) for p in _TRANSITIONS) / safe_wc
    figurative_language_use = sum(tl.count(p) for p in _FIGURATIVE) / safe_wc
    question_usage = text.count("?") / safe_sc

    return {
        "char_count": float(char_count),
        "word_count": float(word_count),
        "sent_count": float(sent_count),
        "avg_word_len": float(avg_word_len),
        "spell_err_count": float(spell_err_count),
        "noun_count": float(noun_count),
        "adj_count": float(adj_count),
        "verb_count": float(verb_count),
        "adv_count": float(adv_count),
        "readability_score": float(readability_score),
        "punctuation_score": float(punctuation_score),
        "vocabulary_richness": float(vocabulary_richness),
        "complex_sentence_ratio": float(complex_sentence_ratio),
        "clause_density": float(clause_density),
        "semantic_coherence": float(semantic_coherence),
        "sentiment_subjectivity": float(sentiment_subjectivity),
        "transitional_phrase_use": float(transitional_phrase_use),
        "figurative_language_use": float(figurative_language_use),
        "question_usage": float(question_usage),
    }


def compute_features_batch(texts, n_jobs: int = 1, cache_path: str | None = None):
    """Compute features for a batch of texts -> (N, 19) float32 array.

    A disk cache is used because recomputation is slow; if cache_path exists
    and matches the input size it is loaded directly.
    """
    import os
    if cache_path and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts) and arr.shape[1] == len(FEATURE_NAMES):
            return arr.astype(np.float32)
    rows = []
    for i, t in enumerate(texts):
        f = compute_features(t)
        rows.append([f[k] for k in FEATURE_NAMES])
        if (i + 1) % 2000 == 0:
            print(f"  features {i + 1}/{len(texts)}", flush=True)
    arr = np.asarray(rows, dtype=np.float32)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, arr)
    return arr


if __name__ == "__main__":
    # Quick self test.
    s = "Dear newspaper. I think computers help us. However, they can be bad! Why? Because we waste time."
    f = compute_features(s)
    for k in FEATURE_NAMES:
        print(f"{k:24s} {f[k]:.4f}")
