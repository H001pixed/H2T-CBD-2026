import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / '01_raw_datasets'
RUNS = ROOT / 'results' / 'runs'
EMB = ROOT / 'results' / 'embeddings'
FEAT_COLS = [
    'char_count', 'word_count', 'sent_count', 'avg_word_len', 'spell_err_count',
    'noun_count', 'adj_count', 'verb_count', 'adv_count', 'readability_score',
    'punctuation_score', 'vocabulary_richness', 'complex_sentence_ratio',
    'clause_density', 'semantic_coherence', 'sentiment_subjectivity',
    'transitional_phrase_use', 'figurative_language_use', 'question_usage',
]
ANCHOR_GROUPS = {
    'length_fluency': ['char_count', 'word_count', 'sent_count'],
    'lexical_sophistication': ['avg_word_len', 'vocabulary_richness', 'adj_count', 'adv_count'],
    'syntactic_complexity': ['complex_sentence_ratio', 'clause_density', 'verb_count', 'noun_count'],
    'mechanics': ['spell_err_count', 'punctuation_score'],
    'coherence_readability': ['readability_score', 'semantic_coherence'],
    'rhetoric_engagement': ['sentiment_subjectivity', 'transitional_phrase_use',
                            'figurative_language_use', 'question_usage'],
}
FACTOR_NAMES = list(ANCHOR_GROUPS)
ANCHOR_DIMS = [[FEAT_COLS.index(f) for f in fs] for fs in ANCHOR_GROUPS.values()]
DATASETS = {'feat': (0, 10), 'asap2': (1, 6), 'dysf': (1, 5)}


def configure(root):
    global ROOT, DATA, RUNS, EMB
    ROOT = Path(root).resolve()
    DATA = ROOT / 'data' / '01_raw_datasets'
    RUNS = ROOT / 'results' / 'runs'
    EMB = ROOT / 'results' / 'embeddings'


def read_config(path=None):
    path = Path(path) if path is not None else Path('configs/Run_Config.json')
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def code_hash():
    code_root = Path(__file__).resolve().parents[2]
    files = sorted((code_root / 'Code').rglob('*.py'))
    return fingerprint({str(p.relative_to(code_root)): file_hash(p) for p in files})


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(name):
    return pd.read_csv(DATA / name / 'data.csv', dtype={'id': str, 'essay_set': str, 'text_hash': str},
                       keep_default_na=False)


def score_step(name, config):
    return .5 if name == 'dysf' and config['ellipse_target'] == 'overall' else 1.0


class ScaleMap:
    def __init__(self, smin, smax, step=1):
        self.smin, self.smax, self.step = float(smin), float(smax), float(step)
        self.levels = np.linspace(smin, smax, int(round((smax - smin) / step)) + 1)
        self.mids = None

    def fit(self, scores):
        scores = np.asarray(scores)
        counts = np.array([np.isclose(scores, level).sum() for level in self.levels])
        if counts.sum() != len(scores) or not len(scores):
            raise ValueError('Scores do not match the declared rating scale.')
        self.mids = (np.cumsum(counts) - counts / 2) / len(scores)
        return self

    def uniform(self):
        self.mids = np.linspace(0, 1, len(self.levels))
        return self

    def to_unit(self, scores):
        index = np.rint((np.asarray(scores) - self.smin) / self.step).astype(int)
        return self.mids[index].astype(np.float32)

    def from_unit(self, values):
        values = np.clip(np.asarray(values), 0, 1)
        return self.levels[np.abs(values[:, None] - self.mids[None, :]).argmin(axis=1)]

    def to_dict(self):
        return {'smin': self.smin, 'smax': self.smax, 'step': self.step, 'mids': self.mids.tolist()}

    @classmethod
    def from_dict(cls, value):
        obj = cls(value['smin'], value['smax'], value['step'])
        obj.mids = np.asarray(value['mids'])
        return obj


def qwk(y_true, y_pred, min_rating=None, max_rating=None, step=1):
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    lo = min_rating if min_rating is not None else min(true.min(), pred.min())
    hi = max_rating if max_rating is not None else max(true.max(), pred.max())
    n = int(round((hi - lo) / step)) + 1
    a = np.rint((true - lo) / step).astype(int)
    b = np.rint((pred - lo) / step).astype(int)
    observed = np.zeros((n, n))
    np.add.at(observed, (a, b), 1)
    expected = np.outer(observed.sum(1), observed.sum(0)) / len(true)
    weights = (np.arange(n)[:, None] - np.arange(n)[None, :]) ** 2
    denominator = (weights * expected).sum()
    return float(1 - (weights * observed).sum() / denominator) if denominator else float('nan')


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pearson(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return None
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def stratified_split(scores, seed, fracs=(.8, .1, .1)):
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for value in np.unique(scores):
        idx = np.flatnonzero(scores == value)
        rng.shuffle(idx)
        a, b = int(round(len(idx) * fracs[0])), int(round(len(idx) * fracs[1]))
        for output, selected in zip(parts, (idx[:a], idx[a:a + b], idx[a + b:])):
            output.extend(selected.tolist())
    return tuple(np.asarray(p, dtype=int) for p in parts)
