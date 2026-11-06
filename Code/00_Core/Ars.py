import numpy as np
from scipy.stats import spearmanr

import Common as C


def compute_ars(features, scores, prompts, mode='oriented', theta=.1, pooled=False):
    """Select groups using only the supplied source-training rows.

    Oriented mode uses each feature's mean finite signed prompt correlation
    to choose one direction for all prompts (zero or missing mean keeps +1).
    Group means and the across-prompt mean-minus-SD rule then stay unchanged.
    """
    if mode not in ('signed', 'absolute', 'oriented'):
        raise ValueError(f'Unknown ARS direction rule: {mode}')
    prompts = np.zeros(len(scores), dtype=int) if pooled else np.asarray(prompts)
    names = np.unique(prompts)
    correlations = np.full((len(names), len(C.FEAT_COLS)), np.nan)
    for p, name in enumerate(names):
        rows = prompts == name
        y = scores[rows]
        for j in range(features.shape[1]):
            x = features[rows, j]
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 10 and np.std(x[valid]) > 0 and np.std(y[valid]) > 0:
                correlations[p, j] = spearmanr(x[valid], y[valid]).statistic
    values = correlations.copy()
    direction = np.ones(len(C.FEAT_COLS))
    if mode == 'absolute':
        values = np.abs(values)
    elif mode == 'oriented':
        for j in range(values.shape[1]):
            observed = values[np.isfinite(values[:, j]), j]
            if len(observed) and observed.mean() < 0:
                direction[j] = -1
        values *= direction
    kept, records = [], []
    for k, dims in enumerate(C.ANCHOR_DIMS):
        by_prompt = []
        for row in values[:, dims]:
            valid = row[np.isfinite(row)]
            by_prompt.append(float(valid.mean()) if len(valid) else None)
        complete = all(v is not None for v in by_prompt)
        mean = float(np.mean(by_prompt)) if complete else None
        std = float(np.std(by_prompt, ddof=0)) if complete else None
        ars = mean - std if complete else None
        keep = ars is not None and ars > theta
        if keep:
            kept.append(k)
        records.append({'group': C.FACTOR_NAMES[k], 'group_index': k, 'mean': mean,
                        'std': std, 'ars': ars, 'keep': keep,
                        'per_prompt': dict(zip(map(str, names), by_prompt))})
    raw = [[float(v) if np.isfinite(v) else None for v in row] for row in correlations]
    return kept, {'mode': mode, 'theta': theta, 'pooled': pooled, 'groups': records,
                  'prompt_names': list(map(str, names)), 'feature_names': C.FEAT_COLS,
                  'signed_correlations': raw, 'feature_direction': direction.tolist()}
