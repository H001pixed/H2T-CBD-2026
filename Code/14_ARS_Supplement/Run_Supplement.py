"""Paired full-anchoring and direct-ARS experiments on the two independent corpora."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'Code/00_Core'))

import Common as C
import Experiments as E

SCALES = {'persuade_nonoverlap': (1, 6, 1), 'cela_single_prompt': (1, 5, .5)}
SOURCE_FILES = [
    'Code/00_Core/Ars.py', 'Code/00_Core/Common.py', 'Code/00_Core/Engine.py',
    'Code/00_Core/Experiments.py', 'Code/00_Core/Models.py',
    'Code/00_Core/Extract_Embeddings.py', 'Code/01_Features/Features.py',
    'Code/01_Features/Canonical_Data.py', 'Code/14_ARS_Supplement/Run_Supplement.py',
    'Code/14_ARS_Supplement/Extract_Supplement_Embeddings.py',
]


def source_hash():
    return C.fingerprint({name: C.file_hash(PROJECT / name) for name in SOURCE_FILES})


def score_map(name, scores, config):
    low, high, step = SCALES[name]
    return C.ScaleMap(low, high, step).fit(scores)


def setup(raw_root, output):
    C.configure(raw_root)
    C.RUNS = output / 'results/runs'
    C.DATASETS.update({name: bounds[:2] for name, bounds in SCALES.items()})
    C.code_hash = source_hash
    E.scale_map = score_map


def freeze(raw_root, output, config):
    import pandas as pd
    datasets = {}
    for name in SCALES:
        path = raw_root / 'data/01_raw_datasets' / name / 'data.csv'
        frame = pd.read_csv(path, dtype={'id': str, 'essay_set': str})
        prompts = sorted(frame.essay_set.unique().tolist())
        datasets[name] = {'rows': len(frame), 'prompts': prompts,
                          'data_sha256': C.file_hash(path), 'score_range': list(SCALES[name]),
                          'protocol': 'P-LOPO' if name == 'persuade_nonoverlap' else 'P-in',
                          'expected_runs': len(config['seeds']) * 2 * (len(prompts) if name == 'persuade_nonoverlap' else 1)}
    manifest = {
        'frozen_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'config': config, 'code_hash': source_hash(),
        'source_files': {name: C.file_hash(PROJECT / name) for name in SOURCE_FILES},
        'datasets': datasets,
        'comparisons': ['h2tcbd', 'selective'],
        'selection': 'Fit oriented ARS on actual fitting rows only; retain ARS > 0.1',
        'p_lopo': 'Hold out one entire prompt; split source prompts 90/10 for fitting/early stopping',
        'p_in': 'Existing score-stratified 80/10/10 split; fit pooled single-prompt ARS on fitting rows',
        'empty_selection': 'Record no_groups_selected; do not add groups or change theta',
        'score_target': 'Original human holistic scores for both corpora; no trait averaging or rounding',
        'reporting': 'Retain every planned seed, prompt, positive/negative pair and failure',
    }
    path = output / 'configs/frozen_protocol.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if any(previous[key] != manifest[key] for key in ('config', 'code_hash', 'datasets')):
            raise ValueError('Frozen configuration, source code, or data differ from this run.')
        return previous
    C.save_json(path, manifest)
    C.save_json(output / 'configs/Run_Config.json', config)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=list(SCALES), default=list(SCALES))
    parser.add_argument('--stage', choices=['freeze', 'train'], default='train')
    args = parser.parse_args()
    raw_root, output = args.root.resolve(), args.output.resolve()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    setup(raw_root, output)
    manifest = freeze(raw_root, output, config)
    print(json.dumps({'stage': args.stage, 'code_hash': manifest['code_hash'],
                      'expected_runs': sum(d['expected_runs'] for d in manifest['datasets'].values())}), flush=True)
    if args.stage == 'freeze':
        return
    for name in args.datasets:
        run_config = copy.deepcopy(config)
        protocol = manifest['datasets'][name]['protocol']
        E.run_suite(run_config, [name], [protocol], ['h2tcbd', 'selective'],
                    pooled=name == 'cela_single_prompt')


if __name__ == '__main__':
    main()
