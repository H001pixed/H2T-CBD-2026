import argparse
import copy
import json
import re
import time
from pathlib import Path

import numpy as np

import Common as C
from Ars import compute_ars
from Engine import load_emb, subset, train_eval


def train_indices(data):
    return np.flatnonzero(data['partition'].astype(str) == 'train')


def source_split(indices, seed):
    indices = indices.copy()
    np.random.default_rng(seed).shuffle(indices)
    count = max(1, int(.1 * len(indices)))
    return indices[count:], indices[:count]


def split_indices(data, protocol, seed, hold=None):
    available = train_indices(data)
    prompts = data['essay_set'].astype(str)
    if protocol == 'P-in-prompt':
        available = available[prompts[available] == str(hold)]
    if protocol in ('P-in', 'P-in-prompt'):
        parts = C.stratified_split(data['score'][available], seed)
        return tuple(available[p] for p in parts)
    if protocol in ('P-LOPO', 'P-LOPO-official'):
        source = available[prompts[available] != str(hold)]
        tr, va = source_split(source, seed)
        test_partition = 'test' if protocol.endswith('official') else 'train'
        te = np.flatnonzero((prompts == str(hold)) & (data['partition'].astype(str) == test_partition))
        return tr, va, te
    if protocol == 'P-official':
        tr, va = source_split(available, seed)
        return tr, va, np.flatnonzero(data['partition'].astype(str) == 'test')
    raise ValueError(f'Unknown protocol: {protocol}')


def clean_overlap(tr, va, te):
    test_hashes = set(te['text_hash'])
    va_keep = np.array([x not in test_hashes for x in va['text_hash']], dtype=bool)
    va_removed = va['id'][~va_keep].tolist()
    va = subset(va, np.flatnonzero(va_keep))
    held_hashes = test_hashes | set(va['text_hash'])
    tr_keep = np.array([x not in held_hashes for x in tr['text_hash']], dtype=bool)
    tr_removed = tr['id'][~tr_keep].tolist()
    return subset(tr, np.flatnonzero(tr_keep)), va, te, {'train': tr_removed, 'validation': va_removed}


def scale_map(name, scores, config):
    return C.ScaleMap(*C.DATASETS[name], step=C.score_step(name, config)).fit(scores)


def run_one(name, variant, protocol, seed, hold, config, data, source_hash,
            source_name=None, source_data=None, fixed_groups=None, pooled=False, overwrite=False):
    source_name = source_name or name
    if source_data is None:
        it, iv, ie = split_indices(data, protocol, seed, hold)
        tr, va, te = (subset(data, p) for p in (it, iv, ie))
        target_calibration = None
    else:
        it, iv = source_split(train_indices(source_data), seed)
        tr, va = subset(source_data, it), subset(source_data, iv)
        calibration, _, ie = split_indices(data, 'P-in', seed)
        target_calibration = subset(data, calibration)
        te = subset(data, ie)
        test_hashes = set(te['text_hash'])
        calibration_keep = np.array([h not in test_hashes for h in target_calibration['text_hash']], dtype=bool)
        target_calibration = subset(target_calibration, np.flatnonzero(calibration_keep))
    tr, va, te, removed = clean_overlap(tr, va, te)
    tag = protocol.replace('P-', '').replace('-', '_')
    group_tag = '' if fixed_groups is None else '_groups' + ''.join(map(str, fixed_groups))
    fold_tag = '' if hold is None else '_fold' + str(hold)
    if hold is not None and not re.fullmatch(r'[A-Za-z0-9_-]+', str(hold)):
        fold_tag = '_fold' + C.fingerprint(str(hold))[:10]
    exp_id = (f'B1_{tag}_{source_name + "2" if source_data is not None else ""}{name}_{variant}'
              f'_s{seed}{fold_tag}_{config["ars_mode"]}_t{config["theta"]:g}'
              f'_{config["anchor_reduction"]}{group_tag}' + ('_pooled' if pooled else ''))
    if name == 'dysf' or source_name == 'dysf':
        exp_id += '_' + config['ellipse_target']
    folder = C.RUNS / exp_id
    spec = {'dataset': name, 'source_dataset': source_name, 'variant': variant, 'protocol': protocol,
            'seed': seed, 'fold': None if hold is None else str(hold), 'config': config,
            'data_hash': data['_fingerprint'], 'source_data_hash': (source_data or data)['_fingerprint'],
            'code_hash': source_hash, 'train_ids': tr['id'].tolist(), 'val_ids': va['id'].tolist(),
            'test_ids': te['id'].tolist(), 'fixed_groups': fixed_groups, 'pooled': pooled}
    spec['calibration_ids'] = [] if target_calibration is None else target_calibration['id'].tolist()
    signature = C.fingerprint(spec)
    result_path = folder / 'result.json'
    if result_path.exists() and not overwrite:
        previous = json.loads(result_path.read_text(encoding='utf-8'))
        if previous.get('run_signature') != signature:
            raise ValueError(f'{exp_id}: configuration/data/code changed; choose a new output root or --overwrite.')
        if previous.get('status') == 'succeeded':
            for required in ('model.pt', 'predictions.npz', 'splits.npz', 'history.json'):
                if not (folder / required).exists():
                    raise FileNotFoundError(folder / required)
        return previous
    if result_path.exists():
        result_path.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    C.save_json(folder / 'run_spec.json', spec)
    C.save_json(folder / 'excluded_duplicates.json', removed)
    np.savez_compressed(folder / 'splits.npz', train_ids=tr['id'], val_ids=va['id'], test_ids=te['id'],
                        calibration_ids=np.asarray(spec['calibration_ids'], dtype=str))
    base = {'exp_id': exp_id, 'dataset': name, 'source_dataset': source_name, 'variant': variant,
            'protocol': protocol, 'seed': seed, 'fold': spec['fold'], 'ars_mode': config['ars_mode'],
            'theta': config['theta'], 'anchor_reduction': config['anchor_reduction'],
            'fixed_groups': fixed_groups, 'pooled': pooled, 'run_signature': signature,
            'ellipse_target': config['ellipse_target'], 'test_qwk': None, 'test_rmse': None}
    if min(len(tr['id']), len(va['id']), len(te['id'])) == 0:
        result = {**base, 'status': 'empty_split', 'kept_groups': []}
        C.save_json(result_path, result)
        return result
    kept, ars = compute_ars(tr['feats'], tr['score'], tr['essay_set'], config['ars_mode'], config['theta'], pooled)
    C.save_json(folder / 'ars.json', ars)
    if variant == 'h2tcbd':
        kept = list(range(6))
    elif variant in ('no_anchor', 'blackbox', 'featconcat'):
        kept = []
    elif fixed_groups is not None:
        kept = list(fixed_groups)
    if variant in ('selective', 'soft_selective') and not kept:
        result = {**base, 'status': 'no_groups_selected', 'kept_groups': []}
        C.save_json(result_path, result)
        return result
    source_map = scale_map(source_name, tr['score'], config)
    target_map = None
    if source_data is not None:
        if config['cross_mapping'] == 'calibrated':
            target_map = scale_map(name, target_calibration['score'], config)
        else:
            target_map = C.ScaleMap(*C.DATASETS[name], step=C.score_step(name, config)).uniform()
    started = time.monotonic()
    metrics = train_eval(variant, tr, va, te, source_map, config, seed, kept, folder, target_map)
    result = {**base, 'status': 'succeeded', 'kept_groups': kept,
              'seconds': time.monotonic() - started, **metrics}
    C.save_json(result_path, result)
    print(f'{exp_id}: QWK={result["test_qwk"]}', flush=True)
    return result


def run_suite(config, datasets, protocols, variants, folds=None, fixed_groups=None,
              pooled=False, overwrite=False):
    source_hash = C.code_hash()
    data = {name: load_emb(name, config) for name in datasets}
    records = []
    for name, values in data.items():
        for protocol in protocols:
            if protocol.startswith('P-cross'):
                continue
            selected_folds = [None]
            if protocol in ('P-LOPO', 'P-LOPO-official', 'P-in-prompt'):
                selected_folds = sorted(np.unique(values['essay_set'][train_indices(values)].astype(str)))
                if folds:
                    selected_folds = [p for p in selected_folds if p in folds]
            for seed in config['seeds']:
                for hold in selected_folds:
                    for variant in variants:
                        records.append(run_one(name, variant, protocol, seed, hold, config, values,
                                               source_hash, fixed_groups=fixed_groups,
                                               pooled=pooled, overwrite=overwrite))
    if any(p.startswith('P-cross') for p in protocols):
        protocol = 'P-cross-' + config['cross_mapping']
        for source, target in [('asap2', 'dysf'), ('asap2', 'feat'), ('dysf', 'asap2'), ('dysf', 'feat')]:
            if source not in data or target not in data:
                continue
            for seed in config['seeds']:
                for variant in variants:
                    records.append(run_one(target, variant, protocol, seed, None, config, data[target],
                                           source_hash, source, data[source], fixed_groups, pooled, overwrite))
    return records


PRESETS = {
    'block1': (['P-in', 'P-LOPO', 'P-cross'], ['blackbox', 'h2tcbd']),
    'ablation_pin': (['P-in'], ['blackbox', 'h2tcbd', 'no_anchor', 'featconcat']),
    'ablation_lopo': (['P-LOPO'], ['blackbox', 'h2tcbd', 'no_anchor']),
    'prompt_pin': (['P-in-prompt'], ['blackbox', 'h2tcbd']),
    'selective': (['P-LOPO'], ['h2tcbd', 'selective']),
    'cross_selective': (['P-LOPO'], ['h2tcbd', 'selective']),
    'theta': (['P-LOPO'], ['h2tcbd', 'selective']),
    'legacy_selective': (['P-LOPO'], ['soft_selective']),
    'legacy_cross': (['P-in'], ['soft_selective']),
}


def run_cli(preset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=C.ROOT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=list(C.DATASETS))
    parser.add_argument('--protocols', nargs='+')
    parser.add_argument('--variants', nargs='+', choices=['blackbox', 'h2tcbd', 'no_anchor', 'featconcat', 'selective', 'soft_selective'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--folds', nargs='+')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--ars-mode', choices=['signed', 'absolute', 'oriented'])
    parser.add_argument('--theta', type=float)
    parser.add_argument('--fixed-groups', nargs='+', type=int, choices=range(6))
    parser.add_argument('--pooled', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    C.configure(args.root)
    config = C.read_config(args.config)
    for name in ('seeds', 'epochs', 'device', 'ars_mode', 'theta'):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.fixed_groups and len(set(args.fixed_groups)) != len(args.fixed_groups):
        parser.error('--fixed-groups contains duplicates')
    protocols, variants = PRESETS[preset]
    default_datasets = ['asap2', 'dysf'] if preset in ('cross_selective', 'legacy_cross') else config['datasets']
    datasets = args.datasets or default_datasets
    if preset.startswith('legacy'):
        config['anchor_reduction'] = 'active_groups'
    if preset == 'block1' and args.protocols is None and args.variants is None:
        for name in datasets:
            variants = ['blackbox', 'h2tcbd' if name == 'feat' else 'no_anchor']
            run_suite(config, [name], ['P-in'], variants, overwrite=args.overwrite)
        if 'feat' in datasets:
            run_suite(config, ['feat'], ['P-LOPO'], ['blackbox', 'h2tcbd'],
                      folds=args.folds, overwrite=args.overwrite)
        run_suite(config, datasets, ['P-cross'], ['blackbox', 'no_anchor'], overwrite=args.overwrite)
        return
    grid = config['theta_grid'] if preset == 'theta' and args.theta is None else [config['theta']]
    for theta in grid:
        run_config = copy.deepcopy(config)
        run_config['theta'] = theta
        run_suite(run_config, datasets, args.protocols or protocols, args.variants or variants,
                  args.folds, args.fixed_groups, args.pooled, args.overwrite)
