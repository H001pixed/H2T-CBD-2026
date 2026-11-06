"""Analyze the fixed ARS supplement runs without fitting or training models."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
import sys
sys.path.insert(0, str(PROJECT / 'Code/00_Core'))
import Common as C

DATASETS = ('persuade_nonoverlap', 'cela_single_prompt')
VARIANTS = ('h2tcbd', 'selective')
SEEDS = (13, 42, 123, 2026, 2027, 2028, 2029, 2030, 2031, 2032)
THETAS = (0.075, 0.1, 0.125)
SCALES = {'persuade_nonoverlap': (1, 6, 1), 'cela_single_prompt': (1, 5, .5)}
TOL = 1e-10


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys or ['status'])
        w.writeheader()
        w.writerows(rows)


def scalar_equal(a, b):
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=TOL)


def run_records(runs_root):
    records = []
    if not runs_root.exists():
        return records
    for result_path in sorted(runs_root.glob('*/result.json')):
        folder = result_path.parent
        rec = {'path': str(folder), 'exp_id': folder.name, 'status': 'missing_result'}
        try:
            result = read_json(result_path)
            rec.update({k: result.get(k) for k in ('dataset', 'variant', 'seed', 'fold', 'status', 'test_qwk', 'test_rmse', 'kept_groups',
                                                  'best_epoch', 'epochs_run', 'n_train', 'n_val', 'n_test')})
            spec_path = folder / 'run_spec.json'
            rec['run_spec_status'] = 'present' if spec_path.exists() else 'missing'
            spec = read_json(spec_path) if spec_path.exists() else {}
            if rec['status'] == 'succeeded' and not spec:
                raise ValueError('successful run is missing its split specification')
            rec['spec'] = spec
            pred_path = folder / 'predictions.npz'
            rec['predictions_status'] = 'present' if pred_path.exists() else 'missing'
            rec['ars_status'] = 'present' if (folder / 'ars.json').exists() else 'missing'
            rec['ars'] = read_json(folder / 'ars.json') if rec['ars_status'] == 'present' else None
            rec['prediction'] = None
            if pred_path.exists():
                with np.load(pred_path, allow_pickle=False) as z:
                    pred = {k: z[k].copy() for k in z.files}
                rec['prediction'] = pred
                required = ('id', 'score_true', 'score_pred')
                absent = [k for k in required if k not in pred]
                rec['predictions_status'] = 'missing_required_' + ','.join(absent) if absent else 'present'
                if absent and rec['status'] == 'succeeded': raise ValueError(f'missing prediction keys: {absent}')
                if not absent:
                    ids = pred['id'].astype(str).tolist()
                    y, p = pred['score_true'].astype(float), pred['score_pred'].astype(float)
                    rec['unique_prediction_levels'] = len(np.unique(p))
                    if len(set(ids)) != len(ids): raise ValueError('duplicate prediction IDs')
                    if len(ids) != len(y) or len(y) != len(p): raise ValueError('prediction array lengths differ')
                    if ids != list(map(str, spec['test_ids'])):
                        raise ValueError('prediction IDs differ from the fixed test split')
                    lo, hi, step = SCALES.get(rec.get('dataset'), (None, None, None))
                    if lo is None:
                        raise ValueError('unknown dataset scale')
                    q = C.qwk(y, p, lo, hi, step)
                    rmse = C.rmse(y, p)
                    rec['recomputed_qwk'], rec['recomputed_rmse'] = q, rmse
                    if not scalar_equal(q if np.isfinite(q) else None, result.get('test_qwk')) or not scalar_equal(rmse, result.get('test_rmse')):
                        raise ValueError(f'metric mismatch stored=({result.get("test_qwk")},{result.get("test_rmse")}) recomputed=({q},{rmse})')
                    rec['metric_status'] = 'undefined_metric' if not np.isfinite(q) else 'verified'
                    rec['recomputed_qwk'] = q if np.isfinite(q) else None
            if rec['status'] == 'succeeded' and rec['predictions_status'] != 'present':
                raise ValueError('successful run has unusable predictions')
        except Exception as e:
            raise ValueError(f'{folder.name}: {type(e).__name__}: {e}') from e
        records.append(rec)
    return records


def key_for(rec):
    s = rec.get('spec') or {}
    return (rec.get('dataset'), int(rec.get('seed')) if rec.get('seed') is not None else None,
            str(s.get('fold', rec.get('fold')) or ''))


def paired_analysis(records, allkeys):
    index = {}
    for r in records:
        if r.get('dataset') in DATASETS and r.get('variant') in VARIANTS:
            index.setdefault((*key_for(r), r.get('variant')), []).append(r)
    pairs, rows = [], []
    for key in allkeys:
        sides = {v: index.get((*key, v), []) for v in VARIANTS}
        row = {'dataset': key[0], 'seed': key[1], 'fold': key[2], 'pair_status': 'missing_side'}
        if len(sides['h2tcbd']) == len(sides['selective']) == 1:
            a, b = sides['h2tcbd'][0], sides['selective'][0]
            row['pair_status'] = 'incomplete_run'
            valid = all(x.get('status') == 'succeeded' and x.get('prediction') is not None and x.get('recomputed_qwk') is not None for x in (a, b))
            if valid:
                sa, sb = a['spec'], b['spec']
                pa, pb = a['prediction'], b['prediction']
                for field in ('train_ids', 'val_ids', 'test_ids'):
                    if list(map(str, sa.get(field, []))) != list(map(str, sb.get(field, []))):
                        raise ValueError(f'{key}: paired {field} mismatch')
                if not np.array_equal(pa['id'].astype(str), pb['id'].astype(str)) or not np.array_equal(pa['score_true'].astype(float), pb['score_true'].astype(float)):
                    raise ValueError(f'{key}: paired prediction IDs or score_true mismatch')
                row.update(pair_status='complete', n_test=len(pa['id']), h2tcbd_qwk=a.get('recomputed_qwk'), selective_qwk=b.get('recomputed_qwk'),
                           delta_qwk=b['recomputed_qwk']-a['recomputed_qwk'], h2tcbd_rmse=a['recomputed_rmse'], selective_rmse=b['recomputed_rmse'],
                           delta_rmse=b['recomputed_rmse']-a['recomputed_rmse'],
                           kept_groups='|'.join(map(str, b['kept_groups'])),
                           n_kept_groups=len(b['kept_groups']), actual_pruning=len(b['kept_groups']) < 6)
                pairs.append((key, a, b, row))
            elif any(x.get('status')=='no_groups_selected' for x in (a,b)): row['pair_status']='no_groups_selected'
            elif any(x.get('metric_status')=='undefined_metric' for x in (a,b)): row['pair_status']='undefined_metric'
        elif len(sides['h2tcbd'])>1 or len(sides['selective'])>1:
            raise ValueError(f'{key}: duplicate run records')
        rows.append(row)
    return pairs, rows


def bootstrap_persuade(pairs):
    by_fold = {}
    for key, _, _, row in pairs:
        if key[0] == 'persuade_nonoverlap':
            by_fold.setdefault(key[2], []).append(row['delta_qwk'])
    fold_means = [float(np.mean(v)) for _, v in sorted(by_fold.items()) if len(v)==10]
    partial = {str(k):len(v) for k,v in by_fold.items() if len(v)!=10}
    if len(fold_means) != 8:
        return {'complete_prompts': len(fold_means), 'partial_folds': partial, 'ci95': None, 'bootstrap_n': 10000}
    rng = np.random.default_rng(20261102)
    samples = rng.choice(fold_means, size=(10000, len(fold_means)), replace=True).mean(axis=1)
    return {'complete_prompts': len(fold_means), 'partial_folds': partial, 'prompt_mean_deltas': fold_means,
            'mean_delta': float(np.mean(fold_means)), 'ci95': np.quantile(samples, [.025, .975]).tolist(), 'bootstrap_n': 10000}


def bootstrap_cela(pairs):
    cela = [p for p in pairs if p[0][0] == 'cela_single_prompt']
    out = {'complete_pairs': len(cela), 'bootstrap_n': 5000, 'valid': 0, 'invalid': 0, 'ci95': None}
    if len(cela) != 10:
        return out
    ids = sorted({str(x) for _, a, _, _ in cela for x in a['prediction']['id']})
    id_index = {x: i for i, x in enumerate(ids)}
    rng = np.random.default_rng(20261102)
    draws = rng.integers(0, len(ids), size=(5000, len(ids)))
    vals = []
    for draw in draws:
        counts = np.bincount(draw, minlength=len(ids))
        ds = []
        valid = True
        for _, a, b, _ in cela:
            pa, pb = a['prediction'], b['prediction']
            mult = np.array([counts[id_index[str(i)]] for i in pa['id']])
            if not mult.sum():
                valid = False
                break
            yt = np.repeat(pa['score_true'].astype(float), mult)
            qa = C.qwk(yt, np.repeat(pa['score_pred'].astype(float), mult), 1, 5, .5)
            qb = C.qwk(yt, np.repeat(pb['score_pred'].astype(float), mult), 1, 5, .5)
            if not np.isfinite(qa) or not np.isfinite(qb):
                valid = False
                break
            ds.append(qb-qa)
        if valid:
            vals.append(float(np.mean(ds)))
    out['valid'], out['invalid'] = len(vals), 5000-len(vals)
    if vals:
        out['ci95'] = np.quantile(vals, [.025, .975]).tolist()
    out['essay_ids'] = len(ids)
    return out


def selection_analysis(records):
    rows = []
    for r in records:
        if r.get('dataset') not in DATASETS or r.get('variant') != 'selective':
            continue
        ars = r.get('ars')
        groups = ars.get('groups', []) if isinstance(ars, dict) else []
        by_index = {int(g['group_index']): g for g in groups if 'group_index' in g}
        vals = [None if by_index.get(i, {}).get('ars') is None else float(by_index[i]['ars']) for i in range(6)]
        chosen = r.get('kept_groups')
        spec = r.get('spec') or {}
        theta = (spec.get('config') or {}).get('theta')
        threshold_choices = {str(t): [i for i, v in enumerate(vals) if v is not None and v > t] for t in THETAS}
        directions = ars.get('feature_direction', []) if isinstance(ars, dict) else []
        rows.append({'dataset': r.get('dataset'), 'seed': r.get('seed'), 'fold': key_for(r)[2], 'theta': theta,
                     **{f'ars_group_{i}': vals[i] for i in range(6)},
                     **{f'mean_correlation_group_{i}': by_index.get(i,{}).get('mean') for i in range(6)},
                     **{f'cross_prompt_sd_group_{i}': by_index.get(i,{}).get('std') for i in range(6)},
                     **{f'selected_group_{i}': int(i in (chosen or [])) if isinstance(chosen, list) else None for i in range(6)},
                     **{f'theta_{t}_groups': '|'.join(map(str, threshold_choices[str(t)])) for t in THETAS},
                     'feature_direction': '|'.join(map(str, directions)),
                     'selection_status': r.get('status'), 'ars_status': r.get('ars_status')})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runs-dir', type=Path, help='Read saved runs from a separate directory.')
    p.add_argument('--manifest', type=Path, help='Read the original frozen protocol from this path.')
    args = p.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    runs = args.runs_dir.resolve() if args.runs_dir else output / 'results/runs'
    records = run_records(runs)
    manifest_path = args.manifest.resolve() if args.manifest else output/'configs/frozen_protocol.json'
    if manifest_path.exists():
        manifest=read_json(manifest_path); plan=[]; seeds=manifest.get('config',{}).get('seeds',SEEDS)
        for ds, meta in manifest['datasets'].items():
            folds=meta['prompts'] if ds=='persuade_nonoverlap' else ['']
            plan.extend((ds,int(seed),str(fold)) for seed in seeds for fold in folds)
    else:
        import pandas as pd
        plan=[]
        for ds in DATASETS:
            data_path=root/'data/01_raw_datasets'/ds/'data.csv'
            if not data_path.exists(): raise FileNotFoundError(data_path)
            folds=sorted(pd.read_csv(data_path,dtype={'essay_set':str}).essay_set.dropna().astype(str).unique()) if ds=='persuade_nonoverlap' else ['']
            plan.extend((ds,seed,fold) for seed in SEEDS for fold in folds)
    plan=sorted(set(plan),key=lambda x:(x[0],x[1],x[2]))
    pairs, pair_rows = paired_analysis(records, plan)
    base = output / 'results/analysis'
    dirs = [base/'01_selection', base/'02_scoring_pairs', base/'03_prompt_and_single_prompt']
    for d in dirs: d.mkdir(parents=True, exist_ok=True)
    run_index = {(*key_for(r), r['variant']): r for r in records}
    run_rows = []
    for dataset, seed, fold in plan:
        for variant in VARIANTS:
            record = run_index.get((dataset, seed, fold, variant), {})
            run_rows.append({'dataset': dataset, 'seed': seed, 'fold': fold, 'variant': variant,
                             'status': record.get('status', 'not_started'),
                             'qwk': record.get('recomputed_qwk'), 'rmse': record.get('recomputed_rmse'),
                             'best_epoch': record.get('best_epoch'), 'epochs_run': record.get('epochs_run'),
                             'n_train': record.get('n_train'), 'n_val': record.get('n_val'), 'n_test': record.get('n_test'),
                             'unique_prediction_levels': record.get('unique_prediction_levels'),
                             'kept_groups': '|'.join(map(str, record.get('kept_groups') or []))})
    write_csv(dirs[1]/'run_status.csv', run_rows)
    sel = selection_analysis(records)
    write_csv(dirs[0]/'selection.csv', sel)
    write_csv(dirs[1]/'paired_detail.csv', pair_rows)
    perspective = bootstrap_persuade(pairs)
    cela = bootstrap_cela(pairs)
    per_fold = {}
    for key, a, b, row in pairs:
        if key[0] != 'persuade_nonoverlap':
            continue
        item = per_fold.setdefault(key[2], {'seed_deltas': [], 'pair_rows': []})
        item['seed_deltas'].append(row['delta_qwk'])
        item['pair_rows'].append(row)
    topic_rows = []
    for f, item in sorted(per_fold.items()):
        v = item['seed_deltas']
        ar=[p[1] for p in pairs if p[0][0]=='persuade_nonoverlap' and p[0][2]==f]
        topic_rows.append({'fold': f, 'n_seeds': len(v),
                           'mean_delta_qwk': float(np.mean(v)), 'std_delta_qwk': float(np.std(v, ddof=1)) if len(v)>1 else 0.0,
                           'h2tcbd_mean_qwk':float(np.mean([x['recomputed_qwk'] for x in ar])),
                           'selective_mean_qwk':float(np.mean([p[2]['recomputed_qwk'] for p in pairs if p[0][0]=='persuade_nonoverlap' and p[0][2]==f])),
                           'h2tcbd_mean_rmse':float(np.mean([x['recomputed_rmse'] for x in ar])),
                           'selective_mean_rmse':float(np.mean([p[2]['recomputed_rmse'] for p in pairs if p[0][0]=='persuade_nonoverlap' and p[0][2]==f])),
                           'positive_pairs': sum(x>0 for x in v), 'negative_pairs': sum(x<0 for x in v),
                           'equal_pairs': sum(x==0 for x in v), 'min_delta_qwk': min(v), 'max_delta_qwk': max(v),
                           'within_minus_002': sum(x>=-.02 for x in v)})
    for row in pair_rows:
        if row['dataset'] == 'persuade_nonoverlap' and row['pair_status'] == 'complete':
            row['fold_seed_mean_delta_qwk'] = float(np.mean(per_fold[row['fold']]['seed_deltas']))
            row['fold_positive_pairs'] = sum(x>0 for x in per_fold[row['fold']]['seed_deltas'])
            row['fold_within_minus_002'] = sum(x>=-.02 for x in per_fold[row['fold']]['seed_deltas'])
    write_csv(dirs[0]/'selection.csv', sel)
    write_csv(dirs[1]/'paired_detail.csv', pair_rows)
    cela_rows = [row for row in pair_rows if row['dataset']=='cela_single_prompt']
    write_csv(dirs[2]/'persuade_fold_summary.csv', topic_rows)
    write_csv(dirs[2]/'cela_seed_summary.csv', cela_rows)
    complete_models = sum(r.get('status') == 'succeeded' for r in records)
    ds_summary={}
    for ds in DATASETS:
        pp=[x for x in pairs if x[0][0]==ds]; dd=[x[3]['delta_qwk'] for x in pp]
        pruned=[x for x in pp if len(x[2]['kept_groups']) < 6]
        pruned_deltas=[x[3]['delta_qwk'] for x in pruned]
        ds_summary[ds]={'complete_score_pairs':len(pp),'h2tcbd_mean_qwk':float(np.mean([x[1]['recomputed_qwk'] for x in pp])) if pp else None,
          'selective_mean_qwk':float(np.mean([x[2]['recomputed_qwk'] for x in pp])) if pp else None,
          'h2tcbd_mean_rmse':float(np.mean([x[1]['recomputed_rmse'] for x in pp])) if pp else None,
          'selective_mean_rmse':float(np.mean([x[2]['recomputed_rmse'] for x in pp])) if pp else None,
          'mean_delta_qwk':float(np.mean(dd)) if dd else None,'sd_delta_qwk':float(np.std(dd,ddof=1)) if len(dd)>1 else (0. if dd else None),
          'positive_pairs':sum(x>0 for x in dd),'negative_pairs':sum(x<0 for x in dd),'equal_pairs':sum(x==0 for x in dd),
          'min_delta_qwk':min(dd) if dd else None,'max_delta_qwk':max(dd) if dd else None,
          'delta_at_least_minus_002':sum(x>=-.02 for x in dd),
          'actual_pruning_pairs':len(pruned), 'all_six_group_pairs':len(pp)-len(pruned),
          'pruned_mean_delta_qwk':float(np.mean(pruned_deltas)) if pruned_deltas else None,
          'pruned_positive_pairs':sum(x>0 for x in pruned_deltas),
          'pruned_negative_pairs':sum(x<0 for x in pruned_deltas),
          'pruned_equal_pairs':sum(x==0 for x in pruned_deltas),
          'kept_group_count_distribution':{str(n):sum(len(x[2]['kept_groups'])==n for x in pp) for n in range(1,7)},
          'constant_h2tcbd_models':sum(r.get('unique_prediction_levels')==1 for r in records if r.get('dataset')==ds and r.get('variant')=='h2tcbd'),
          'constant_selective_models':sum(r.get('unique_prediction_levels')==1 for r in records if r.get('dataset')==ds and r.get('variant')=='selective')}
    theta_counts={}
    theta_model_changes={}
    for ds in DATASETS:
      rr=[r for r in sel if r['dataset']==ds]
      for t in THETAS:
       theta_model_changes[f'{ds}|{t}']=sum(r[f'theta_{t}_groups'] != r['theta_0.1_groups'] for r in rr)
       for g in range(6):
        theta_counts[f'{ds}|{t}|{g}']={'selected':sum(r.get(f'ars_group_{g}') is not None and r[f'ars_group_{g}']>t for r in rr),
          'different_from_0.1':sum(r.get(f'ars_group_{g}') is not None and (r[f'ars_group_{g}']>t)!=(r[f'ars_group_{g}']>.1) for r in rr)}
    all_recorded=all((*k,v) in {(*key_for(r),r.get('variant')) for r in records} for k in plan for v in VARIANTS)
    summary = {'expected_runs': 180, 'planned_pair_rows':len(plan), 'discovered_runs': len(records), 'successful_runs': complete_models,
               'complete_pairs': len(pairs), 'expected_pairs': 90, 'pair_status_counts': {s: sum(x['pair_status']==s for x in pair_rows) for s in sorted({x['pair_status'] for x in pair_rows})},
               'metric_recomputed': True, 'metric_tolerance': TOL, 'dataset_score_summary':ds_summary,'persuade': perspective, 'cela': cela,
               'selection_rows': len(sel), 'theta_grid': list(THETAS), 'theta_group_selection_counts':theta_counts,
               'theta_model_selection_changes':theta_model_changes,
               'all_attempts_recorded':all_recorded,'score_pairs_complete':len(pairs)==90,
               'status': 'all_attempts_recorded' if all_recorded else 'incomplete_attempts',
               'unresolved_records': [{'exp_id': r['exp_id'], 'status': r.get('status'), 'error': r.get('error'), 'run_spec_status': r.get('run_spec_status'), 'predictions_status': r.get('predictions_status'), 'ars_status': r.get('ars_status')} for r in records if r.get('status') != 'succeeded' or r.get('error')]}
    dump_json(base/'summary.json', summary)
    for d, text in zip(dirs, [f'ARS values and selected groups for every corpus, seed and fold. Records: {len(sel)}.',
                              f'Per-run status and paired scoring checks. Runs: {len(records)}; complete pairs: {len(pairs)}.',
                              f'PERSUADE prompt summaries, CELA seed summaries and fixed-design intervals. Complete PERSUADE prompts: {perspective["complete_prompts"]}; complete CELA pairs: {cela["complete_pairs"]}.']):
        readme = d/'README.md'
        if not readme.exists():
            readme.write_text(text+'\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'runs': len(records), 'complete_pairs': len(pairs), 'output': str(base)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
