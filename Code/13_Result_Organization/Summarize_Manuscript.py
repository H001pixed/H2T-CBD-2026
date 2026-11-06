"""Section tables from completed runs, preserving expected coverage."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Expected_Experiments import expected_set, train_folds

PROJECT=Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=PROJECT / "workspace")
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()
ROOT = args.root.resolve()
OUT = args.output_dir.resolve() if args.output_dir else ROOT / 'results/analysis'
OUT.mkdir(parents=True, exist_ok=True)
FIELDS = ['dataset','source_dataset','protocol','variant','fold','seed','theta']
allowed = expected_set(train_folds(ROOT))
EXPECTED = pd.DataFrame(sorted(allowed, key=str), columns=FIELDS)
records=[]
for p in (ROOT/'results/runs').glob('*/result.json'):
    r=json.loads(p.read_text(encoding='utf-8'))
    if tuple(r.get(k) for k in FIELDS) not in allowed:
        continue
    records.append({**{k:r.get(k) for k in FIELDS}, 'status':r['status'],
        'qwk':r.get('test_qwk'), 'rmse':r.get('test_rmse'), 'kept_groups':json.dumps(r.get('kept_groups')),
        'run_id':p.parent.name, 'epochs_run':r.get('epochs_run'), 'best_epoch':r.get('best_epoch')})
frame=pd.DataFrame(records)
frame['fold']=frame.fold.fillna('ALL')
EXPECTED['fold']=EXPECTED.fold.fillna('ALL')
if frame.duplicated(FIELDS).any():
    raise ValueError('Duplicate experimental identities; do not select one silently')
G=['dataset','source_dataset','protocol','variant','theta']
summary=[]
for key, expected in EXPECTED.groupby(G,dropna=False):
    observed=frame
    for k,v in zip(G,key):
        observed=observed.loc[observed[k]==v]
    complete=(len(observed)==len(expected) and set(map(tuple,observed[FIELDS].to_numpy()))==set(map(tuple,expected[FIELDS].to_numpy()))
              and (observed.status=='succeeded').all() and np.isfinite(observed.qwk).all())
    row=dict(zip(G,key))
    row.update(n_expected=len(expected),n_observed=len(observed),complete=bool(complete),
        mean_qwk=float(observed.qwk.mean()) if complete else None,
        mean_rmse=float(observed.rmse.mean()) if complete else None,
        mean_epochs_run=float(observed.epochs_run.mean()) if complete else None)
    summary.append(row)
pd.DataFrame(summary).to_csv(OUT/'model_summary.csv',index=False,encoding='utf-8-sig')
contrasts=[]
pair_rows=[]
contexts=['dataset','source_dataset','protocol']
for ctx, expected in EXPECTED.loc[EXPECTED.theta==.1].groupby(contexts,dropna=False):
    data=frame.loc[frame.theta==.1]
    for k,v in zip(contexts,ctx): data=data.loc[data[k]==v]
    protocol=ctx[2]
    comparisons={'P-in':[('blackbox','h2tcbd'),('blackbox','no_anchor'),('blackbox','featconcat'),('no_anchor','h2tcbd')],
        'P-LOPO':[('blackbox','h2tcbd'),('blackbox','no_anchor'),('no_anchor','h2tcbd'),('h2tcbd','selective')],
        'P-in-prompt':[('blackbox','h2tcbd')], 'P-cross-calibrated':[('blackbox','no_anchor')],
        'P-official':[('h2tcbd','selective')]}[protocol]
    for baseline, variant in comparisons:
        exp=expected.loc[expected.variant==baseline,['fold','seed']]
        left=data.loc[data.variant==baseline]
        right=data.loc[data.variant==variant]
        pair=exp.merge(left[['fold','seed','status','qwk','rmse','run_id']],on=['fold','seed'],how='left').merge(
            right[['fold','seed','status','qwk','rmse','run_id']],on=['fold','seed'],how='left',suffixes=('_baseline','_comparison'))
        pair['delta_qwk']=pair.qwk_comparison-pair.qwk_baseline
        pair['complete']=(pair.status_baseline=='succeeded') & (pair.status_comparison=='succeeded') & np.isfinite(pair.delta_qwk)
        for k,v in zip(contexts,ctx): pair[k]=v
        pair['baseline']=baseline; pair['comparison']=variant
        pair_rows.extend(pair.to_dict('records'))
        row={**dict(zip(contexts,ctx)), 'baseline':baseline,'comparison':variant,'n_expected':len(pair),
            'n_complete':int(pair.complete.sum()),'status':'complete' if pair.complete.all() else 'partial'}
        if pair.complete.all():
            means=pair.groupby('fold').delta_qwk.mean()
            values=means.to_numpy() if len(means)>1 else pair.delta_qwk.to_numpy()
            rng=np.random.default_rng(42)
            draws=values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1)
            lo,hi=np.quantile(draws,[.025,.975])
            row.update(baseline_qwk=float(pair.qwk_baseline.mean()),comparison_qwk=float(pair.qwk_comparison.mean()),
                mean_delta=float(pair.delta_qwk.mean()),ci_lower=float(lo),ci_upper=float(hi),
                bootstrap_unit='prompt_label' if len(means)>1 else 'training_seed',n_units=len(values),
                positive_pairs=int((pair.delta_qwk>0).sum()),positive_prompt_labels=int((means>0).sum()))
        contrasts.append(row)
pd.DataFrame(contrasts).to_csv(OUT/'paired_comparisons.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(pair_rows).to_csv(OUT/'seed_pairs.csv',index=False,encoding='utf-8-sig')
print(json.dumps({'model_cells':len(summary),'complete_model_cells':sum(r['complete'] for r in summary),
                  'comparisons':len(contrasts),'complete_comparisons':sum(r['status']=='complete' for r in contrasts)},ensure_ascii=False))
