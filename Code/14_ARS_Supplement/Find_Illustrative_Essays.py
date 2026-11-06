"""Select three illustrative essays from saved ARS predictions without training or plotting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
GROUPS = ['length_fluency', 'lexical_complexity', 'syntactic_complexity', 'mechanics', 'cohesion_readability', 'rhetoric_engagement']
ROLES = [('high', 'high', 6), ('middle', 'mid', 3), ('low', 'low', 2)]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def profile_masks(percentiles):
    return {
        'high': percentiles.min(axis=1) >= .6,
        'middle': (percentiles.min(axis=1) <= .4) & (percentiles.max(axis=1) >= .6),
        'low': percentiles.max(axis=1) <= .4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT/'workspace')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--prompt', default='Distance learning')
    args = parser.parse_args()
    root = args.root.resolve()
    runs = root/'supplement/results/runs'
    output = root/'supplement/results/cases'
    output.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(root/'data/01_raw_datasets/persuade_nonoverlap/data.csv', dtype={'id': str}).set_index('id')

    scan = []
    selected_folder = None
    for result_path in sorted(runs.glob('B1_LOPO_persuade_nonoverlap_selective_*/result.json')):
        result = read_json(result_path)
        if result['status'] != 'succeeded':
            raise ValueError(f'Unfinished source run: {result_path.parent.name}')
        with np.load(result_path.with_name('predictions.npz'), allow_pickle=False) as p:
            contribution = p['score_terms'][:, result['kept_groups']]
            percentile = pd.DataFrame(contribution).rank(pct=True).to_numpy()
            masks = profile_masks(percentile)
            human, pred = p['score_true'], p['score_pred']
            bands = {'high': human >= 5, 'middle': (human >= 3) & (human <= 4), 'low': human <= 2}
            counts = {role: int((masks[role] & band & (pred == human)).sum()) for role, band in bands.items()}
        scan.append({'exp_id': result_path.parent.name, 'seed': result['seed'], 'prompt': result['fold'],
                     'high_profile_candidates': counts['high'], 'middle_profile_candidates': counts['middle'],
                     'low_profile_candidates': counts['low'], 'all_three_available': all(counts.values())})
        if result['seed'] == args.seed and result['fold'] == args.prompt:
            if selected_folder is not None:
                raise ValueError('More than one source model matches the requested identity.')
            selected_folder = result_path.parent
    pd.DataFrame(scan).to_csv(output/'candidate_scan.csv', index=False, encoding='utf-8-sig')
    if selected_folder is None:
        raise ValueError('Requested model not found.')

    result = read_json(selected_folder/'result.json')
    checkpoint = torch.load(selected_folder/'model.pt', map_location='cpu', weights_only=False)
    with np.load(selected_folder/'predictions.npz', allow_pickle=False) as z:
        p = {name: z[name].copy() for name in z.files}
    groups = result['kept_groups']
    state = checkpoint['state_dict']
    weight = state['readout.weight'].numpy()[0]
    mask = state['score_mask'].numpy()
    bias = float(state['readout.bias'][0])
    terms = p['z'] * mask * weight
    if not np.allclose(terms, p['score_terms'], atol=1e-6, rtol=0):
        raise ValueError('Stored contributions do not match checkpoint weights and coordinates.')
    unit = 1/(1+np.exp(-(terms.sum(axis=1)+bias)))
    if not np.allclose(unit, p['pred_unit'], atol=2e-7, rtol=1e-6):
        raise ValueError('Stored contributions do not reconstruct the continuous model output.')

    contribution = terms[:, groups]
    percentile = pd.DataFrame(contribution).rank(pct=True).to_numpy()
    profiles = profile_masks(percentile)
    human, pred = p['score_true'], p['score_pred']
    ids = p['id'].astype(str)
    chosen, rows = [], []
    for role, label, grade in ROLES:
        candidates = np.flatnonzero((human == grade) & (pred == human) & profiles[role])
        if not len(candidates):
            raise ValueError(f'No {role} candidate with human and model score {grade}.')
        center = np.median(percentile[human == grade], axis=0)
        distance = np.square(percentile[candidates]-center).sum(axis=1)
        i = int(candidates[np.lexsort((ids[candidates], distance))[0]])
        original = source.loc[ids[i]]
        if float(original['score']) != human[i] or original['essay_set'] != args.prompt:
            raise ValueError('Source essay labels differ from the saved prediction.')
        case = {'role': label, 'id': ids[i], 'human_score': float(human[i]), 'model_score': float(pred[i]),
                'word_count': int(original['word_count']), 'contributions': contribution[i].astype(float).tolist(),
                'contribution_percentiles': (percentile[i]*100).tolist(), 'pred_unit': float(p['pred_unit'][i]),
                'intercept': bias, 'n_matching_exact_grade_candidates': len(candidates), 'text': original['text']}
        chosen.append(case)
        rows.append({k: v for k, v in case.items() if k not in ('text','contributions','contribution_percentiles')} |
                    {f'{GROUPS[g]}': float(contribution[i,j]) for j,g in enumerate(groups)} |
                    {f'{GROUPS[g]}_percentile': float(percentile[i,j]*100) for j,g in enumerate(groups)})
    pd.DataFrame(rows).to_csv(output/'case_contributions.csv', index=False, encoding='utf-8-sig')
    payload = {'dataset': 'persuade_nonoverlap', 'prompt': args.prompt, 'seed': args.seed,
               'exp_id': selected_folder.name, 'kept_groups': groups, 'group_names': [GROUPS[g] for g in groups],
               'score_scale': [1,6], 'reference_population': len(ids),
               'selection_type': 'post-hoc illustrative cases, selected for requested profile shapes',
               'profile_definition': 'Same held-out prompt and fixed model. High: all contribution percentiles >=60; low: all <=40; middle: at least one <=40 and one >=60. Human and model scores agree.',
               'within_profile_selection': 'Nearest to the whole exact-human-grade median percentile profile; ID breaks ties.',
               'model_choice': 'Exploratory scan of 80 frozen models; Distance learning / seed 2026 selected for illustrating all three requested profiles. No retraining or evaluation changes.',
               'models_scanned': len(scan), 'models_with_all_three_profiles': sum(x['all_three_available'] for x in scan),
               'cases': chosen}
    (output/'illustrative_cases.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    md = ['# Illustrative essay cases', '',
          f'Prompt: {args.prompt}. Selective-anchoring model, seed {args.seed}.', '',
          'Contributions are signed terms before the score mapping. Percentiles use all test essays on this prompt.', '',
          'The cases were selected after training by their contribution profiles. Human and model scores agree.', '',
          '| Case | Human score | Model score | Word count |', '|---|---:|---:|---:|']
    for case in chosen:
        md.append(f"| {case['role']} | {case['human_score']:g} | {case['model_score']:g} | {case['word_count']} |")
    for case in chosen:
        md += ['', '## ' + case['role'] + ' essay', '', case['text']]
    (output/'illustrative_cases.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    (output/'README.md').write_text(
        '# Case outputs\n\n'
        '- `case_contributions.csv`: contribution values and within-prompt percentiles.\n'
        '- `illustrative_cases.json`: model identity, selection rules, and selected essay text.\n'
        '- `illustrative_cases.md`: locally selected essay examples.\n'
        '- `candidate_scan.csv`: candidate coverage across all scanned models.\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'source_model': selected_folder.name,
                      'cases': [{k:v for k,v in x.items() if k!='text'} for x in chosen]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
