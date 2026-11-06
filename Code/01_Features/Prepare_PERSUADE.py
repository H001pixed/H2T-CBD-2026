"""Prepare the official PERSUADE training release and remove overlapping prompts and essays."""
import argparse
import collections
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

from Canonical_Data import text_hash


def comparison_key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def prompt_key(text):
    return comparison_key(text).strip('"\'“”‘’ ')


def prepare(source, existing_root, output):
    csv.field_size_limit(100_000_000)
    existing = collections.defaultdict(list)
    known_prompts = set()
    for dataset in ('feat', 'asap2'):
        with (existing_root / dataset / 'data.csv').open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                existing[comparison_key(row['text'])].append(f"{dataset}:{row['id']}")
                if dataset == 'asap2':
                    known_prompts.add(prompt_key(row['essay_set']))

    essays = {}
    annotation_rows = 0
    with source.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            annotation_rows += 1
            essay_id = row['essay_id_comp'] or row['essay_id']
            item = (row['full_text'], float(row['holistic_essay_score']), row['prompt_name'])
            if essay_id in essays and essays[essay_id] != item:
                raise ValueError(f'Conflicting source rows for essay {essay_id}')
            essays[essay_id] = item

    kept, removed = [], []
    retained_texts = {}
    prompt_counts = collections.defaultdict(collections.Counter)
    for essay_id, (text, score, prompt) in essays.items():
        if not text.strip() or score not in range(1, 7) or not prompt.strip():
            raise ValueError(f'Invalid source essay {essay_id}')
        key = comparison_key(text)
        matches = existing.get(key, [])
        reason = ''
        if prompt_key(prompt) in known_prompts:
            reason = 'overlapping_prompt'
        elif matches:
            reason = 'overlapping_essay'
        elif key in retained_texts:
            if retained_texts[key][1] != score:
                raise ValueError(f'Identical text with conflicting scores: {essay_id}')
            reason = 'duplicate_text_within_source'
            matches = [f'persuade:{retained_texts[key][0]}']
        prompt_counts[prompt]['source_essays'] += 1
        prompt_counts[prompt]['text_matches_existing'] += bool(existing.get(key))
        if reason:
            prompt_counts[prompt][reason] += 1
            removed.append({'id': essay_id, 'essay_set': prompt, 'reason': reason,
                            'matched_ids': '|'.join(matches), 'text_hash': text_hash(text)})
        else:
            retained_texts[key] = (essay_id, score)
            kept.append({'id': essay_id, 'text': text, 'score': score,
                         'essay_set': prompt, 'partition': 'train', 'text_hash': text_hash(text)})
            prompt_counts[prompt]['retained'] += 1

    output.mkdir(parents=True, exist_ok=True)
    tables = [
        ('data.csv', ['id', 'text', 'score', 'essay_set', 'partition', 'text_hash'], kept),
        ('removed_overlaps_and_duplicates.csv', ['id', 'essay_set', 'reason', 'matched_ids', 'text_hash'], removed),
        ('prompt_retention.csv', ['essay_set', 'source_essays', 'text_matches_existing',
                             'overlapping_prompt', 'overlapping_essay',
                             'duplicate_text_within_source', 'retained'],
         [{'essay_set': prompt, **counts} for prompt, counts in sorted(prompt_counts.items())]),
    ]
    for name, fields, rows in tables:
        with (output / name).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, restval=0)
            writer.writeheader()
            writer.writerows(rows)
    with source.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    summary = {
        'source': 'PERSUADE 2.0 official training release',
        'source_url': 'https://github.com/scrosseye/persuade_corpus_2.0',
        'source_file': source.name, 'source_sha256': digest,
        'source_annotation_rows': annotation_rows, 'source_unique_essays': len(essays),
        'source_prompt_count': len(prompt_counts), 'retained_essays': len(kept),
        'retained_prompt_count': len({row['essay_set'] for row in kept}),
        'removal_counts': dict(collections.Counter(row['reason'] for row in removed)),
        'comparison_corpora': ['ASAP-1', 'ASAP-2.0, including train and test text'],
        'text_comparison': 'Unicode NFKC, casefold, collapse whitespace; no fuzzy match',
        'prompt_exclusion': 'Remove entire prompts present in ASAP-2.0',
        'stored_text': 'Original text preserved; canonical SHA256 collapses whitespace only',
        'score_range': [1, 6], 'partition': 'Official train only; official test not downloaded',
        'status': 'Deduplicated text and labels; features, embeddings and training not run',
    }
    (output / 'metadata.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--existing-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.existing_root, args.output)
