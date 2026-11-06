"""Prepare the original CELA release, retaining holistic scores and checking text overlap."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from Canonical_Data import text_hash
from Prepare_PERSUADE import comparison_key


def prepare(source, existing_root, output):
    existing = {}
    for dataset in ('feat', 'asap2', 'persuade_nonoverlap'):
        with (existing_root / dataset / 'data.csv').open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                existing[comparison_key(row['text'])] = f"{dataset}:{row['id']}"
    with source.open(encoding='utf-8-sig', newline='') as handle:
        source_rows = list(csv.DictReader(handle))
    kept, removed, seen = [], [], {}
    for number, row in enumerate(source_rows, 1):
        essay_id = f'cela_row{number:04d}'
        text, score = row['Essays'], float(row['Holistic'])
        key = comparison_key(text)
        if not text.strip() or not 1 <= score <= 5 or score * 2 != int(score * 2):
            raise ValueError(f'Invalid original score or text: {essay_id}')
        match = existing.get(key) or seen.get(key)
        if match:
            removed.append({'id': essay_id, 'matched_id': match, 'text_hash': text_hash(text)})
        else:
            kept.append({'id': essay_id, 'text': text, 'score': score,
                         'essay_set': 'CELA_children_limit', 'partition': 'train',
                         'text_hash': text_hash(text)})
            seen[key] = essay_id
    output.mkdir(parents=True, exist_ok=True)
    for name, fields, rows in [
        ('data.csv', ['id', 'text', 'score', 'essay_set', 'partition', 'text_hash'], kept),
        ('removed_overlaps.csv', ['id', 'matched_id', 'text_hash'], removed),
    ]:
        with (output / name).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with source.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    summary = {
        'source': 'CELA, original author release',
        'source_url': 'https://github.com/gzutxy/CELA',
        'paper_doi': '10.1109/ACCESS.2021.3110683', 'source_sha256': digest,
        'source_essays': len(source_rows), 'retained_essays': len(kept),
        'removed_duplicates': len(removed), 'prompt_count': 1,
        'prompt': 'Should a government be allowed to limit the number of children a family can have?',
        'target': 'Original Holistic column; no averaging of analytic dimensions or rounding',
        'score_step': .5, 'observed_score_range': [min(r['score'] for r in kept), max(r['score'] for r in kept)],
        'id_rule': 'Original CSV data-row number, starting at 1; source has no essay identifier',
        'text_comparison': 'Unicode NFKC, casefold, collapse whitespace; no fuzzy match',
        'comparison_corpora': ['ASAP-1', 'ASAP-2.0', 'PERSUADE non-overlapping prompts'],
        'partition': 'All retained source rows; no train/validation/test split assigned yet',
        'status': 'Text and original labels prepared; features, embeddings and training not run',
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
