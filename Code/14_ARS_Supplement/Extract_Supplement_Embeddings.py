"""Extract full-text embeddings for the independent corpora with explicit GPU precision."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'Code/00_Core'))
import Common as C
import Extract_Embeddings as B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    enc = config['encoder']
    names = list(reversed(config['datasets']))
    paths = B._resolve_inputs(root, names)
    hashes = {name: B._sha256(path) for name, path in paths.items()}
    frames = {name: B._read_dataset(path, C.FEAT_COLS) for name, path in paths.items()}

    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[enc['dtype']]
    device = torch.device(enc['device'])
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Supplement encoding requires the configured CUDA device.')
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('Configured bfloat16 is unavailable on this GPU.')
    revision = B._resolve_revision(enc['model'], enc['revision'])
    model_config = AutoConfig.from_pretrained(enc['model'], revision=revision, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(enc['model'], revision=revision, padding_side='left', trust_remote_code=False)
    limits = {'configured': enc['max_tokens'], 'model': model_config.max_position_embeddings,
              'tokenizer': tokenizer.model_max_length, 'supported': B.SUPPORTED_CONTEXT[enc['model']]}
    context = min(limits.values())
    lengths = {name: B._scan_lengths(tokenizer, frame.text.tolist()) for name, frame in frames.items()}
    for name, tokens in lengths.items():
        if tokens.min() < 1 or tokens.max() > context:
            raise ValueError(f'{name}: input exceeds the untruncated context limit.')
        print(json.dumps({'dataset': name, 'rows': len(tokens), 'max_tokens': int(tokens.max()),
                          'dtype': str(dtype), 'gpu': torch.cuda.get_device_name(0)}), flush=True)
    parameters = {
        'pooling': 'last_token', 'normalization': 'l2_float32', 'padding_side': 'left',
        'add_special_tokens': True, 'truncation': False, 'instruction': None,
        'context_limits': limits, 'effective_max_tokens': context,
        'dtype': str(dtype), 'device': str(device), 'batch_size': enc['batch_size'],
        'attn_implementation': 'sdpa', 'use_cache': False,
    }
    versions = {name: version(name) for name in
                ('torch', 'transformers', 'tokenizers', 'huggingface-hub', 'numpy', 'pandas')}
    output = root / 'results/embeddings'
    output.mkdir(parents=True, exist_ok=True)
    pending = {}
    for name, frame in frames.items():
        arrays = B._row_arrays(frame, C.FEAT_COLS, lengths[name])
        expected = {
            'cache_schema_version': B.CACHE_SCHEMA_VERSION, 'input_sha256': hashes[name],
            'model': enc['model'], 'revision': revision, 'feature_columns': C.FEAT_COLS,
            'rows': len(frame), 'embedding_dim': model_config.hidden_size,
            'max_tokens': int(lengths[name].max()), 'encoder_parameters': parameters,
            'runtime_versions': versions,
        }
        if not B._check_cache(output / f'{name}.json', expected, output / f'{name}.npz', arrays):
            pending[name] = expected, arrays
    if not pending:
        return
    model = AutoModel.from_pretrained(enc['model'], revision=revision, config=model_config,
                                      torch_dtype=dtype, attn_implementation='sdpa',
                                      trust_remote_code=False).to(device).eval()
    model.requires_grad_(False)
    for name, (expected, arrays) in pending.items():
        rows = []
        started = time.monotonic()
        for start in range(0, len(frames[name]), 100):
            stop = min(start + 100, len(frames[name]))
            rows.append(B._pool_full_text(model, tokenizer, frames[name].text.iloc[start:stop].tolist(),
                                         enc['batch_size'], device, context, lengths[name][start:stop]))
            print(f'{name}: encoded {stop}/{len(frames[name])}; seconds={time.monotonic()-started:.1f}', flush=True)
        emb = np.concatenate(rows)
        if emb.shape != (len(frames[name]), model_config.hidden_size) or B._sha256(paths[name]) != hashes[name]:
            raise ValueError(f'{name}: encoded shape or source data changed.')
        path = output / f'{name}.npz'
        np.savez_compressed(path, emb=emb, **arrays)
        C.save_json(path.with_suffix('.json'), {**expected, 'requested_revision': enc['revision'],
                                               'source_config': config, 'embedding_sha256': B._sha256(path)})
        print(f'{name}: embedding saved', flush=True)


if __name__ == '__main__':
    main()
