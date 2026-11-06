import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import Common as C
from Models import make_model


def load_emb(name, config=None):
    path = C.EMB / f'{name}.npz'
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if meta['feature_columns'] != C.FEAT_COLS:
        raise ValueError(f'{name}: cached feature order differs from Common.FEAT_COLS.')
    if config is not None:
        encoder = config['encoder']
        if meta['model'] != encoder['model']:
            raise ValueError(f'{name}: cached encoder differs from the run configuration.')
        revision = encoder.get('revision')
        if revision is not None and revision not in (meta['revision'], meta.get('requested_revision')):
            raise ValueError(f'{name}: cached model revision differs from the run configuration.')
    csv_path = C.DATA / name / 'data.csv'
    if meta['input_sha256'] != C.file_hash(csv_path):
        raise ValueError(f'{name}: embeddings do not match the prepared data.')
    digest = C.file_hash(path)
    if meta['embedding_sha256'] != digest:
        raise ValueError(f'{name}: embedding file changed after extraction.')
    with np.load(path, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    if len(np.unique(data['id'])) != len(data['id']):
        raise ValueError(f'{name}: essay IDs are not unique.')
    if data['feats'].shape != (len(data['id']), len(C.FEAT_COLS)):
        raise ValueError(f'{name}: feature shape does not match essay IDs.')
    if not np.isfinite(data['emb']).all() or not np.isfinite(data['feats']).all():
        raise ValueError(f'{name}: non-finite embeddings or features.')
    if name == 'dysf' and config is not None:
        prepared = json.loads((C.DATA / name / 'metadata.json').read_text(encoding='utf-8'))
        if prepared['target'] != config['ellipse_target']:
            raise ValueError('ELLIPSE target differs between prepared data and run configuration.')
    data['_fingerprint'] = digest
    return data


def subset(data, indices):
    return {key: value[indices] for key, value in data.items() if isinstance(value, np.ndarray)}


class FeatNorm:
    def fit(self, values):
        self.mu = values.mean(0)
        self.sd = values.std(0) + 1e-6
        return self

    def transform(self, values):
        return (values - self.mu) / self.sd

    def to_dict(self):
        return {'mu': self.mu.tolist(), 'sd': self.sd.tolist()}

    @classmethod
    def from_dict(cls, value):
        obj = cls()
        obj.mu, obj.sd = np.asarray(value['mu']), np.asarray(value['sd'])
        return obj


def anchor_loss(output, targets, reduction='fixed_six'):
    predictions = output.get('anchor_pred', {})
    if not predictions:
        return targets.new_tensor(0.)
    loss = sum(F.mse_loss(pred, targets[:, C.ANCHOR_DIMS[k]]) for k, pred in predictions.items())
    return loss / (6 if reduction == 'fixed_six' else len(predictions))


@torch.inference_mode()
def predict(model, data, norm, scoremap, device, batch_size=256):
    model.eval()
    unit_scores, z_rows, decoded_rows, term_rows = [], [], [], []
    for start in range(0, len(data['score']), batch_size):
        rows = slice(start, start + batch_size)
        x = torch.as_tensor(data['emb'][rows], dtype=torch.float32, device=device)
        f = torch.as_tensor(norm.transform(data['feats'][rows]), dtype=torch.float32, device=device)
        output = model(x, feats=f)
        unit_scores.append(output['score'].cpu().numpy())
        if 'z' in output:
            z_rows.append(output['z'].cpu().numpy())
            term_rows.append(output['score_terms'].cpu().numpy())
            decoded = np.full((len(x), 19), np.nan, dtype=np.float32)
            for k, values in output['anchor_pred'].items():
                decoded[:, C.ANCHOR_DIMS[k]] = values.cpu().numpy()
            decoded_rows.append(decoded)
    unit = np.concatenate(unit_scores)
    result = {'id': data['id'], 'score_true': data['score'], 'score_pred': scoremap.from_unit(unit),
              'pred_unit': unit, 'true_features': data['feats']}
    if z_rows:
        decoded = np.concatenate(decoded_rows)
        result.update(z=np.concatenate(z_rows), decoded_standard=decoded,
                      decoded=decoded * norm.sd + norm.mu, score_terms=np.concatenate(term_rows),
                      active_mask=model.score_mask.detach().cpu().numpy().astype(bool))
    return result


def train_eval(variant, tr, va, te, scoremap_tr, config, seed, kept_groups, run_dir,
               te_scoremap=None):
    device = torch.device(config['device'])
    C.set_seed(seed)
    model_args = {'in_dim': tr['emb'].shape[1], 'hidden': config['hidden'], 'dropout': config['dropout']}
    model = make_model(variant, kept_groups=kept_groups, **model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    norm = FeatNorm().fit(tr['feats'])
    x = torch.as_tensor(tr['emb'], dtype=torch.float32)
    y = torch.as_tensor(scoremap_tr.to_unit(tr['score']), dtype=torch.float32)
    features = torch.as_tensor(norm.transform(tr['feats']), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed + 991)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed + 991)
    best, best_epoch, best_state = -float('inf'), -1, None
    history = []
    for epoch in range(config['epochs']):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        total = 0.
        for indices in order.split(config['batch_size']):
            xb, yb, fb = x[indices].to(device), y[indices].to(device), features[indices].to(device)
            output = model(xb, feats=fb)
            loss = F.mse_loss(output['score'], yb)
            loss += config['anchor_weight'] * anchor_loss(output, fb, config['anchor_reduction'])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        validation = predict(model, va, norm, scoremap_tr, device, config['batch_size'])
        vq = C.qwk(va['score'], validation['score_pred'], scoremap_tr.smin, scoremap_tr.smax, scoremap_tr.step)
        history.append({'epoch': epoch, 'train_loss': total / len(y),
                        'val_qwk': float(vq) if np.isfinite(vq) else None})
        if np.isfinite(vq) and vq > best:
            best, best_epoch = float(vq), epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= config['patience']:
            break
    if best_state is None:
        raise ValueError('Validation QWK was undefined; no checkpoint was selected.')
    model.load_state_dict(best_state)
    target_map = te_scoremap or scoremap_tr
    predictions = predict(model, te, norm, target_map, device, config['batch_size'])
    tq = C.qwk(te['score'], predictions['score_pred'], target_map.smin, target_map.smax, target_map.step)
    metrics = {'test_qwk': float(tq) if np.isfinite(tq) else None,
               'test_rmse': C.rmse(te['score'], predictions['score_pred']),
               'test_pearson': C.pearson(te['score'], predictions['score_pred']),
               'val_qwk': best, 'best_epoch': best_epoch, 'epochs_run': len(history),
               'n_train': len(tr['score']), 'n_val': len(va['score']), 'n_test': len(te['score'])}
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'model_args': model_args, 'variant': variant,
                'kept_groups': list(kept_groups), 'feat_norm': norm.to_dict(),
                'score_map': scoremap_tr.to_dict(), 'test_score_map': target_map.to_dict(),
                'config': config, 'seed': seed}, run_dir / 'model.pt')
    np.savez_compressed(run_dir / 'predictions.npz', **predictions)
    C.save_json(run_dir / 'history.json', history)
    return metrics


def load_model(run_dir, device='cpu'):
    saved = torch.load(Path(run_dir) / 'model.pt', map_location=device, weights_only=False)
    model = make_model(saved['variant'], kept_groups=saved['kept_groups'], **saved['model_args'])
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval(), saved
