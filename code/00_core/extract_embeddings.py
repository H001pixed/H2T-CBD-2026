"""Extract 768d embeddings from frozen DeBERTa-v3-base for all three datasets.

This is the only GPU-heavy step. Results are cached as .npz files in
runs/_emb/ — all downstream experiments train on these cached embeddings.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

import common as C
from metrics_logger import MetricsLogger

MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 512
BATCH_SIZE = 64


@torch.no_grad()
def embed_texts(model, tok, texts, device, ml=None, tag=""):
    """Mean-pool token embeddings; texts longer than MAX_LEN are truncated."""
    embs = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i + BATCH_SIZE]
        enc = tok(chunk, max_length=MAX_LEN, truncation=True, padding=True,
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs[i:i + len(chunk)] = pooled.float().cpu().numpy()
        if ml is not None and (i // BATCH_SIZE) % 10 == 0:
            done = i + len(chunk)
            rate = done / max(time.time() - t0, 1e-6)
            ml.log(stage=tag, done=done, total=len(texts), rate_per_s=round(rate, 1))
    return embs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    C.EMB.mkdir(parents=True, exist_ok=True)
    ml = MetricsLogger(C.RUNS / "_emb_extract")

    todo = [n for n in C.DATASETS if not (C.EMB / f"{n}.npz").exists()]
    if not todo:
        print("all embeddings cached, skip")
        return 0

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    if device == "cuda":
        model = model.half()

    for name in C.DATASETS:
        outp = C.EMB / f"{name}.npz"
        if outp.exists():
            print(f"{name}: cached, skip")
            continue
        df = C.load_dataset(name)
        print(f"{name}: embedding {len(df)} texts ...")
        t0 = time.time()
        emb = embed_texts(model, tok, df["text"].tolist(), device, ml=ml, tag=name)
        save = {"emb": emb, "score": df["score"].values.astype(np.int64)}
        if name == "feat":
            save["essay_set"] = df["essay_set"].values.astype(np.int64)
            save["feats"] = df[C.FEAT_COLS].values.astype(np.float32)
        np.savez_compressed(outp, **save)
        dt = time.time() - t0
        print(f"{name}: done {emb.shape} in {dt:.1f}s -> {outp.name}")
        ml.log(stage=name, status="done", seconds=round(dt, 1), shape=list(emb.shape))

    ml.done(status="all_done", datasets=list(C.DATASETS))
    print("EMBEDDING EXTRACTION COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
