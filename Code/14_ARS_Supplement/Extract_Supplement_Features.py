"""Extract the original 19 linguistic features for the two supplement datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "Code" / "01_Features"))
from Features import FEATURE_NAMES, compute_features  # noqa: E402
from Canonical_Data import text_hash  # noqa: E402

DATASETS = ("persuade_nonoverlap", "cela_single_prompt")
BASE_COLUMNS = ["id", "text", "score", "essay_set", "partition", "text_hash"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--root", default=str(PROJECT / "workspace"), help="Workspace containing data/01_raw_datasets")
    ap.add_argument("--work-dir", help="Backup and log directory; defaults to ars-supplement under the system temporary directory")
    args = ap.parse_args()
    root = Path(args.root) / "data" / "01_raw_datasets"
    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.gettempdir()) / "ars-supplement"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "supplement_extract.log"
    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, value):
            for stream in self.streams: stream.write(value); stream.flush()
        def flush(self):
            for stream in self.streams: stream.flush()
    log_stream = log_path.open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_stream)
    print(f"LOG {log_path}", flush=True)
    for name in args.datasets:
        folder = root / name
        csv_path, meta_path = folder / "data.csv", folder / "metadata.json"
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if frame.columns[:6].tolist() != BASE_COLUMNS:
            raise ValueError(f"{name}: unexpected base columns/order: {frame.columns.tolist()}")
        if frame[BASE_COLUMNS].isna().any().any():
            raise ValueError(f"{name}: missing base field")
        if not frame["text_hash"].eq(frame["text"].map(text_hash)).all():
            raise ValueError(f"{name}: text_hash mismatch")
        before_hash = sha256(csv_path)
        complete = all(c in frame.columns for c in FEATURE_NAMES)
        if complete and frame.columns[6:25].tolist() == FEATURE_NAMES:
            vals = frame[FEATURE_NAMES].apply(pd.to_numeric, errors="raise").to_numpy(float)
            if np.isfinite(vals).all():
                print(f"SKIP {name}: complete 19 features; csv_sha256={before_hash}", flush=True)
                continue
        extras = [c for c in frame.columns if c not in BASE_COLUMNS + FEATURE_NAMES]
        if extras:
            raise ValueError(f"{name}: unexpected columns prevent safe append: {extras}")
        original = frame[BASE_COLUMNS].copy(deep=True)
        backup = work_dir / f"{name}.data.before.csv"
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(csv_path.read_bytes())
        print(f"START {name}: rows={len(frame)} source_sha256={before_hash} backup={backup}", flush=True)
        rows = []
        for i, value in enumerate(frame["text"], 1):
            feats = compute_features(value)
            rows.append([feats[k] for k in FEATURE_NAMES])
            if i % 100 == 0 or i == len(frame):
                print(f"PROGRESS {name}: {i}/{len(frame)}", flush=True)
        arr = np.asarray(rows, dtype=np.float64)
        if arr.shape != (len(frame), 19) or not np.isfinite(arr).all():
            raise ValueError(f"{name}: extracted feature shape/value validation failed")
        out = pd.concat([original, pd.DataFrame(arr, columns=FEATURE_NAMES), frame.drop(columns=BASE_COLUMNS + FEATURE_NAMES, errors="ignore")], axis=1)
        if not out[BASE_COLUMNS].equals(original):
            raise ValueError(f"{name}: base fields changed before write")
        fd, temp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=folder)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            out.to_csv(temp_path, index=False, encoding="utf-8-sig", lineterminator="\n")
            check = pd.read_csv(temp_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            if check.shape[0] != len(frame) or not check[BASE_COLUMNS].equals(original):
                raise ValueError(f"{name}: temporary CSV base-field verification failed")
            checked = check[FEATURE_NAMES].apply(pd.to_numeric, errors="raise").to_numpy(float)
            if checked.shape != (len(frame), 19) or not np.isfinite(checked).all():
                raise ValueError(f"{name}: temporary feature verification failed")
            os.replace(temp_path, csv_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["feature_names"] = FEATURE_NAMES
        metadata["status"] = "Original 19 linguistic features extracted; source labels and row order preserved; no ARS, split, or training performed"
        if name == "cela_single_prompt":
            metadata["score_range"] = [1, 5]
            metadata["score_step"] = 0.5
        meta_tmp = meta_path.with_suffix(".json.tmp")
        meta_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(meta_tmp, meta_path)
        print(f"DONE {name}: rows={len(out)} features=19 csv_sha256={sha256(csv_path)}", flush=True)


if __name__ == "__main__":
    main()
