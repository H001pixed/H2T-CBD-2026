"""Append the original 19 features without changing row identities.

Accept a canonical CSV path, or --root and --datasets (default: feat).
No normalization, downloading, dataset preparation, or cache reuse occurs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Canonical_Data import DATASETS, ORIGINAL19, REQUIRED, dataset_names, read_canonical


def fix(path: str | Path) -> None:
    path = Path(path)
    frame = read_canonical(path)
    metadata_path = path.with_name("metadata.json")
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"{metadata_path}: expected a JSON object")
    # Lazy NLP import keeps argument parsing and canonical validation independent.
    from Features import FEATURE_NAMES, compute_features_batch
    if FEATURE_NAMES != ORIGINAL19:
        raise ValueError("feature order differs from the original release")
    values = compute_features_batch(frame["text"].tolist())
    if values.shape != (len(frame), len(ORIGINAL19)):
        raise ValueError("feature matrix shape does not match the input rows")
    for index, name in enumerate(ORIGINAL19):
        frame[name] = values[:, index]
    extras = [name for name in frame if name not in REQUIRED + ORIGINAL19]
    frame = frame[REQUIRED + ORIGINAL19 + extras]
    metadata_text = None
    if metadata is not None:
        metadata["feature_names"] = list(ORIGINAL19)
        metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # Complete extraction/validation before touching the original CSV.
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    if metadata_text is not None:
        metadata_path.write_text(metadata_text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", choices=[*DATASETS, "all3"], default=None)
    args = parser.parse_args(argv)
    if args.path is not None:
        if args.root is not None or args.datasets is not None:
            parser.error("use a CSV path OR --root/--datasets")
        paths = [args.path]
    else:
        root = args.root or Path(__file__).resolve().parents[2]
        names = dataset_names(args.datasets if args.datasets is not None else ["feat"])
        paths = [root / "data/01_raw_datasets" / name / "data.csv" for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing prepared CSV(s): " + ", ".join(missing))
    for path in paths:
        fix(path)


if __name__ == "__main__":
    main()
