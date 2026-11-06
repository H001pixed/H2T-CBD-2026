"""Prepare canonical data.csv + metadata.json from local official sources.

Default: train-only, without features. Select an ELLIPSE target explicitly
when dysf is included. Use Fix_Feat.py or --with-features for feat's NLP step.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from Canonical_Data import DATASETS, ORIGINAL19, canonicalize, dataset_names, text_hash


# Fixed original-release observed ranges, NOT rubric minima/maxima. Never fit
# these to the current subset or to official test labels.
ASAP1_RANGES = {
    1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3),
    5: (0, 4), 6: (0, 4), 7: (2, 24), 8: (10, 60),
}
ELLIPSE_TRAITS = ["Cohesion", "Syntax", "Vocabulary", "Phraseology", "Grammar", "Conventions"]
ELLIPSE_TARGETS = ("legacy_mean_round", "overall")


def _hash(text: str) -> str:
    return text_hash(text)


def _decode(raw: bytes, *, asap1: bool = False) -> tuple[str, str, int]:
    try:
        return raw.decode("utf-8-sig"), "utf-8-sig", 0
    except UnicodeDecodeError:
        if not asap1:
            raise
    # Match the release's cp1252 replacement policy only for ASAP-1.
    decoded = raw.decode("cp1252", errors="replace")
    return decoded, "cp1252", decoded.count("\ufffd")


def _read_csv(raw: bytes, source: str, *, asap1: bool = False) -> pd.DataFrame:
    decoded, encoding, replacements = _decode(raw, asap1=asap1)
    frame = pd.read_csv(io.StringIO(decoded), sep="\t" if asap1 else ",",
                        dtype=str, keep_default_na=False)
    frame.attrs["source"] = {
        "path": source, "encoding": encoding,
        "undefined_cp1252_bytes_replaced": replacements,
    }
    return frame


def _zip_csv(path: Path, password: str | None = None) -> pd.DataFrame:
    if password:
        import pyzipper
        archive_type = pyzipper.AESZipFile
    else:
        archive_type = zipfile.ZipFile
    with archive_type(path) as archive:
        names = sorted(name for name in archive.namelist()
                       if name.lower().endswith(".csv")
                       and "__macosx" not in name.lower()
                       and not Path(name).name.startswith("._"))
        if len(names) != 1:
            raise ValueError(f"expected one CSV in {path}, found {names}")
        raw = archive.read(names[0], pwd=password.encode("utf-8") if password else None)
    return _read_csv(raw, f"{path.name}!{names[0]}")


def _require(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{source}: missing source columns {missing}")


def _scores(frame: pd.DataFrame, columns: list[str], partition: str) -> pd.DataFrame:
    """Missing labels are allowed only for explicitly included official test."""
    if partition == "train":
        _require(frame, columns, "training labels")
    values = pd.DataFrame(index=frame.index)
    for name in columns:
        raw = frame[name] if name in frame else pd.Series("", index=frame.index)
        values[name] = pd.to_numeric(raw.replace("", np.nan), errors="raise").astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("infinite source score")
    if partition == "train" and values.isna().any().any():
        raise ValueError("missing training labels")
    return values


def _frame(raw: pd.DataFrame, source: str, partition: str, id_column: str,
           text_column: str, prompt_column: str, score: pd.Series) -> pd.DataFrame:
    _require(raw, [id_column, text_column, prompt_column], source)
    # In the author releases, set is the official split, never the prompt ID.
    if source in {"asap2", "ellipse"}:
        _require(raw, ["set"], source)
        if not raw["set"].eq(partition).all():
            raise ValueError(f"{source}: set disagrees with official source partition {partition}")
    ids = raw[id_column].astype(str)
    if ids.str.strip().eq("").any():
        raise ValueError(f"{source}: empty source ID")
    return pd.DataFrame({
        "id": source + ":" + partition + ":" + ids,
        "text": raw[text_column], "score": score,
        "essay_set": raw[prompt_column], "partition": partition,
        "_source_id": ids,
    })


def _combine(frames: list[pd.DataFrame], sources: list[dict]) -> pd.DataFrame:
    frame = pd.concat(frames, ignore_index=True)
    # Prefixing train/test must not hide repeated source identities.
    if frame["_source_id"].duplicated().any():
        ids = frame.loc[frame["_source_id"].duplicated(keep=False), "_source_id"].tolist()
        raise ValueError(f"duplicate source IDs within/across official partitions: {ids[:10]}")
    frame = frame.drop(columns="_source_id")
    frame["text_hash"] = frame["text"].map(text_hash)
    frame.attrs["sources"] = sources
    return frame


def _asap1(root: Path, include_test: bool) -> pd.DataFrame:
    src = root / "data/sources/asap1_official"
    paths = [(src / "training_set_rel3.tsv", "train")]
    if include_test:
        paths.append((src / "test_set.tsv", "test"))
    frames, sources = [], []
    for path, partition in paths:
        raw = _read_csv(path.read_bytes(), path.name, asap1=True)
        _require(raw, ["essay_id", "essay", "essay_set"], path.name)
        # Exact prompt labels; do not truncate malformed values such as 1.5.
        prompts = raw["essay_set"]
        if not prompts.isin([str(key) for key in ASAP1_RANGES]).all():
            raise ValueError(f"{path.name}: unknown ASAP-1 essay_set")
        scores = _scores(raw, ["domain1_score"], partition)["domain1_score"]
        skipped_unlabelled = 0
        if partition == "test":
            # The official ASAP-1 test source is retained on disk. Unlabelled
            # rows do not belong in the scored CSV consumed by embeddings.
            labelled = scores.notna()
            skipped_unlabelled = int((~labelled).sum())
            raw = raw.loc[labelled]
            scores = scores.loc[labelled]
            prompts = raw["essay_set"]
        lower = prompts.map({str(key): bounds[0] for key, bounds in ASAP1_RANGES.items()})
        upper = prompts.map({str(key): bounds[1] for key, bounds in ASAP1_RANGES.items()})
        normalized = np.round((scores - lower) / (upper - lower) * 10)
        frames.append(_frame(raw, "asap1", partition, "essay_id", "essay", "essay_set", normalized))
        sources.append({**raw.attrs["source"], "partition": partition,
                        "skipped_unlabelled_rows": skipped_unlabelled})
    return _combine(frames, sources)


def _asap2(root: Path, include_test: bool) -> pd.DataFrame:
    src = root / "data/sources/asap2_official"
    inputs = [(src / "ASAP_2_Final_github_train.zip", "train", None)]
    if include_test:
        inputs.append((src / "ASAP_2_Final_github_test.zip", "test", "asap2_test"))
    frames, sources = [], []
    for path, partition, password in inputs:
        raw = _zip_csv(path, password)
        scores = _scores(raw, ["score"], partition)["score"]
        frames.append(_frame(raw, "asap2", partition, "essay_id", "full_text", "prompt_name", scores))
        sources.append({**raw.attrs["source"], "partition": partition})
    return _combine(frames, sources)


def _ellipse(root: Path, target: str, include_test: bool) -> pd.DataFrame:
    if target not in ELLIPSE_TARGETS:
        raise ValueError("explicit ellipse_target must be legacy_mean_round or overall")
    src = root / "data/sources/ellipse_official"
    path = src / "ELLIPSE_Final_github_train.csv"
    inputs = [(_read_csv(path.read_bytes(), path.name), "train")]
    if include_test:
        inputs.append((_zip_csv(src / "ELLIPSE_Final_github_test.zip", "ellipse_test"), "test"))
    frames, sources = [], []
    for raw, partition in inputs:
        columns = ["Overall"] if target == "overall" else ELLIPSE_TRAITS
        values = _scores(raw, columns, partition)
        # Keep all six traits and the original NumPy half-to-even rounding.
        # skipna=False prevents incomplete test traits becoming invented labels.
        scores = values["Overall"] if target == "overall" else np.round(values.mean(axis=1, skipna=False), 0)
        frames.append(_frame(raw, "ellipse", partition, "text_id_kaggle", "full_text", "prompt", scores))
        sources.append({**raw.attrs["source"], "partition": partition})
    return _combine(frames, sources)


def _add_features(frame: pd.DataFrame) -> pd.DataFrame:
    from Features import FEATURE_NAMES, compute_features_batch
    if FEATURE_NAMES != ORIGINAL19:
        raise ValueError("feature order differs from the original release")
    values = compute_features_batch(frame["text"].tolist())
    out = frame.copy()
    for index, name in enumerate(ORIGINAL19):
        out[name] = values[:, index]
    return out


def _metadata(name: str, frame: pd.DataFrame, target: str) -> dict:
    if name == "feat":
        lower, upper, step = 0.0, 10.0, 1.0
    elif name == "asap2":
        lower, upper, step = 1.0, 6.0, 1.0
    else:
        lower, upper, step = 1.0, 5.0, 0.5 if target == "overall" else 1.0
    train_scores = frame.loc[frame["partition"].eq("train"), "score"]
    if train_scores.empty:
        raise ValueError("training partition is empty")
    grid = (train_scores.to_numpy() - lower) / step
    if ((train_scores < lower) | (train_scores > upper)).any() or not np.allclose(grid, np.round(grid), rtol=0, atol=1e-8):
        raise ValueError(f"{name}: training scores do not match target grid [{lower}, {upper}] step {step}")
    meta = {
        "dataset": name, "target": target,
        "score_min": lower, "score_max": upper, "score_step": step,
        # Legacy aliases; downstream code should prefer score_min/max/step.
        "smin": lower, "smax": upper, "step": step, "score_range": [lower, upper],
        "score_levels": [float(value) for value in np.arange(lower, upper + step / 2, step)],
        "observed_train_score_range": [float(train_scores.min()), float(train_scores.max())],
        "include_official_test": any(source["partition"] == "test" for source in frame.attrs.get("sources", [])),
        "partition_counts": {part: int(frame["partition"].eq(part).sum()) for part in ("train", "test")},
        "source_columns": {
            "id": "essay_id" if name != "dysf" else "text_id_kaggle",
            "text": "essay" if name == "feat" else "full_text",
            "essay_set": "essay_set" if name == "feat" else ("prompt_name" if name == "asap2" else "prompt"),
            "partition": "source file" if name == "feat" else "source file, checked against set",
            "score": "domain1_score" if name == "feat" else (
                "score" if name == "asap2" else ("Overall" if target == "overall" else ELLIPSE_TRAITS)),
        },
        "sources": frame.attrs.get("sources", []),
        "feature_names": [column for column in ORIGINAL19 if column in frame],
        "text_hash_algorithm": "sha256(whitespace-normalized text, UTF-8)",
        "missing_test_scores": "ASAP-1 unlabelled official-test rows excluded from scored CSV; source file unchanged",
    }
    if name == "feat":
        meta["normalization"] = {
            "formula": "numpy.round((domain1_score - lo) / (hi - lo) * 10)",
            "rounding": "half to even", "clip": False,
            "input_ranges": {str(key): list(bounds) for key, bounds in ASAP1_RANGES.items()},
            "range_source": "original-release observed ranges, fixed; not rubric maxima",
            "test_policy": "same fixed ranges; scores outside the original observed ranges are not clipped",
        }
    elif name == "dysf":
        meta["target_formula"] = ("Overall unchanged" if target == "overall" else
                                  "numpy.round(mean(Cohesion, Syntax, Vocabulary, Phraseology, Grammar, Conventions), 0)")
        meta["score_scale_source"] = ("official-train Overall levels 1.0, 1.5, ..., 5.0"
                                       if target == "overall" else "original-release rounded six-trait mean")
    return meta


def _write(root: Path, name: str, frame: pd.DataFrame, target: str) -> None:
    frame = canonicalize(frame)
    meta = _metadata(name, frame, target)
    metadata_text = json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    out = root / "data/01_raw_datasets" / name
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "data.csv", index=False, encoding="utf-8", lineterminator="\n")
    (out / "metadata.json").write_text(metadata_text, encoding="utf-8")


def prepare(root: str | Path, datasets: list[str], ellipse_target: str | None = None,
            include_official_test: bool = False, with_features: bool = False) -> None:
    root = Path(root).resolve()
    names = dataset_names(datasets)
    if "dysf" in names and ellipse_target not in ELLIPSE_TARGETS:
        raise ValueError("--ellipse-target is required for dysf: legacy_mean_round or overall")
    for name in names:
        if name == "feat":
            frame, target = _asap1(root, include_official_test), "asap1_normalized_0_10"
        elif name == "asap2":
            frame, target = _asap2(root, include_official_test), "score"
        else:
            frame, target = _ellipse(root, ellipse_target, include_official_test), ellipse_target
        # A canonical file is available even when explicit NLP extraction fails.
        _write(root, name, frame, target)
        if with_features and name == "feat":
            _write(root, name, _add_features(frame), target)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--datasets", nargs="+", choices=[*DATASETS, "all3"], default=list(DATASETS))
    parser.add_argument("--ellipse-target", choices=ELLIPSE_TARGETS)
    parser.add_argument("--include-official-test", action="store_true")
    parser.add_argument("--with-features", action="store_true", help="also compute feat's original 19 features")
    args = parser.parse_args(argv)
    if "dysf" in args.datasets or "all3" in args.datasets:
        if args.ellipse_target is None:
            parser.error("--ellipse-target is required for dysf/all3")
    prepare(args.root, args.datasets, args.ellipse_target, args.include_official_test, args.with_features)


if __name__ == "__main__":
    main()
