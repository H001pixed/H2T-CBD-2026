"""Frozen, untruncated Qwen embeddings from the canonical prepared CSVs.

Resolve one Hub commit for config/tokenizer/weights, scan every input before
loading weights, and reuse only caches with matching provenance and row arrays.
No instruction or chat template is added to essay text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("id", "text", "score", "essay_set", "partition", "text_hash")
TEXT_COLUMNS = ("id", "text", "essay_set", "partition", "text_hash")
DEFAULT_CONFIG = "configs/Run_Config.json"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"
# The model card advertises 32K; a larger max_tokens must not silently extend it.
SUPPORTED_CONTEXT = {DEFAULT_MODEL: 32768}
CACHE_SCHEMA_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dataset_names(value: Any) -> list[str]:
    if value is None:
        names = ["feat", "asap2", "dysf"]
    elif isinstance(value, str):
        names = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        names = value
    else:
        raise ValueError("datasets must be a comma-separated string or a JSON string list")
    if not names or any(not isinstance(name, str) or
                        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None
                        for name in names):
        raise ValueError("Dataset names must be nonempty simple directory names")
    if len(set(names)) != len(names):
        raise ValueError("Dataset names must be unique")
    return names


def _resolve_inputs(root: Path, names: list[str]) -> dict[str, Path]:
    paths = {name: root / "data" / "01_raw_datasets" / name / "data.csv" for name in names}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prepared dataset(s): " + ", ".join(missing))
    return paths


def _read_dataset(path: Path, feat_cols: list[str]) -> pd.DataFrame:
    # Preserve literal text/IDs, leading zeroes, and strings such as "NA".
    frame = pd.read_csv(path, dtype={name: str for name in TEXT_COLUMNS},
                        keep_default_na=False)
    missing = [column for column in (*REQUIRED_COLUMNS, *feat_cols) if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"Dataset contains no rows: {path}")
    for column in ("id", "text", "partition", "text_hash"):
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Empty {column} in {path}")
    if frame["id"].duplicated().any():
        raise ValueError(f"Duplicate essay IDs in {path}")
    for column in ("score", *feat_cols):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame["score"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Scores must be finite in {path}")
    if not np.isfinite(frame[feat_cols].to_numpy(dtype=np.float32)).all():
        raise ValueError(f"All original features must be finite float32 values in {path}")
    return frame


def _resolve_revision(model_id: str, revision: str) -> str:
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(model_id)
    if Path(model_id).exists():
        raise ValueError("Local model paths cannot establish Hub revision provenance; use a Hub model ID")
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    from huggingface_hub import HfApi

    # Resolve a branch/tag exactly once, with no fallback to main or another model.
    commit = HfApi().model_info(model_id, revision=revision).sha
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise ValueError("Hub did not return an immutable model commit SHA")
    return commit.lower()


def _tokenize(tokenizer: Any, texts: list[str], *, padding: bool) -> Any:
    # Both passes use the same special-token policy and the exact original text.
    return tokenizer(texts, add_special_tokens=True, padding=padding,
                     truncation=False, return_attention_mask=True,
                     return_tensors="pt" if padding else None)


def _scan_lengths(tokenizer: Any, texts: list[str]) -> np.ndarray:
    lengths: list[int] = []
    for start in range(0, len(texts), 128):
        encoded = _tokenize(tokenizer, texts[start:start + 128], padding=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return np.asarray(lengths, dtype=np.int64)


def _row_arrays(frame: pd.DataFrame, feat_cols: list[str],
                lengths: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "id": frame["id"].to_numpy(dtype="U"),
        "text_hash": frame["text_hash"].to_numpy(dtype="U"),
        "score": frame["score"].to_numpy(dtype=np.float64),
        "essay_set": frame["essay_set"].to_numpy(dtype="U"),
        "partition": frame["partition"].to_numpy(dtype="U"),
        "feats": frame[feat_cols].to_numpy(dtype=np.float32),
        "token_lengths": lengths,
    }


def _cache_error(metadata_path: Path, reason: str) -> ValueError:
    return ValueError(f"Invalid embedding cache {metadata_path}: {reason}; use --overwrite to replace it")


def _check_cache(metadata_path: Path, expected: dict[str, Any], npz_path: Path,
                 arrays: dict[str, np.ndarray]) -> bool:
    if not metadata_path.exists() and not npz_path.exists():
        return False
    if not metadata_path.is_file() or not npz_path.is_file():
        raise _cache_error(metadata_path, "cache files are incomplete")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _cache_error(metadata_path, f"metadata unreadable ({exc})") from exc
    if not isinstance(metadata, dict):
        raise _cache_error(metadata_path, "metadata must be a JSON object")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise _cache_error(metadata_path, f"{key} does not match current inputs/encoder")
    # This is an output checksum, not a member of the input fingerprint.
    checksum = metadata.get("embedding_sha256")
    if not isinstance(checksum, str) or _sha256(npz_path) != checksum:
        raise _cache_error(metadata_path, "embedding_sha256 does not match the NPZ")
    with np.load(npz_path, allow_pickle=False) as cached:
        for key, value in arrays.items():
            if key not in cached or cached[key].dtype != value.dtype or not np.array_equal(cached[key], value):
                raise _cache_error(metadata_path, f"{key} is missing or differs from the prepared rows")
        if "emb" not in cached:
            raise _cache_error(metadata_path, "emb is missing")
        emb = cached["emb"]
        if (emb.shape != (expected["rows"], expected["embedding_dim"])
                or emb.dtype != np.float32 or not np.isfinite(emb).all()):
            raise _cache_error(metadata_path, "emb has invalid shape, dtype, or values")
        if not np.allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-4, atol=1e-5):
            raise _cache_error(metadata_path, "emb is not L2-normalized")
    return True


def _pool_full_text(model: Any, tokenizer: Any, texts: list[str], batch_size: int,
                    device: Any, context_limit: int, lengths: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = _tokenize(tokenizer, texts[start:start + batch_size], padding=True)
            attention = batch["attention_mask"]
            token_counts = attention.sum(dim=1).numpy()
            if not np.array_equal(token_counts, lengths[start:start + batch_size]):
                raise ValueError(f"Token lengths changed between preflight and encoding at row {start}")
            if attention.shape[1] > context_limit or not bool(attention[:, -1].eq(1).all()):
                raise ValueError(f"Invalid context length or left padding at row {start}")
            batch = {name: value.to(device) for name, value in batch.items()}
            hidden = model(**batch, use_cache=False, return_dict=True,
                           output_hidden_states=False, output_attentions=False).last_hidden_state
            # With verified left padding, every final position is a real token.
            pooled = hidden[:, -1].float()
            norms = pooled.norm(dim=1)
            if (not bool(torch.isfinite(pooled).all()) or not bool(torch.isfinite(norms).all())
                    or bool(norms.le(1e-12).any())):
                raise ValueError(f"Nonfinite or zero embedding at row {start}")
            rows.append(F.normalize(pooled, p=2, dim=1).cpu().numpy())
    return np.concatenate(rows).astype(np.float32, copy=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[2]
    config_path = args.config if args.config is not None else Path(DEFAULT_CONFIG)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    config = _load_config(config_path)
    encoder = config.get("encoder", {})
    if not isinstance(encoder, dict):
        raise ValueError("encoder must be a JSON object")
    model_id = args.model if args.model is not None else encoder.get("model", DEFAULT_MODEL)
    revision = args.revision if args.revision is not None else encoder.get("revision")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("encoder.model must be a nonempty Hugging Face model ID")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("A model revision is required via --revision or config encoder.revision")
    batch_size = _positive_int(args.batch_size if args.batch_size is not None
                               else encoder.get("batch_size", 1), "batch_size")
    configured_context = _positive_int(encoder.get("max_tokens", 32768), "max_tokens")
    device_name = args.device if args.device is not None else encoder.get("device", "cuda")
    if not isinstance(device_name, str):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    names = _dataset_names(args.datasets if args.datasets is not None else config.get("datasets"))

    # Share the feature order used by the training engine.
    from Common import FEAT_COLS

    feat_cols = list(FEAT_COLS)
    if (len(feat_cols) != 19 or any(not isinstance(col, str) for col in feat_cols)
            or len(set(feat_cols)) != 19 or set(feat_cols).intersection(REQUIRED_COLUMNS)):
        raise ValueError("Common.FEAT_COLS must contain the 19 unique original feature names")
    inputs = _resolve_inputs(root, names)
    input_hashes = {name: _sha256(path) for name, path in inputs.items()}
    frames = {name: _read_dataset(path, feat_cols) for name, path in inputs.items()}
    for name, path in inputs.items():
        if _sha256(path) != input_hashes[name]:
            raise ValueError(f"Prepared CSV changed while reading: {path}")

    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import torch

    device = torch.device(device_name)
    if device.type not in {"cpu", "cuda"} or (device.type == "cpu" and device.index is not None):
        raise ValueError("Only cpu, cuda, and cuda:<index> are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; choose --device cpu explicitly")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Requested CUDA device does not support bfloat16")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    resolved_revision = _resolve_revision(model_id, revision)
    model_config = AutoConfig.from_pretrained(model_id, revision=resolved_revision, trust_remote_code=False)
    model_context = _positive_int(getattr(model_config, "max_position_embeddings", None),
                                   "model max_position_embeddings")
    embedding_dim = _positive_int(getattr(model_config, "hidden_size", None), "model hidden_size")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=resolved_revision,
                                               padding_side="left", trust_remote_code=False)
    if tokenizer.padding_side != "left" or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must provide a pad token and support left padding")
    tokenizer_context = _positive_int(tokenizer.model_max_length, "tokenizer model_max_length")
    context_limits = {"configured": configured_context, "model": model_context,
                      "tokenizer": tokenizer_context}
    if model_id in SUPPORTED_CONTEXT:
        context_limits["supported"] = SUPPORTED_CONTEXT[model_id]
    context_limit = min(context_limits.values())
    lengths_by_name = {name: _scan_lengths(tokenizer, frame["text"].tolist())
                       for name, frame in frames.items()}
    violations = []
    for name, lengths in lengths_by_name.items():
        bad = np.flatnonzero((lengths < 1) | (lengths > context_limit))
        if bad.size:
            violations.append(f"{name}: {bad.size} rows; first id={frames[name].iloc[int(bad[0])]['id']!r}, "
                              f"tokens={int(lengths[bad[0]])}")
    if violations:
        raise ValueError(f"Full-text token lengths must be 1..{context_limit} ({context_limits}); "
                         "no truncation is allowed. " + "; ".join(violations))

    encoder_parameters = {
        "pooling": "last_token", "normalization": "l2_float32", "padding_side": "left",
        "add_special_tokens": True, "truncation": False, "instruction": None,
        "context_limits": context_limits, "effective_max_tokens": context_limit,
        "dtype": str(dtype), "device": str(device), "batch_size": batch_size,
        "attn_implementation": "sdpa", "use_cache": False,
    }
    runtime_versions = {package: version(package) for package in
                        ("torch", "transformers", "tokenizers", "huggingface-hub", "numpy", "pandas")}
    output_dir = root / "results" / "embeddings"
    pending = {}
    for name, frame in frames.items():
        lengths = lengths_by_name[name]
        arrays = _row_arrays(frame, feat_cols, lengths)
        expected = {
            "cache_schema_version": CACHE_SCHEMA_VERSION, "input_sha256": input_hashes[name],
            "model": model_id, "revision": resolved_revision, "feature_columns": feat_cols,
            "rows": len(frame), "embedding_dim": embedding_dim, "max_tokens": int(lengths.max()),
            "encoder_parameters": encoder_parameters, "runtime_versions": runtime_versions,
        }
        if not args.overwrite and _check_cache(output_dir / f"{name}.json", expected,
                                               output_dir / f"{name}.npz", arrays):
            continue
        pending[name] = (expected, arrays)
    if not pending:
        return

    # Only now may the 8B weights be downloaded/loaded: all CSVs, full lengths,
    # actual context bounds, and existing caches have already been checked.
    model = AutoModel.from_pretrained(model_id, revision=resolved_revision, config=model_config,
                                      torch_dtype=dtype, attn_implementation="sdpa",
                                      trust_remote_code=False).to(device).eval()
    model.requires_grad_(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (expected, arrays) in pending.items():
        emb = _pool_full_text(model, tokenizer, frames[name]["text"].tolist(), batch_size,
                              device, context_limit, lengths_by_name[name])
        if emb.shape != (len(frames[name]), embedding_dim):
            raise ValueError(f"Unexpected embedding shape for {name}: {emb.shape}")
        if _sha256(inputs[name]) != input_hashes[name]:
            raise ValueError(f"Prepared CSV changed during encoding: {inputs[name]}")
        npz_path = output_dir / f"{name}.npz"
        np.savez_compressed(npz_path, emb=emb, **arrays)
        metadata = {**expected, "requested_revision": revision, "source_config": config,
                    "embedding_sha256": _sha256(npz_path)}
        (output_dir / f"{name}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
