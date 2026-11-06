"""Locate local official sources; download missing files only with --download.

ASAP-1 is a Kaggle competition release. ASAP-2.0 and ELLIPSE are the author
repository releases recorded in data/sources/source_manifest.csv; the Kaggle
competition counterparts have different filenames/schemas.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen


DATASETS = ("feat", "asap2", "dysf")
AUTHOR_SOURCES = {
    "asap2": ("asap2_official", "https://raw.githubusercontent.com/scrosseye/ASAP_2.0/main/",
              "ASAP_2_Final_github_train.zip", "ASAP_2_Final_github_test.zip"),
    "dysf": ("ellipse_official", "https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/",
             "ELLIPSE_Final_github_train.csv", "ELLIPSE_Final_github_test.zip"),
}


def _find(root: Path, filename: str) -> Path:
    if root.is_file():
        if root.name == filename:
            return root
        raise FileNotFoundError(f"expected {filename}, got {root}")
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {filename} under {root}, found {len(matches)}")
    return matches[0]


def _kaggle_file(cache: Path, filename: str, target: Path) -> None:
    if cache.is_file() and cache.suffix.lower() == ".zip":
        # Read only the requested member, without extractall/path traversal.
        with zipfile.ZipFile(cache) as archive:
            members = [name for name in archive.namelist()
                       if Path(name).name == filename and "__macosx" not in name.lower()]
            if len(members) != 1:
                raise FileNotFoundError(f"expected one {filename} in {cache}")
            with archive.open(members[0]) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    else:
        shutil.copyfile(_find(cache, filename), target)


def main(root: str | Path | None = None, datasets: list[str] | None = None,
         include_official_test: bool = False, download: bool = False) -> list[Path]:
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    names = list(DATASETS) if datasets is None or datasets == ["all3"] else list(datasets)
    if not names or any(name not in DATASETS for name in names) or len(set(names)) != len(names):
        raise ValueError("select unique feat/asap2/dysf datasets, or all3 on its own")
    requested = []
    for name in names:
        if name == "feat":
            files = ["training_set_rel3.tsv"] + (["test_set.tsv"] if include_official_test else [])
            requested.extend((name, root / "data/sources/asap1_official" / filename, None) for filename in files)
        else:
            directory, base_url, train_file, test_file = AUTHOR_SOURCES[name]
            files = [train_file] + ([test_file] if include_official_test else [])
            requested.extend((name, root / "data/sources" / directory / filename, base_url + filename) for filename in files)
    missing = [str(path) for _, path, _ in requested if not path.is_file()]
    if missing and not download:
        raise FileNotFoundError("missing local official sources (use --download explicitly to fetch): " + ", ".join(missing))
    kaggle_cache = None
    for name, path, url in requested:
        if path.is_file():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # Stage in the destination directory; incomplete downloads never masquerade
        # as complete local sources on a subsequent invocation.
        with tempfile.TemporaryDirectory(prefix=".source-", dir=path.parent) as staging:
            temporary = Path(staging) / path.name
            if name == "feat":
                if kaggle_cache is None:
                    import kagglehub
                    kaggle_cache = Path(kagglehub.competition_download("asap-aes"))
                _kaggle_file(kaggle_cache, path.name, temporary)
            else:
                with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if temporary.stat().st_size == 0:
                raise ValueError(f"empty downloaded source: {path.name}")
            temporary.replace(path)
    return [path for _, path, _ in requested]


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", choices=[*DATASETS, "all3"], default=list(DATASETS))
    parser.add_argument("--include-official-test", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    main(args.root, args.datasets, args.include_official_test, args.download)


if __name__ == "__main__":
    cli()
