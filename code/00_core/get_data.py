"""Download the three official Kaggle datasets used by the paper.

The script uses kagglehub, which reads the same Kaggle API credentials as the
kaggle CLI. Put your kaggle.json (from https://www.kaggle.com/settings/account)
into the ~/.kaggle directory, or set KAGGLE_USERNAME and KAGGLE_KEY.

Outputs:
  data/01_raw_datasets/feat/training_set_rel3.tsv   ASAP-1
  data/01_raw_datasets/asap2/train.csv             ASAP-2.0
  data/01_raw_datasets/dysf/train.csv              Feedback Prize ELL
"""

from pathlib import Path
import shutil
import tempfile
import zipfile

import kagglehub

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "01_raw_datasets"


def _extract_if_zip(path: Path) -> Path:
    """Return a directory of files, extracting a zip when kagglehub returns one."""
    if path.is_dir():
        return path
    if path.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="kaggle_"))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        return tmp
    return path


def _find(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return matches[0]


def _download(handle: str, dest: str, filename: str) -> None:
    print(f"downloading {handle}")
    cache = Path(kagglehub.competition_download(handle))
    root = _extract_if_zip(cache)
    src = _find(root, filename)
    out = DATA / dest / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    print(f"  saved {out}")


def main() -> None:
    _download("asap-aes", "feat", "training_set_rel3.tsv")
    _download("learning-agency-lab-automated-essay-scoring-2", "asap2", "train.csv")
    _download("feedback-prize-english-language-learning", "dysf", "train.csv")


if __name__ == "__main__":
    main()
