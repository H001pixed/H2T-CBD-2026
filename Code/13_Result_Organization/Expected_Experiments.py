"""Expected identities for the manuscript's ASAP-1 and ASAP-2.0 experiments."""
import csv
from pathlib import Path

SEEDS = [13, 42, 123, 2026, 2027, 2028, 2029, 2030, 2031, 2032]
DATASETS = ("feat", "asap2")


def train_folds(root: Path):
    folds = {}
    for dataset in DATASETS:
        path = root / "data/01_raw_datasets" / dataset / "data.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            folds[dataset] = sorted({row["essay_set"] for row in csv.DictReader(handle)
                                     if row["partition"] == "train"})
    return folds


def expected_set(folds):
    result = set()
    for dataset in DATASETS:
        for protocol, selected_folds, variants in [
            ("P-in", [None], ("blackbox", "no_anchor", "h2tcbd", "featconcat")),
            ("P-LOPO", folds[dataset], ("blackbox", "no_anchor", "h2tcbd", "selective")),
        ]:
            for fold in selected_folds:
                for variant in variants:
                    for seed in SEEDS:
                        result.add((dataset, dataset, protocol, variant, fold, seed, 0.1))
    for seed in SEEDS:
        for variant in ("h2tcbd", "selective"):
            result.add(("asap2", "asap2", "P-official", variant, None, seed, 0.1))
    return result
