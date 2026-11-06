"""Run the experiments used in the current manuscript."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent


def commands(root, stage, device=None):
    main = PROJECT / "configs/main.json"
    supplement = PROJECT / "configs/supplement.json"
    common = ["--root", str(root), "--config", str(main)]
    jobs = []
    if stage in ("embeddings", "all"):
        jobs.append(["Code/00_Core/Extract_Embeddings.py", *common])
        jobs.append(["Code/14_ARS_Supplement/Extract_Supplement_Embeddings.py",
                     "--root", str(root), "--config", str(supplement)])
    if stage in ("main", "all"):
        extra = ["--device", device] if device else []
        jobs.extend([
            ["Code/02_Experiments/Run_Within_Corpus.py", *common,
             "--datasets", "feat", "asap2", *extra],
            ["Code/02_Experiments/Run_Leave_One_Prompt_Out.py", *common,
             "--datasets", "feat", "asap2", "--variants",
             "blackbox", "no_anchor", "h2tcbd", "selective", *extra],
            ["Code/02_Experiments/Run_Selective_Anchoring.py", *common,
             "--datasets", "asap2", "--protocols", "P-official", *extra],
        ])
    if stage in ("supplement", "all"):
        jobs.append(["Code/14_ARS_Supplement/Run_Supplement.py", "--root", str(root),
                     "--output", str(root / "supplement"), "--config", str(supplement)])
    if stage in ("analysis", "all"):
        jobs.append(["Code/13_Result_Organization/Summarize_Manuscript.py", "--root", str(root)])
        jobs.append(["Code/14_ARS_Supplement/Analyze_Supplement.py", "--root", str(root),
                     "--output", str(root / "supplement")])
    return [[sys.executable, str(PROJECT / script), *args] for script, *args in jobs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "workspace")
    parser.add_argument("--stage", choices=["embeddings", "main", "supplement", "analysis", "all"], default="main")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Override the main scoring-model device only.")
    parser.add_argument("--list", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()
    root = args.root.resolve()
    jobs = commands(root, args.stage, args.device)
    if not args.list:
        (root / "configs").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / "configs/main.json", root / "configs/Run_Config.json")
    for job in jobs:
        print(subprocess.list2cmdline(job), flush=True)
        if not args.list:
            subprocess.run(job, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
