import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_Core'))
from Experiments import run_cli


if __name__ == '__main__':
    run_cli('selective')

