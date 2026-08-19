"""Streaming epoch-level metrics logger for training runs."""
from __future__ import annotations

import json
import time
from pathlib import Path


def log_epoch(exp_dir, **metrics) -> None:
    """Append one line to <exp_dir>/metrics.jsonl."""
    d = Path(exp_dir)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"wall_ts": time.time(), **metrics}
    line = json.dumps(rec, ensure_ascii=False)
    with (d / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


class MetricsLogger:
    """Thin wrapper that remembers exp_dir so callers don't repeat it."""

    def __init__(self, exp_dir):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def log(self, **metrics) -> None:
        log_epoch(self.exp_dir, **metrics)

    def done(self, **final) -> None:
        final = {"wall_ts": time.time(), **final}
        (self.exp_dir / "metrics_final.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
        )
