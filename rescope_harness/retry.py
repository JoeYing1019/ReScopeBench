"""Which unfinished arm may be rerun in place after an infrastructure failure."""

from __future__ import annotations

import json
from pathlib import Path

# Attempts per task, counting the first one.
MAX_TASK_ATTEMPTS = 5


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def retryable_arm(arm_root: Path) -> bool:
    """An unfinished arm whose current task failed on infrastructure, under the attempt cap."""
    summary = read_json(arm_root / "summary.json")
    if summary and summary.get("complete") is True:
        return False
    pointer = read_json(arm_root / "current.json")
    manifest = read_json(arm_root / pointer["checkpoint"] / "manifest.json") if pointer else None
    position = int(manifest["position"]) if manifest else 0
    attempts = sorted(arm_root.glob(f"{position + 1:04d}-*/attempt-*"))
    if not attempts or len(attempts) >= MAX_TASK_ATTEMPTS:
        return False
    failure = read_json(attempts[-1] / "failure.json")
    return bool(failure) and failure.get("status") == "infrastructure_error"
