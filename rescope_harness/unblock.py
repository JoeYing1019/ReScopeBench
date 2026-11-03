"""Restart arms a batch controller has given up on while nothing else runs them.

batch_eval keeps an arm "blocked" for the rest of the controller's life once its
process exits non-zero. Arms blocked by a transient condition (another runner held
the lock, the disk was full) would otherwise idle until every other arm finishes.
For each blocked, unfinished arm with no live runner this starts
`python -m rescope_harness supervised-retry` detached, with the current environment;
the controller sees the arm complete when it is done.

    python -m rescope_harness.unblock --controller <batch_eval --output dir> \
        --credentials-file <file> [--dry-run]

Run it with the same environment variables as the batch (model API, sandbox).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PKG = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", required=True, type=Path)
    parser.add_argument("--credentials-file", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(PKG))
    from rescopebench.batch_eval import running

    status = json.loads((args.controller / "status.json").read_text())
    log_dir = args.controller / "unblock"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = 0
    for row in status["arms"]:
        if row["state"] != "blocked":
            continue
        arm_root = Path(row["output"]) / row["arm"]
        summary = arm_root / "summary.json"
        if summary.is_file() and json.loads(summary.read_text()).get("complete") is True:
            continue
        if running(arm_root):
            continue
        command = [sys.executable, "-m", "rescope_harness", "supervised-retry",
                   "--case", row["case"], "--output", row["output"], "--arm", row["arm"],
                   "--execution-mode", row["execution_mode"], "--credentials-file",
                   str(Path(args.credentials_file).resolve()), "--retry-infrastructure"]
        if row.get("binding"):
            at = command.index("--execution-mode")
            command[at:at] = ["--binding", row["binding"]]
        print(("would start " if args.dry_run else "start ") + f"{row['case']}/{row['arm']}")
        if args.dry_run:
            continue
        log = open(log_dir / f"{row['case']}-{row['arm']}.log", "a")
        subprocess.Popen(command, cwd=PKG, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        started += 1
    print(f"started {started}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
