"""Remove stopped containers and unused volumes owned by ReScopeBench.

The command never performs a global Docker prune.  Containers must carry the
frozen Hermes image ownership label; volumes must carry both ReScopeBench run
and role labels.  Running containers and in-use volumes are always retained.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any


CONTAINER_LABEL = "io.rescopebench.hermes.commit"
VOLUME_LABELS = ("rescope.run", "rescope.role")
SCHEMA = "rescopebench-docker-resource-cleanup/v1"


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=check, timeout=120)


def _inspect(kind: str, identifiers: list[str]) -> list[dict[str, Any]]:
    rows = []
    for start in range(0, len(identifiers), 100):
        chunk = identifiers[start:start + 100]
        if chunk:
            rows.extend(json.loads(_run(["docker", kind, "inspect", *chunk]).stdout))
    return rows


def _created_seconds(value: str) -> float:
    rendered = value.strip().replace("Z", "+00:00")
    if "." in rendered:
        head, tail = rendered.split(".", 1)
        zone = "+00:00" if "+" not in tail and "-" not in tail else tail[tail.find("+") if "+" in tail else tail.find("-"):]
        fraction = tail[:len(tail) - len(zone)] if zone != "+00:00" or tail.endswith("Z") else tail
        rendered = f"{head}.{fraction[:6]}{zone}"
    return datetime.fromisoformat(rendered).timestamp()


def _ids(command: list[str]) -> list[str]:
    output = _run(command).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def clean(*, min_age_minutes: float, apply: bool) -> dict[str, Any]:
    cutoff = time.time() - min_age_minutes * 60
    container_ids = _ids([
        "docker", "ps", "-aq", "--filter", f"label={CONTAINER_LABEL}",
    ])
    container_candidates = []
    for row in _inspect("container", container_ids):
        labels = row.get("Config", {}).get("Labels") or {}
        state = row.get("State") or {}
        if not labels.get(CONTAINER_LABEL) or state.get("Running") is True:
            continue
        created = _created_seconds(row["Created"])
        if created > cutoff:
            continue
        container_candidates.append({
            "id": row["Id"],
            "name": str(row.get("Name", "")).lstrip("/"),
            "created": row["Created"],
            "status": state.get("Status"),
        })

    removed_containers = []
    if apply:
        for row in container_candidates:
            result = _run(["docker", "rm", "-f", row["id"]], check=False)
            if result.returncode == 0:
                removed_containers.append(row)

    volume_ids = _ids([
        "docker", "volume", "ls", "-q",
        "--filter", f"label={VOLUME_LABELS[0]}",
        "--filter", f"label={VOLUME_LABELS[1]}",
    ])
    volume_candidates = []
    for row in _inspect("volume", volume_ids):
        labels = row.get("Labels") or {}
        if any(not labels.get(label) for label in VOLUME_LABELS):
            continue
        created = _created_seconds(row["CreatedAt"])
        if created > cutoff:
            continue
        volume_candidates.append({
            "name": row["Name"],
            "created": row["CreatedAt"],
            "run": labels[VOLUME_LABELS[0]],
            "role": labels[VOLUME_LABELS[1]],
        })

    removed_volumes = []
    retained_in_use_volumes = []
    if apply:
        for row in volume_candidates:
            result = _run(["docker", "volume", "rm", row["name"]], check=False)
            if result.returncode == 0:
                removed_volumes.append(row)
            else:
                retained_in_use_volumes.append({**row, "reason": result.stderr.strip()[:500]})

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if apply else "dry_run",
        "minimum_age_minutes": min_age_minutes,
        "container_candidates": len(container_candidates),
        "containers_removed": len(removed_containers),
        "volume_candidates": len(volume_candidates),
        "volumes_removed": len(removed_volumes),
        "volumes_retained_in_use": len(retained_in_use_volumes),
        "containers": container_candidates,
        "volumes": volume_candidates,
        "in_use_volumes": retained_in_use_volumes,
        "global_prune_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-age-minutes", type=float, default=60)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.min_age_minutes < 1:
        parser.error("--min-age-minutes must be at least 1")
    result = clean(min_age_minutes=args.min_age_minutes, apply=args.apply)
    rendered = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
