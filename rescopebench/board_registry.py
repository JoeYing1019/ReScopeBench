"""Register benchmark runs and maintain review status for the live board."""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
from typing import Any

from rescopebench import pathing


CONFIG_SCHEMA = "rescopebench-live-board-config/v1"
REGISTRY_SCHEMA = "rescopebench-live-board-registry/v1"
CONFIG_PATH = pathing.OUTPUT_ROOT / "live-board.json"
REGISTRY_NAME = "board-registry.json"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def configured_run_root(config_path: Path | None = None) -> Path | None:
    """Resolve the active board root from the environment or host config."""
    override = os.environ.get("RESCOPE_BENCH_LIVE_RUN_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    config_path = (config_path or CONFIG_PATH).expanduser().resolve()
    if not config_path.is_file():
        return None
    config = _read_json(config_path, {})
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"invalid live-board config: {config_path}")
    value = config.get("run_root")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"live-board config has no run_root: {config_path}")
    root = Path(value).expanduser()
    return (config_path.parent / root).resolve() if not root.is_absolute() else root.resolve()


def _root(value: Path | None) -> Path:
    root = value.expanduser().resolve() if value is not None else configured_run_root()
    if root is None:
        raise ValueError(
            "no live board configured; pass --run-root or set RESCOPE_BENCH_LIVE_RUN_ROOT"
        )
    return root


def _validate_run_id(run_id: str) -> str:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run id must contain only letters, digits, '.', '_' or '-'")
    return run_id


def read_registry(run_root: Path) -> dict[str, Any]:
    value = _read_json(run_root / REGISTRY_NAME, {})
    if not isinstance(value, dict):
        return {"schema": REGISTRY_SCHEMA, "runs": {}}
    runs = value.get("runs")
    return {
        "schema": REGISTRY_SCHEMA,
        "runs": runs if isinstance(runs, dict) else {},
    }


class _RegistryLock:
    def __init__(self, run_root: Path):
        self.path = run_root / ".board-registry.lock"
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_):
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def register(
    run_root: Path,
    run: Path,
    *,
    run_id: str | None = None,
    arm: str | None = None,
) -> dict[str, Any]:
    """Idempotently expose one prepared run to the live board."""
    run_root = run_root.expanduser().resolve()
    original = run.expanduser()
    derived_id = original.name
    run = original.resolve()
    run_id = _validate_run_id(run_id or derived_id)
    if not (run / "run-plan.json").is_file():
        raise ValueError(f"cannot register an unprepared run: {run}")
    if arm is not None and arm not in {"native", "reset"}:
        raise ValueError("arm must be native or reset")

    runs = run_root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    link = runs / run_id
    with _RegistryLock(run_root):
        if link.exists() or link.is_symlink():
            if link.resolve() != run:
                raise ValueError(f"board run id already points elsewhere: {run_id}")
        elif link != run:
            temporary = runs / f".{run_id}.{os.getpid()}.tmp"
            try:
                temporary.symlink_to(run, target_is_directory=True)
                temporary.replace(link)
            finally:
                if temporary.is_symlink():
                    temporary.unlink()

        registry = read_registry(run_root)
        entries = registry["runs"]
        entry = entries.get(run_id, {})
        if not isinstance(entry, dict):
            entry = {}
        arms = entry.get("arms_started", [])
        arms = [value for value in arms if value in {"native", "reset"}]
        if arm is not None and arm not in arms:
            arms.append(arm)
        entry.update(
            {
                "target": str(run),
                "registered_at": entry.get("registered_at") or _now(),
                "last_seen_at": _now(),
                "arms_started": sorted(arms),
                "finalized": entry.get("finalized") is True,
            }
        )
        entries[run_id] = entry
        _write_json(run_root / REGISTRY_NAME, registry)
    return {"status": "registered", "run_id": run_id, **entry}


def register_if_configured(
    run: Path,
    *,
    run_root: Path | None = None,
    run_id: str | None = None,
    arm: str | None = None,
    disabled: bool = False,
) -> dict[str, Any] | None:
    if disabled:
        return None
    root = run_root.expanduser().resolve() if run_root is not None else configured_run_root()
    return register(root, run, run_id=run_id, arm=arm) if root is not None else None


def mark_finalized(
    run_root: Path,
    run_id: str,
    *,
    finalized: bool,
    note: str | None = None,
) -> dict[str, Any]:
    """Set a human review decision without changing execution evidence."""
    run_root = run_root.expanduser().resolve()
    run_id = _validate_run_id(run_id)
    link = run_root / "runs" / run_id
    if not link.is_dir():
        raise ValueError(f"board run does not exist: {run_id}")
    with _RegistryLock(run_root):
        registry = read_registry(run_root)
        entry = registry["runs"].get(run_id, {})
        if not isinstance(entry, dict):
            entry = {}
        entry.update(
            {
                "target": str(link.resolve()),
                "registered_at": entry.get("registered_at") or _now(),
                "last_seen_at": _now(),
                "arms_started": entry.get("arms_started", []),
                "finalized": finalized,
                "finalized_at": _now() if finalized else None,
            }
        )
        if note is not None:
            entry["review_note"] = note
        registry["runs"][run_id] = entry
        _write_json(run_root / REGISTRY_NAME, registry)
    return {"status": "updated", "run_id": run_id, **entry}


def unregister(run_root: Path, run_id: str) -> dict[str, Any]:
    """Remove a board entry without deleting its target run."""
    run_root = run_root.expanduser().resolve()
    run_id = _validate_run_id(run_id)
    link = run_root / "runs" / run_id
    with _RegistryLock(run_root):
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise ValueError("refusing to remove a real run directory from the board")
        registry = read_registry(run_root)
        registry["runs"].pop(run_id, None)
        _write_json(run_root / REGISTRY_NAME, registry)
    return {"status": "unregistered", "run_id": run_id}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--run-root", type=Path)
    register_parser.add_argument("--run", required=True, type=Path)
    register_parser.add_argument("--run-id")
    register_parser.add_argument("--arm", choices=("native", "reset"))

    mark_parser = subparsers.add_parser("mark")
    mark_parser.add_argument("--run-root", type=Path)
    mark_parser.add_argument("--run-id", required=True)
    state = mark_parser.add_mutually_exclusive_group(required=True)
    state.add_argument("--finalized", action="store_true")
    state.add_argument("--in-review", action="store_true")
    mark_parser.add_argument("--note")

    remove_parser = subparsers.add_parser("remove")
    remove_parser.add_argument("--run-root", type=Path)
    remove_parser.add_argument("--run-id", required=True)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--run-root", type=Path)

    args = parser.parse_args()
    root = _root(args.run_root)
    if args.command == "register":
        result = register(root, args.run, run_id=args.run_id, arm=args.arm)
    elif args.command == "mark":
        result = mark_finalized(
            root,
            args.run_id,
            finalized=args.finalized,
            note=args.note,
        )
    elif args.command == "remove":
        result = unregister(root, args.run_id)
    else:
        result = read_registry(root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
