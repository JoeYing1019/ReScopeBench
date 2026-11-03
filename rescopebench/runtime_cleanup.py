"""Host-only cleanup compatibility for a frozen driver's missing close call."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rescopebench import stream_engine


def install(driver) -> bool:
    if driver.case_id != "carbon_credits":
        return False
    if getattr(driver, "_owned_runtime_cleanup_v1", False):
        return True
    original = driver.close

    def close(world):
        original(world)
        root = Path(world.runtime_root)
        source = Path(world.source).resolve()
        if root.is_symlink():
            raise ValueError("refusing to clean a replaced runtime-directory symlink")
        if not root.exists():
            return
        resolved = root.resolve()
        if (not root.name.startswith("rescope_carbon_credits_")
                or resolved == source or resolved in source.parents or source in resolved.parents):
            raise ValueError("runtime directory is not an independent owned copy")
        # Only the driver's private temporary database is disposed. The engine
        # has already journaled the result and checkpoint on normal completion.
        world.close()
        if root.exists():
            raise OSError("environment close did not remove its temporary database")

    driver.close = close
    driver._owned_runtime_cleanup_v1 = True
    return True


def record(output: Path, driver) -> None:
    if not getattr(driver, "_owned_runtime_cleanup_v1", False):
        return
    receipt = {
        "schema": "rescopebench-runtime-cleanup/v1",
        "case_id": driver.case_id,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "host-owned temporary database only",
        "hook": "driver.close after normal result/checkpoint commit",
        "frozen_task_and_grader_sources_unchanged": True,
    }
    stream_engine._write_once(Path(output) / "processing-addenda/runtime-cleanup-v1.json", receipt)
