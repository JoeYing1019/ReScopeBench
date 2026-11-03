"""Integrity checks for one installed, frozen environment implementation.

These checks bind executable files to the selected contract. They do not infer
learning quality from reference execution or add conditions to task grading.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import re

from rescopebench import pathing


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def checked_member(root: Path, name: str) -> Path:
    if not isinstance(name, str):
        raise ValueError("case member name must be a string")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or not relative.parts or ".." in relative.parts
            or relative.as_posix() != name or "\\" in name):
        raise ValueError("unsafe case package member")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("case package member escapes its root")
    for member in (path, *path.parents):
        if member == root:
            break
        if member.is_symlink():
            raise ValueError("case package members cannot be symlinks")
    return path


class ReleasedCase:
    """Common file and contract validation, independent of business behavior."""

    def _case_directory(self) -> Path:
        return Path(inspect.getfile(type(self))).resolve().parent

    def _manifest(self):
        root = self._case_directory()
        manifest = json.loads((root / "case-manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("schema") != "rescopebench-case-files/v1"
                or manifest.get("case_id") != self.case_id
                or manifest.get("case_version") != self.case_version):
            raise ValueError("case manifest identity differs")
        for key in ("files", "shared_sources"):
            records = manifest.get(key)
            if not isinstance(records, dict) or not records:
                raise ValueError("case manifest has no " + key)
            if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in records.values()):
                raise ValueError("invalid case manifest checksum")
        return manifest

    def source_files(self) -> list[Path]:
        manifest = self._manifest()
        root = self._case_directory()
        paths = {root / "case-manifest.json"}
        paths.update(checked_member(root, name) for name in manifest["files"])
        paths.update(checked_member(pathing.FINAL_ROOT, name) for name in manifest["shared_sources"])
        if any(not path.is_file() for path in paths):
            raise ValueError("case package contains a missing source or fixture")
        return sorted(paths)

    def validate(self) -> dict:
        root = self._case_directory()
        manifest = self._manifest()
        errors = []
        for base, records in ((root, manifest["files"]),
                              (pathing.FINAL_ROOT, manifest["shared_sources"])):
            for name, expected in records.items():
                path = checked_member(base, name)
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    errors.append("source or fixture checksum mismatch: " + name)
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
                  if path.is_file() and path.name != "case-manifest.json"
                  and "__pycache__" not in path.parts}
        if actual != set(manifest["files"]):
            errors.append("case package inventory differs")
        contract = json.loads((root / "data/contract.json").read_text(encoding="utf-8"))
        reference = json.loads((root / "data/reference.json").read_text(encoding="utf-8"))
        world = self.create_world()
        try:
            initial = world.snapshot()
            world.restore(deepcopy(initial))
            restored = world.snapshot()
        finally:
            self.close(world)
        bound = getattr(self, "binding_spec", None)
        expected_binding = json.loads(checked_member(root, contract["binding"]).read_text(encoding="utf-8"))
        ids = [task["private_task_id"] for task in self.tasks]
        policies = reference.get("policies", {})
        checks = {
            "source_and_data_integrity": not errors,
            "contract_identity": contract.get("schema") == "rescopebench-case-contract/v1"
                and contract.get("case_id") == self.case_id and contract.get("case_version") == self.case_version,
            "task_count": len(self.tasks) == contract["task_count"],
            "public_task_contract": digest(self.tasks) == contract["tasks_sha256"],
            "tool_contract": digest(self.tools) == contract["tools_sha256"],
            "prompt_contract": digest(self.prompt) == contract["prompt_sha256"],
            "lifecycle_schedule": digest(self.lifecycle_schedule) == contract["schedule_sha256"],
            "frozen_initial_state": digest(initial) == contract["initial_sha256"],
            "initial_snapshot_round_trip": initial == restored,
            "released_data_directory": self.directory == root / "data",
            "loaded_binding_exact": bound is None or bound == expected_binding,
            "reference_coverage": reference.get("schema") == "rescopebench-reference-calls/v1"
                and reference.get("case_id") == self.case_id
                and reference.get("case_version") == self.case_version and bool(policies)
                and all([row["task"] for row in rows] == ids for rows in policies.values()),
        }
        return {"status": "passed" if all(checks.values()) else "failed",
                "case_version": self.case_version, "checks": checks, "errors": errors,
                "scope": "Frozen package integrity and reference coverage, not natural model performance."}
