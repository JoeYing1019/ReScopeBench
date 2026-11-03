"""Portable path contracts for the standalone final benchmark bundle."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


FINAL_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(
    os.environ.get("RESCOPE_BENCH_OUTPUT_ROOT", FINAL_ROOT.parent / "benchmark_outputs")
).expanduser().resolve()


def resolve_path(value: str | Path) -> Path:
    """Resolve a benchmark-relative path without allowing bundle escape."""
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Path must be relative to rescopebench: {value}")
    resolved = (FINAL_ROOT / relative).resolve()
    try:
        resolved.relative_to(FINAL_ROOT)
    except ValueError as error:
        raise ValueError(f"Path escapes rescopebench: {value}") from error
    return resolved


def relative_path(value: str | Path) -> str:
    """Encode a file or directory as a portable benchmark-relative path."""
    path = Path(value)
    resolved = (FINAL_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        relative = resolved.relative_to(FINAL_ROOT)
    except ValueError as error:
        raise ValueError(f"External dependency is not portable: {resolved}") from error
    return relative.as_posix() or "."


def normalize_inventory(files: Mapping[str, str]) -> dict[str, str]:
    """Convert an absolute source inventory into stable relative keys."""
    normalized: dict[str, str] = {}
    for filename, sha256 in files.items():
        key = relative_path(filename)
        if key in normalized and normalized[key] != sha256:
            raise ValueError(f"Conflicting hashes for portable source path: {key}")
        normalized[key] = sha256
    return dict(sorted(normalized.items()))


def inventory_paths(files: Mapping[str, str]) -> dict[Path, str]:
    """Resolve and validate a portable source inventory."""
    return {resolve_path(filename): sha256 for filename, sha256 in files.items()}
