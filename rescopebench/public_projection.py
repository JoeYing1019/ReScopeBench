"""Canonical hashes and the public/private boundary for benchmark inputs."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


_SEMANTIC_DRIFT_MARKERS = re.compile(
    r"(?i)(?:\bpre[-_ ]?drift\b|\bpost[-_ ]?drift\b|\bdrift[ _-]?onset\b|"
    r"\bchanged[ _-]?atom\b|\bpreserved[ _-]?atom\b)"
)
_PRIVATE_EPISODE_RE = re.compile(
    r"(?<![A-Za-z0-9])T(?:0[1-9]|1[0-6])(?![A-Za-z0-9]|:[0-5][0-9])",
    re.IGNORECASE,
)
_PRIVATE_ARM_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:native|reset|reflect)\d{2}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_PRIVATE_RUN_PATH_RE = re.compile(
    r"(?i)(?:benchmark_outputs[/\\]|(?<![A-Za-z0-9_])runs[/\\][A-Za-z0-9_.-]+|"
    r"attempt-\d{4,})"
)
_PRIVATE_KEYS = {
    "answer",
    "arm",
    "arm_policy",
    "changed_atoms",
    "expected_score",
    "golden",
    "golden_trace",
    "lifecycle_position",
    "next_episode_id",
    "position",
    "preserved_atoms",
    "previous_episode_id",
    "private_position",
    "private_episode_id",
    "private_task_id",
    "rubric",
    "rubric_weights",
    "score_weights",
    "structure_atoms",
    "rule_atoms",
    "run_path",
}


class PacketContractError(ValueError):
    """Raised when a learner-visible packet exposes evaluator-only metadata."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _walk(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")


def assert_hard_public_projection(value: Any) -> None:
    """Reject unambiguous private lifecycle and evaluation data on public surfaces."""
    for path, item in _walk(value):
        if isinstance(item, Mapping):
            private = sorted(str(key) for key in item if str(key).lower() in _PRIVATE_KEYS)
            if private:
                raise PacketContractError(f"private keys at {path}: {private}")
        if not isinstance(item, str):
            continue
        if _SEMANTIC_DRIFT_MARKERS.search(item):
            raise PacketContractError(f"semantic drift marker at {path}")
        if _PRIVATE_EPISODE_RE.search(item):
            raise PacketContractError(f"private episode identifier at {path}")
        if _PRIVATE_ARM_RE.search(item):
            raise PacketContractError(f"private arm label at {path}")
        if _PRIVATE_RUN_PATH_RE.search(item):
            raise PacketContractError(f"private benchmark run path at {path}")
