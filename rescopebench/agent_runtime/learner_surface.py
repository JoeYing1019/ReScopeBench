"""Opaque learner identities and lifecycle-leakage checks for Robot."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from rescopebench.public_projection import assert_hard_public_projection


OPAQUE_EVENT_RE = re.compile(r"^evt-[0-9a-f]{20}$")
_PRIVATE_EPISODE_RE = re.compile(
    r"(?<![A-Za-z0-9])T(?:0[1-9]|1[0-6])(?![A-Za-z0-9]|:[0-5][0-9])",
    re.IGNORECASE,
)
_ARM_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:native|reset|reflect)\d{2}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_RUN_PATH_RE = re.compile(
    r"(?:benchmark_outputs[/\\]robot|robot-final-v\d+[/\\]20\d{6}-\d{6})",
    re.IGNORECASE,
)
_PRIVATE_KEYS = frozenset(
    {
        "arm",
        "arm_policy",
        "lifecycle_position",
        "next_episode_id",
        "position",
        "previous_episode_id",
        "private_episode_id",
        "run_path",
    }
)


def opaque_event_id(work_orders: Any) -> str:
    """Derive a stable, non-sequential public ID from the public order set."""

    if (
        not isinstance(work_orders, list)
        or not work_orders
        or any(not isinstance(value, str) or not value for value in work_orders)
        or len(set(work_orders)) != len(work_orders)
    ):
        raise ValueError("opaque event IDs require unique non-empty work orders")
    payload = json.dumps(
        ["robot-final-public-event/v1", *sorted(work_orders)],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return "evt-" + hashlib.sha256(payload).hexdigest()[:20]


def leakage_hits(value: Any) -> dict[str, list[str]]:
    """Report private lifecycle labels in a learner/provider-visible value."""

    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
    private_keys: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            private_keys.update(str(key) for key in item if key in _PRIVATE_KEYS)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return {
        "private_episode_ids": sorted(set(_PRIVATE_EPISODE_RE.findall(encoded))),
        "arm_labels": sorted(set(_ARM_RE.findall(encoded))),
        "run_paths": sorted(set(_RUN_PATH_RE.findall(encoded))),
        "private_keys": sorted(private_keys),
    }


def assert_opaque_learner_surface(value: Any, *, surface: str) -> None:
    hits = leakage_hits(value)
    present = {name: rows for name, rows in hits.items() if rows}
    if present:
        raise ValueError(f"{surface} exposes private lifecycle metadata: {present}")
    try:
        assert_hard_public_projection(value)
    except ValueError as error:
        raise ValueError(f"{surface} exposes private lifecycle metadata: {error}") from error


__all__ = [
    "OPAQUE_EVENT_RE",
    "assert_opaque_learner_surface",
    "leakage_hits",
    "opaque_event_id",
]
