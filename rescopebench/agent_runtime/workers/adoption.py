"""Opt-in ownership handoff for skills created inside a benchmark episode."""

from __future__ import annotations

import json
import re
from typing import Any, Callable


WILDCARD = "*"
SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def validate_allowlist(value: Any) -> list[str]:
    """Return a canonical exact-name allowlist; ``*`` means every new local create."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("curator_adoption_allowlist must be a list")
    names = []
    for item in value:
        if not isinstance(item, str) or (item != WILDCARD and not SKILL_NAME.fullmatch(item)):
            raise ValueError("Invalid curator adoption allowlist entry")
        if item not in names:
            names.append(item)
    if WILDCARD in names and len(names) != 1:
        raise ValueError("The curator adoption wildcard cannot be combined with skill names")
    return names


def operation_names(arguments: Any) -> list[str]:
    if not isinstance(arguments, dict) or not isinstance(arguments.get("operations"), list):
        return []
    return list(dict.fromkeys(
        operation.get("name")
        for operation in arguments["operations"]
        if isinstance(operation, dict) and isinstance(operation.get("name"), str)
    ))


def requested_creates(arguments: Any) -> list[str]:
    if not isinstance(arguments, dict) or not isinstance(arguments.get("operations"), list):
        return []
    return list(dict.fromkeys(
        operation.get("name")
        for operation in arguments["operations"]
        if (isinstance(operation, dict) and operation.get("action") == "create"
            and isinstance(operation.get("name"), str))
    ))


class CuratorAdoptionGate:
    """Adopt only allowlisted skills whose first observed operation is a successful create."""

    def __init__(
        self,
        allowlist: Any,
        *,
        find_skill: Callable[[str], Any],
        adopt_skill: Callable[[str], tuple[bool, str]],
        record: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.allowlist = frozenset(validate_allowlist(allowlist))
        self.find_skill = find_skill
        self.adopt_skill = adopt_skill
        self.record = record or (lambda _receipt: None)
        self.preexisting: set[str] = set()
        self.adopted: set[str] = set()
        self.receipts: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []

    def _allowed(self, name: str) -> bool:
        return WILDCARD in self.allowlist or name in self.allowlist

    def invoke(self, arguments: Any, handler: Callable[[], Any], *, foreground: bool) -> Any:
        """Run the native handler, then hand successful first-time creates to the curator."""
        if not foreground or not self.allowlist:
            return handler()

        names = operation_names(arguments)
        for name in names:
            if self.find_skill(name) is not None:
                self.preexisting.add(name)
        candidates = [
            name for name in requested_creates(arguments)
            if self._allowed(name) and name not in self.preexisting and name not in self.adopted
        ]
        result = handler()
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
        except (TypeError, json.JSONDecodeError):
            return result
        if (not isinstance(parsed, dict) or parsed.get("success") is not True
                or parsed.get("staged") is True):
            return result
        successful_creates = {
            row.get("name") for row in parsed.get("results") or []
            if (isinstance(row, dict) and row.get("action") == "create"
                and row.get("success") is True)
        }
        for name in candidates:
            if name not in successful_creates:
                continue
            try:
                ok, message = self.adopt_skill(name)
            except Exception as error:  # Fail the episode later without altering the tool response.
                ok, message = False, f"{type(error).__name__}: {error}"
            receipt = {
                "skill": name,
                "success": bool(ok),
                "message": str(message),
                "policy": "allowlisted_foreground_first_create",
            }
            self.receipts.append(receipt)
            self.record(receipt)
            if ok:
                self.adopted.add(name)
            else:
                self.failures.append(receipt)
        return result

    def require_success(self) -> None:
        if self.failures:
            names = ", ".join(row["skill"] for row in self.failures)
            raise RuntimeError("Benchmark curator adoption failed for: " + names)
