"""Shared execution-mode and claim-scope policy for common stream runs."""

from __future__ import annotations

from typing import Any, Mapping


EXECUTION_MODES = ("formal", "candidate")
CLAIM_SCOPE_BY_MODE = {
    "formal": "formal-benchmark",
    "candidate": "candidate-only",
}


def normalize_execution_mode(execution_mode: str) -> str:
    if execution_mode not in EXECUTION_MODES:
        choices = ", ".join(EXECUTION_MODES)
        raise ValueError(f"execution mode must be one of: {choices}")
    return execution_mode


def claim_scope_for_mode(execution_mode: str) -> str:
    return CLAIM_SCOPE_BY_MODE[normalize_execution_mode(execution_mode)]


def _policy(execution_mode: str, formal_launch_authorized: bool) -> dict[str, Any]:
    mode = normalize_execution_mode(execution_mode)
    if mode == "formal" and formal_launch_authorized is not True:
        raise ValueError("formal execution requires formal_launch_authorized=true on the selected binding")
    return {
        "execution_mode": mode,
        "claim_scope": claim_scope_for_mode(mode),
        "formal_launch_authorized": formal_launch_authorized,
        "candidate_only": mode == "candidate",
    }


def validate_driver_policy(driver: Any, execution_mode: str = "formal") -> dict[str, Any]:
    """Resolve policy for the operator-selected execution mode."""
    spec = getattr(driver, "binding_spec", None)
    if not isinstance(spec, Mapping):
        raise ValueError("driver has no selected execution binding policy")
    mode = normalize_execution_mode(execution_mode)
    # Selecting formal execution is the explicit operator authorization. The
    # binding's construction-time flag may remain false until model validation.
    authorized = mode == "formal"
    return _policy(execution_mode, authorized)


def validate_plan_policy(plan: Mapping[str, Any], execution_mode: str | None = None) -> dict[str, Any]:
    """Validate a frozen plan claim, optionally against an explicitly requested mode."""
    if not isinstance(plan, Mapping):
        raise ValueError("run plan must be an object")
    frozen_mode = normalize_execution_mode(plan.get("execution_mode"))
    if execution_mode is not None and normalize_execution_mode(execution_mode) != frozen_mode:
        raise ValueError("requested execution mode differs from the frozen run plan")
    authorized = plan.get("formal_launch_authorized")
    if type(authorized) is not bool:
        raise ValueError("run plan formal authorization must be a boolean")
    expected = _policy(frozen_mode, authorized)
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("run plan execution policy is inconsistent")
    return expected
