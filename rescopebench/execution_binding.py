"""Load each environment's configuration and verify prepared run contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rescopebench import execution_policy, pathing, stream_engine
from rescopebench.case_names import case_identity, case_name


BINDING_SCHEMA = "rescopebench-formal-binding/v1"
ACTIVE_BINDING = "formal-binding.json"


def _case_root(case_id: str) -> Path:
    if (not isinstance(case_id, str) or not case_id
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in case_id)):
        raise ValueError("invalid case identifier")
    return (pathing.FINAL_ROOT / "cases" / case_name(case_id)).resolve()


def _binding_path(case_id: str, value: str | Path) -> Path:
    case_root = _case_root(case_id)
    path = Path(value)
    if not path.is_absolute():
        path = case_root / path
    path = path.resolve()
    if not path.is_relative_to(case_root):
        raise ValueError("candidate binding must stay inside its case package")
    return path


def _binding_spec(case_id: str, path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"execution binding is absent: {path}")
    spec = stream_engine.read_json(path)
    if (not isinstance(spec, dict) or spec.get("schema") != BINDING_SCHEMA
            or spec.get("case_id") != case_identity(case_id) or not isinstance(spec.get("case_version"), str)):
        raise ValueError(f"invalid execution binding: {path}")
    return spec


def _driver_argument(case_id: str, path: Path) -> str | None:
    case_root = _case_root(case_id)
    relative = path.relative_to(case_root).as_posix()
    return None if relative == ACTIVE_BINDING else relative


def new_run_binding(case_id: str, requested: str | Path | None = None, *,
                    execution_mode: str = "formal") -> str | None:
    """Use the environment's single configuration for a new run."""
    case_root = _case_root(case_id)
    execution_policy.normalize_execution_mode(execution_mode)
    if requested is None:
        path = case_root / ACTIVE_BINDING
    else:
        path = _binding_path(case_id, requested)
    _binding_spec(case_id, path)
    return _driver_argument(case_id, path)


def _binding_sources(case_id: str, plan: dict) -> dict[Path, str]:
    case_root = _case_root(case_id)
    sources = plan.get("runtime_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("run plan has no frozen runtime source inventory")
    result: dict[Path, str] = {}
    for key, expected in sources.items():
        if not isinstance(key, str) or not isinstance(expected, str):
            raise ValueError("run plan runtime source inventory is malformed")
        try:
            path = pathing.resolve_path(key)
        except ValueError:
            continue
        if (path.parent == case_root and path.name.startswith("formal-binding")
                and path.suffix == ".json"):
            if not path.is_file():
                raise ValueError(f"frozen execution binding is absent: {key}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"frozen execution binding hash differs: {key}")
            result[path] = actual
    if not result:
        raise ValueError("run plan does not freeze an execution binding")
    return result


def _infer_plan_path(case_id: str, plan: dict) -> Path:
    frozen = _binding_sources(case_id, plan)
    matching = []
    for path in sorted(frozen):
        spec = _binding_spec(case_id, path)
        if spec["case_version"] == plan.get("case_version"):
            matching.append(path)

    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        names = ", ".join(path.name for path in matching)
        raise ValueError(f"run plan execution binding is ambiguous: {names}")
    raise ValueError("run plan has no binding matching its case identity")


def planned_run_binding(run: str | Path, case_id: str | None = None,
                        requested: str | Path | None = None, *,
                        execution_mode: str = "formal") -> str | None:
    """Recover the exact binding frozen by a prepared plan and reject overrides."""
    run = Path(run).expanduser().resolve()
    plan = stream_engine.read_json(run / "run-plan.json")
    if plan.get("schema") != stream_engine.SCHEMA + "/plan":
        raise ValueError("invalid run plan schema")
    execution_policy.validate_plan_policy(plan, execution_mode)
    planned_case = plan.get("case_id")
    if case_id is not None and planned_case != case_identity(case_id):
        raise ValueError("requested case differs from the prepared run plan")
    selected = _infer_plan_path(planned_case, plan)
    if requested is not None:
        explicit = _binding_path(planned_case, requested)
        _binding_spec(planned_case, explicit)
        if explicit != selected:
            raise ValueError("explicit binding differs from the prepared run plan")
    return _driver_argument(planned_case, selected)
