"""Load the selected environment through its explicit execution binding."""

from __future__ import annotations

from copy import deepcopy
import importlib
import inspect
from pathlib import Path

from rescopebench import pathing, stream_engine
from rescopebench.case_names import case_identity, case_name


def _symbol(name: str):
    module, attribute = name.split(":", 1)
    return getattr(importlib.import_module(module), attribute)


def load_driver(case_id: str, binding_path: Path | str | None = None):
    case_id = case_identity(case_id)
    case_root = (pathing.FINAL_ROOT / "cases" / case_name(case_id)).resolve()
    if binding_path is None:
        from rescopebench.execution_binding import new_run_binding

        binding_path = new_run_binding(case_id) or "formal-binding.json"
    path = Path(binding_path)
    if not path.is_absolute():
        path = case_root / path
    path = path.resolve()
    if not path.is_relative_to(case_root):
        raise ValueError("candidate binding must stay inside its case package")
    spec = stream_engine.read_json(path)
    if spec.get("schema") != "rescopebench-formal-binding/v1" or spec.get("case_id") != case_id:
        raise ValueError("invalid case binding")
    factory = spec.get("driver_factory")
    if factory:
        constructor = _symbol(factory)
        kwargs = {}
        artifact_root = spec.get("artifact_root")
        signature = inspect.signature(constructor)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if artifact_root is not None and ("artifact_root" in signature.parameters or accepts_kwargs):
            root = (case_root / artifact_root).resolve()
            if not root.is_relative_to(case_root):
                raise ValueError("candidate artifact root escapes its case package")
            kwargs["artifact_root"] = root
        driver = constructor(**kwargs)
    else:
        raise ValueError("execution binding requires an explicit driver_factory")
    if driver.case_id != case_id or driver.case_version != spec.get("case_version"):
        raise ValueError("driver identity differs from its binding")
    driver.binding_path = path
    driver.binding_name = str(path.relative_to(case_root))
    driver.binding_spec = deepcopy(spec)
    driver.formal_launch_authorized = spec.get("formal_launch_authorized") is True
    return driver
