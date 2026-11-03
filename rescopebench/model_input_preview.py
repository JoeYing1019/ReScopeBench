"""Read-only rendering of the exact first provider request for a formal task."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from rescopebench import execution_binding, native_executor, stream_engine
from rescopebench.agent_runtime import history as native_history
from rescopebench.agent_runtime import native_runtime
from rescopebench.formal_driver import load_driver


SCHEMA = "rescopebench-model-input-preview/v1"


def parse_json_argument(value: str, *, name: str) -> Any:
    """Parse inline JSON, with ``@path`` and plain JSON-file paths as conveniences."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be JSON or a JSON file path")
    source = value
    if value.startswith("@"):
        path = Path(value[1:]).expanduser().resolve()
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"{name} JSON file cannot be read: {path}") from error
    else:
        try:
            return json.loads(value)
        except json.JSONDecodeError as inline_error:
            path = Path(value).expanduser()
            if not path.is_file():
                raise ValueError(f"{name} is not valid JSON") from inline_error
            try:
                source = path.resolve().read_text(encoding="utf-8")
            except OSError as error:
                raise ValueError(f"{name} JSON file cannot be read: {path}") from error
    try:
        return json.loads(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON") from error


def render_driver_input(
    driver: Any,
    position: int,
    *,
    model: str,
    carriers: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    tool_execution_mode: str = "serial",
) -> dict[str, Any]:
    """Render the admitted request and local carrier surfaces without executing it."""

    if type(position) is not int or not 1 <= position <= len(driver.tasks):
        raise ValueError(f"position must be between 1 and {len(driver.tasks)}")
    if carriers is not None and not isinstance(carriers, dict):
        raise ValueError("carriers must be a JSON object")
    if history is not None and not isinstance(history, list):
        raise ValueError("history must be a JSON array")

    assets = deepcopy(carriers or {})
    episodes = deepcopy(history or [])
    runtime = native_executor.for_model(model, prompt=driver.prompt, tool_execution_mode=tool_execution_mode)
    archive = native_history.HistoryArchive(driver.tools, episodes)
    handler = archive.session(assets)
    selected_task = driver.tasks[position - 1]
    if selected_task.get("position") != position:
        raise ValueError("driver task order differs from the requested position")
    task = stream_engine.public_task(selected_task)
    packet = runtime.packet_for(
        task,
        driver.tools,
        assets,
        history_handler=handler,
        condition="native_memory_skill",
    )

    grouped_tools = {
        "business": native_runtime.surface.project(packet["business_tools"]),
        "history": native_runtime.surface.project(packet["history_tools"]),
        "native": native_runtime.surface.project(native_runtime.surface.NATIVE_TOOLS),
    }
    tools = native_runtime.expected_tools(packet)
    if tools != [
        *grouped_tools["business"],
        *grouped_tools["history"],
        *grouped_tools["native"],
    ]:
        raise ValueError("grouped tools differ from the runtime provider surface")

    raw = {
        "model": packet["model"],
        "seed": packet["seed"],
        "messages": [
            {"role": "system", "content": packet["system"]},
            {"role": "user", "content": packet["request"]},
        ],
        "tools": tools,
        "max_completion_tokens": runtime.MAX_TOKENS,
        **deepcopy(packet["request_options"]),
    }
    admitted, conversions = runtime.provider_body(raw, tools)
    if admitted.get("messages") != raw["messages"] or admitted.get("tools") != tools:
        raise ValueError("provider admission changed the rendered messages or tools")
    request_options = {
        key: deepcopy(value)
        for key, value in admitted.items()
        if key not in {"model", "messages", "tools"}
    }

    return {
        "schema": SCHEMA,
        "case_id": driver.case_id,
        "case_version": driver.case_version,
        "binding": getattr(driver, "binding_name", None),
        "position": position,
        "event_id": task["event_id"],
        "model": admitted["model"],
        "system": packet["system"],
        "request": packet["request"],
        "messages": deepcopy(admitted["messages"]),
        "tools": grouped_tools,
        "admitted_request_options": request_options,
        "provider_conversions": list(conversions),
        "carriers": deepcopy(packet["assets"]),
        "history": episodes,
        "history_archive": deepcopy(handler.bundle),
        "credentials_loaded": False,
        "model_called": False,
        "container_started": False,
    }


def render_model_input(
    case_id: str,
    position: int,
    *,
    binding: str | Path | None = None,
    model: str = native_runtime.MODEL,
    carriers: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    tool_execution_mode: str = "serial",
) -> dict[str, Any]:
    """Load the selected new-run binding and render one real task's model input."""

    selected = execution_binding.new_run_binding(case_id, binding, execution_mode="candidate")
    driver = load_driver(case_id, selected)
    return render_driver_input(
        driver,
        position,
        model=model,
        carriers=carriers,
        history=history,
        tool_execution_mode=tool_execution_mode,
    )


__all__ = [
    "SCHEMA",
    "parse_json_argument",
    "render_driver_input",
    "render_model_input",
]
