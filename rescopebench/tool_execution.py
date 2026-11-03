"""Declared tool batching, independent of environment prices and model quality.

This module is also mounted standalone into the isolated Hermes worker.
"""

from copy import deepcopy
from functools import wraps


MODES = ("serial", "batched")
SCHEMA = "rescopebench-tool-dispatch/v1"


def contract(mode="serial"):
    if mode not in MODES:
        raise ValueError(f"tool execution mode must be one of {MODES}")
    return {
        "schema": SCHEMA,
        "mode": mode,
        "parallel_tool_calls": mode == "batched",
        "execution_order": "response_call_index",
        "concurrent_state_mutation": False,
        "business_meter": "environment_per_invocation",
        "history_meter": "history_session_per_invocation",
        "argument_deduplication": False,
    }


def plan_mode(plan):
    # Plans made before batching have the original serial semantics.
    value = plan.get("tool_execution")
    if value is None:
        return "serial"
    if not isinstance(value, dict) or value != contract(value.get("mode")):
        raise ValueError("run plan tool execution contract differs")
    return value["mode"]


def resource_state(world):
    """Read the current adapters' scalar meters, never copy their business data.

    These are observations, not a second price schedule. Units stay separate;
    unknown adapters can provide their own projection on the host facade.
    """
    values = vars(world)
    result = {}

    def counter(unit, candidates, direction):
        for name in candidates:
            value = values.get(name)
            if type(value) in (int, float):
                result[unit] = {"value": value, "direction": direction}
                return True
        return False

    if counter("read_work_units", ("read_used", "_read_units"), "used"):
        counter("mutation_attempts", ("mutation_used", "_write_attempts"), "used")
    elif not counter("work_units", ("work_units", "_work_units"), "used"):
        counter("work_units", ("remaining", "_remaining"), "remaining")
    counter("business_calls", ("_calls",), "used")
    if "mutation_attempts" not in result:
        counter("mutation_attempts", ("mutation_used", "_write_attempts", "_writes"), "used")
    runtime = values.get("runtime")
    if runtime is not None and hasattr(runtime, "business_call_count"):
        result["business_calls"] = {"value": runtime.business_call_count, "direction": "used"}
        result["battery"] = {"value": runtime.adapter._battery(), "direction": "remaining"}
    return result


def cost_receipt(before, after, *, dispatched):
    return {
        "schema": "rescopebench-invocation-cost/v1",
        "dispatched": dispatched,
        "before": before, "after": after,
        "consumed": {
            unit: (after[unit]["value"] - value["value"])
            * (1 if value["direction"] == "used" else -1)
            for unit, value in before.items()
            if unit in after and value["direction"] == after[unit]["direction"]
        },
    }


def install_ordered_dispatch(agent_class, *, emit):
    """Keep native handlers/limits but disable batch races and argument dedup.

    Separate call IDs denote separate invocations, even for equal arguments.
    The host commits only complete episodes; this is not external exactly-once IO.
    """
    original = agent_class._execute_tool_calls
    sequence = 0

    @wraps(original)
    def ordered(self, assistant_message, messages, effective_task_id, api_call_count=0):
        nonlocal sequence
        sequence += 1
        batch_id = sequence
        calls = assistant_message.tool_calls or []
        record = {
            "batch_id": batch_id,
            "api_call_count": api_call_count,
            "calls": [{"call_index": index, "tool_call_id": call.id,
                       "name": call.function.name, "arguments": call.function.arguments}
                      for index, call in enumerate(calls)],
        }
        emit(batch_id, {**record, "status": "started"})
        offset = len(messages)
        self._executing_tools = True
        status = "raised"
        try:
            result = self._execute_tool_calls_sequential(
                assistant_message, messages, effective_task_id, api_call_count)
            status = "returned"
            return result
        finally:
            self._executing_tools = False
            emit(batch_id, {**record, "status": status,
                            "results": deepcopy([m for m in messages[offset:] if m.get("role") == "tool"])})

    agent_class._execute_tool_calls = ordered
    agent_class._deduplicate_tool_calls = staticmethod(lambda calls: calls)
