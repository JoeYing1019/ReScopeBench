"""Hermes' tool-argument repairs, applied on the host before Hermes sees a response.

Added by the rescope harness; not part of the original release.

Hermes rewrites model-emitted arguments before dispatch: an empty string becomes
"{}" (agent/turn_tool_validation.py) and string values are coerced to the tool's
JSON-Schema type, e.g. "250" -> 250 and "null" -> None for a nullable field
(tools/arg_coercion.py). The host event trace records the executed value while
Hermes' transcript keeps the original text, and the frozen binding check treats
that difference as a failed task. Applying the same repairs here first makes the
transcript equal to what runs. This is a port of the Hermes code pinned by the release
(hermes_commit in rescopebench/vendor/image-manifest.json); keep it in step with
that commit, not with newer Hermes.
"""

from __future__ import annotations

import json
from typing import Any


def coerce_tool_args(args: Any, parameters: Any) -> Any:
    if not args or not isinstance(args, dict):
        return args
    properties = (parameters or {}).get("properties") if isinstance(parameters, dict) else None
    if not properties:
        return args
    args = dict(args)
    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        expected = prop_schema.get("type")
        is_container = isinstance(value, (list, tuple))
        if expected == "array" and value is not None and not is_container:
            if isinstance(value, str):
                coerced = _coerce_value(value, expected, schema=prop_schema)
                if coerced is not value:
                    args[key] = coerced
                    continue
            args[key] = [value]
            continue
        if not isinstance(value, str):
            if (expected == "array" and is_container) or (expected == "object" and isinstance(value, dict)):
                args[key] = _normalize_json_strings_for_schema(value, prop_schema)
            continue
        if not expected and not _schema_allows_null(prop_schema):
            continue
        coerced = _coerce_value(value, expected, schema=prop_schema)
        if coerced is not value:
            args[key] = coerced
            if isinstance(coerced, (list, tuple, dict)):
                args[key] = _normalize_json_strings_for_schema(coerced, prop_schema)
    return args


def _schema_accepts_kind(schema: Any, kind: str) -> bool:
    if not isinstance(schema, dict):
        return False
    t = schema.get("type")
    if t == kind or (isinstance(t, list) and kind in t):
        return True
    return any(isinstance(branches := schema.get(union_key), list)
               and any(_schema_accepts_kind(b, kind) for b in branches)
               for union_key in ("anyOf", "oneOf", "allOf"))


def _normalize_json_strings_for_schema(value: Any, schema: Any) -> Any:
    if not isinstance(schema, dict):
        return value
    if isinstance(value, str):
        trimmed = value.strip()
        expects_array = _schema_accepts_kind(schema, "array")
        expects_object = _schema_accepts_kind(schema, "object")
        if not ((expects_array and trimmed.startswith("[")) or (expects_object and trimmed.startswith("{"))):
            return value
        try:
            parsed = json.loads(trimmed)
        except (ValueError, TypeError):
            return value
        if not ((isinstance(parsed, list) and expects_array) or (isinstance(parsed, dict) and expects_object)):
            return value
        value = parsed
    if isinstance(value, list):
        items_schema = schema.get("items")
        if not isinstance(items_schema, dict):
            return value
        out = [_normalize_json_strings_for_schema(item, items_schema) for item in value]
        return out if any(n is not o for n, o in zip(out, value)) else value
    if isinstance(value, dict):
        props = schema.get("properties")
        if not isinstance(props, dict):
            return value
        out = dict(value)
        for k, prop_schema in props.items():
            if k in value and isinstance(prop_schema, dict):
                out[k] = _normalize_json_strings_for_schema(value[k], prop_schema)
        return out if any(out[k] is not v for k, v in value.items()) else value
    return value


def _coerce_value(value: str, expected_type, schema: dict | None = None):
    if _schema_allows_null(schema) and value.strip().lower() == "null":
        return None
    if isinstance(expected_type, list):
        return next((r for t in expected_type if (r := _coerce_value(value, t, schema=schema)) is not value), value)
    coercer = _SCALAR_COERCERS.get(expected_type)
    if coercer is not None:
        return coercer(value)
    return None if expected_type == "null" and value.strip().lower() == "null" else value


def _schema_allows_null(schema: dict | None) -> bool:
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    if schema_type == "null" or (isinstance(schema_type, list) and "null" in schema_type):
        return True
    if schema.get("nullable") is True:
        return True
    return any(isinstance(variants := schema.get(union_key), list)
               and any(isinstance(v, dict) and v.get("type") == "null" for v in variants)
               for union_key in ("anyOf", "oneOf"))


def _coerce_json(value: str, expected_python_type: type):
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    return parsed if isinstance(parsed, expected_python_type) else value


def _coerce_number(value: str, integer_only: bool = False):
    try:
        f = float(value)
    except (ValueError, OverflowError):
        return value
    if f != f or f in (float("inf"), float("-inf")):
        return value
    return int(f) if f == int(f) else value if integer_only else f


def _coerce_boolean(value: str):
    return {"true": True, "false": False}.get(value.strip().lower(), value)


_SCALAR_COERCERS = {
    "integer": lambda v: _coerce_number(v, integer_only=True),
    "number": _coerce_number,
    "boolean": _coerce_boolean,
    "array": lambda v: _coerce_json(v, list),
    "object": lambda v: _coerce_json(v, dict),
}


def normalize_tool_arguments(body: dict, response: dict) -> list[dict]:
    """Apply Hermes' repairs to the response's tool calls in place; return what changed."""
    schemas = {(tool.get("function") or {}).get("name"): (tool.get("function") or {}).get("parameters")
               for tool in body.get("tools") or []}
    changes = []
    message = (response.get("choices") or [{}])[0].get("message") or {}
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        original = function.get("arguments")
        if isinstance(original, str) and not original.strip():
            function["arguments"] = "{}"
            changes.append({"tool_call_id": call.get("id"), "from": original, "to": "{}"})
            continue
        try:
            args = json.loads(original) if isinstance(original, str) else original
        except ValueError:
            continue
        coerced = coerce_tool_args(args, schemas.get(function.get("name")))
        if coerced != args:
            function["arguments"] = json.dumps(coerced, ensure_ascii=False)
            changes.append({"tool_call_id": call.get("id"), "from": original, "to": function["arguments"]})
    return changes
