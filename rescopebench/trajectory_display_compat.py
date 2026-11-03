"""Treat assistant display text as non-authoritative during evidence binding.

The provider response is already retained verbatim. Benchmark claims depend on
structured tool calls, replayed business events, terminal state, and grading,
not on a second byte-for-byte comparison with Hermes' displayed assistant text.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from rescopebench import annotation_recovery
from rescopebench import pathing, stream_engine


ERROR = "Foreground assistant messages differ from actual provider responses"
AMENDMENT_KIND = "hermes_assistant_display_text_non_authoritative_v3"
_ROUTING_TOKEN = re.compile(r"\b(route-)([0-9a-f]{24})\b")
native = stream_engine.native_runtime
_BASE_VALIDATOR = native.protocol.validate_trajectory_binding
_ASSISTANT = _BASE_VALIDATOR.__globals__["_assistant"]


def _masked_routing_tokens(
    text: str,
    suffix_length: int = 2,
    *,
    keep_hyphen: bool = True,
) -> tuple[str, int]:
    count = 0

    if suffix_length not in {2, 3, 4}:
        raise ValueError("routing-token display suffix must be 2, 3, or 4 hex characters")

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        prefix = match.group(1) if keep_hyphen else match.group(1).removesuffix("-")
        return f"{prefix}...{match.group(2)[-suffix_length:]}"

    return _ROUTING_TOKEN.sub(replace, text), count


def _routing_token_display_equivalent(expected: str, actual: str) -> int | None:
    """Return the number of independently masked token occurrences, if exact otherwise."""
    pattern: list[str] = []
    tokens: list[str] = []
    cursor = 0
    for index, match in enumerate(_ROUTING_TOKEN.finditer(expected)):
        pattern.append(re.escape(expected[cursor : match.start()]))
        full = match.group(0)
        suffixes = [match.group(2)[-length:] for length in (2, 3, 4)]
        masked = [
            f"{prefix}...{suffix}"
            for prefix in (match.group(1), match.group(1).removesuffix("-"))
            for suffix in suffixes
        ]
        group = f"token_{index}"
        choices = "|".join(re.escape(value) for value in (full, *masked))
        pattern.append(f"(?P<{group}>{choices})")
        tokens.append(full)
        cursor = match.end()
    if not tokens:
        return None
    pattern.append(re.escape(expected[cursor:]))
    matched = re.fullmatch("".join(pattern), actual)
    if matched is None:
        return None
    transformations = sum(
        matched.group(f"token_{index}") != full
        for index, full in enumerate(tokens)
    )
    return transformations if transformations else None


def validate_trajectory_binding(
    messages: list[dict[str, Any]],
    events: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate structured provider binding while ignoring assistant prose display."""
    actual = [_ASSISTANT(message) for message in messages if message.get("role") == "assistant"]
    adjusted = deepcopy(responses)
    expected = []
    for response in adjusted:
        choices = response.get("choices") or []
        if len(choices) != 1:
            raise ValueError("Provider response must have one choice")
        expected.append(_ASSISTANT(choices[0].get("message") or {}))
    if len(actual) != len(expected):
        raise ValueError(ERROR)

    content_differences = 0
    routing_token_transformations = 0
    for actual_message, expected_message, response in zip(actual, expected, adjusted):
        if actual_message == expected_message:
            continue
        actual_shape = {**actual_message, "content": ""}
        expected_shape = {**expected_message, "content": ""}
        expected_content = expected_message.get("content")
        actual_content = actual_message.get("content")
        if (
            actual_shape != expected_shape
            or not isinstance(expected_content, str)
            or not isinstance(actual_content, str)
        ):
            raise ValueError(ERROR)
        response["choices"][0]["message"]["content"] = actual_content
        content_differences += 1
        routing_token_transformations += (
            _routing_token_display_equivalent(expected_content, actual_content) or 0
        )

    result = _BASE_VALIDATOR(messages, events, adjusted, tools)
    return {
        **result,
        "assistant_content_differences_ignored": content_differences,
        "routing_token_display_transformations": routing_token_transformations,
    }


def binding_validator():
    return validate_trajectory_binding


def patch_runtime_validator() -> None:
    """Patch both the default executor and the template cloned by ``for_model``."""
    native.execute_episode_base.__globals__["validate_trajectory_binding"] = (
        validate_trajectory_binding
    )
    native.protocol.execute_episode.__globals__["validate_trajectory_binding"] = (
        validate_trajectory_binding
    )


def recover(run: Path, arm: str):
    return annotation_recovery.recover(
        run, arm, validator=validate_trajectory_binding, error=ERROR,
        amendment_kind=AMENDMENT_KIND, validator_path=Path(__file__),
    )


def install(output: Path, arm: str) -> dict[str, Any]:
    """Install the host-only validator and record its exact scope."""
    annotation_recovery.install(output, arm)
    plan = stream_engine.read_json(Path(output) / "run-plan.json")
    receipt = {
        "schema": "rescopebench-processing-addendum/v1",
        "arm": arm,
        "kind": AMENDMENT_KIND,
        "plan_sha256": stream_engine.digest(plan),
        "scope": ["host_trajectory_validation"],
        "accepted_transform": "assistant prose display is non-authoritative",
        "still_strict": [
            "assistant message count and roles",
            "structured tool calls and arguments",
            "business event trace and results",
            "terminal state replay",
            "grading",
        ],
        "business_events_state_grading_and_provider_tool_calls_unchanged": True,
        "model_requests_reissued": 0,
        "sources": {
            pathing.relative_path(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(annotation_recovery.__file__))
        },
    }
    stream_engine._write_once(
        Path(output) / "processing-addenda" / f"trajectory-display-v3-{arm}.json",
        receipt,
    )
    # Patch the current default executor and the source function cloned for
    # model-specific runtimes created after this installation.
    patch_runtime_validator()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=stream_engine.ARMS)
    args = parser.parse_args()
    print(json.dumps(recover(args.output, args.arm), sort_keys=True))


if __name__ == "__main__":
    main()
