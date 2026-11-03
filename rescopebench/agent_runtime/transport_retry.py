"""Retry transient LLM failures at the upstream boundary, never business actions."""

from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
from datetime import timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
import random
import time

import httpx

from . import core


SCHEMA = "llm-provider-transient-retry/v2"
QUOTA_CODES = {
    "insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached",
    "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
}
PROVIDER_TIMEOUT_MARKERS = (
    "connection timed out", "i/o timeout", "context deadline exceeded",
    "timeout awaiting response headers", "read timeout", "read timed out",
)
MAX_UNKNOWN_HTTP400_RETRIES = 3
HARD_HTTP400_MARKERS = (
    "content_filter", "content_policy_violation", "responsibleaipolicyviolation",
    "safety policy", "content management policy", "context_length_exceeded",
    "maximum context length", "invalid_api_key", "authentication_error",
    "invalid_parameter", "unsupported_parameter", "invalid schema",
    "invalid_json", "invalid_tool", "tool_call_id", "missing required parameter",
)


class TransportRequestFailed(RuntimeError):
    def __init__(self, attempts, last_error, reason):
        super().__init__(f"LLM transport failed after {attempts} attempt(s): {reason}")
        self.attempts, self.last_error, self.reason = attempts, last_error, reason


class TransportRetryExhausted(TransportRequestFailed):
    pass


def retry_policy(*, rate_limit_retries=8, server_retries=5, connection_retries=5,
                 backoff_seconds=2.0, backoff_multiplier=2.0, backoff_cap_seconds=60.0,
                 jitter_fraction=0.25, request_timeout_seconds=60.0,
                 max_elapsed_seconds=300.0):
    values = dict(rate_limit_retries=rate_limit_retries, server_retries=server_retries,
                  connection_retries=connection_retries, backoff_seconds=backoff_seconds,
                  backoff_multiplier=backoff_multiplier, backoff_cap_seconds=backoff_cap_seconds,
                  jitter_fraction=jitter_fraction, request_timeout_seconds=request_timeout_seconds,
                  max_elapsed_seconds=max_elapsed_seconds)
    for name in ("rate_limit_retries", "server_retries", "connection_retries"):
        if type(values[name]) is not int or values[name] < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in set(values) - {"rate_limit_retries", "server_retries", "connection_retries"}:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if backoff_multiplier < 1 or not 0 <= jitter_fraction <= 1:
        raise ValueError("Invalid backoff multiplier or jitter fraction")
    if request_timeout_seconds <= 0 or max_elapsed_seconds <= 0:
        raise ValueError("Transport timeouts must be positive")
    return {"schema": SCHEMA, "sdk_max_retries": 0, **values}


def validate_policy(policy):
    if not isinstance(policy, dict):
        raise ValueError("Retry policy must be an object")
    value = deepcopy(policy)
    sdk_retries = value.pop("sdk_max_retries", None)
    if value.pop("schema", None) != SCHEMA or type(sdk_retries) is not int or sdk_retries != 0:
        raise ValueError("Unknown retry policy or layered SDK retries")
    try:
        expected = retry_policy(**value)
    except TypeError as error:
        raise ValueError("Unknown retry policy fields") from error
    if expected != policy:
        raise ValueError("Incomplete retry policy")
    return expected


def status_code(error):
    value = getattr(error, "status_code", None)
    if type(value) is int:
        return value
    value = getattr(getattr(error, "response", None), "status_code", None)
    return value if type(value) is int else None


def error_body(error):
    value = getattr(error, "body", None)
    if value is None:
        response = getattr(error, "response", None)
        try:
            value = response.json() if response is not None else None
        except (TypeError, ValueError):
            value = None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    if not isinstance(value, dict):
        return {}
    nested = value.get("error")
    return nested if isinstance(nested, dict) else value


def public_error_fields(error):
    fields = {}
    secrets = [value for key, value in os.environ.items()
               if key.startswith("RESCOPE_PROVIDER_API_KEY") and value]
    for name in ("code", "type", "message"):
        value = error_body(error).get(name)
        if not isinstance(value, (str, int, float, bool)):
            continue
        value = str(value)
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        fields[name] = value[:3000]
    return fields


def failure_class(error):
    """Vendor timeout handling is deliberately narrower than all HTTP 406s."""
    status = status_code(error)
    payload = error_body(error)
    code, kind = payload.get("code"), payload.get("type")
    if status == 400:
        detail = " ".join(str(payload.get(key, "")) for key in ("code", "type", "message")).lower()
        if any(marker in detail for marker in HARD_HTTP400_MARKERS):
            return None, "non_transient_http_error"
        return "server", "http400_bounded_retry"
    if status == 429:
        if str(code) in QUOTA_CODES or str(kind) in QUOTA_CODES:
            return None, "quota_or_billing"
        return "rate_limit", "http_rate_limit"
    if status == 408:
        return "connection", "http_timeout"
    if status in {500, 502, 503, 504}:
        return "server", "transient_server_error"
    if status == 406 and str(code) == "-4334":
        message = payload.get("message")
        if isinstance(message, str) and any(marker in message.lower() for marker in PROVIDER_TIMEOUT_MARKERS):
            return "connection", "provider_wrapped_upstream_timeout"
    if status is not None:
        return None, "non_transient_http_error"
    if isinstance(error, (TimeoutError, asyncio.TimeoutError, ConnectionError, httpx.TimeoutException,
                          httpx.NetworkError, httpx.RemoteProtocolError)):
        return "connection", "network_timeout_or_connection"
    if any(base.__name__ in {"APIConnectionError", "APITimeoutError"} for base in type(error).__mro__):
        return "connection", "sdk_timeout_or_connection"
    return None, "non_transport_error"


def retry_after_seconds(error, wall_time):
    response = getattr(error, "response", None)
    raw = getattr(response, "headers", None) or {}
    headers = {str(key).lower(): str(value) for key, value in raw.items()}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            seconds = float(value) * scale
            if math.isfinite(seconds) and seconds >= 0:
                return seconds
        except ValueError:
            if name == "retry-after":
                try:
                    when = parsedate_to_datetime(value)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    return max(0.0, when.timestamp() - wall_time)
                except (TypeError, ValueError, OverflowError):
                    pass
    return None


def request_with_transient_retry(*, create, body, logical_turn, policy, record,
                                 sleep=time.sleep, monotonic=time.monotonic,
                                 wall_clock=time.time, random_value=random.random):
    """create receives (unchanged_body, timeout_seconds); timeout is not an API field."""
    policy = validate_policy(policy)
    admitted = deepcopy(body)
    request_hash = core.digest(admitted)
    failures, attempt, last_error = Counter(), 0, None
    started_at = monotonic()
    deadline = started_at + policy["max_elapsed_seconds"]
    common = {"logical_turn": logical_turn, "request_sha256": request_hash}

    def exhausted(reason):
        record({"kind": "transport_retry_exhausted", **common, "attempts": attempt,
                "reason": reason, "elapsed_seconds": monotonic() - started_at})
        raise TransportRetryExhausted(attempt, last_error, reason) from None

    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            exhausted("elapsed_time_budget")
        attempt += 1
        timeout = min(policy["request_timeout_seconds"], remaining)
        record({"kind": "transport_attempt_started", **common, "attempt": attempt,
                "timeout_seconds": timeout})
        try:
            response = create(deepcopy(admitted), timeout)
        except Exception as error:
            last_error = error
            category, reason = failure_class(error)
            summary = {"error_type": type(error).__name__, "http_status": status_code(error),
                       "failure_reason": reason}
            if status_code(error) == 400:
                summary["public_error_fields"] = public_error_fields(error)
                if category is not None:
                    failures["http400"] += 1
                    if failures["http400"] > MAX_UNKNOWN_HTTP400_RETRIES:
                        exhausted("http400_retry_limit")
            if category is None:
                record({"kind": "transport_attempt_failed_non_retryable", **common,
                        "attempt": attempt, **summary})
                raise TransportRequestFailed(attempt, error, reason) from None
            failures[category] += 1
            record({"kind": "transport_attempt_failed_retryable", **common, "attempt": attempt,
                    "failure_class": category, "failure_number": failures[category], **summary})
            if failures[category] > policy[category + "_retries"]:
                exhausted(category + "_retry_limit")
            delay = policy["backoff_seconds"]
            for _ in range(failures[category] - 1):
                delay = min(delay * policy["backoff_multiplier"], policy["backoff_cap_seconds"])
            delay = min(delay, policy["backoff_cap_seconds"])
            delay *= 1 - policy["jitter_fraction"] + policy["jitter_fraction"] * random_value()
            server_delay = retry_after_seconds(error, wall_clock())
            if server_delay is not None:
                delay = max(delay, server_delay)
            if delay >= deadline - monotonic():
                exhausted("retry_wait_exceeds_time_budget")
            record({"kind": "transport_retry_wait", **common, "after_attempt": attempt,
                    "seconds": delay, "server_minimum_seconds": server_delay,
                    "failure_class": category})
            sleep(delay)
            continue
        if monotonic() >= deadline:
            record({"kind": "transport_late_response_discarded", **common, "attempt": attempt})
            exhausted("elapsed_time_budget")
        record({"kind": "transport_attempt_succeeded", **common, "attempt": attempt,
                "elapsed_seconds": monotonic() - started_at})
        return response
