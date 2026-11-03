"""Shared audited upstream boundary for task and reflection model requests."""

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
import uuid

import httpx
import openai

from . import core
from . import transport_retry as retry


AZURE_API_VERSION = "2024-02-01"


def api_keys_from_environment():
    """Load an ordered credential pool without persisting credential values."""
    names = ["RESCOPE_PROVIDER_API_KEY", "RESCOPE_PROVIDER_API_KEY_SECONDARY"]
    numbered = sorted(
        name for name in os.environ
        if name.startswith("RESCOPE_PROVIDER_API_KEY_") and name not in names
    )
    values = []
    for name in [*names, *numbered]:
        value = os.environ.get(name)
        if isinstance(value, str) and value and value not in values:
            values.append(value)
    if not values:
        raise RuntimeError(
            "RESCOPE_PROVIDER_API_KEY or RESCOPE_PROVIDER_API_KEY_SECONDARY is required"
        )
    return tuple(values)


def _normalize_keys(value):
    values = [value] if isinstance(value, str) else list(value)
    if not values or any(not isinstance(key, str) or not key for key in values):
        raise ValueError("At least one non-empty API credential is required")
    return tuple(dict.fromkeys(values))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(core.encoded(value) + "\n")


class Provider:
    """Synchronous caller contract, cancellable async network I/O.

    The provider owns its dedicated event loop and client, including an injected
    AsyncOpenAI test client. Call from a synchronous host thread, not an async loop.
    """

    def __init__(self, key, directory, *, base_url=None, azure_endpoint=None,
                 api_version=AZURE_API_VERSION, policy=None, client=None):
        self.directory = Path(directory)
        self.policy = retry.validate_policy(policy if policy is not None else retry.retry_policy())
        self.count, self.wires, self.expected = 0, 0, None
        self.lock = threading.Lock()
        self.closed, self.capture_failure = False, None
        self.keys = _normalize_keys(key)
        if (base_url is None) == (azure_endpoint is None):
            raise ValueError("Specify exactly one of base_url or azure_endpoint")
        if client is not None and not isinstance(client, openai.AsyncOpenAI):
            raise TypeError("The upstream boundary requires an AsyncOpenAI client")
        if client is not None and client.max_retries != 0:
            raise ValueError("SDK retries must be zero; the upstream boundary owns retries")
        if client is not None and len(self.keys) != 1:
            raise ValueError("An injected client supports exactly one credential")
        write(self.directory / "transport-policy.json", self.policy)
        write(self.directory / "credential-policy.json", {
            "schema": "provider-credential-pool/v1",
            "credential_slots": len(self.keys),
            "selection": "logical-turn round-robin; advance slot on retry",
            "secret_values_persisted": False,
            "protocol": "azure-openai" if azure_endpoint is not None else "openai-compatible",
            "api_version": api_version if azure_endpoint is not None else None,
        })
        if client is not None:
            self.clients = (client,)
        elif azure_endpoint is not None:
            self.clients = tuple(
                openai.AsyncAzureOpenAI(
                    api_key=value, azure_endpoint=azure_endpoint, api_version=api_version,
                    max_retries=0, timeout=self.policy["request_timeout_seconds"],
                    default_headers={"X-Request-ID": "rescope-llm-" + uuid.uuid4().hex},
                )
                for value in self.keys
            )
        else:
            self.clients = tuple(
                openai.AsyncOpenAI(
                    api_key=value, base_url=base_url, max_retries=0,
                    timeout=self.policy["request_timeout_seconds"],
                    default_headers={"X-Request-ID": "rescope-llm-" + uuid.uuid4().hex},
                )
                for value in self.keys
            )
        self.client = self.clients[0]
        self.loop = asyncio.new_event_loop()
        self.hook = self.capture
        for value in self.clients:
            value._client.event_hooks["request"].append(self.hook)
        self._physical_attempt = 0
        self._record = None

    async def capture(self, request):
        try:
            actual = json.loads(request.content)
            if actual != self.expected:
                raise ValueError("Provider serialization changed the admitted request")
            self.wires += 1
            write(self.directory / "wire" / f"{self.wires:04d}.json", actual)
        except Exception:
            # The SDK wraps request-hook errors as connection failures. Preserve
            # local validation/storage failures so they cannot trigger a retry.
            self.capture_failure = ValueError("Provider request audit failed")
            raise self.capture_failure from None

    async def _attempt(self, body, timeout_seconds):
        slot = ((self.count - 1) + self._physical_attempt) % len(self.clients)
        self._physical_attempt += 1
        self.client = self.clients[slot]
        if self._record is not None:
            self._record({
                "kind": "transport_credential_selected",
                "logical_turn": self.count,
                "attempt": self._physical_attempt,
                "credential_slot": f"slot-{slot + 1}",
                "credential_slots": len(self.clients),
            })
        timeout = httpx.Timeout(timeout_seconds, connect=min(30.0, timeout_seconds),
                                write=min(30.0, timeout_seconds), pool=min(30.0, timeout_seconds))
        return await asyncio.wait_for(
            self.client.with_options(timeout=timeout, max_retries=0).chat.completions.create(**body),
            timeout=timeout_seconds,
        )

    def _create(self, body, timeout_seconds):
        self.capture_failure = None
        try:
            # wait_for completes cancellation before another attempt can start.
            return self.loop.run_until_complete(self._attempt(body, timeout_seconds))
        except Exception:
            if self.capture_failure is not None:
                raise self.capture_failure from None
            raise

    @staticmethod
    def _require_sync_thread():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError("Call the synchronous LLM provider outside an active event loop")

    def __call__(self, body):
        self._require_sync_thread()
        with self.lock:
            if self.closed:
                raise RuntimeError("LLM provider is closed")
            if body.get("stream", False):
                raise ValueError("Audited evaluation requests must be non-streaming")
            self.count += 1
            self.expected = deepcopy(body)
            self._physical_attempt = 0
            write(self.directory / "requests" / f"{self.count:04d}.json", self.expected)
            events = []

            def record(row):
                events.append(row)
                write(self.directory / "transport" / f"{self.count:04d}-{len(events):02d}.json", row)

            self._record = record
            try:
                value = retry.request_with_transient_retry(
                    create=self._create, body=self.expected, logical_turn=self.count,
                    policy=self.policy, record=record,
                )
            finally:
                self._record = None
            raw = value.model_dump(mode="json", exclude_none=True)
            write(self.directory / "responses" / f"{self.count:04d}.json", raw)
            choices = raw.get("choices") or []
            if len(choices) != 1 or choices[0].get("finish_reason") not in {"stop", "tool_calls"}:
                raise RuntimeError("Model completion malformed or truncated; not a transport retry")
            return raw

    def close(self):
        self._require_sync_thread()
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                for client in self.clients:
                    hooks = client._client.event_hooks["request"]
                    if self.hook in hooks:
                        hooks.remove(self.hook)
                    self.loop.run_until_complete(client.close())
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            finally:
                self.loop.close()

    def secret_values(self):
        return self.keys

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
