"""Explicit provider adapters for export rehearsal, not installed in live runs.

The current audited I/O, retry loop and Responses conversation are reused. No
case, environment variable, Keychain entry or running transport is modified.
"""

from copy import deepcopy
import json
import os
import threading

from rescopebench import responses_provider
from rescopebench.agent_runtime import model_api
from rescopebench.portable_provider_config import ProviderConfig, CredentialPool, load_credentials, normalize_chat_request


def transport(config: ProviderConfig, *, reasoning=None, policy=None):
    record = {**config.public_record(), "schema": "rescopebench-portable-transport/v1",
              "retry_policy": model_api.retry.validate_policy(
                  model_api.resilient_retry_policy() if policy is None else policy)}
    if config.api == "responses":
        if not isinstance(reasoning, dict):
            raise ValueError("Supply an explicit reasoning mapping for Responses")
        record.update(reasoning=deepcopy(reasoning), store=False,
                      seed="omitted_unsupported", tool_schema_strict=False,
                      output_replay="all_items_for_retained_assistant_messages")
    elif reasoning is not None:
        raise ValueError("Chat reasoning settings belong in the unchanged request")
    return record


def validate_transport(record):
    """Validate a saved plan against itself, not an operator's changed shell."""
    config = ProviderConfig.from_record(record)
    expected = transport(config, reasoning=record.get("reasoning"), policy=record.get("retry_policy"))
    if json.dumps(record, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("Portable provider transport differs from its declared contract")
    return expected


def transport_from_environment(api="chat_completions", *, environ=None):
    values = os.environ if environ is None else environ
    config = ProviderConfig.from_environment(api, values)
    reasoning = None
    if api == "responses":
        try:
            reasoning = json.loads(values["RESCOPE_RESPONSES_REASONING_JSON"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Set RESCOPE_RESPONSES_REASONING_JSON explicitly for Responses") from None
    return transport(config, reasoning=reasoning)


def provider_from_transport(directory, record, *, pool=None, credentials_file=None,
                            environ=None, client=None):
    frozen = validate_transport(record)
    config = ProviderConfig.from_record(frozen)
    if pool is None:
        pool = load_credentials(config, path=credentials_file, environ=environ)
    return create_provider(config, pool, directory, reasoning=frozen.get("reasoning"),
                           policy=frozen["retry_policy"], client=client)


def redact(value, keys):
    if isinstance(value, str):
        for key in sorted(keys, key=len, reverse=True):
            value = value.replace(key, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: redact(item, keys) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, keys) for item in value]
    return value


class _RedactedErrors:
    def _create(self, body, timeout_seconds):
        try:
            return super()._create(body, timeout_seconds)
        except Exception as error:
            # Keep the original exception class/status for the existing retry
            # taxonomy, while protecting explicitly supplied, non-env keys.
            for name in ("body", "message"):
                if hasattr(error, name):
                    setattr(error, name, redact(getattr(error, name), self.keys))
            error.args = tuple(redact(item, self.keys) for item in error.args)
            raise


class PortableChatProvider(_RedactedErrors, model_api.RotatingProvider):
    def __init__(self, config, pool, directory, *, client=None, policy=None):
        if config.api != "chat_completions":
            raise ValueError("Chat provider requires chat_completions configuration")
        record = transport(config, policy=policy)
        self.portable_config = config
        self.normalization_lock = threading.Lock()
        super().__init__(pool.keys, directory, **config.connection_kwargs(),
                         policy=record["retry_policy"], client=client)
        model_api.provider_base.write(self.directory / "portable-configuration.json",
                                      {"transport": record, "credentials": pool.public_record()})

    def __call__(self, body):
        with self.normalization_lock:
            if self.closed:
                raise RuntimeError("provider is closed")
            request, normalization = normalize_chat_request(body, self.portable_config)
            model_api.provider_base.write(
                self.directory / "normalization" / f"{self.count + 1:04d}.json", normalization)
            return super().__call__(request)


class PortableResponsesProvider(_RedactedErrors, responses_provider.ResponsesProvider):
    def __init__(self, config, pool, directory, *, reasoning, client=None, policy=None):
        if config.api != "responses" or config.protocol != "openai":
            raise ValueError("Responses provider requires explicit OpenAI-compatible Responses")
        if len(pool.keys) != 1:
            raise ValueError("Pinned Responses transport supports one credential per worker")
        self.config = transport(config, reasoning=reasoning, policy=policy)
        self.conversation = responses_provider.Conversation()
        # Only construction changes: use the explicit endpoint rather than the
        # original company-specific validate_transport()/credential factory.
        model_api.provider_base.Provider.__init__(
            self, pool.keys, directory, **config.connection_kwargs(),
            policy=self.config["retry_policy"], client=client)
        model_api.provider_base.write(self.directory / "portable-configuration.json",
                                      {"transport": self.config, "credentials": pool.public_record()})


def create_provider(config: ProviderConfig, pool: CredentialPool, directory, *,
                    reasoning=None, client=None, policy=None):
    if config.api == "responses":
        return PortableResponsesProvider(config, pool, directory, reasoning=reasoning,
                                         client=client, policy=policy)
    if reasoning is not None:
        raise ValueError("Chat reasoning settings belong in the unchanged request")
    return PortableChatProvider(config, pool, directory, client=client, policy=policy)
