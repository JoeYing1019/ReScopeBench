"""Provider configuration for a future clean export; not installed in live runs.

Connection/authentication differ between deployments, not business contracts.
No case imports, model-name heuristics, credential stores or network calls here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
import os
from pathlib import Path
import re
import shlex
import stat
from typing import Mapping, MutableMapping
from urllib.parse import urlsplit


KEY_NAME = re.compile(r"(?:OPENAI_API_KEY|AZURE_OPENAI_API_KEY)(?:_SECONDARY|_[1-9][0-9]*)?\Z")
LEGACY_KEY_NAME = re.compile(r"RESCOPE_PROVIDER_API_KEY(?:_SECONDARY|_[1-9][0-9]*)?\Z")
APIS = {"chat_completions", "responses"}
TOKEN_FIELDS = {"max_tokens", "max_completion_tokens"}


def endpoint(value):
    if not isinstance(value, str) or value != value.strip() or any(ord(char) < 33 for char in value):
        raise ValueError("Provider endpoint must be a URL without whitespace")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError:
        raise ValueError("Invalid provider endpoint") from None
    if (parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError("Provider endpoint must not contain credentials, query or fragment")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Non-loopback provider endpoints require HTTPS")
    return value.rstrip("/")


@dataclass(frozen=True)
class ProviderConfig:
    api: str
    protocol: str
    endpoint: str
    api_version: str | None = None
    output_token_field: str = "max_tokens"
    seed_mode: str = "send"

    def __post_init__(self):
        if self.api not in APIS or self.protocol not in {"openai", "azure"}:
            raise ValueError("Unsupported explicitly selected provider API/protocol")
        if self.endpoint != endpoint(self.endpoint):
            raise ValueError("Provider endpoint must be normalized")
        if self.protocol == "azure":
            if self.api != "chat_completions":
                raise ValueError("Azure Responses integration is not verified by this export helper")
            if not isinstance(self.api_version, str) or not self.api_version.strip():
                raise ValueError("Azure requires an explicit API version")
        elif self.api_version is not None:
            raise ValueError("API version is only meaningful for Azure")
        if self.output_token_field not in TOKEN_FIELDS or self.seed_mode not in {"send", "omit"}:
            raise ValueError("Unsupported output-token field or seed policy")
        if self.api == "responses" and (self.seed_mode != "omit" or self.output_token_field != "max_tokens"):
            raise ValueError("Responses uses its existing max_output_tokens conversion and omits seed")

    @classmethod
    def from_environment(cls, api="chat_completions", environ=None):
        values = os.environ if environ is None else environ
        protocol = values.get("RESCOPE_PROVIDER_PROTOCOL", "openai")
        if protocol == "azure":
            target = values.get("AZURE_OPENAI_ENDPOINT", "")
            version = values.get("OPENAI_API_VERSION")
        else:
            target = values.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            version = None
        seed = values.get("RESCOPE_SEED_MODE", "omit" if api == "responses" else "send")
        return cls(api, protocol, endpoint(target), version,
                   values.get("RESCOPE_OUTPUT_TOKEN_FIELD", "max_tokens"), seed)

    def connection_kwargs(self):
        if self.protocol == "azure":
            return {"azure_endpoint": self.endpoint, "api_version": self.api_version}
        return {"base_url": self.endpoint}

    def public_record(self):
        return {"schema": "rescopebench-portable-provider-configuration/v1", **asdict(self),
                "changes_business_budget": False, "model_name_inference": False}

    @classmethod
    def from_record(cls, value):
        """Decode a frozen connection, without consulting current environment defaults."""
        names = {item.name for item in fields(cls)}
        if not isinstance(value, dict) or not names.issubset(value):
            raise ValueError("Incomplete portable provider record")
        config = cls(**{name: value[name] for name in names})
        return config


@dataclass(frozen=True)
class CredentialPool:
    keys: tuple[str, ...] = field(repr=False)
    source_names: tuple[str, ...]
    explicit_file_used: bool

    def public_record(self):
        return {"credential_slots": len(self.keys), "source_names": list(self.source_names),
                "explicit_file_used": self.explicit_file_used, "system_store_used": False,
                "secret_values_persisted": False}

    def bind_legacy_environment(self, environ: MutableMapping[str, str]):
        # The pinned worker reads these variable names. Replace only that key
        # pool in memory; never merge an unrelated inherited credential pool.
        for name in tuple(environ):
            if LEGACY_KEY_NAME.fullmatch(name) or name == "RESCOPE_RESPONSES_API_KEY":
                del environ[name]
        for index, value in enumerate(self.keys):
            name = ("RESCOPE_PROVIDER_API_KEY" if index == 0 else
                    "RESCOPE_PROVIDER_API_KEY_SECONDARY" if index == 1 else f"RESCOPE_PROVIDER_API_KEY_{index + 1}")
            environ[name] = value


def _key_order(name, prefix):
    if name == prefix:
        return 0, name
    if name == prefix + "_SECONDARY":
        return 1, name
    return int(name.rsplit("_", 1)[1]) + 1, name


def load_credentials(config: ProviderConfig, *, environ: Mapping[str, str] | None = None,
                     path: Path | None = None):
    values = dict(os.environ if environ is None else environ)
    prefix = "AZURE_OPENAI_API_KEY" if config.protocol == "azure" else "OPENAI_API_KEY"

    def selected(name):
        return KEY_NAME.fullmatch(name) and (name == prefix or name.startswith(prefix + "_"))

    from_file = False
    if path is not None:
        path = Path(path)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("Credential file must be a private regular file")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            name, separator, value = line.partition("=")
            name = name.strip()
            if not separator or not selected(name):
                continue
            try:
                parsed = shlex.split(value, comments=True)
            except ValueError:
                raise ValueError("Malformed credential assignment") from None
            if len(parsed) != 1 or not parsed[0].strip():
                raise ValueError("Credential assignment must contain one nonempty value")
            values[name] = parsed[0]
            from_file = True
    names = sorted((name for name in values if selected(name)), key=lambda name: _key_order(name, prefix))
    keys, sources = [], []
    for name in names:
        value = values[name]
        if not isinstance(value, str):
            raise ValueError("Credential values must be strings")
        if not value.strip():
            continue
        if any(char.isspace() for char in value):
            raise ValueError("Credential values must not contain whitespace")
        if value not in keys:
            keys.append(value)
            sources.append(name)
    if not keys:
        raise ValueError(f"Supply {prefix} for this explicit provider protocol")
    return CredentialPool(tuple(keys), tuple(sources), from_file)


def normalize_chat_request(body, config: ProviderConfig):
    if config.api != "chat_completions":
        raise ValueError("Use the pinned Responses conversation converter for Responses")
    result = deepcopy(body)
    changes = []
    present = [name for name in TOKEN_FIELDS if name in result]
    if len(present) != 1:
        raise ValueError("Expected exactly one explicit output-token allowance")
    old = present[0]
    allowance = result[old]
    if type(allowance) is not int or allowance < 1:
        raise ValueError("Output-token allowance must be a positive integer")
    if old != config.output_token_field:
        result[config.output_token_field] = result.pop(old)
        changes.append({"field": "output_token_field", "from": old, "to": config.output_token_field,
                        "allowance_unchanged": True})
    if config.seed_mode == "omit" and "seed" in result:
        del result["seed"]
        changes.append({"field": "seed", "action": "explicit_omission"})
    return result, {"configuration": config.public_record(), "changes": changes,
                    "messages_tools_parallel_and_model_unchanged": True,
                    "reasoning_sampling_and_other_parameters_unchanged": True}
