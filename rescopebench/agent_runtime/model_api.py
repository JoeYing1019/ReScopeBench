"""Host-wide provider scheduling with per-credential concurrency limits."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import asyncio
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
import uuid

import httpx

from . import provider as provider_base
from . import transport_retry as retry
from rescopebench import scheduling


SCHEDULER_SCHEMA = "provider-credential-scheduler/v2"
DEFAULT_MAX_CONCURRENCY_PER_KEY = 4
DEFAULT_MIN_START_INTERVAL_SECONDS = 0.25
DEFAULT_CAPACITY_WAIT_SECONDS = 60.0
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 2.0
DEFAULT_RECOVERY_SUCCESSES = 4

api_keys_from_environment = provider_base.api_keys_from_environment


def _environment_number(name, default, *, integer=False, positive=False):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def scheduler_config_from_environment():
    """Read only non-secret scheduler controls from the process environment."""
    return {
        "schema": SCHEDULER_SCHEMA,
        "max_concurrency_per_key": _environment_number(
            "RESCOPE_LLM_MAX_CONCURRENCY_PER_KEY",
            DEFAULT_MAX_CONCURRENCY_PER_KEY,
            integer=True,
            positive=True,
        ),
        "min_start_interval_seconds": _environment_number(
            "RESCOPE_LLM_MIN_START_INTERVAL_SECONDS",
            DEFAULT_MIN_START_INTERVAL_SECONDS,
        ),
        "capacity_wait_seconds": _environment_number(
            "RESCOPE_LLM_SCHEDULER_WAIT_SECONDS",
            DEFAULT_CAPACITY_WAIT_SECONDS,
            positive=True,
        ),
        "rate_limit_cooldown_seconds": _environment_number(
            "RESCOPE_LLM_RATE_LIMIT_COOLDOWN_SECONDS",
            DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
            positive=True,
        ),
        "recovery_successes": _environment_number(
            "RESCOPE_LLM_RECOVERY_SUCCESSES",
            DEFAULT_RECOVERY_SUCCESSES,
            integer=True,
            positive=True,
        ),
    }


def validate_scheduler_config(config):
    if not isinstance(config, dict) or config.get("schema") != SCHEDULER_SCHEMA:
        raise ValueError("Unknown credential scheduler policy")
    names = {
        "schema",
        "max_concurrency_per_key",
        "min_start_interval_seconds",
        "capacity_wait_seconds",
        "rate_limit_cooldown_seconds",
        "recovery_successes",
    }
    if set(config) != names:
        raise ValueError("Unknown credential scheduler policy fields")
    for name in ("max_concurrency_per_key", "recovery_successes"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in (
        "min_start_interval_seconds",
        "capacity_wait_seconds",
        "rate_limit_cooldown_seconds",
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config["min_start_interval_seconds"] < 0:
        raise ValueError("min_start_interval_seconds must be non-negative")
    if config["capacity_wait_seconds"] <= 0 or config["rate_limit_cooldown_seconds"] <= 0:
        raise ValueError("Scheduler wait and cooldown must be positive")
    return deepcopy(config)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    )
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


@contextmanager
def _file_lock(path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class _CredentialLease:
    def __init__(self, scheduler, slot, lane, descriptor, waited_seconds):
        self.scheduler = scheduler
        self.slot = slot
        self.lane = lane
        self.descriptor = descriptor
        self.waited_seconds = waited_seconds
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)
            marker = (str(self.scheduler.directory), self.slot, self.lane)
            with self.scheduler._local_guard:
                self.scheduler._local_active.discard(marker)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class SharedCredentialScheduler:
    """Cross-thread/process round-robin admission with adaptive 429 pressure."""

    _local_guard = threading.Lock()
    _local_active = set()

    def __init__(self, directory, credential_slots, *, config=None):
        if type(credential_slots) is not int or credential_slots <= 0:
            raise ValueError("credential_slots must be a positive integer")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.credential_slots = credential_slots
        self.config = validate_scheduler_config(
            config if config is not None else scheduler_config_from_environment()
        )
        self.policy = {
            **self.config,
            "credential_slots": credential_slots,
            "maximum_total_concurrency": (
                credential_slots * self.config["max_concurrency_per_key"]
            ),
            "selection": "cross-process round-robin",
            "scope": "host-global across runs and environments",
            "pressure_control": "429 halves slot concurrency; verified successes restore one lane",
            "secret_values_persisted": False,
        }
        with _file_lock(self.directory / "coordinator.lock"):
            policy_path = self.directory / "policy.json"
            if policy_path.exists():
                if json.loads(policy_path.read_text(encoding="utf-8")) != self.policy:
                    raise ValueError("Credential scheduler policy differs within one run")
            else:
                _atomic_json(policy_path, self.policy)
            state_path = self.directory / "state.json"
            if not state_path.exists():
                _atomic_json(state_path, self._initial_state())

    def _initial_state(self):
        return {
            "schema": SCHEDULER_SCHEMA,
            "next_slot": 0,
            "slots": [
                {
                    "next_start_at": 0.0,
                    "effective_concurrency": self.config["max_concurrency_per_key"],
                    "consecutive_rate_limits": 0,
                    "successes_since_pressure": 0,
                }
                for _ in range(self.credential_slots)
            ],
        }

    def _state(self):
        value = json.loads((self.directory / "state.json").read_text(encoding="utf-8"))
        if (
            value.get("schema") != SCHEDULER_SCHEMA
            or type(value.get("next_slot")) is not int
            or not isinstance(value.get("slots"), list)
            or len(value["slots"]) != self.credential_slots
        ):
            raise ValueError("Credential scheduler state is invalid")
        return value

    def acquire(self, timeout_seconds=None):
        wait_limit = self.config["capacity_wait_seconds"]
        if timeout_seconds is not None:
            if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
                raise ValueError("Scheduler timeout must be positive")
            wait_limit = min(wait_limit, float(timeout_seconds))
        started = time.monotonic()
        deadline = started + wait_limit
        while True:
            earliest = None
            with _file_lock(self.directory / "coordinator.lock"):
                state = self._state()
                now = time.time()
                first = state["next_slot"] % self.credential_slots
                for offset in range(self.credential_slots):
                    slot = (first + offset) % self.credential_slots
                    slot_state = state["slots"][slot]
                    available_at = float(slot_state["next_start_at"])
                    if available_at > now:
                        earliest = available_at if earliest is None else min(earliest, available_at)
                        continue
                    lane_limit = int(slot_state["effective_concurrency"])
                    for lane in range(lane_limit):
                        marker = (str(self.directory), slot, lane)
                        with self._local_guard:
                            if marker in self._local_active:
                                continue
                            self._local_active.add(marker)
                        descriptor = os.open(
                            self.directory / f"slot-{slot + 1:02d}-lane-{lane + 1:02d}.lock",
                            os.O_RDWR | os.O_CREAT,
                            0o600,
                        )
                        try:
                            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            os.close(descriptor)
                            with self._local_guard:
                                self._local_active.discard(marker)
                            continue
                        state["next_slot"] = (slot + 1) % self.credential_slots
                        slot_state["next_start_at"] = (
                            now + self.config["min_start_interval_seconds"]
                        )
                        _atomic_json(self.directory / "state.json", state)
                        return _CredentialLease(
                            self, slot, lane, descriptor, time.monotonic() - started
                        )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("LLM credential scheduler capacity wait exceeded")
            delay = 0.05
            if earliest is not None:
                delay = max(0.005, min(delay, earliest - time.time()))
            time.sleep(min(delay, remaining))

    def report_failure(self, slot, category, *, retry_after_seconds=None):
        if category not in {"rate_limit", "server", "connection"}:
            return
        with _file_lock(self.directory / "coordinator.lock"):
            state = self._state()
            slot_state = state["slots"][slot]
            if category == "rate_limit":
                slot_state["consecutive_rate_limits"] += 1
                slot_state["successes_since_pressure"] = 0
                slot_state["effective_concurrency"] = max(
                    1,
                    math.ceil(slot_state["effective_concurrency"] / 2),
                )
                cooldown = min(
                    60.0,
                    self.config["rate_limit_cooldown_seconds"]
                    * (2 ** (slot_state["consecutive_rate_limits"] - 1)),
                )
                if retry_after_seconds is not None:
                    cooldown = max(cooldown, min(60.0, retry_after_seconds))
                slot_state["next_start_at"] = max(
                    float(slot_state["next_start_at"]), time.time() + cooldown
                )
            _atomic_json(self.directory / "state.json", state)

    def report_success(self, slot):
        with _file_lock(self.directory / "coordinator.lock"):
            state = self._state()
            slot_state = state["slots"][slot]
            slot_state["consecutive_rate_limits"] = 0
            if slot_state["effective_concurrency"] < self.config["max_concurrency_per_key"]:
                slot_state["successes_since_pressure"] += 1
                if slot_state["successes_since_pressure"] >= self.config["recovery_successes"]:
                    slot_state["effective_concurrency"] += 1
                    slot_state["successes_since_pressure"] = 0
            _atomic_json(self.directory / "state.json", state)


class Provider(provider_base.Provider):
    """Frozen audited provider plus run-wide credential admission control."""

    def __init__(self, key, directory, *, scheduler_directory, scheduler_config=None, **kwargs):
        keys = provider_base._normalize_keys(key)
        self.scheduler = SharedCredentialScheduler(
            scheduler_directory,
            len(keys),
            config=scheduler_config,
        )
        super().__init__(keys, directory, **kwargs)
        policy_path = self.directory / "credential-policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy.update(
            selection="shared cross-process round-robin; retries reacquire capacity",
            scheduler=self.scheduler.policy,
        )
        _atomic_json(policy_path, policy)

    async def _attempt(self, body, timeout_seconds):
        self._physical_attempt += 1
        attempt = self._physical_attempt
        started = time.monotonic()
        with self.scheduler.acquire(timeout_seconds) as lease:
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("LLM credential scheduler consumed request timeout")
            self.client = self.clients[lease.slot]
            if self._record is not None:
                self._record({
                    "kind": "transport_credential_selected",
                    "logical_turn": self.count,
                    "attempt": attempt,
                    "credential_slot": f"slot-{lease.slot + 1}",
                    "credential_slots": len(self.clients),
                    "credential_lane": lease.lane + 1,
                    "scheduler_wait_seconds": lease.waited_seconds,
                })
            timeout = httpx.Timeout(
                remaining,
                connect=min(30.0, remaining),
                write=min(30.0, remaining),
                pool=min(30.0, remaining),
            )
            try:
                response = await asyncio.wait_for(
                    self.client.with_options(timeout=timeout, max_retries=0)
                    .chat.completions.create(**body),
                    timeout=remaining,
                )
            except Exception as error:
                category, _reason = retry.failure_class(error)
                server_delay = retry.retry_after_seconds(error, time.time())
                self.scheduler.report_failure(
                    lease.slot,
                    category,
                    retry_after_seconds=server_delay,
                )
                raise
            self.scheduler.report_success(lease.slot)
            return response


class RotatingProvider(provider_base.Provider):
    """Direct provider: randomize the first credential, then rotate on retries.

    This intentionally avoids host-wide adaptive throttling. The transport layer
    still applies the configured timeout, backoff, and retry limits, while the
    base provider advances to the next credential after every failed attempt.
    """

    def __init__(self, key, directory, **kwargs):
        keys = provider_base._normalize_keys(key)
        offset = secrets.randbelow(len(keys))
        rotated = keys[offset:] + keys[:offset]
        super().__init__(rotated, directory, **kwargs)
        policy_path = self.directory / "credential-policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy.update(
            selection="random initial credential; rotate credential on every retry",
            host_wide_throttling=False,
        )
        _atomic_json(policy_path, policy)


def resilient_retry_policy():
    """Keep each attempt fail-fast while allowing transient outages to recover."""

    return retry.retry_policy(
        rate_limit_retries=40,
        server_retries=20,
        connection_retries=20,
        backoff_seconds=1.0,
        backoff_multiplier=1.5,
        backoff_cap_seconds=30.0,
        request_timeout_seconds=60.0,
        max_elapsed_seconds=1800.0,
    )


def global_scheduler_directory() -> Path:
    return scheduling.model_api_directory()
