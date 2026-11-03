"""Host-wide admission pools for lightweight benchmark work."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from rescopebench import pathing


POLICY_SCHEMA = "rescopebench-host-scheduler/v1"
DEFAULT_LIGHT_CONTAINER_SLOTS = 20
DEFAULT_CPU_HEAVY_SLOTS = 1
DEFAULT_RESOURCE_WAIT_SECONDS = 3600.0


def _number(name, default, *, integer=False, positive=False):
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
    return value


def scheduler_root() -> Path:
    return Path(
        os.environ.get(
            "RESCOPE_BENCH_SCHEDULER_ROOT",
            pathing.OUTPUT_ROOT / ".scheduler",
        )
    ).expanduser().resolve()


def policy_from_environment() -> dict:
    return {
        "schema": POLICY_SCHEMA,
        "light_container_slots": _number(
            "RESCOPE_LIGHT_CONTAINER_SLOTS",
            DEFAULT_LIGHT_CONTAINER_SLOTS,
            integer=True,
            positive=True,
        ),
        "cpu_heavy_slots": _number(
            "RESCOPE_CPU_HEAVY_SLOTS",
            DEFAULT_CPU_HEAVY_SLOTS,
            integer=True,
            positive=True,
        ),
        "resource_wait_seconds": _number(
            "RESCOPE_RESOURCE_WAIT_SECONDS",
            DEFAULT_RESOURCE_WAIT_SECONDS,
            positive=True,
        ),
        "pool_accounting": {
            "light_container": "Docker or ReAct episode execution",
            "cpu_heavy": "path search, bulk replay, synthesis, or exhaustive validation",
            "independent": True,
        },
    }


def validate_policy(policy):
    if not isinstance(policy, dict) or policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("Unknown host scheduler policy")
    expected = policy_from_environment()
    if set(policy) != set(expected) or policy.get("pool_accounting") != expected["pool_accounting"]:
        raise ValueError("Invalid host scheduler policy fields")
    for name in ("light_container_slots", "cpu_heavy_slots"):
        if type(policy[name]) is not int or policy[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    wait = policy["resource_wait_seconds"]
    if isinstance(wait, bool) or not isinstance(wait, (int, float)) or not math.isfinite(wait) or wait <= 0:
        raise ValueError("resource_wait_seconds must be positive and finite")
    return json.loads(json.dumps(policy))


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    )
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


@contextmanager
def _exclusive(path: Path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class _Lease:
    def __init__(self, pool, slot, descriptor, waited_seconds):
        self.pool = pool
        self.slot = slot
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
            with self.pool._guard:
                self.pool._active.discard((str(self.pool.directory), self.slot))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class SharedSlotPool:
    """Fair file-lock slot pool shared by threads and local processes."""

    _guard = threading.Lock()
    _active = set()

    def __init__(self, directory: Path, capacity: int, *, wait_seconds: float):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Pool capacity must be a positive integer")
        if not isinstance(wait_seconds, (int, float)) or wait_seconds <= 0:
            raise ValueError("Pool wait must be positive")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.capacity = capacity
        self.wait_seconds = float(wait_seconds)
        with _exclusive(self.directory / "coordinator.lock"):
            policy = {"schema": "rescopebench-slot-pool/v1", "capacity": capacity}
            policy_path = self.directory / "policy.json"
            if policy_path.exists():
                if json.loads(policy_path.read_text(encoding="utf-8")) != policy:
                    raise ValueError("Host pool capacity changed while scheduler state exists")
            else:
                _atomic_json(policy_path, policy)
            state_path = self.directory / "state.json"
            if not state_path.exists():
                _atomic_json(state_path, {"next_slot": 0})

    def acquire(self, timeout_seconds=None):
        timeout = self.wait_seconds if timeout_seconds is None else min(
            self.wait_seconds, float(timeout_seconds)
        )
        if timeout <= 0:
            raise ValueError("Pool timeout must be positive")
        started = time.monotonic()
        deadline = started + timeout
        while True:
            with _exclusive(self.directory / "coordinator.lock"):
                state_path = self.directory / "state.json"
                state = json.loads(state_path.read_text(encoding="utf-8"))
                first = int(state["next_slot"]) % self.capacity
                for offset in range(self.capacity):
                    slot = (first + offset) % self.capacity
                    marker = (str(self.directory), slot)
                    with self._guard:
                        if marker in self._active:
                            continue
                        self._active.add(marker)
                    descriptor = os.open(
                        self.directory / f"slot-{slot + 1:03d}.lock",
                        os.O_RDWR | os.O_CREAT,
                        0o600,
                    )
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        os.close(descriptor)
                        with self._guard:
                            self._active.discard(marker)
                        continue
                    state["next_slot"] = (slot + 1) % self.capacity
                    _atomic_json(state_path, state)
                    return _Lease(self, slot, descriptor, time.monotonic() - started)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Host scheduler capacity wait exceeded")
            time.sleep(min(0.05, remaining))


def light_container_pool(policy: dict | None = None) -> SharedSlotPool:
    policy = validate_policy(policy or policy_from_environment())
    return SharedSlotPool(
        scheduler_root() / "light-containers",
        policy["light_container_slots"],
        wait_seconds=policy["resource_wait_seconds"],
    )


def cpu_heavy_pool(policy: dict | None = None) -> SharedSlotPool:
    policy = validate_policy(policy or policy_from_environment())
    return SharedSlotPool(
        scheduler_root() / "cpu-heavy",
        policy["cpu_heavy_slots"],
        wait_seconds=policy["resource_wait_seconds"],
    )


def model_api_directory() -> Path:
    return scheduler_root() / "model-api"
