"""Run the Hermes worker in a remote sandbox instead of a local container.

The host loop stays unchanged: it still talks to a local ``exchange`` directory
and a Popen-like object. This module translates the release's ``docker run``
arguments into one sandbox from the configured provider and mirrors the exchange
directory in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import threading
import time
import uuid

from rescope_harness.sandbox.providers import SandboxClient, SandboxFatalError, SandboxSettings, client_for

REMOTE_STATE = "/tmp/rescope_harness"
REMOTE_PYTHON = "/usr/local/bin/python3"
# Env baked into the release image rescopebench-portable:reviewed18-20261015; `docker run`
# starts from it, so the sandbox process starts from the same set instead of provider defaults.
IMAGE_ENV = {
    "PATH": "/opt/hermes/.venv/bin:/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "PYTHON_VERSION": "3.12.14",
    "UV_PROJECT_ENVIRONMENT": "/opt/hermes/.venv",
    "UV_PYTHON_DOWNLOADS": "never",
    "UV_LINK_MODE": "copy",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
}
IMAGE_WORKDIR = "/agent/work"
# Hermes timeouts and context windows are set in the image (hermes_image/), not here:
# the worker starts under `env -i`, so an environment variable would never reach it.

# How long a task keeps asking for a sandbox before giving up. Capacity shortages
# under heavy load clear within minutes; an hour covers a bad stretch.
PROVISION_DEADLINE_SECONDS = 3600.0
# Under hundreds of concurrent sandboxes a provider may fail to deliver some execs
# although the sandbox is healthy. Every exec here is a listing, a publish or setup,
# all safe to repeat, so retry them for this long before calling the sandbox lost.
EXEC_RETRY_SECONDS = 900.0

# Long-poll listing of /exchange. The exit code is read before listing, so a
# listing that reports an exit code already contains every file the worker wrote.
SYNC_SCRIPT = r'''
import hashlib, json, os, sys, time
root, state = "/exchange", "/tmp/rescope_harness"
previous, wait = sys.argv[1], float(sys.argv[2])
deadline = time.monotonic() + wait
while True:
    try:
        with open(state + "/exit_code") as handle:
            code = handle.read().strip()
    except FileNotFoundError:
        code = None
    files = []
    for directory, _, names in os.walk(root):
        for name in names:
            if name.endswith((".tmp", ".part")):
                continue
            path = os.path.join(directory, name)
            relative = os.path.relpath(path, root)
            if relative.split(os.sep)[0] == "responses":
                continue
            try:
                info = os.stat(path)
            except FileNotFoundError:
                continue
            if info.st_size == 0:
                # Mid-creation: the worker has created but not yet written the file.
                continue
            files.append([relative, info.st_mtime_ns, info.st_size])
    files.sort()
    fingerprint = hashlib.sha256(json.dumps([files, code]).encode()).hexdigest()
    if fingerprint != previous or time.monotonic() >= deadline:
        break
    time.sleep(0.1)
print(json.dumps({"fingerprint": fingerprint, "files": files, "exit_code": code}))
'''


@dataclass
class RunSpec:
    name: str = ""
    env: dict[str, str] = field(default_factory=dict)
    mounts: list[tuple[str, str, str]] = field(default_factory=list)
    tmpfs: list[str] = field(default_factory=list)
    user: str = ""
    entrypoint: str = ""
    image: str = ""
    args: list[str] = field(default_factory=list)
    memory: str = ""
    cpus: str = ""
    pids_limit: int | None = None
    flags: list[str] = field(default_factory=list)


def parse_run(command: list[str]) -> RunSpec:
    if command[:2] != ["docker", "run"]:
        raise ValueError("Only `docker run` commands can be moved to a remote sandbox")
    spec, index = RunSpec(), 2
    valued = {"--name", "-e", "-v", "--tmpfs", "--entrypoint", "--user", "-u"}
    while index < len(command):
        part = command[index]
        if part in valued:
            key, value = part, command[index + 1]
            index += 2
        elif part.startswith("--") and "=" in part:
            key, value = part.split("=", 1)
            index += 1
        elif part.startswith("-"):
            spec.flags.append(part)
            index += 1
            continue
        else:
            spec.image, spec.args = part, list(command[index + 1:])
            break
        if key == "--name":
            spec.name = value
        elif key == "-e":
            name, separator, item = value.partition("=")
            spec.env[name] = item if separator else os.environ.get(name, "")
        elif key == "-v":
            host, container, *mode = value.split(":")
            spec.mounts.append((host, container, mode[0] if mode else "rw"))
        elif key == "--tmpfs":
            spec.tmpfs.append(value.split(":", 1)[0])
        elif key == "--entrypoint":
            spec.entrypoint = value
        elif key in {"--user", "-u"}:
            spec.user = value
        elif key == "--memory":
            spec.memory = value
        elif key == "--cpus":
            spec.cpus = value
        elif key == "--pids-limit":
            spec.pids_limit = int(value)
        else:
            spec.flags.append(f"{key}={value}")
    if not spec.image or not spec.name:
        raise ValueError("docker run command has no image or container name")
    return spec


def docker_memory_bytes(value: str) -> int:
    units = {"b": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}
    value = value.strip().lower()
    return int(float(value[:-1]) * units[value[-1]]) if value and value[-1] in units else int(value or 0)


def sandbox_resources(spec: RunSpec) -> tuple[str, str]:
    """Sandbox request from the run's --cpus / --memory; RESCOPE_SANDBOX_CPU / _MEMORY override it."""
    cpu = os.environ.get("RESCOPE_SANDBOX_CPU") or f"{max(1, round(float(spec.cpus or 1) * 1000))}m"
    memory = os.environ.get("RESCOPE_SANDBOX_MEMORY") or f"{max(1, docker_memory_bytes(spec.memory or '4g') // 1024 ** 2)}Mi"
    return cpu, memory


def hermes_commit() -> str:
    manifest = Path(__file__).resolve().parents[2] / "rescopebench" / "vendor" / "image-manifest.json"
    return json.loads(manifest.read_text(encoding="utf-8"))["hermes_commit"]


# Containers by docker name, kept after exit so `docker inspect` receipts work.
REGISTRY: dict[str, "RemoteContainer"] = {}
_registry_lock = threading.Lock()


class RemoteContainer:
    """Popen-compatible handle for one worker run inside a remote sandbox."""

    def __init__(self, command: list[str], *, provider: str, stdout=None, settings: SandboxSettings | None = None):
        self.spec = parse_run(command)
        self.settings = settings or SandboxSettings.from_environment()
        self.provider = provider
        self.log = stdout
        self.returncode: int | None = None
        self.pid = None
        self.args = command
        self._error: BaseException | None = None
        self._error_reported = False
        self._stop = threading.Event()
        self._known: dict[str, tuple[int, int]] = {}
        self._uploaded: set[str] = set()
        self._deleted = False
        self.exchange_local, self.exchange_remote = self._exchange_mount()
        self.cpu, self.memory = sandbox_resources(self.spec)
        self.receipt = {"sandbox_id": None, "provider": provider, "image": self.settings.image,
                        "cpu": self.cpu, "memory": self.memory, "created_at": time.time()}
        with _registry_lock:
            REGISTRY[self.spec.name] = self
        self.client: SandboxClient = client_for(provider, self.settings)
        try:
            self._start()
        except BaseException:
            self._delete("startup failed")
            raise
        self._threads = [threading.Thread(target=self._guard, args=(target,), daemon=True,
                                          name=f"sandbox-{self.spec.name}-{target.__name__}")
                         for target in (self._download_loop, self._upload_loop, self._heartbeat_loop)]
        for thread in self._threads:
            thread.start()

    def _exchange_mount(self) -> tuple[Path, str]:
        for host, container, mode in self.spec.mounts:
            if mode == "rw" and Path(host).is_dir():
                return Path(host), container
        raise ValueError("Worker command has no read-write exchange directory mount")

    def _exec(self, commands: list[str], *, timeout: float = 120.0) -> dict:
        """client.exec, repeating calls the provider failed to deliver; any other result is returned as is."""
        deadline = time.monotonic() + EXEC_RETRY_SECONDS
        wait = 2.0
        while True:
            result = self.client.exec(commands, timeout=timeout)
            if not result.get("transient") or time.monotonic() + wait > deadline:
                return result
            self.receipt["exec_retries"] = self.receipt.get("exec_retries", 0) + 1
            if self._stop.wait(wait):
                return result
            wait = min(wait * 2, 30.0)

    def _sh(self, script: str, *, timeout: float = 120.0) -> dict:
        result = self._exec(["sh", "-c", script], timeout=timeout)
        if result.get("error") or result.get("exit_code") not in (None, 0):
            raise RuntimeError(f"Sandbox setup command failed: {json.dumps(result)[:800]}")
        return result

    def _provision(self) -> None:
        """Get a ready sandbox, retrying capacity failures before the task starts.

        A create that times out or a sandbox that never becomes ready says nothing
        about the model, and nothing has run yet, so the half-made sandbox is deleted
        and a fresh one is requested with backoff. SandboxFatalError fails at once.
        """
        deadline = time.monotonic() + PROVISION_DEADLINE_SECONDS
        wait, attempt = 15.0, 0
        while True:
            attempt += 1
            try:
                self.receipt["sandbox_id"] = self.client.create(
                    self.settings.image, cpu=self.cpu, memory=self.memory,
                    labels={"rescope/container": self.spec.name[:63]})
                self.client.wait_ready()
                self.receipt["provision_attempts"] = attempt
                return
            except (SandboxFatalError, NotImplementedError):
                raise
            except Exception as error:  # noqa: BLE001
                if time.monotonic() + wait > deadline:
                    raise
                self.receipt.setdefault("provision_failures", []).append(f"{type(error).__name__}: {error}"[:300])
                if self.client.sandbox_id:
                    try:
                        self.client.delete(reason="provision retry")
                    except Exception:  # noqa: BLE001
                        pass
                    self.client.sandbox_id = None
                time.sleep(wait)
                wait = min(wait * 2, 300.0)

    def _start(self) -> None:
        self._provision()
        self.receipt["ready_at"] = time.time()
        directories = {REMOTE_STATE, self.exchange_remote + "/responses"}
        uploads: list[tuple[Path, str]] = []
        for host, container, mode in self.spec.mounts:
            path = Path(host)
            if container == self.exchange_remote:
                continue
            if path.is_dir():
                directories.add(container)
                for item in sorted(path.rglob("*")):
                    if item.is_file():
                        target = f"{container}/{item.relative_to(path).as_posix()}"
                        directories.add(str(Path(target).parent))
                        uploads.append((item, target))
            else:
                directories.add(str(Path(container).parent))
                uploads.append((path, container))
        quoted = " ".join(shlex.quote(item) for item in sorted(directories))
        writable = " ".join(shlex.quote(item) for item in self.spec.tmpfs)
        self._sh(
            f"set -e; mkdir -p {quoted} {shlex.quote(self.exchange_remote)} {writable}; "
            + (f"chmod 1777 {writable}; " if writable else "")
            + f"chown 1000:1000 {shlex.quote(self.exchange_remote)}; chmod 0777 {shlex.quote(self.exchange_remote)}; "
            + "command -v setpriv >/dev/null")
        with tempfile.TemporaryDirectory() as scratch:
            sync = Path(scratch) / "sync.py"
            sync.write_text(SYNC_SCRIPT, encoding="utf-8")
            run = Path(scratch) / "run.sh"
            run.write_text(self._run_script(), encoding="utf-8")
            uploads += [(sync, f"{REMOTE_STATE}/sync.py"), (run, f"{REMOTE_STATE}/run.sh")]
            for local, remote in uploads:
                self.client.upload(local, remote)
        mounted = " ".join(shlex.quote(remote) for _, remote in uploads)
        self._sh(f"set -e; chown root:root {mounted}; chmod 0444 {mounted}")
        self.receipt["task_id"] = self.client.async_exec(["sh", f"{REMOTE_STATE}/run.sh"])
        self.receipt["started_at"] = time.time()

    def _run_script(self) -> str:
        """Start the worker as `docker run` would: image env, the run's env, its user, read-only mounts."""
        user = self.spec.user or "1000:1000"
        uid, _, gid = user.partition(":")
        env = {**IMAGE_ENV, **self.spec.env}
        assignments = " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
        program = " ".join(shlex.quote(part) for part in [self.spec.entrypoint, *self.spec.args] if part)
        return (
            "#!/bin/sh\n"
            f"cd {shlex.quote(IMAGE_WORKDIR)} 2>/dev/null || cd /\n"
            f"env -i {assignments} setpriv --reuid={uid} --regid={gid or uid} --clear-groups --no-new-privs -- "
            f"{program} > {REMOTE_STATE}/worker.log 2>&1\n"
            "code=$?\n"
            f"echo \"$code\" > {REMOTE_STATE}/exit_code.part && mv {REMOTE_STATE}/exit_code.part {REMOTE_STATE}/exit_code\n"
        )

    def _guard(self, target) -> None:
        try:
            target()
        except BaseException as error:
            if not self._stop.is_set():
                self._error = error
                self._stop.set()

    def _download_loop(self) -> None:
        fingerprint = "-"
        while not self._stop.is_set():
            result = self._exec([REMOTE_PYTHON, "-I", "-S", f"{REMOTE_STATE}/sync.py", fingerprint, "15"],
                                timeout=90.0)
            lines = (result.get("stdout") or "").strip().splitlines()
            if not lines:
                raise RuntimeError(f"Exchange listing returned no output: {json.dumps(result)[:800]}")
            listing = json.loads(lines[-1])
            fingerprint = listing["fingerprint"]
            for relative, mtime, size in sorted(listing["files"], key=lambda row: (row[1], row[0])):
                if self._known.get(relative) != (mtime, size):
                    self._fetch(relative, mtime, size)
            if listing["exit_code"] is not None:
                self._finish(int(listing["exit_code"]))
                return

    def _fetch(self, relative: str, mtime: int, size: int) -> None:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe exchange path from sandbox: {relative}")
        content = self.client.download(f"{self.exchange_remote}/{relative}")
        complete = len(content) == size
        if complete and relative.endswith(".json"):
            # The listing can catch a file mid-write; a short copy would make the host
            # loop's json.loads fail. Publish only complete, parseable JSON.
            try:
                json.loads(content)
            except ValueError:
                complete = False
        if not complete:
            self._known[relative] = (-1, -1)
            return
        target = self.exchange_local / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.part")
        partial.write_bytes(content)
        os.utime(partial, ns=(mtime, mtime))
        os.replace(partial, target)
        self._known[relative] = (mtime, size)

    def _upload_loop(self) -> None:
        responses = self.exchange_local / "responses"
        while not self._stop.is_set():
            for path in sorted(responses.glob("*.json")) if responses.is_dir() else []:
                if path.name in self._uploaded:
                    continue
                target = f"{self.exchange_remote}/responses/{path.name}"
                try:
                    if not json.loads(path.read_text()):
                        continue
                except (OSError, ValueError):
                    continue   # not fully written yet; retry on the next pass
                self.client.upload(path, target + ".part")
                # A retried mv may find its first try already done.
                part, final = shlex.quote(target + ".part"), shlex.quote(target)
                moved = self._exec(["sh", "-c", f"mv -f {part} {final} 2>/dev/null || test -f {final}"])
                if moved.get("error") or moved.get("exit_code") not in (None, 0):
                    raise RuntimeError(f"Cannot publish response {path.name}: {json.dumps(moved)[:500]}")
                self._uploaded.add(path.name)
            self._stop.wait(0.05)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(60):
            try:
                self.client.heartbeat()
            except Exception:  # noqa: BLE001
                pass

    def _finish(self, code: int) -> None:
        try:
            text = self.client.download(f"{REMOTE_STATE}/worker.log").decode("utf-8", errors="replace")
        except Exception as error:  # noqa: BLE001
            text = f"[rescope_harness] worker.log unavailable: {type(error).__name__}: {error}\n"
        if self.log is not None:
            self.log.write(text)
            self.log.flush()
        self.receipt.update(exit_code=code, finished_at=time.time())
        self.returncode = code
        self._stop.set()
        self._delete("worker exited")

    def _delete(self, reason: str) -> None:
        if self._deleted:
            return
        self._deleted = True
        try:
            self.client.delete(reason=reason)
            self.receipt["deleted"] = True
        except Exception as error:  # noqa: BLE001
            self.receipt["delete_error"] = f"{type(error).__name__}: {error}"[:500]
        finally:
            self.client.close()

    # Popen interface used by the host loop.
    def poll(self):
        # Raise a mirroring failure once inside the host loop; the host's finally
        # block polls again and must not have its evidence writes interrupted.
        if self._error is not None and not self._error_reported:
            self._error_reported = True
            self.terminate()
            raise RuntimeError(f"Remote sandbox failed for {self.spec.name}") from self._error
        return self.returncode

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(0.1)
        return self.returncode

    def terminate(self):
        self._stop.set()
        self._delete("terminated by host")
        if self.returncode is None:
            self.returncode = -15

    kill = terminate

    def inspect(self) -> dict:
        """docker-inspect-shaped record for the release's container receipts."""
        return {
            "Id": self.receipt["sandbox_id"],
            "Image": self.settings.image,
            "State": {"Running": self.returncode is None, "ExitCode": self.returncode,
                      "OOMKilled": self.returncode == 137},
            "HostConfig": {"Memory": docker_memory_bytes(self.spec.memory), "MemorySwap": docker_memory_bytes(self.spec.memory),
                           "NanoCpus": int(float(self.spec.cpus or 0) * 1e9), "PidsLimit": self.spec.pids_limit},
            "Config": {"Labels": {"io.rescopebench.hermes.commit": hermes_commit()}},
            "RemoteSandbox": dict(self.receipt),
        }
