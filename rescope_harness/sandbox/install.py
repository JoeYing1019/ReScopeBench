"""Install the remote sandbox executor into an unmodified rescopebench.

Only module attributes are replaced at import time; no benchmark source file is
edited, so frozen source hashes in plans stay identical to the release.
"""

from __future__ import annotations

from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess as _subprocess
from types import SimpleNamespace

from rescope_harness.sandbox.providers import SandboxSettings
from rescope_harness.sandbox.remote import REGISTRY, RemoteContainer, parse_run, sandbox_resources


def _tolerant_read() -> None:
    """A mirrored exchange file can be observed before its content lands; wait briefly, then fail.

    The benchmark binds ``execute_episode`` through FunctionType with a *copy* of the
    module globals, so patching one module object is not enough: every globals dict
    that carries the original ``_read`` is patched, and every failure is logged.
    """
    import sys
    import time
    from rescopebench.agent_runtime import protocol as stage

    original = stage._read
    log = Path(os.environ.get("RESCOPE_BRIDGE_LOG", "/tmp/rescope_bridge_read.log"))
    patched = set()

    def wrap(original_read):
        def read(path):
            deadline = time.monotonic() + float(os.environ.get("RESCOPE_BRIDGE_READ_WAIT", "5"))
            while True:
                try:
                    return original_read(path)
                except ValueError as error:
                    if time.monotonic() >= deadline:
                        with log.open("a") as handle:
                            handle.write(json.dumps({
                                "time": time.time(), "path": str(path),
                                "size": Path(path).stat().st_size if Path(path).exists() else None,
                                "content": Path(path).read_text(errors="replace")[:400] if Path(path).exists() else None,
                                "error": str(error),
                            }) + "\n")
                        raise
                    time.sleep(0.05)
        return read

    def patch_globals(namespace):
        current = namespace.get("_read")
        if current is None or id(current) not in patched:
            if current is not None:
                patched.add(id(current))
            namespace["_read"] = wrap(current or original)

    for module in list(sys.modules.values()):
        if module is None:
            continue
        namespace = getattr(module, "__dict__", None)
        if isinstance(namespace, dict) and namespace.get("_read") is original:
            patch_globals(namespace)
    from rescopebench.agent_runtime import native_runtime
    for value in list(vars(native_runtime).values()):
        namespace = getattr(value, "__globals__", None)
        if isinstance(namespace, dict) and namespace.get("_read") is original:
            patch_globals(namespace)


def _completed(command, code=0, stdout="", stderr=""):
    return _subprocess.CompletedProcess(command, code, stdout, stderr)


def _subprocess_shim(provider: str, settings: SandboxSettings) -> SimpleNamespace:
    """subprocess for native_executor: `docker run` becomes a RemoteContainer, and the
    other docker calls it makes (image inspect, inspect, stop, kill, rm) are answered here."""

    def docker(command):
        if not isinstance(command, (list, tuple)) or len(command) < 2 or command[0] != "docker":
            return None
        verb = list(command[1:])
        if verb[:2] == ["image", "inspect"]:
            return _completed(command, 0, settings.image + "\n")
        if verb[0] == "inspect":
            found = [REGISTRY[name].inspect() for name in verb[1:] if name in REGISTRY]
            return _completed(command, 0, json.dumps(found)) if found else _completed(command, 1, "", "No such container")
        if verb[0] in {"stop", "kill"}:
            for name in verb[1:]:
                if name in REGISTRY:
                    REGISTRY[name].terminate()
            return _completed(command)
        if verb[0] == "rm":
            for name in (name for name in verb[1:] if not name.startswith("-")):
                container = REGISTRY.pop(name, None)
                if container is not None:
                    container.terminate()
            return _completed(command)
        raise RuntimeError(f"Unsupported docker command under the remote sandbox executor: {command[:3]}")

    def run(command, *args, **kwargs):
        handled = docker(command)
        if handled is None:
            return _subprocess.run(command, *args, **kwargs)
        if kwargs.get("check") and handled.returncode:
            raise _subprocess.CalledProcessError(handled.returncode, command, handled.stdout, handled.stderr)
        return handled

    def popen(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and list(command[:2]) == ["docker", "run"]:
            return RemoteContainer(list(command), provider=provider, stdout=kwargs.get("stdout"), settings=settings)
        return _subprocess.Popen(command, *args, **kwargs)

    return SimpleNamespace(**{**vars(_subprocess), "run": run, "Popen": popen})


def install_remote(provider: str) -> None:
    from rescopebench import native_executor

    if getattr(native_executor, "_remote_sandbox_installed", False):
        return
    # A module that registers a custom provider (providers.register_provider) at import.
    if os.environ.get("RESCOPE_SANDBOX_PLUGIN"):
        importlib.import_module(os.environ["RESCOPE_SANDBOX_PLUGIN"])
    _tolerant_read()
    settings = SandboxSettings.from_environment()
    shim = _subprocess_shim(provider, settings)
    original_clone = native_executor._clone

    def clone(function, overrides):
        return original_clone(function, {**overrides, "subprocess": shim})

    def container_contract():
        command = native_executor.docker_command(Path("/public-placeholder"), Path("/exchange-placeholder"),
                                                 "placeholder")
        spec = parse_run(command)
        cpu, memory = sandbox_resources(spec)
        return {"image": settings.image, "image_id": settings.image, "source_image_id": spec.image,
                "resources": deepcopy(native_executor.RESOURCES), "network": f"{provider}-sandbox",
                "readonly_root": False, "user": spec.user or "1000:1000", "pids_limit": None,
                "executor": "remote_sandbox/v1",
                "remote_sandbox": {"provider": provider, "cpu": cpu, "memory": memory}}

    native_executor._clone = clone
    native_executor.container_contract = container_contract
    native_executor.subprocess = shim
    native_executor._remote_sandbox_installed = True
