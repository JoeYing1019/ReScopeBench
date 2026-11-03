"""Model- and resource-isolated binding of the reviewed native Hermes worker."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import FunctionType, SimpleNamespace
import uuid

from rescopebench.agent_runtime import native_runtime as native


RESOURCES = {"memory": "192m", "memory_swap": "192m", "cpus": "0.1"}


def _write_receipt(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def docker_command(public, exchange, name):
    command = native.native_task.docker_command(public, exchange, name)
    command = [part for part in command if part != "--rm"
               and not part.startswith(("--memory=", "--memory-swap=", "--cpus="))]
    index = command.index("--entrypoint")
    command[index:index] = [f"--memory={RESOURCES['memory']}", f"--memory-swap={RESOURCES['memory_swap']}",
                            f"--cpus={RESOURCES['cpus']}"]
    return command


def container_contract():
    command = docker_command(Path("/public-placeholder"), Path("/exchange-placeholder"), "placeholder")
    image = command[command.index("--entrypoint") + 2]
    result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                            capture_output=True, text=True, check=True, timeout=20)
    return {"image": image, "image_id": result.stdout.strip(), "resources": deepcopy(RESOURCES),
            "network": "none", "readonly_root": True, "user": "1000:1000", "pids_limit": 96}


def _clone(function, overrides):
    bound = FunctionType(function.__code__, {**function.__globals__, **overrides}, function.__name__,
                         function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return bound


def for_model(model, *, prompt=None, tool_execution_mode="serial"):
    runtime = native.for_model(model, prompt=prompt, tool_execution_mode=tool_execution_mode)
    base = runtime.execute_episode.__globals__["execute_episode_base"]
    task_module = SimpleNamespace(**vars(native.native_task))
    task_module.docker_command = docker_command
    bound_base = _clone(base, {"native_task": task_module})
    bound = _clone(runtime.execute_episode, {"execute_episode_base": bound_base})

    def execute(directory, *args, **kwargs):
        try:
            return bound(directory, *args, **kwargs)
        finally:
            started = Path(directory) / "started.json"
            if started.is_file():
                name = json.loads(started.read_text(encoding="utf-8"))["container"]
                receipt_path = Path(directory) / "container-receipt.json"
                record = {"container": name, "cleanup": {"status": "not_attempted"}}
                try:
                    result = subprocess.run(
                        ["docker", "inspect", name], capture_output=True, text=True, timeout=20
                    )
                    if result.returncode != 0:
                        record["cleanup"] = {"status": "already_absent"}
                    else:
                        inspected = json.loads(result.stdout)[0]
                        labels = inspected.get("Config", {}).get("Labels") or {}
                        record.update({
                            "id": inspected["Id"],
                            "state": inspected["State"],
                            "image_id": inspected["Image"],
                            "limits": {
                                key: inspected["HostConfig"][key]
                                for key in ("Memory", "MemorySwap", "NanoCpus", "PidsLimit")
                            },
                        })
                        if not labels.get("io.rescopebench.hermes.commit"):
                            record["cleanup"] = {
                                "status": "refused",
                                "reason": "container lacks ReScopeBench ownership label",
                            }
                        else:
                            removed = subprocess.run(
                                ["docker", "rm", "-f", name],
                                capture_output=True,
                                text=True,
                                timeout=30,
                            )
                            remaining = subprocess.run(
                                ["docker", "inspect", name],
                                capture_output=True,
                                text=True,
                                timeout=20,
                            )
                            record["cleanup"] = {
                                "status": "removed" if removed.returncode == 0 and remaining.returncode != 0 else "failed",
                                "docker_rm_exit_code": removed.returncode,
                                "stderr": removed.stderr.strip()[:800],
                            }
                    _write_receipt(receipt_path, record)
                except Exception as cleanup_error:
                    record["cleanup"] = {
                        "status": "failed",
                        "error_type": type(cleanup_error).__name__,
                        "error": str(cleanup_error)[:800],
                    }
                    try:
                        _write_receipt(receipt_path, record)
                    except Exception:
                        pass

    runtime.execute_episode = execute
    return runtime
