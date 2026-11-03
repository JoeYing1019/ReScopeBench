"""Isolated upstream Hermes task + its automatic native background review."""

from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import threading
import time

try:
    import driver_base as base
    import driver_adoption as adoption
    import driver_surface as surface
except ModuleNotFoundError:
    from . import worker_base as base
    from . import adoption
    from .. import surface


PHASE = "task"
EXPECTED = None
OPEN = threading.Event()
GATE = threading.Lock()
TOOL_EVENTS = []


def now():
    return datetime.now(timezone.utc).isoformat()


def snapshot():
    # Filesystem lock files are transient runtime state, not transferable notes.
    return {name: item for name, item in base.snapshot().items() if not name.endswith(".lock")}


class Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        with GATE:
            try:
                if not OPEN.is_set() or self.path != "/v1/chat/completions":
                    raise ValueError("Provider bridge closed or unknown route")
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if body.get("tools") != EXPECTED:
                    raise ValueError("Hermes changed the reviewed tool surface")
                value = base.exchange("model", {"phase": PHASE, "request": body})
                choices = value.get("choices") or []
                if len(choices) != 1 or choices[0].get("finish_reason") not in {"stop", "tool_calls"}:
                    raise ValueError("Incomplete model response")
                status = 200
            except Exception as error:
                value = {"error": {"message": str(error), "type": type(error).__name__}}
                status = 500
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def record_tool(name, arguments, result):
    TOOL_EVENTS.append({"phase": PHASE, "name": name, "arguments": deepcopy(arguments), "result": result})
    base.write(base.EXCHANGE / "native-tools" / f"{len(TOOL_EVENTS):04d}.json", TOOL_EVENTS[-1])


def register_inline_audits(executors):
    # Hermes memory uses the agent's live store and bypasses the registry.
    # Observe that dispatch without replacing the handler or validating notes.
    for name in surface.NATIVE_NAMES & executors.keys():
        handler = executors[name]
        if getattr(handler, "_rescope_observed", False):
            continue

        def audited(agent, arguments, context, _handler=handler, _name=name):
            result = _handler(agent, arguments, context)
            record_tool(_name, arguments, result)
            return result

        audited._rescope_observed = True
        executors[name] = audited


def register_tools(registry, packet, adopter=None):
    for schema in packet["business_tools"] + packet["history_tools"]:
        name = schema["function"]["name"]
        kind = "history" if name in surface.HISTORY_NAMES else "native"

        def rpc(arguments, _name=name, _kind=kind, **_kwargs):
            if _kind == "native" and PHASE != "task":
                return json.dumps({"error": "Business actions are unavailable after task delivery"})
            return json.dumps(base.exchange(_kind, {"phase": PHASE, "name": _name, "arguments": arguments}))

        if registry.get_entry(name) is not None:
            raise ValueError("Tool collision")
        registry.register(name=name, toolset="rescope_history" if kind == "history" else "rescope_robot",
                          schema=deepcopy(schema["function"]), handler=rpc)
    # Preserve upstream schemas and handlers; the wrapper only records calls.
    for schema in surface.NATIVE_TOOLS:
        name = schema["function"]["name"]
        entry = registry.get_entry(name)
        if entry is None:
            raise RuntimeError("Native experience tool was not registered: " + name)
        handler = entry.handler

        def audited(arguments, _handler=handler, _name=name, **kwargs):
            invoke = lambda: _handler(arguments, **kwargs)
            result = (adopter.invoke(arguments, invoke, foreground=PHASE == "task")
                      if adopter is not None and _name == "skill_manage" else invoke())
            record_tool(_name, arguments, result)
            return result

        registry.register(name=name, toolset=entry.toolset, schema=deepcopy(entry.schema), handler=audited)


def main():
    global PHASE, EXPECTED
    packet = json.loads(base.PUBLIC.read_text())
    surface.validate_assets(packet["assets"])
    adoption_allowlist = adoption.validate_allowlist(
        packet.get("curator_adoption_allowlist")
    )
    base.PROFILE.mkdir(parents=True, exist_ok=True)
    for part in ("skills", "memories"):
        (base.PROFILE / part).mkdir(exist_ok=True)
    (base.PROFILE / "config.yaml").write_text(packet["profile_config"])
    base.install_assets(packet["assets"])
    if snapshot() != packet["assets"]:
        raise ValueError("Entry native assets differ from the declared checkpoint")
    os.environ.update(HERMES_HOME=str(base.PROFILE), HOME="/work")
    os.chdir("/work")
    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port

    def guard(event, values):
        if event == "socket.connect":
            address = values[1]
            if not isinstance(address, tuple) or address[:2] != ("127.0.0.1", port):
                raise PermissionError("Only the local model bridge is reachable")
        if event in {"subprocess.Popen", "os.system"}:
            raise PermissionError("Learner subprocesses are disabled")

    sys.addaudithook(guard)
    sys.path.insert(0, "/opt/hermes")
    from run_agent import AIAgent
    if packet.get("completion_evidence_profile") is not None:
        from agent import prompt_builder
        from driver_completion_evidence import install_hermes_contract

        base.write(base.EXCHANGE / "completion-contract.json", install_hermes_contract(
            prompt_builder, packet["completion_evidence_profile"]))
    if packet.get("request_options", {}).get("parallel_tool_calls") is True:
        from driver_tool_execution import install_ordered_dispatch

        install_ordered_dispatch(
            AIAgent,
            emit=lambda index, record: base.write(
                base.EXCHANGE / "tool-batches" / f"{index:04d}.json", {"phase": PHASE, **record}),
        )
    from hermes_state import SessionDB
    from tools.registry import registry
    from tools import (  # Register native tools and use the official ownership handoff.
        skills_tool,
        skill_manager_tool,
        skill_usage,
        memory_tool,
    )

    adoption_index = 0

    def record_adoption(receipt):
        nonlocal adoption_index
        adoption_index += 1
        base.write(
            base.EXCHANGE / "curator-adoptions" / f"{adoption_index:04d}.json",
            receipt,
        )

    adopter = adoption.CuratorAdoptionGate(
        adoption_allowlist,
        find_skill=skill_manager_tool._find_skill,
        adopt_skill=skill_usage.adopt_skill,
        record=record_adoption,
    )
    register_tools(registry, packet, adopter)
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
    register_inline_audits(INLINE_TOOL_EXECUTORS)
    database = SessionDB()
    started = now()
    agent = AIAgent(
        model=packet["model"], provider="custom", api_mode="chat_completions",
        api_key="local-bridge-placeholder", base_url=f"http://127.0.0.1:{port}/v1",
        enabled_toolsets=["rescope_robot", "rescope_history", "memory", "skills"],
        skip_context_files=True, load_soul_identity=False, skip_memory=False,
        skip_background_review=False, ephemeral_system_prompt=packet["system"], quiet_mode=True,
        request_overrides={"seed": packet["seed"], **packet["request_options"]},
        max_iterations=100, max_tokens=65536,
        run_budget_seconds=3600, session_id=packet["session_id"], session_db=database,
    )
    upstream_tools = [{"type": "function", "function": deepcopy(registry.get_entry(row["function"]["name"]).schema)}
                      for row in surface.NATIVE_TOOLS]
    EXPECTED = surface.project(packet["business_tools"] + packet["history_tools"] + upstream_tools)
    if {tool["function"]["name"] for tool in agent.tools} != {tool["function"]["name"] for tool in EXPECTED}:
        raise ValueError("Hermes advertised undeclared tools")
    agent.tools = deepcopy(EXPECTED)
    agent.valid_tool_names = {tool["function"]["name"] for tool in EXPECTED}
    agent._invalidate_system_prompt()
    spawn = agent._spawn_background_review

    def observed_spawn(*args, **kwargs):
        global PHASE
        PHASE = "native_review"
        base.write(base.EXCHANGE / "before-native-review.json", {
            "at": now(), "assets": snapshot(),
            "review_memory": kwargs.get("review_memory"), "review_skills": kwargs.get("review_skills"),
        })
        return spawn(*args, **kwargs)

    agent._spawn_background_review = observed_spawn
    base.write(base.EXCHANGE / "surface.json", {
        "tools": EXPECTED, "entry_assets": packet["assets"], "upstream": surface.UPSTREAM,
        "skip_memory": agent.skip_memory if hasattr(agent, "skip_memory") else False,
        "skip_background_review": agent.skip_background_review,
        "native_handlers": True, "skill_nudge_interval": agent._skill_nudge_interval,
        "curator_adoption_allowlist": adoption_allowlist,
        "task_started_at": started,
    })
    OPEN.set()
    try:
        result = agent.run_conversation(user_message=packet["request"], conversation_history=[], task_id=packet["session_id"])
        if not result.get("completed") or result.get("failed") or result.get("interrupted"):
            raise RuntimeError("Hermes task incomplete")
        base.write(base.EXCHANGE / "foreground.json", result)
        PHASE = "native_review"
        deadline = time.monotonic() + 3600
        while any(t.name == "bg-review" and t.is_alive() for t in threading.enumerate()):
            if time.monotonic() > deadline:
                raise TimeoutError("Native background review did not settle")
            time.sleep(0.1)
        if getattr(agent, "_background_review_run", None) is not None:
            raise RuntimeError("Native review lifecycle remains open")
        adopter.require_success()
        assets = snapshot()
        surface.validate_assets(assets)
    finally:
        with GATE:
            OPEN.clear()
        agent.close()
        database.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)
    if thread.is_alive():
        raise RuntimeError("Provider bridge failed to close")
    base.write(base.EXCHANGE / "completed.json", {
        "completed": True, "terminal_assets": assets,
        "native_review_triggered": (base.EXCHANGE / "before-native-review.json").exists(),
        "curator_adoptions": adopter.receipts,
        "provider_server_closed": True, "provider_server_joined": True, "closed_at": now(),
    })


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        base.write(base.EXCHANGE / "worker-failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise
