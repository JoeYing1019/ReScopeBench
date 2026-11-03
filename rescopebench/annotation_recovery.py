"""Recover closed episodes rejected only by an overly narrow Hermes decoder.

This changes no learner inputs and issues no model or business requests. The
original failure remains intact; a separate amendment binds the recovered result
to the original messages, provider responses, action trace, and terminal state.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path

from rescopebench import execution_binding, pathing, stream_engine
from rescopebench.formal_driver import load_driver


native = stream_engine.native_runtime
ERROR = "Unknown Hermes business-result annotation"
AMENDMENT_KIND = "hermes_failure_warning_count_ge_3"
_ORIGINAL_DECODER = native.native_task.decode_business_result
_BASE_VALIDATOR = native.protocol.validate_trajectory_binding


def decode_business_result(content, name):
    try:
        return _ORIGINAL_DECODER(content, name)
    except ValueError:
        value, end = json.JSONDecoder().raw_decode(content)
        note = native.native_task.SAME_TOOL_FAILURE_NOTE.fullmatch(content[end:])
        direct_error = isinstance(value, dict) and value.get("status") == "error"
        envelope_error = (isinstance(value, dict) and value.get("type") == "tool_result"
                          and value.get("tool") == name and isinstance(value.get("content"), dict)
                          and value["content"].get("status") == "error")
        if note is not None and (direct_error or envelope_error):
            count = int(note.group(1))
            if count >= 3 and int(note.group(3)) == count and note.group(2) == name:
                return value, True
        raise ValueError(ERROR) from None


def binding_validator():
    return partial(_BASE_VALIDATOR, decoder=decode_business_result)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover(run: Path, arm: str, *, validator=None, error=ERROR,
            amendment_kind=AMENDMENT_KIND, validator_path=None):
    run = Path(run).resolve()
    plan = stream_engine.read_json(run / "run-plan.json")
    execution_mode = plan.get("execution_mode")
    driver = load_driver(
        plan["case_id"],
        execution_binding.planned_run_binding(
            run,
            plan["case_id"],
            execution_mode=execution_mode,
        ),
    )
    stream_engine.verify_plan(run, driver, execution_mode=execution_mode)
    root = run / arm
    with stream_engine._arm_lock(root):
        state, assets, episodes, checkpoint = stream_engine.current_checkpoint(root, plan, reset=arm == "reset")
        if checkpoint["test_injected"] or checkpoint["position"] >= len(plan["tasks"]):
            raise ValueError("recovery requires an incomplete formal arm")
        task = plan["tasks"][checkpoint["position"]]
        task_root = root / f"{task['position']:04d}-{task['event_id']}"
        attempts = sorted(task_root.glob("attempt-*"))
        if not attempts:
            raise ValueError("there is no failed attempt to recover")
        attempt = attempts[-1]
        episode = attempt / "episode"
        if (episode / "episode-result-private.json").exists():
            raise ValueError("a semantic result already exists; use normal checkpoint recovery")
        failure = stream_engine.read_json(episode / "failure.json")
        if failure.get("error") != error or failure.get("error_type") != "ValueError":
            raise ValueError("this failure is outside the annotation amendment")
        identity = stream_engine.read_json(attempt / "attempt.json")
        expected = {"plan_sha256": stream_engine.digest(plan), "event_id": task["event_id"],
                    "position": task["position"], "reset_control": arm == "reset", "test_injected": False,
                    "previous_checkpoint_sha256": stream_engine.digest(checkpoint)}
        if identity != expected:
            raise ValueError("failed attempt does not extend the current checkpoint")
        if (stream_engine.read_json(attempt / "assets-before.json") != assets
                or stream_engine.read_json(attempt / "history-before.json") != episodes):
            raise ValueError("entry experience differs from the durable checkpoint")
        container = stream_engine.read_json(episode / "container-receipt.json")
        lifecycle = container["state"]
        if (lifecycle.get("Running") is not False or lifecycle.get("ExitCode") != 0
                or lifecycle.get("OOMKilled") is not False or container["image_id"] != plan["container"]["image_id"]):
            raise ValueError("worker did not finish cleanly in the frozen image")
        exchange = episode / "exchange"
        completed = stream_engine.read_json(exchange / "completed.json")
        if not all(completed.get(key) is True for key in ("completed", "provider_server_closed", "provider_server_joined")):
            raise ValueError("worker lifecycle is incomplete")
        packet = stream_engine.read_json(episode / "public/packet.json")
        native.validate_surface(stream_engine.read_json(exchange / "surface.json"), packet)
        sources = stream_engine.read_json(episode / "sources-before.json")
        if sources != stream_engine.read_json(episode / "sources-after.json"):
            raise ValueError("source files changed during the episode")
        if sources["sha256"] != native.core.digest(sources["files"]):
            raise ValueError("source manifest digest differs")
        for relative, expected_hash in sources["files"].items():
            if _sha(pathing.resolve_path(relative)) != expected_hash:
                raise ValueError(f"source is no longer frozen: {relative}")
        foreground = stream_engine.read_json(exchange / "foreground.json")
        events = stream_engine.read_json(episode / "trace.json")["events"]
        event_files = [stream_engine.read_json(path) for path in sorted((episode / "events").glob("*.json"))]
        if event_files != events:
            raise ValueError("event journal differs from the trace")
        wires = [stream_engine.read_json(path) for path in sorted((episode / "hermes-wire").glob("*.json"))]
        responses, reviews, phases = [], [], []
        for index, wire in enumerate(wires, 1):
            if wire["admitted"] != stream_engine.read_json(episode / f"provider-admitted/requests/{index:04d}.json"):
                raise ValueError("provider request differs from the original wire")
            response = stream_engine.read_json(episode / f"provider-admitted/responses/{index:04d}.json")
            native.native_task.validate_provider_response(response, native.expected_tools(packet))
            phases.append(wire["phase"])
            (responses if wire["phase"] == "task" else reviews).append(response)
        validate = binding_validator() if validator is None else validator
        binding = validate(foreground["messages"], events, responses, driver.tools)
        seen = {path.name for path in (exchange / "requests").glob("*.json")}
        bridge = native.native_task.validate_exchange_complete(exchange, seen)
        before = stream_engine.read_json(episode / "before-state-private.json")
        terminal = stream_engine.read_json(episode / "final-state-private.json")
        world = driver.create_world()
        try:
            world.restore(deepcopy(state))
            driver.begin(world, task)
            if stream_engine.canonical(world.snapshot()) != stream_engine.canonical(before):
                raise ValueError("episode entry differs from the checkpoint plus task boundary")
            facade = stream_engine.Facade(driver, world, task)
            parameters = native.tool_parameters(driver.tools)
            for event in events:
                valid, arguments = native.validate_arguments(parameters[event["name"]], event["arguments"])
                response = facade.call(event["name"], arguments) if valid else {
                    "status": "error", "data": {}, "message": "INVALID_ARGUMENTS: does not match the declared schema"}
                if response != event["response"]:
                    raise ValueError("deterministic replay differs from the observed business event")
            if stream_engine.canonical(world.snapshot()) != stream_engine.canonical(terminal):
                raise ValueError("replayed terminal world differs from the recorded world")
            grade = facade.grade()
        finally:
            driver.close(world)
        history_events = [stream_engine.read_json(path) for path in (episode / "history-events").glob("*.json")]
        used = max((row.get("response", {}).get("history_budget", {}).get("used", 0) for row in history_events), default=0)
        if not 0 <= used <= 30:
            raise ValueError("history usage exceeds the frozen operation budget")
        terminal_assets = native.surface.validate_assets(completed["terminal_assets"])
        amendment = {"schema": "rescopebench-postprocessing-amendment/v1", "kind": amendment_kind,
                     "validator_sha256": _sha(Path(validator_path or __file__)), "original_failure_sha256": _sha(episode / "failure.json"),
                     "model_requests_reissued": 0, "business_replay_is_private": True,
                     "trajectory_sha256": native.core.digest(foreground["messages"]),
                     "terminal_state_sha256": native.core.digest(terminal), "plan_sha256": stream_engine.digest(plan)}
        task_messages = [row for row in foreground["messages"] if row.get("role") == "assistant"]
        result = {"schema": native.PROTOCOL, "status": "semantic_result", "score": grade["score"], "grade": grade,
                  "feedback": native.public_feedback(grade), "trajectory": foreground["messages"], "assets": terminal_assets,
                  "model_calls": len(phases), "task_model_calls": len(responses), "native_review_model_calls": len(reviews),
                  "agent_tool_rounds": sum(bool(row.get("tool_calls")) for row in task_messages),
                  "agent_business_tool_rounds": sum(any(call.get("function", {}).get("name") in parameters
                      for call in row.get("tool_calls") or []) for row in task_messages),
                  "business_tool_calls": len(events), "history_tool_calls": len(history_events),
                  "native_experience_tool_calls": len(list((exchange / "native-tools").glob("*.json"))),
                  "native_review_triggered": completed["native_review_triggered"],
                  "curator_adoptions": completed.get("curator_adoptions", []),
                  "provider_wire_attempts": len(list((attempt / "provider/wire").glob("*.json"))),
                  "source_sha256": sources["sha256"], "sources_unchanged": True,
                  "entry_state_sha256": native.core.digest(before), "terminal_state_sha256": native.core.digest(terminal),
                  "task_completed": True, "max_iterations": plan["max_iterations"],
                  "history_budget": {"limit": 30, "used": used, "remaining": 30 - used},
                  "postprocessing_amendment": amendment}
        stop = exchange / "semantic-stop.json"
        if stop.is_file():
            receipt = stream_engine.read_json(stop)
            result.update(task_completed=False, termination=receipt["kind"], turn_exit_reason=receipt["turn_exit_reason"],
                          max_iterations=receipt["effective_max_iterations"])
        for filename, value in (("postprocessing-amendment.json", amendment), ("assets.json", terminal_assets),
                                ("grade-private.json", grade), ("feedback.json", result["feedback"]),
                                ("bridge-integrity.json", {**bridge, "trajectory_binding": binding, "model_phases": phases,
                                                          "no_post_task_business_actions": True}),
                                ("episode-result-private.json", result)):
            stream_engine._write_once(episode / filename, value)
        return {"status": "recovered_not_yet_promoted", "arm": arm, "position": task["position"],
                "score": grade["score"], "model_requests_reissued": 0, "amendment": amendment}


def install(output: Path, arm: str):
    """Record and install host-only decoding, without changing learner output."""
    output = Path(output)
    plan = stream_engine.read_json(output / "run-plan.json")
    receipt = {
        "schema": "rescopebench-processing-addendum/v1", "arm": arm,
        "kind": "exact_hermes_failure_warning_compatibility",
        "plan_sha256": stream_engine.digest(plan),
        "scope": ["host_trajectory_validation", "declared_business_only_history_materialization"],
        "learner_prompt_tools_observations_and_grading_unchanged": True,
        "model_requests_reissued": 0,
        "sources": {pathing.relative_path(Path(__file__)): _sha(Path(__file__))},
    }
    stream_engine._write_once(output / "processing-addenda" / f"hermes-warning-{arm}.json", receipt)
    native.native_task.decode_business_result = decode_business_result
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=stream_engine.ARMS)
    args = parser.parse_args()
    print(json.dumps(recover(args.output, args.arm), sort_keys=True))


if __name__ == "__main__":
    main()
