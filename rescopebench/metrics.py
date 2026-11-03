"""Compute descriptive Native/Reset lifecycle metrics from a frozen stream run.

The formal unit is one matched pair of 16-, 24-, or 32-task longitudinal arms.  This module
only consumes durable promotion journals, so incomplete or ambiguous attempts
fail closed instead of being silently interpreted as zero-score episodes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Iterable
import uuid

from rescopebench import execution_policy, pathing, stream_engine as engine
from rescopebench.lifecycle import event_window, validate_schedule


DISPLAY_ADDENDUM_KINDS = {
    "hermes_final_response_routing_token_display_masking",
    "hermes_final_response_routing_token_display_masking_v2",
    "hermes_assistant_display_text_non_authoritative_v3",
}


@dataclass(frozen=True)
class Episode:
    position: int
    event_id: str
    score: float
    completed_artifacts: int | None
    business_tool_calls: int | None
    agent_tool_rounds: int | None
    history_tool_calls: int | None
    model_calls: int | None
    termination: str | None

    def public_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "event_id": self.event_id,
            "score": self.score,
            "completed_artifacts": self.completed_artifacts,
            "business_tool_calls": self.business_tool_calls,
            "agent_tool_rounds": self.agent_tool_rounds,
            "history_tool_calls": self.history_tool_calls,
            "model_calls": self.model_calls,
            "termination": self.termination,
        }


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _optional_count(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _score(result: dict[str, Any]) -> float:
    value = result.get("score")
    if value is None and isinstance(result.get("grade"), dict):
        value = result["grade"].get("score")
    score = _number(value, label="episode score")
    if not 0.0 <= score <= 100.0:
        raise ValueError("episode score must be in [0, 100]")
    if isinstance(result.get("grade"), dict) and result["grade"].get("score") != score:
        raise ValueError("episode score differs from its terminal grade")
    return score


def _k8s_complete_count(result: dict[str, Any], task: dict[str, Any]) -> int | None:
    """Recover the legacy omitted scalar from journal-bound artifact facts."""
    from rescopebench.planned_artifacts import public_targets

    targets, _ = public_targets(task, "k8s_operations")
    release_ids = [target["release_id"] for target in targets]
    milestones = {"deployment_committed", "pods_validly_placed",
                  "service_contract_persisted", "capacity_conserved"}
    components = result.get("grade", {}).get("public_components", {})
    expected = {f"{release_id}:{milestone}" for release_id in release_ids for milestone in milestones}
    if len(release_ids) != len(set(release_ids)) or set(components) != expected:
        return None
    if any(not isinstance(row, dict) or type(row.get("correct")) is not int
           or row["correct"] not in (0, 1) or type(row.get("total")) is not int or row["total"] != 1
           for row in components.values()):
        return None
    # The K8s grader defines completion as all four actual milestones, not a score bin.
    return sum(all(components[f"{release_id}:{name}"]["correct"] == 1 for name in milestones)
               for release_id in release_ids)


def _completed_artifacts(result: dict[str, Any], *, case_id=None, task=None) -> int | None:
    value = result.get("completed_artifacts")
    grade = result.get("grade")
    if value is None and isinstance(grade, dict):
        value = grade.get("completed_artifacts")
    if value is None and case_id == "k8s_operations" and task is not None:
        value = _k8s_complete_count(result, task)
    return _optional_count(value, label="completed_artifacts")


def _episode(position: int, event_id: str, result: dict[str, Any], *, case_id=None, task=None) -> Episode:
    if result.get("status") != "semantic_result":
        raise ValueError(f"position {position} is not a semantic result")
    return Episode(
        position=position,
        event_id=event_id,
        score=_score(result),
        completed_artifacts=_completed_artifacts(result, case_id=case_id, task=task),
        business_tool_calls=_optional_count(
            result.get("business_tool_calls"), label="business_tool_calls"
        ),
        agent_tool_rounds=_optional_count(
            result.get("agent_tool_rounds"), label="agent_tool_rounds"
        ),
        history_tool_calls=_optional_count(
            result.get("history_tool_calls"), label="history_tool_calls"
        ),
        model_calls=_optional_count(result.get("model_calls"), label="model_calls"),
        termination=(
            str(result["termination"])
            if result.get("termination") is not None
            else None
        ),
    )


def _plan(run_root: Path) -> dict[str, Any]:
    plan = _read(run_root / "run-plan.json")
    if plan.get("schema") != engine.SCHEMA + "/plan" or plan.get("arms") != list(engine.ARMS):
        raise ValueError("invalid formal run plan")
    execution_policy.validate_plan_policy(plan)
    tasks = plan.get("tasks", [])
    count = len(tasks)
    validate_schedule(plan.get("lifecycle_schedule"), count)
    if ([row.get("position") for row in tasks] != list(range(1, count + 1))
            or len({row.get("event_id") for row in tasks}) != count):
        raise ValueError("invalid frozen task identities")
    for task in tasks:
        engine.public_task(task)
    checks = {
        "tasks": plan.get("tasks_sha256") == engine.digest(tasks),
        "tools": plan.get("tools_sha256") == engine.digest(plan.get("tools")),
        "initial": plan.get("initial_snapshot_sha256") == engine.digest(_read(run_root / "initial-snapshot.json")),
        "admission": plan.get("validation_sha256") == engine.digest(_read(run_root / "admission.json")),
    }
    if not all(checks.values()):
        raise ValueError(f"run plan payload integrity failed: {[key for key, valid in checks.items() if not valid]}")
    if _read(run_root / "admission.json").get("status") != "passed":
        raise ValueError("case admission did not pass")
    sources = plan.get("runtime_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("frozen runtime source inventory is missing")
    for relative, expected in sources.items():
        if hashlib.sha256(pathing.resolve_path(relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen runtime source integrity failed: {relative}")
    for path in (run_root / "processing-addenda").glob("*.json"):
        addendum = _read(path)
        if addendum.get("schema") == "rescopebench-http400-retry-addendum/v1":
            _transport_addendum(run_root, plan, addendum)
            continue
        if addendum.get("schema") == "rescopebench-robot-observed-capability-recovery/v1":
            _robot_restore_addendum(run_root, plan, addendum)
            continue
        if addendum.get("schema") == "rescopebench-emergency-receipt-gap-recovery/v1":
            _emergency_restore_addendum(run_root, plan, addendum)
            continue
        if addendum.get("schema") == "rescopebench-runtime-cleanup/v1":
            checks = {
                "case": addendum.get("case_id") == plan.get("case_id"),
                "implementation": isinstance(addendum.get("implementation_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", addendum["implementation_sha256"]) is not None,
                "scope": addendum.get("scope") == "host-owned temporary database only",
                "hook": addendum.get("hook") == "driver.close after normal result/checkpoint commit",
                "frozen": addendum.get("frozen_task_and_grader_sources_unchanged") is True,
            }
            if not all(checks.values()):
                raise ValueError("invalid runtime cleanup addendum")
            continue
        if addendum.get("kind") in DISPLAY_ADDENDUM_KINDS:
            sources = addendum.get("sources")
            checks = {
                "schema": addendum.get("schema") == "rescopebench-processing-addendum/v1",
                "arm": addendum.get("arm") in engine.ARMS,
                "plan": addendum.get("plan_sha256") == engine.digest(plan),
                "no_repeat": addendum.get("model_requests_reissued") == 0,
                "business": addendum.get(
                    "business_events_state_grading_and_provider_tool_calls_unchanged"
                ) is True,
                "sources": isinstance(sources, dict)
                and bool(sources)
                and all(
                    isinstance(value, str)
                    and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                    for value in sources.values()
                ),
            }
            if not all(checks.values()):
                raise ValueError("invalid assistant-display processing addendum")
            continue
        if (addendum.get("schema") != "rescopebench-processing-addendum/v1"
                or addendum.get("kind") != "exact_hermes_failure_warning_compatibility"
                or addendum.get("arm") not in engine.ARMS or addendum.get("plan_sha256") != engine.digest(plan)
                or addendum.get("learner_prompt_tools_observations_and_grading_unchanged") is not True
                or addendum.get("model_requests_reissued") != 0 or not addendum.get("sources")):
            raise ValueError("invalid host processing addendum")
        for relative, expected in addendum["sources"].items():
            if hashlib.sha256(pathing.resolve_path(relative).read_bytes()).hexdigest() != expected:
                raise ValueError("host processing addendum source integrity failed")
    return plan


def _robot_restore_addendum(run_root: Path, plan: dict, addendum: dict) -> None:
    checksum = addendum.get("implementation_sha256")
    position = addendum.get("start_after_position")
    valid = (
        plan.get("case_version") == "robot-rule-v9.2"
        and addendum.get("arm") in engine.ARMS
        and addendum.get("plan_sha256") == engine.digest(plan)
        and type(position) is int and 0 < position < len(plan["tasks"])
        and addendum.get("frozen_source_files_modified") is False
        and addendum.get("learner_prompt_tools_observations_and_grading_unchanged") is True
        and addendum.get("model_requests_reissued") == 0
        and isinstance(checksum, str) and re.fullmatch(r"[0-9a-f]{64}", checksum)
    )
    if not valid:
        raise ValueError("invalid Robot restore-only addendum")
    checkpoint = _read(run_root / addendum["arm"] / "checkpoints" / f"{position:04d}" / "manifest.json")
    source = run_root / "repair-sources" / (checksum + ".py")
    if (checkpoint["snapshot_sha256"] != addendum.get("checkpoint_snapshot_sha256")
            or source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != checksum):
        raise ValueError("Robot restore-only source or checkpoint differs")


def _emergency_restore_addendum(run_root: Path, plan: dict, addendum: dict) -> None:
    checksum = addendum.get("implementation_sha256")
    evidence_hash = addendum.get("evidence_sha256")
    position = addendum.get("start_after_position")
    if not (
        plan.get("case_version") == "emergency-response-permission-v7.3"
        and addendum.get("arm") in engine.ARMS
        and addendum.get("plan_sha256") == engine.digest(plan)
        and type(position) is int and 0 < position < len(plan["tasks"])
        and addendum.get("frozen_source_files_modified") is False
        and addendum.get("learner_prompt_tools_observations_and_grading_unchanged") is True
        and addendum.get("model_requests_reissued") == 0
        and all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)
                for h in (checksum, evidence_hash))
    ):
        raise ValueError("invalid Emergency restore-only addendum")
    checkpoint = _read(run_root / addendum["arm"] / "checkpoints" / f"{position:04d}" / "manifest.json")
    source = run_root / "repair-sources" / (checksum + ".py")
    evidence = _read(run_root / "repair-evidence" / (evidence_hash + ".json"))
    if (checkpoint["snapshot_sha256"] != addendum.get("checkpoint_snapshot_sha256")
            or source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != checksum
            or engine.digest(evidence) != evidence_hash
            or evidence.get("terminal_sha256") != checkpoint["snapshot_sha256"]
            or evidence.get("plan_sha256") != engine.digest(plan)
            or evidence.get("arm") != addendum["arm"] or evidence.get("position") != position):
        raise ValueError("Emergency restore-only source, evidence or checkpoint differs")


def _transport_addendum(run_root: Path, plan: dict, addendum: dict) -> None:
    """Audit transport metadata offline; never inspect or constrain learner prose."""
    checksum = addendum.get("implementation_sha256")
    position = addendum.get("start_after_position")
    valid = (
        addendum.get("arm") in engine.ARMS
        and addendum.get("plan_sha256") == engine.digest(plan)
        and type(position) is int and 0 <= position < 16
        and addendum.get("max_http400_retries") == 3
        and addendum.get("business_tool_retry_added") is False
        and addendum.get("model_request_changed") is False
        and addendum.get("frozen_source_files_modified") is False
        and isinstance(checksum, str) and re.fullmatch(r"[0-9a-f]{64}", checksum)
    )
    if not valid:
        raise ValueError("invalid transport-only retry addendum")
    source = run_root / "transport-sources" / (checksum + ".py")
    if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
        raise ValueError("transport amendment source capture differs")


def _checkpoint(arm_root, plan, arm, position):
    directory = arm_root / "checkpoints" / f"{position:04d}"
    manifest = _read(directory / "manifest.json")
    if (manifest.get("schema") != engine.SCHEMA + "/checkpoint" or manifest.get("position") != position
            or manifest.get("plan_sha256") != engine.digest(plan)
            or manifest.get("reset_control") is not (arm == "reset")):
        raise ValueError("checkpoint identity integrity failed")
    if manifest.get("test_injected") is not False:
        raise ValueError("test-injected evidence is not a formal model result")
    state, assets, history, _ = engine._load_checkpoint(directory, manifest)
    if position == 0 and (manifest.get("promotion_sha256") is not None
                          or manifest["snapshot_sha256"] != plan["initial_snapshot_sha256"]
                          or assets or history):
        raise ValueError("initial checkpoint differs from the frozen initial world")
    return state, assets, history, manifest


def _amendment(path, result, plan):
    amendment = result.get("postprocessing_amendment")
    if amendment is None:
        return
    from rescopebench import annotation_recovery

    episode = path.parent / "episode"
    kind = amendment.get("kind")
    if kind in DISPLAY_ADDENDUM_KINDS:
        checks = {
            "receipt": _read(episode / "postprocessing-amendment.json") == amendment,
            "no_repeat": amendment.get("model_requests_reissued") == 0,
            "plan": amendment.get("plan_sha256") == engine.digest(plan),
            "validator": isinstance(amendment.get("validator_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", amendment["validator_sha256"]) is not None,
            "failure": amendment.get("original_failure_sha256")
            == hashlib.sha256((episode / "failure.json").read_bytes()).hexdigest(),
            "trajectory": amendment.get("trajectory_sha256")
            == engine.native_runtime.core.digest(result["trajectory"]),
            "terminal": amendment.get("terminal_state_sha256")
            == engine.native_runtime.core.digest(
                _read(episode / "final-state-private.json")
            ),
        }
        if not all(checks.values()):
            raise ValueError("assistant-display amendment integrity failed")
        return
    checks = {
        "receipt": _read(episode / "postprocessing-amendment.json") == amendment,
        "kind": kind == "hermes_failure_warning_count_ge_3",
        "no_repeat": amendment.get("model_requests_reissued") == 0,
        "plan": amendment.get("plan_sha256") == engine.digest(plan),
        "validator": amendment.get("validator_sha256")
            == hashlib.sha256(Path(annotation_recovery.__file__).read_bytes()).hexdigest(),
        "failure": amendment.get("original_failure_sha256") == hashlib.sha256((episode / "failure.json").read_bytes()).hexdigest(),
        "trajectory": amendment.get("trajectory_sha256") == engine.native_runtime.core.digest(result["trajectory"]),
        "terminal": amendment.get("terminal_state_sha256") == engine.native_runtime.core.digest(_read(episode / "final-state-private.json")),
    }
    if not all(checks.values()):
        raise ValueError("postprocessing amendment integrity failed")


def load_arm(run_root: Path, arm: str, *, plan=None) -> dict[int, Episode]:
    if arm not in engine.ARMS:
        raise ValueError("unknown formal arm")
    plan = _plan(Path(run_root)) if plan is None else plan
    arm_root = Path(run_root).resolve() / arm
    summary_path = arm_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing {arm} summary: {summary_path}")
    summary = _read(summary_path)
    policy = execution_policy.validate_plan_policy(plan)
    if summary.get("arm") != arm:
        raise ValueError(f"{arm} summary identifies another arm")
    count = len(plan["tasks"])
    if summary.get("complete") is not True or summary.get("position") != count:
        raise ValueError(f"{arm} is not a complete {count}-task stream")
    if summary.get("test_injected") is not False:
        raise ValueError("test-injected evidence is not a formal model result")
    if (summary.get("schema") != engine.SCHEMA + "/arm-summary" or summary.get("model") != plan["model"]
            or summary.get("plan_sha256") != engine.digest(plan)):
        raise ValueError("arm summary integrity failed")
    if any(summary.get(key) != value for key, value in policy.items()):
        raise ValueError("arm summary execution policy differs from the frozen plan")

    episodes: dict[int, Episode] = {}
    previous = _checkpoint(arm_root, plan, arm, 0)[3]
    promotion_paths = sorted(arm_root.glob("[0-9][0-9][0-9][0-9]-*/attempt-*/promotion.json"))
    for path in promotion_paths:
        promotion = _read(path)
        position = promotion.get("position")
        event_id = promotion.get("event_id")
        result = promotion.get("result")
        if type(position) is not int or not 1 <= position <= count:
            raise ValueError(f"invalid promotion position: {path}")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"invalid promotion event id: {path}")
        if not isinstance(result, dict):
            raise ValueError(f"invalid promotion result: {path}")
        if position in episodes:
            raise ValueError(f"multiple durable promotions for {arm} position {position}")
        task = plan["tasks"][position - 1]
        if path.parent.parent.name != f"{position:04d}-{task['event_id']}":
            raise ValueError("promotion directory identity differs from the plan")
        checks = {
            "schema": promotion.get("schema") == engine.SCHEMA + "/promotion",
            "event": event_id == task["event_id"],
            "plan": promotion.get("plan_sha256") == engine.digest(plan),
            "arm": promotion.get("reset_control") is (arm == "reset"),
            "previous": previous["position"] == position - 1
            and promotion.get("previous_checkpoint_sha256") == engine.digest(previous),
        }
        if promotion.get("test_injected") is not False or result.get("test_injected") is not False:
            raise ValueError("test-injected evidence is not a formal model result")
        checks.update({key: promotion.get(f"{key}_sha256") == engine.digest(promotion.get(key))
                       for key in ("result", "snapshot", "assets", "history")})
        if not all(checks.values()):
            raise ValueError(f"promotion integrity failed: {[key for key, valid in checks.items() if not valid]}")
        if _read(path.parent / "result.json") != result or promotion["assets"] != result.get("assets"):
            raise ValueError("semantic result differs from its durable journal")
        checkpoint_state, checkpoint_assets, checkpoint_history, current = _checkpoint(arm_root, plan, arm, position)
        if (current.get("promotion_sha256") != engine.digest(promotion) or checkpoint_state != promotion["snapshot"]
                or checkpoint_assets != ({} if arm == "reset" else promotion["assets"])
                or checkpoint_history != ([] if arm == "reset" else promotion["history"])):
            raise ValueError("checkpoint differs from its promotion journal")
        _amendment(path, result, plan)
        episodes[position] = _episode(position, event_id, result, case_id=plan["case_id"], task=task)
        previous = current

    expected = set(range(1, count + 1))
    if set(episodes) != expected:
        missing = sorted(expected - set(episodes))
        extra = sorted(set(episodes) - expected)
        raise ValueError(f"{arm} promotion coverage mismatch: missing={missing}, extra={extra}")
    final = engine.current_checkpoint(arm_root, plan, reset=arm == "reset")[3]
    if final != previous:
        raise ValueError("current checkpoint does not identify the completed chain")
    return episodes


def _mean(values: Iterable[float]) -> float:
    return round(fmean(values), 6)


def _optional_mean(values: Iterable[int | None]) -> float | None:
    rows = list(values)
    if not rows or any(value is None for value in rows):
        return None
    return _mean(float(value) for value in rows if value is not None)


def _window(
    native: dict[int, Episode], reset: dict[int, Episode], positions: tuple[int, ...]
) -> dict[str, Any]:
    native_scores = [native[position].score for position in positions]
    reset_scores = [reset[position].score for position in positions]
    contrasts = [left - right for left, right in zip(native_scores, reset_scores)]
    return {
        "positions": list(positions),
        "native_mean": _mean(native_scores),
        "reset_mean": _mean(reset_scores),
        "mean_contrast": _mean(contrasts),
        "native_business_tool_calls_mean": _optional_mean(
            native[position].business_tool_calls for position in positions
        ),
        "reset_business_tool_calls_mean": _optional_mean(
            reset[position].business_tool_calls for position in positions
        ),
    }


def analyze_run(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    plan = _plan(run_root)
    policy = execution_policy.validate_plan_policy(plan)
    native = load_arm(run_root, "native", plan=plan)
    reset = load_arm(run_root, "reset", plan=plan)
    return _analyze_scheduled_run(run_root, plan, native, reset, policy)


def _analyze_scheduled_run(run_root, plan, native, reset, policy):
    count = len(plan["tasks"])
    for position in range(1, count + 1):
        if native[position].event_id != reset[position].event_id:
            raise ValueError(f"arm event mismatch at position {position}")
    events = []
    for event in plan["lifecycle_schedule"]["events"]:
        window = event_window(plan["lifecycle_schedule"], event["event_id"])
        onset = event["before_task"]
        native_drop = native[onset - 1].score - native[onset].score
        reset_drop = reset[onset - 1].score - reset[onset].score
        events.append({
            **event, "censored_at": window["censored_at"], "censor_reason": window["censor_reason"],
            "windows": {name: _window(native, reset, tuple(window[name]))
                        for name in ("pre", "onset", "post_four", "post")},
            "w_exp_immediate_drop": native_drop, "wo_exp_immediate_drop": reset_drop,
            "excess_immediate_drop": native_drop - reset_drop,
        })
    return {
        "schema": "rescopebench-event-metrics/v1", "run_root": str(run_root),
        "case_id": plan["case_id"], "case_version": plan["case_version"], "model": plan["model"],
        **policy, "formal_claim_eligible": not policy["candidate_only"],
        "arm_labels": {"native": "w exp", "reset": "w/o exp"},
        "interpretation": "Descriptive longitudinal contrast. Business worlds may diverge. "
            "Event windows are censored at the next event or stream end. Neither score drops "
            "nor recovery to a previous score establish experience revision.",
        "integrity": {"status": "passed", "plan_sha256": engine.digest(plan),
                      "verified_promotions": 2 * count, "test_injected": False,
                      "runtime_source_hashes_verified": True, "checkpoint_chains_verified": True,
                      "execution_policy_verified": True},
        "lifecycle_schedule": plan["lifecycle_schedule"], "events": events,
        "windows": {"all_post_initial": _window(native, reset, tuple(range(2, count + 1)))},
        "metrics": {"mean_position_contrast": _window(native, reset, tuple(range(2, count + 1)))["mean_contrast"],
                    "mean_event_excess_drop": _mean(row["excess_immediate_drop"] for row in events)},
        "positions": [{"position": p, "event_id": native[p].event_id,
                       "native": native[p].public_dict(), "reset": reset[p].public_dict(),
                       "score_contrast": round(native[p].score - reset[p].score, 6)}
                      for p in range(1, count + 1)],
    }


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_run(args.run)
    if args.output is not None:
        _write_atomic(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
