"""Small process-level queue for policy-gated stream arms; no API-key scheduler."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from rescopebench.case_names import case_identity

from rescopebench import (
    annotation_recovery,
    execution_binding,
    execution_policy,
    formal_evidence_cleanup,
    pathing,
    stream_engine as engine,
    trajectory_display_compat,
)


BATCH_REGISTRY_SCHEMA = "rescopebench-batch-registry/v1"
CONTROLLER_PLAN_SCHEMA = "rescopebench-batch-controller/v1"


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def load_registry(
    registry: Path,
    output: Path,
    execution_mode: str = "formal",
) -> dict:
    """Validate and normalize the launch queue before any child can start."""
    execution_policy.claim_scope_for_mode(execution_mode)
    payload = engine.read_json(Path(registry).expanduser().resolve())
    if not isinstance(payload, dict) or payload.get("schema") != BATCH_REGISTRY_SCHEMA:
        raise ValueError(f"registry schema must be {BATCH_REGISTRY_SCHEMA}")
    entries = payload.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("registry runs must be a nonempty list")

    controller = Path(output).expanduser().resolve()
    if _overlap(controller, pathing.FINAL_ROOT):
        raise ValueError("batch controller output must be outside the code package")
    normalized, seen = [], set()
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or not set(entry) <= {"case", "binding", "output"}:
            raise ValueError(f"invalid registry entry fields at position {index}")
        case = entry.get("case")
        run_value = entry.get("output")
        if (not isinstance(case, str) or not case
                or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in case)):
            raise ValueError(f"invalid registry case at position {index}")
        case = case_identity(case)
        if not isinstance(run_value, str) or not run_value.strip():
            raise ValueError(f"invalid registry output at position {index}")
        run = Path(run_value).expanduser().resolve()
        if _overlap(run, pathing.FINAL_ROOT):
            raise ValueError("stream run output must be outside the code package")
        if _overlap(controller, run):
            raise ValueError("controller and formal run outputs must not overlap")
        if any(_overlap(run, previous) for previous in seen):
            raise ValueError("registered run outputs must not overlap")
        seen.add(run)

        plan = engine.read_json(run / "run-plan.json")
        if (plan.get("case_id") != case or plan.get("schema") != engine.SCHEMA + "/plan"
                or plan.get("arms") != list(engine.ARMS)):
            raise ValueError("registry differs from prepared case identity")
        policy = execution_policy.validate_plan_policy(plan, execution_mode)

        requested = entry.get("binding")
        if requested is not None and (not isinstance(requested, str) or not requested.strip()):
            raise ValueError(f"invalid candidate binding at position {index}")
        binding = execution_binding.planned_run_binding(
            run,
            case,
            requested,
            execution_mode=execution_mode,
        )
        normalized.append({
            "case": case,
            "output": str(run),
            **policy,
            **({"binding": binding} if binding else {}),
        })
    return {"schema": BATCH_REGISTRY_SCHEMA, "runs": normalized}


def running(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return False


def status(run, arm, *, retry_infrastructure=False):
    root = run / arm
    pointer = root / "current.json"
    position = 0
    if pointer.is_file():
        checkpoint = engine.read_json(root / engine.read_json(pointer)["checkpoint"] / "manifest.json")
        if checkpoint.get("test_injected") is not False:
            raise ValueError("test-injected run cannot enter the formal batch")
        position = checkpoint["position"]
    if running(root):
        return {"state": "running", "position": position}
    summary = root / "summary.json"
    if summary.is_file() and engine.read_json(summary).get("complete") is True:
        return {"state": "complete", "position": position}
    attempts = sorted(root.glob(f"{position + 1:04d}-*/attempt-*"))
    if attempts:
        last = attempts[-1]
        if (last / "promotion.json").is_file() or (last / "episode/episode-result-private.json").is_file():
            return {"state": "queued_recovery", "position": position}
        failure = last / "failure.json"
        failure_payload = engine.read_json(failure) if failure.is_file() else {}
        if retry_infrastructure and failure_payload.get("status") == "infrastructure_error":
            return {"state": "queued_recovery", "position": position}
        reason = failure_payload.get("error", "unclear interrupted attempt")
        if reason not in {annotation_recovery.ERROR, trajectory_display_compat.ERROR,
                          "Business tool result is not a valid direct result"}:
            return {"state": "blocked", "position": position, "reason": reason}
    return {"state": "queued", "position": position}


def run_batch(
    registry,
    output,
    workers,
    credentials_file,
    execution_mode="formal",
    retry_infrastructure=False,
):
    if type(workers) is not int or not 1 <= workers <= 36:
        raise ValueError("workers must be in [1, 36]")
    output = Path(output).expanduser().resolve()
    frozen_registry = load_registry(registry, output, execution_mode)
    entries = frozen_registry["runs"]
    output.mkdir(parents=True, exist_ok=True)
    children, blocked, scored = {}, {}, set()
    with engine._arm_lock(output):
        controller_plan = {
            "schema": CONTROLLER_PLAN_SCHEMA,
            "registry": frozen_registry,
            "registry_sha256": engine.digest(frozen_registry),
            "max_concurrent_arms": workers,
            "credential_source": "private_file" if credentials_file else "environment",
            "retry_infrastructure": retry_infrastructure,
            "execution_mode": execution_mode,
            "claim_scope": execution_policy.claim_scope_for_mode(execution_mode),
            "candidate_only": execution_mode == "candidate",
        }
        engine._write_once(output / "controller-plan.json", controller_plan)
        while True:
            rows, seen = [], set()
            for entry in entries:
                run = Path(entry["output"]).resolve()
                binding = entry.get("binding")
                for arm in engine.ARMS:
                    key = (str(run), arm)
                    if key in seen:
                        raise ValueError("duplicate registered arm")
                    seen.add(key)
                    if key in children:
                        process, log, logfile = children[key]
                        code = process.poll()
                        if code is not None:
                            log.close()
                            del children[key]
                            if code != 0:
                                blocked[key] = {"exit_code": code, "log": str(logfile)}
                    item = status(
                        run,
                        arm,
                        retry_infrastructure=retry_infrastructure,
                    )
                    if key in children:
                        item["state"] = "running"
                    if key in blocked and item["state"] != "complete":
                        item.update(state="blocked", controller_failure=blocked[key])
                    rows.append({"case": entry["case"], "binding": binding,
                                 "execution_mode": entry["execution_mode"],
                                 "claim_scope": entry["claim_scope"],
                                 "candidate_only": entry["candidate_only"],
                                 "output": str(run), "arm": arm, **item})
            active = sum(row["state"] == "running" for row in rows)
            for row in rows:
                if active >= workers:
                    break
                if row["state"] not in {"queued", "queued_recovery"}:
                    continue
                if shutil.disk_usage(row["output"]).free < 20 * 1024 ** 3:
                    row.update(state="storage_wait", reason="less than 20 GiB free; running arms are not stopped")
                    continue
                key = (row["output"], row["arm"])
                logfile = output / f"{row['case']}-{row['arm']}-{uuid.uuid4().hex[:8]}.log"
                log = logfile.open("x")
                command = [sys.executable, "-m", "rescopebench.supervised_eval", "--case", row["case"],
                           "--output", row["output"], "--arm", row["arm"],
                           "--execution-mode", row["execution_mode"]]
                if row.get("binding"):
                    command.extend(["--binding", row["binding"]])
                if credentials_file:
                    command.extend(["--credentials-file", str(Path(credentials_file).resolve())])
                if retry_infrastructure:
                    command.append("--retry-infrastructure")
                process = subprocess.Popen(command, cwd=pathing.FINAL_ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
                children[key] = (process, log, logfile)
                row.update(state="running", pid=process.pid, log=str(logfile))
                active += 1
            for entry in entries:
                run = str(Path(entry["output"]).resolve())
                matching = [row for row in rows if row["output"] == run]
                if run not in scored and all(row["state"] == "complete" for row in matching):
                    try:
                        formal_evidence_cleanup.finalize_completed_run(Path(run))
                    except Exception as error:
                        engine.write_json(Path(run) / "metrics-failure.json", {"error_type": type(error).__name__, "error": str(error)[:1800]})
                    scored.add(run)
            counts = {state: sum(row["state"] == state for row in rows) for state in {row["state"] for row in rows}}
            report = {"schema": "rescopebench-batch-status/v1", "timestamp": time.time(),
                      "max_concurrent_arms": workers, "counts": counts, "arms": rows,
                      "execution_mode": controller_plan["execution_mode"],
                      "claim_scope": controller_plan["claim_scope"],
                      "candidate_only": controller_plan["candidate_only"],
                      "registry_sha256": controller_plan["registry_sha256"],
                      "scores_are_not_inferred_from_process_exit": True}
            engine.write_json(output / "status.json", report)
            print(json.dumps({"time": report["timestamp"], "counts": counts}), flush=True)
            if rows and all(row["state"] in {"complete", "blocked"} for row in rows):
                return report
            time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--execution-mode",
        choices=execution_policy.EXECUTION_MODES,
        default="formal",
        help="Candidate execution requires explicit opt-in and remains candidate-only.",
    )
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--retry-infrastructure", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 36:
        parser.error("workers must be in [1, 36]")
    run_batch(
        args.registry,
        args.output,
        args.workers,
        args.credentials_file,
        args.execution_mode,
        args.retry_infrastructure,
    )


if __name__ == "__main__":
    main()
