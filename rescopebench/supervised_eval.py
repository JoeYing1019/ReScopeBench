"""Run a full formal arm, recovering only audited postprocessing failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from rescopebench import (
    annotation_recovery,
    board_registry,
    execution_binding,
    execution_policy,
    formal_evidence_cleanup,
    runtime_cleanup,
    stream_engine,
    trajectory_display_compat,
)
from rescopebench.credentials import load_credentials
from rescopebench.formal_driver import load_driver


def _execution_mode(requested: str | None) -> str:
    return requested or "formal"


def recover_pending(output, arm):
    root = output / arm
    pointer = root / "current.json"
    if not pointer.is_file():
        return
    checkpoint = stream_engine.read_json(root / stream_engine.read_json(pointer)["checkpoint"] / "manifest.json")
    position = checkpoint["position"] + 1
    failures = sorted(root.glob(f"{position:04d}-*/attempt-*/episode/failure.json"))
    if not failures or (failures[-1].parent / "episode-result-private.json").exists():
        return
    failure = stream_engine.read_json(failures[-1])
    if failure.get("error") == trajectory_display_compat.ERROR:
        print(json.dumps(trajectory_display_compat.recover(output, arm), sort_keys=True), flush=True)
    elif failure.get("error") == annotation_recovery.ERROR:
        print(json.dumps(annotation_recovery.recover(output, arm), sort_keys=True), flush=True)


def empty_history_failure_retry(output, arm):
    root = output / arm
    if not (root / "current.json").is_file():
        return False
    pointer = stream_engine.read_json(root / "current.json")
    position = stream_engine.read_json(root / pointer["checkpoint"] / "manifest.json")["position"] + 1
    attempts = sorted(root.glob(f"{position:04d}-*/attempt-*"))
    if not attempts:
        return False
    attempt = attempts[-1]
    path = attempt / "failure.json"
    if not path.is_file() or (attempt / "episode").exists() or (attempt / "provider").exists():
        return False
    failure = stream_engine.read_json(path)
    if failure.get("error") != "Business tool result is not a valid direct result":
        return False
    receipt = {"schema": "rescopebench-empty-attempt-retry/v1", "reason": failure["error"],
               "previous_failure_sha256": stream_engine.digest(failure), "model_requests_reissued": 0,
               "no_episode_or_provider_started": True, "position": position}
    stream_engine._write_once(attempt / "empty-retry-authorization.json", receipt)
    return True


def cleanup_position(output, arm, position):
    promotions = list((output / arm).glob(f"{position:04d}-*/attempt-*/promotion.json"))
    if len(promotions) != 1:
        raise RuntimeError("Expected one committed promotion before evidence cleanup")
    return formal_evidence_cleanup.cleanup_promoted_attempt(
        output, arm, promotions[0].parent, apply=True
    )


def run_with_cleanup(output, arm, driver, *, execution_mode, retry_infrastructure=False):
    """Bound disk growth between committed tasks without changing episode behavior."""
    recovery_attempts = 0
    while True:
        free = shutil.disk_usage(output).free
        if free < 20 * 1024 ** 3:
            raise RuntimeError("No new episode: less than 20 GiB disk space")
        if free < 30 * 1024 ** 3:
            print(json.dumps({"disk_warning_free_bytes": free}), flush=True)
        recover_pending(output, arm)
        try:
            result = stream_engine.run_arm(
                output, arm, driver, limit=1,
                retry_infrastructure=retry_infrastructure or empty_history_failure_retry(output, arm),
                execution_mode=execution_mode,
            )
        except ValueError as error:
            if str(error) not in {annotation_recovery.ERROR, trajectory_display_compat.ERROR}:
                raise
            recovery_attempts += 1
            if recovery_attempts >= 17:
                raise RuntimeError("postprocessing recovery made no progress") from error
            continue
        retry_infrastructure = False
        recovery_attempts = 0
        print(json.dumps(result, sort_keys=True), flush=True)
        cleanup = cleanup_position(output, arm, result["position"])
        print(json.dumps({"task_cleanup": cleanup}, sort_keys=True), flush=True)
        if result["complete"]:
            finalization = formal_evidence_cleanup.finalize_completed_run(output)
            print(json.dumps(finalization, sort_keys=True), flush=True)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--binding",
        type=Path,
        help="Pin the execution binding already frozen by the prepared plan.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=execution_policy.EXECUTION_MODES,
        help="Run formally by default; candidate execution requires explicit opt-in.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=stream_engine.ARMS)
    parser.add_argument(
        "--retry-infrastructure",
        action="store_true",
        help="Retry an incomplete infrastructure attempt while retaining audited compatibility recovery.",
    )
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument(
        "--board-root",
        type=Path,
        help="Override the configured live-board run root.",
    )
    parser.add_argument("--board-run-id", help="Stable label shown on the live board.")
    parser.add_argument(
        "--no-board-register",
        action="store_true",
        help="Disable automatic board registration for this invocation.",
    )
    args = parser.parse_args()
    execution_mode = _execution_mode(args.execution_mode)
    binding = execution_binding.planned_run_binding(
        args.output, args.case, args.binding, execution_mode=execution_mode
    )
    driver = load_driver(args.case, binding)
    plan = stream_engine.verify_plan(args.output, driver, execution_mode=execution_mode)
    board_registration = board_registry.register_if_configured(
        args.output,
        run_root=args.board_root,
        run_id=args.board_run_id,
        arm=args.arm,
        disabled=args.no_board_register,
    )
    load_credentials(args.credentials_file, transport=plan["transport"])
    trajectory_display_compat.install(args.output, args.arm)
    runtime_cleanup.install(driver)
    runtime_cleanup.record(args.output, driver)
    run_with_cleanup(args.output, args.arm, driver, execution_mode=execution_mode,
                     retry_infrastructure=args.retry_infrastructure)
    if board_registration is not None:
        print(json.dumps({"board_registration": board_registration}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
