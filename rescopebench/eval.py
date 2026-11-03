"""Standalone formal-stream command line; each arm runs in its own process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from rescopebench import (
    board_registry,
    execution_binding,
    execution_policy,
    formal_evidence_cleanup,
    model_input_preview,
    native_executor,
    pathing,
    runtime_cleanup,
    stream_engine,
    trajectory_display_compat,
    tool_execution,
)
from rescopebench.credentials import load_credentials
from rescopebench.formal_driver import load_driver


def _execution_mode(requested: str | None) -> str:
    return requested or "formal"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "prepare", "render-input", "run-arm", "verify"),
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--arm", choices=stream_engine.ARMS)
    parser.add_argument("--position", type=int)
    parser.add_argument("--model", default=stream_engine.native_runtime.MODEL)
    parser.add_argument("--provider-api", choices=("chat_completions", "responses"),
                        help="Freeze upstream API when preparing a new run.")
    parser.add_argument("--tool-execution", choices=tool_execution.MODES,
                        help="New runs/input previews default to batched; reference clones and resumes inherit the frozen plan. Use serial for an explicit single-call control.")
    parser.add_argument("--carriers", help="Inline JSON object, @file, or JSON file path.")
    parser.add_argument("--history", help="Inline JSON array, @file, or JSON file path.")
    parser.add_argument("--retry-infrastructure", action="store_true")
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument(
        "--binding",
        type=Path,
        help="Pin an execution binding; existing plans reject a different binding.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=execution_policy.EXECUTION_MODES,
        help="Run formally by default; candidate execution requires explicit opt-in.",
    )
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
    if args.provider_api is not None and args.command != "prepare":
        parser.error("--provider-api can only be selected when preparing a new run")
    if args.tool_execution is not None and args.command not in {"prepare", "render-input"}:
        parser.error("--tool-execution is only set when preparing or rendering a new run")
    output = args.output
    if args.command in {"run-arm", "verify"} and output is None:
        parser.error(f"{args.command} requires --output")
    execution_command = args.command in {"prepare", "run-arm", "verify"}
    execution_mode = _execution_mode(args.execution_mode) if execution_command else "candidate"
    if args.command == "prepare" and args.reference_run is not None:
        binding = execution_binding.planned_run_binding(
            args.reference_run, args.case, args.binding, execution_mode=execution_mode
        )
    elif args.command in {"run-arm", "verify"}:
        binding = execution_binding.planned_run_binding(
            output, args.case, args.binding, execution_mode=execution_mode
        )
    else:
        binding = execution_binding.new_run_binding(
            args.case, args.binding, execution_mode=execution_mode
        )
    driver = load_driver(args.case, binding)
    if args.command == "render-input":
        if args.position is None:
            parser.error("render-input requires --position")
        try:
            carriers = (
                model_input_preview.parse_json_argument(args.carriers, name="carriers")
                if args.carriers is not None
                else None
            )
            completed_history = (
                model_input_preview.parse_json_argument(args.history, name="history")
                if args.history is not None
                else None
            )
            result = model_input_preview.render_driver_input(
                driver,
                args.position,
                model=args.model,
                carriers=carriers,
                history=completed_history,
                tool_execution_mode=args.tool_execution or "batched",
            )
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return
    runtime_cleanup.install(driver)
    if args.command == "preflight":
        result = driver.validate()
    elif args.command == "prepare":
        if output is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = pathing.OUTPUT_ROOT / args.case / driver.case_version / f"{stamp}-{uuid.uuid4().hex[:8]}"
        execution_policy.validate_driver_policy(driver, execution_mode)
        if args.reference_run is None:
            plan = stream_engine.create_plan(
                output,
                driver,
                model=args.model,
                container=native_executor.container_contract(),
                execution_mode=execution_mode,
                tool_execution_mode=args.tool_execution or "batched",
                provider_api=args.provider_api or "chat_completions",
            )
        else:
            plan = stream_engine.clone_plan(
                args.reference_run,
                output,
                driver,
                model=args.model,
                execution_mode=execution_mode,
                tool_execution_mode=args.tool_execution,
                provider_api=args.provider_api,
            )
        runtime_cleanup.record(output, driver)
        result = {"status": "prepared_not_run", "output": str(output.resolve()),
                  "case_id": driver.case_id, "model": args.model, "plan_sha256": stream_engine.digest(plan),
                  "execution_mode": plan["execution_mode"], "claim_scope": plan["claim_scope"],
                  "candidate_only": plan["candidate_only"]}
    else:
        if args.command == "verify":
            plan = stream_engine.verify_plan(output, driver, execution_mode=execution_mode)
            result = {"status": "passed", "plan_sha256": stream_engine.digest(plan),
                      "execution_mode": plan["execution_mode"], "claim_scope": plan["claim_scope"],
                      "candidate_only": plan["candidate_only"]}
        else:
            if args.arm is None:
                parser.error("run-arm requires --arm")
            plan = stream_engine.verify_plan(output, driver, execution_mode=execution_mode)
            board_registration = board_registry.register_if_configured(
                output,
                run_root=args.board_root,
                run_id=args.board_run_id,
                arm=args.arm,
                disabled=args.no_board_register,
            )
            trajectory_display_compat.install(output, args.arm)
            runtime_cleanup.record(output, driver)
            load_credentials(args.credentials_file, transport=plan["transport"])
            result = stream_engine.run_arm(
                output,
                args.arm,
                driver,
                retry_infrastructure=args.retry_infrastructure,
                execution_mode=execution_mode,
            )
            if board_registration is not None:
                result = {**result, "board_registration": board_registration}
            if result.get("complete") is True:
                result = {**result, "finalization": formal_evidence_cleanup.finalize_completed_run(output)}
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
