"""Turn Hermes iteration exhaustion into a scored semantic stop."""

from __future__ import annotations

from copy import deepcopy
import importlib
import os
import re
import sys
from typing import Any


ITERATION_EXIT = re.compile(r"max_iterations_reached\((\d+)/(\d+)\)")


def exchange_path(module: Any):
    exchange = getattr(module, "EXCHANGE", None)
    if exchange is None:
        exchange = module.base.EXCHANGE
    return exchange


def normalize_iteration_result(
    result: dict[str, Any], max_iterations: int
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    reason = result.get("turn_exit_reason")
    match = ITERATION_EXIT.fullmatch(reason) if isinstance(reason, str) else None
    exhausted = (
        result.get("completed") is not True
        and result.get("failed") is not True
        and result.get("interrupted") is not True
        and match is not None
        and int(match.group(2)) == max_iterations
        and int(match.group(1)) >= max_iterations
    )
    if not exhausted:
        return result, None
    normalized = deepcopy(result)
    normalized.update(completed=True, failed=False, interrupted=False)
    receipt = {
        "schema": "robot-hermes-semantic-limit/final-v1",
        "kind": "agent_iteration_budget_exhausted",
        "turn_exit_reason": reason,
        "task_completed": False,
        "score_terminal_business_state": True,
        "effective_max_iterations": max_iterations,
    }
    normalized["rescope_semantic_stop"] = deepcopy(receipt)
    return normalized, receipt


def run_wrapped_driver(
    module_name: str, max_iterations: int, *, learner_wall_clock_budget: bool = True
) -> None:
    module = importlib.import_module(module_name)
    exchange = exchange_path(module)
    # The wrapped driver normally sets these immediately before importing Hermes.
    # This wrapper imports Hermes first to patch its iteration limit, so establish
    # the same writable home before that import occurs.
    profile = getattr(getattr(module, "base", None), "PROFILE", None)
    if profile is not None:
        os.environ.update(HERMES_HOME=str(profile), HOME="/work")
    sys.path.insert(0, "/opt/hermes")
    from run_agent import AIAgent

    original_init = AIAgent.__init__
    original_run = AIAgent.run_conversation

    def patched_init(self, *args, **kwargs):
        requested = kwargs.get("max_iterations")
        requested_seconds = kwargs.get("run_budget_seconds")
        kwargs["max_iterations"] = max_iterations
        if not learner_wall_clock_budget:
            kwargs["run_budget_seconds"] = None
        original_init(self, *args, **kwargs)
        if not learner_wall_clock_budget:
            # Hermes can fall back to config during init. Host delays must not
            # become learner-visible instructions; the outer timeout remains.
            self.run_budget_seconds = None
        module.base.write(
            exchange / "runtime-budget-contract.json",
            {
                "schema": "rescopebench-runtime-budget/v1",
                "requested_by_wrapped_driver_seconds": requested_seconds,
                "effective_learner_wall_clock_seconds": self.run_budget_seconds,
                "host_timeout_policy": "infrastructure_failure_not_business_score",
                "business_budget_unchanged": True,
            },
        )
        module.base.write(
            exchange / "iteration-contract.json",
            {
                "schema": "robot-hermes-semantic-limit/final-v1",
                "requested_by_wrapped_driver": requested,
                "effective_max_iterations": max_iterations,
                "exhaustion_policy": "score_terminal_business_state",
            },
        )

    def patched_run(self, *args, **kwargs):
        raw = original_run(self, *args, **kwargs)
        module.base.write(exchange / "foreground-original.json", raw)
        normalized, receipt = normalize_iteration_result(raw, max_iterations)
        if receipt is not None:
            module.base.write(exchange / "semantic-stop.json", receipt)
        return normalized

    AIAgent.__init__ = patched_init
    AIAgent.run_conversation = patched_run
    try:
        module.main()
    except BaseException as error:
        module.base.write(
            exchange / "worker-failure.json",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        raise
