"""Entry points with the harness adjustments installed.

    python -m rescope_harness eval ...              same arguments as rescopebench.eval
    python -m rescope_harness supervised ...        same arguments as rescopebench.supervised_eval
    python -m rescope_harness batch ...             same arguments as rescopebench.batch_eval
    python -m rescope_harness provider-check ...    same arguments as rescopebench.provider_check
    python -m rescope_harness supervised-retry ...  supervised, rerun in place after an infrastructure failure

Installed before the benchmark module runs, in this order:
    sandbox           RESCOPE_SANDBOX=docker keeps the release's local docker executor;
                      any other value runs the worker in that remote sandbox provider
    rescope_model_api four native model APIs, retries, feedback rounds (rescopebench)
Only module attributes are replaced; no rescopebench file is edited.
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys

MODULES = {"eval": "rescopebench.eval", "supervised": "rescopebench.supervised_eval",
           "batch": "rescopebench.batch_eval", "provider-check": "rescopebench.provider_check"}


def _route_batch_children() -> None:
    """batch_eval spawns each arm as `-m rescopebench.supervised_eval`; keep the harness on.

    The rewrite is installed on the subprocess module itself: runpy reloads
    batch_eval in a fresh namespace, so patching a batch_eval attribute would be lost.
    Only that exact argv is changed; every other Popen call is untouched.
    """
    if getattr(subprocess, "_rescope_harness_routed", False):
        return
    original = subprocess.Popen

    def popen(command, *args, **kwargs):
        if (isinstance(command, list) and len(command) >= 3
                and command[1:3] == ["-m", "rescopebench.supervised_eval"]):
            command = [command[0], "-m", "rescope_harness", "supervised-retry", *command[3:]]
        return original(command, *args, **kwargs)

    subprocess.Popen = popen
    subprocess._rescope_harness_routed = True


def _supervised_retry(args: list[str]) -> int:
    """Run one arm, rerunning it in place while its current task failed on infrastructure.

    batch_eval marks an arm whose process exited non-zero as blocked until the whole
    controller ends, so one sandbox hiccup idled the arm for hours. Rerunning it here
    keeps the arm's slot; the per-task attempt cap is retry.MAX_TASK_ATTEMPTS.
    """
    from pathlib import Path
    import shutil
    import time
    from rescopebench.batch_eval import running
    from rescope_harness.retry import retryable_arm

    arm_root = Path(args[args.index("--output") + 1]) / args[args.index("--arm") + 1]
    fruitless = 0
    while True:
        before = len(list(arm_root.glob("*/attempt-*")))
        code = subprocess.call([sys.executable, "-m", "rescope_harness", "supervised", *args])
        if code == 0:
            return code
        if shutil.disk_usage(arm_root).free < 25 * 1024 ** 3:
            # stream_engine refuses a new episode under 20 GiB free; wait for space.
            print(f"[rescope_harness] {arm_root}: disk nearly full, waiting", flush=True)
            time.sleep(300)
            continue
        if running(arm_root):
            # A runner left by an earlier controller still holds the arm; let it finish.
            print(f"[rescope_harness] {arm_root}: another runner holds the arm, waiting", flush=True)
            time.sleep(120)
            continue
        # A rerun that fails before starting a new attempt will fail the same way again.
        fruitless = fruitless + 1 if len(list(arm_root.glob("*/attempt-*"))) == before else 0
        if fruitless >= 3:
            print(f"[rescope_harness] {arm_root}: three reruns started no attempt, giving up", flush=True)
            return code
        if not retryable_arm(arm_root):
            return code
        print(f"[rescope_harness] {arm_root}: infrastructure failure, rerunning the arm", flush=True)
        time.sleep(60)


def _run_supervised() -> None:
    """supervised_eval with a fresh attempt when the release's post-hoc recovery cannot run.

    Before each task supervised_eval re-validates a "Foreground assistant messages
    differ" or "Unknown Hermes business-result annotation" failure through
    annotation_recovery, which requires every source file to hash as it did during
    that episode. Once a harness file has changed that never holds again, so the arm
    would fail forever. Such an attempt is an infrastructure failure; run_arm with
    --retry-infrastructure starts a new attempt for the task under the attempt cap.
    The module is imported (not run with runpy) so its recover_pending can be wrapped.
    """
    from rescopebench import supervised_eval

    original = supervised_eval.recover_pending

    def recover_pending(output, arm):
        try:
            return original(output, arm)
        except ValueError as error:
            message = str(error)
            if not (message.startswith("source is no longer frozen:") or message in {
                    "worker did not finish cleanly in the frozen image", "worker lifecycle is incomplete"}):
                # Do not turn integrity/replay mismatches into model resampling.
                raise
            print(json.dumps({"recovery_skipped": str(error)[:300], "fresh_attempt": True}), flush=True)

    supervised_eval.recover_pending = recover_pending
    supervised_eval.main()


def _limit_tasks() -> None:
    """RESCOPE_LIMIT=N stops run-arm after N tasks (smoke tests); the run stays resumable."""
    from rescopebench import stream_engine

    limit = int(os.environ["RESCOPE_LIMIT"])
    original = stream_engine.run_arm

    def run_arm(*args, **kwargs):
        return original(*args, **{**kwargs, "limit": limit})

    stream_engine.run_arm = run_arm


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "supervised-retry":
        raise SystemExit(_supervised_retry(sys.argv[2:]))
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        raise SystemExit(__doc__)
    command = sys.argv[1]
    from rescope_harness import sandbox
    from rescopebench import rescope_model_api

    if command != "provider-check":
        sandbox.install()
    rescope_model_api.install()
    if command == "batch":
        _route_batch_children()
    if os.environ.get("RESCOPE_LIMIT"):
        _limit_tasks()
    sys.argv = [MODULES[command], *sys.argv[2:]]
    if command == "supervised":
        _run_supervised()
        return
    runpy.run_module(MODULES[command], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
