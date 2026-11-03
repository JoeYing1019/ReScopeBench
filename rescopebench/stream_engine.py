"""Environment-neutral, durable Native/Reset stream execution."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Protocol
import uuid

from rescopebench import credentials, execution_policy, native_executor, pathing, responses_provider, tool_execution
from rescopebench import portable_provider_config, portable_provider_runtime
from rescopebench import lifecycle as evolution_schedule
from rescopebench.agent_runtime import history, learner_surface, model_api, native_runtime
from rescopebench import case_names, public_projection


SCHEMA = "rescopebench-formal-stream/v1"
ARMS = ("native", "reset")


class Driver(Protocol):
    case_id: str
    case_version: str
    tools: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    prompt: dict[str, str] | None

    def validate(self) -> dict[str, Any]: ...
    def source_files(self) -> list[Path]: ...
    def create_world(self) -> Any: ...
    def begin(self, world: Any, task: dict[str, Any]) -> dict[str, Any]: ...
    def call(self, world: Any, name: str, arguments: dict[str, Any]) -> Any: ...
    def grade(self, world: Any, task: dict[str, Any]) -> dict[str, Any]: ...
    def public_grade(self, grade: dict[str, Any]) -> dict[str, Any]: ...
    def close(self, world: Any) -> None: ...


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_once(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"durable record differs: {path}")
        return
    write_json(path, value)


def _files(driver: Driver) -> dict[str, str]:
    paths = set(map(Path, driver.source_files()))
    binding_path = getattr(driver, "binding_path", None)
    if binding_path is not None:
        paths.add(Path(binding_path))
    paths.update(Path(module.__file__) for module in (history, learner_surface, model_api, native_runtime))
    paths.update((Path(__file__), Path(execution_policy.__file__), Path(pathing.__file__), Path(native_executor.__file__),
                  Path(public_projection.__file__), Path(case_names.__file__), Path(model_api.scheduling.__file__),
                  native_runtime.native_task.WORKER, native_runtime.native_task.NATIVE_BASE,
                  native_runtime.native_task.WORKER_BASE, native_runtime.native_task.ADOPTION_FILE,
                  native_runtime.native_task.SEMANTIC_LIMIT, Path(native_runtime.native_task.__file__),
                  native_runtime.native_task.COMPLETION_EVIDENCE,
                  Path(evolution_schedule.__file__),
                  Path(tool_execution.__file__), Path(responses_provider.__file__), Path(credentials.__file__),
                  Path(portable_provider_config.__file__), Path(portable_provider_runtime.__file__)))
    # Include every shared helper and mounted worker in the frozen run identity.
    paths.update(Path(native_runtime.__file__).parent.rglob("*.py"))
    return pathing.normalize_inventory({
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)
    })


def public_task(task: dict[str, Any]) -> dict[str, str]:
    result = {"event_id": task["event_id"], "query": task["query"]}
    if any(not isinstance(value, str) or not value.strip() for value in result.values()):
        raise ValueError("public event and query must be nonempty strings")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}", result["event_id"]):
        raise ValueError("public event id must be safe as a directory component")
    learner_surface.assert_opaque_learner_surface(result, surface="formal episode")
    return result


def _runtime(driver: Driver, model: str, tool_execution_mode: str = "serial"):
    return native_executor.for_model(model, prompt=driver.prompt, tool_execution_mode=tool_execution_mode)


def _validate_driver(driver: Driver) -> dict[str, Any]:
    validation = driver.validate()
    if validation.get("status") != "passed":
        raise ValueError(f"case admission failed: {validation}")
    schedule = getattr(driver, "lifecycle_schedule", None)
    if schedule is not None:
        evolution_schedule.validate_schedule(schedule, len(driver.tasks))
    elif len(driver.tasks) != 16:
        raise ValueError("a non-legacy stream requires an explicit lifecycle schedule")
    if len({task["event_id"] for task in driver.tasks}) != len(driver.tasks):
        raise ValueError("formal stream requires unique task event identifiers")
    for position, task in enumerate(driver.tasks, 1):
        if task.get("position") != position:
            raise ValueError("private task positions must be consecutive")
        public_task(task)
    native_runtime.tool_parameters(driver.tools)
    return validation


def create_plan(output: Path, driver: Driver, *, model: str = native_runtime.MODEL,
                container: dict[str, Any] | None = None,
                execution_mode: str = "formal", tool_execution_mode: str = "serial",
                provider_api: str = "chat_completions") -> dict[str, Any]:
    policy = execution_policy.validate_driver_policy(driver, execution_mode)
    validation = _validate_driver(driver)
    runtime = _runtime(driver, model, tool_execution_mode)
    world = driver.create_world()
    try:
        initial = deepcopy(world.snapshot())
    finally:
        driver.close(world)
    output = Path(output).resolve()
    if output.is_relative_to(pathing.FINAL_ROOT):
        raise ValueError("formal output must be outside the code package")
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": SCHEMA + "/plan", "case_id": driver.case_id, "case_version": driver.case_version,
        **policy,
        "model": model, "seed": runtime.SEED, "max_tokens": runtime.MAX_TOKENS,
        "max_iterations": runtime.MAX_ITERATIONS, "arms": list(ARMS),
        "tool_execution": deepcopy(runtime.tool_execution),
        "tasks": deepcopy(driver.tasks), "tasks_sha256": digest(driver.tasks),
        "tools": deepcopy(driver.tools), "tools_sha256": digest(driver.tools),
        "prompt": runtime.prompt_contract(), "runtime_sources": _files(driver),
        "validation_sha256": digest(validation), "initial_snapshot_sha256": digest(initial),
        "container": deepcopy(container),
        "transport": responses_provider.transport(provider_api),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if getattr(driver, "lifecycle_schedule", None) is not None:
        plan["lifecycle_schedule"] = deepcopy(driver.lifecycle_schedule)
    write_json(output / "initial-snapshot.json", initial)
    write_json(output / "admission.json", validation)
    write_json(output / "run-plan.json", plan)
    return plan


def verify_plan(output: Path, driver: Driver, *, execution_mode: str = "formal") -> dict[str, Any]:
    output = Path(output).resolve()
    plan = read_json(output / "run-plan.json")
    plan_policy = execution_policy.validate_plan_policy(plan, execution_mode)
    driver_policy = execution_policy.validate_driver_policy(driver, execution_mode)
    validation = _validate_driver(driver)
    runtime = _runtime(driver, plan["model"], tool_execution.plan_mode(plan))
    checks = {
        "schema": plan.get("schema") == SCHEMA + "/plan",
        "execution_policy": plan_policy == driver_policy,
        "case": plan.get("case_id") == driver.case_id and plan.get("case_version") == driver.case_version,
        "arms": plan.get("arms") == list(ARMS),
        "tasks": plan.get("tasks") == driver.tasks and plan.get("tasks_sha256") == digest(driver.tasks),
        "lifecycle_schedule": plan.get("lifecycle_schedule") == getattr(driver, "lifecycle_schedule", None),
        "tools": plan.get("tools") == driver.tools and plan.get("tools_sha256") == digest(driver.tools),
        "prompt": plan.get("prompt") == runtime.prompt_contract(),
        "runtime": plan.get("runtime_sources") == _files(driver),
        "settings": plan.get("seed") == runtime.SEED and plan.get("max_tokens") == runtime.MAX_TOKENS
        and plan.get("max_iterations") == runtime.MAX_ITERATIONS,
        "admission": plan.get("validation_sha256") == digest(validation),
        "initial": plan.get("initial_snapshot_sha256") == digest(read_json(output / "initial-snapshot.json")),
        "resources": plan.get("container") is None or plan["container"].get("resources") == native_executor.RESOURCES,
        "transport": plan.get("transport") == responses_provider.validate_transport(plan.get("transport", {})),
    }
    if not all(checks.values()):
        raise ValueError(f"run plan identity failed: {[key for key, value in checks.items() if not value]}")
    return plan


def clone_plan(reference: Path, output: Path, driver: Driver, *, model: str,
               execution_mode: str = "formal", tool_execution_mode: str | None = None,
               provider_api: str | None = None) -> dict[str, Any]:
    """Preserve the frozen world; change only model and an explicitly selected dispatch mode."""
    original = verify_plan(reference, driver, execution_mode=execution_mode)
    mode = tool_execution_mode if tool_execution_mode is not None else tool_execution.plan_mode(original)
    runtime = _runtime(driver, model, mode)
    if runtime.prompt_contract() != original["prompt"]:
        raise ValueError("model comparison changed the experience scaffold")
    output = Path(output).resolve()
    if output.is_relative_to(pathing.FINAL_ROOT):
        raise ValueError("formal output must be outside the code package")
    output.mkdir(parents=True, exist_ok=False)
    plan = {**deepcopy(original), "model": model, "paired_input_plan_sha256": digest(original),
            "created_at": datetime.now(timezone.utc).isoformat()}
    if tool_execution_mode is not None:
        plan["tool_execution"] = deepcopy(runtime.tool_execution)
    if provider_api is not None:
        plan["transport"] = responses_provider.transport(provider_api)
    for name in ("initial-snapshot.json", "admission.json"):
        write_json(output / name, read_json(Path(reference) / name))
    write_json(output / "run-plan.json", plan)
    verify_plan(output, driver, execution_mode=execution_mode)
    return plan


@contextmanager
def _arm_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this arm already has a live runner") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _checkpoint(root: Path, position: int, state: dict[str, Any], assets: dict[str, Any],
                episodes: list[dict[str, Any]], *, reset: bool, plan_hash: str,
                test_injected: bool, promotion_hash: str | None) -> None:
    assets = {} if reset else native_runtime.surface.validate_assets(assets)
    episodes = [] if reset else deepcopy(episodes)
    manifest = {
        "schema": SCHEMA + "/checkpoint", "position": position, "reset_control": reset,
        "snapshot_sha256": digest(state), "assets_sha256": digest(assets), "history_sha256": digest(episodes),
        "plan_sha256": plan_hash, "test_injected": test_injected, "promotion_sha256": promotion_hash,
    }
    destination = root / "checkpoints" / f"{position:04d}"
    if destination.exists():
        if read_json(destination / "manifest.json") != manifest:
            raise ValueError("existing checkpoint has different content")
        _load_checkpoint(destination, manifest)
    else:
        staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        staging.mkdir(parents=True)
        try:
            with (staging / "snapshot.json.gz").open("xb") as stream:
                stream.write(gzip.compress(canonical(state), compresslevel=1, mtime=0))
                stream.flush()
                os.fsync(stream.fileno())
            write_json(staging / "assets.json", assets)
            write_json(staging / "history.json", episodes)
            write_json(staging / "manifest.json", manifest)
            os.replace(staging, destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    write_json(root / "current.json", {"checkpoint": str(destination.relative_to(root)), "manifest_sha256": digest(manifest)})


def _load_checkpoint(directory: Path, manifest: dict[str, Any]):
    with gzip.open(directory / "snapshot.json.gz", "rb") as stream:
        state = json.load(stream)
    assets, episodes = read_json(directory / "assets.json"), read_json(directory / "history.json")
    checks = {"snapshot": digest(state) == manifest["snapshot_sha256"],
              "assets": digest(assets) == manifest["assets_sha256"],
              "history": digest(episodes) == manifest["history_sha256"]}
    if manifest["reset_control"]:
        checks["empty_reset"] = assets == {} and episodes == []
    if not all(checks.values()):
        raise ValueError("checkpoint payload integrity failed")
    return state, assets, episodes, manifest


def current_checkpoint(root: Path, plan: dict[str, Any], *, reset: bool):
    pointer = read_json(root / "current.json")
    relative = Path(pointer["checkpoint"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe checkpoint path")
    directory = root / relative
    manifest = read_json(directory / "manifest.json")
    position = manifest.get("position")
    if (manifest.get("schema") != SCHEMA + "/checkpoint" or type(position) is not int
            or not 0 <= position <= len(plan["tasks"]) or relative.parts != ("checkpoints", f"{position:04d}")
            or type(manifest.get("test_injected")) is not bool):
        raise ValueError("invalid checkpoint identity")
    if digest(manifest) != pointer["manifest_sha256"] or manifest["plan_sha256"] != digest(plan):
        raise ValueError("checkpoint plan or pointer integrity failed")
    if manifest.get("reset_control") is not reset:
        raise ValueError("checkpoint belongs to another arm")
    loaded = _load_checkpoint(directory, manifest)
    if position == 0 and (manifest.get("promotion_sha256") is not None
                          or manifest["snapshot_sha256"] != plan["initial_snapshot_sha256"]
                          or loaded[1] != {} or loaded[2] != []):
        raise ValueError("initial checkpoint differs from the frozen initial world")
    return loaded


class Facade:
    def __init__(self, driver: Driver, world: Any, task: dict[str, Any]):
        self.driver, self.world, self.task = driver, world, task
        self.tools = deepcopy(driver.tools)

    def call(self, name, arguments):
        # Public observations may echo learner arguments or previously written
        # text. Audit supplied projections at construction, not after actions.
        return self.driver.call(self.world, name, arguments)

    def snapshot(self):
        return self.world.snapshot()

    def resource_state(self):
        return tool_execution.resource_state(self.world)

    def grade(self):
        before = digest(self.world.snapshot())
        value = self.driver.public_grade(self.driver.grade(self.world, self.task))
        if digest(self.world.snapshot()) != before:
            raise ValueError("grading changed business state")
        learner_surface.assert_opaque_learner_surface(value, surface="public feedback")
        native_runtime.public_feedback(value)
        return value

    def source_files(self):
        return [pathing.resolve_path(path) for path in _files(self.driver)]


def _provider(directory: Path, config=None):
    return credentials.provider(directory, config if config is not None else responses_provider.transport())


def _make_promotion(plan, task, result, state, assets, episodes, *, test_injected, reset, previous):
    payload = {
        "schema": SCHEMA + "/promotion", "position": task["position"], "event_id": task["event_id"],
        "plan_sha256": digest(plan), "test_injected": test_injected, "result": result,
        "reset_control": reset, "previous_checkpoint_sha256": digest(previous),
        "snapshot": state, "assets": assets, "history": episodes,
    }
    return {**payload, **{f"{key}_sha256": digest(payload[key]) for key in ("result", "snapshot", "assets", "history")}}


def _promote(root: Path, attempt: Path, promotion, plan, task, *, reset, test_injected, previous):
    checks = {
        "schema": promotion.get("schema") == SCHEMA + "/promotion",
        "position": promotion.get("position") == task["position"],
        "event": promotion.get("event_id") == task["event_id"],
        "plan": promotion.get("plan_sha256") == digest(plan),
        "semantic": promotion.get("result", {}).get("status") == "semantic_result",
        "arm": promotion.get("reset_control") is reset,
        "test_injected": promotion.get("test_injected") is test_injected
        and promotion.get("result", {}).get("test_injected") is test_injected,
        "previous": promotion.get("previous_checkpoint_sha256") == digest(previous)
        and previous.get("position") == task["position"] - 1,
    }
    checks.update({key: promotion.get(f"{key}_sha256") == digest(promotion.get(key)) for key in ("result", "snapshot", "assets", "history")})
    if not all(checks.values()):
        raise ValueError(f"invalid promotion journal: {[key for key, value in checks.items() if not value]}")
    native_runtime.public_feedback(promotion["result"]["grade"])
    native_runtime.surface.validate_assets(promotion["assets"])
    if promotion["assets"] != promotion["result"]["assets"]:
        raise ValueError("promotion assets differ from the semantic result")
    _write_once(attempt / "result.json", promotion["result"])
    _checkpoint(root, task["position"], promotion["snapshot"], promotion["assets"], promotion["history"],
                reset=reset, plan_hash=digest(plan), test_injected=promotion["test_injected"], promotion_hash=digest(promotion))


def _completed_evidence(attempt: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Recover a fully closed native episode, never reissue its model requests."""
    episode = attempt / "episode"
    result_path = episode / "episode-result-private.json"
    if not result_path.is_file():
        return None
    result = read_json(result_path)
    state_path = episode / "final-state-private.json"
    required = ("history_budget", "task_completed", "sources_unchanged", "terminal_state_sha256")
    if not state_path.is_file() or any(key not in result for key in required):
        raise RuntimeError("semantic evidence is incomplete; do not rerun the model")
    state = read_json(state_path)
    if result.get("status") != "semantic_result" or result["sources_unchanged"] is not True:
        raise ValueError("native completion evidence is not a valid semantic result")
    if result["terminal_state_sha256"] != native_runtime.core.digest(state):
        raise ValueError("native terminal state integrity failed")
    if read_json(episode / "sources-before.json") != read_json(episode / "sources-after.json"):
        raise ValueError("sources changed in the native completion evidence")
    return result, state


def _journal_result(driver, world, task, result, assets, episodes, attempt, plan, checkpoint, *, reset, injected):
    facade = Facade(driver, world, task)
    if result.get("status") != "semantic_result" or result.get("grade") != facade.grade():
        raise ValueError("semantic result does not match terminal grade")
    terminal_assets = native_runtime.surface.validate_assets(result["assets"])
    private_grade = driver.grade(world, task)
    result = {**result, "completed_artifacts": private_grade.get("completed_artifacts"), "test_injected": injected}
    write_json(attempt / "grade-private.json", private_grade)
    archive = history.HistoryArchive(driver.tools, episodes)
    archive.append({"episode_id": task["event_id"], "status": "semantic_result", "task": task["query"],
                    "trajectory": result["trajectory"], "grade": result["grade"],
                    "assets_before": assets, "assets_after": terminal_assets})
    promotion = _make_promotion(plan, task, result, world.snapshot(), terminal_assets, archive.episodes,
                               test_injected=injected, reset=reset, previous=checkpoint)
    write_json(attempt / "promotion.json", promotion)
    return promotion


def run_arm(output: Path, arm: str, driver: Driver, *, limit: int | None = None,
            retry_infrastructure: bool = False, episode_executor: Callable | None = None,
            provider_factory: Callable | None = None,
            execution_mode: str = "formal") -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("arm must be native or reset")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    output = Path(output).resolve()
    plan = verify_plan(output, driver, execution_mode=execution_mode)
    runtime = _runtime(driver, plan["model"], tool_execution.plan_mode(plan))
    injected = episode_executor is not None or provider_factory is not None
    execute = episode_executor or runtime.execute_episode
    factory = provider_factory or (lambda directory: _provider(directory, plan["transport"]))
    root, reset = output / arm, arm == "reset"
    with _arm_lock(root):
        if not (root / "current.json").exists():
            _checkpoint(root, 0, read_json(output / "initial-snapshot.json"), {}, [], reset=reset,
                        plan_hash=digest(plan), test_injected=injected, promotion_hash=None)
        state, assets, episodes, checkpoint = current_checkpoint(root, plan, reset=reset)
        if checkpoint["test_injected"] != injected:
            raise ValueError("test-injected and formal runs cannot share a checkpoint")
        if not injected and plan.get("container") != native_executor.container_contract():
            raise ValueError("formal run requires the frozen container image and limits")
        completed = 0
        for task in plan["tasks"][checkpoint["position"]:]:
            if limit is not None and completed >= limit:
                break
            if plan["runtime_sources"] != _files(driver):
                raise ValueError("implementation changed before next episode")
            task_root = root / f"{task['position']:04d}-{task['event_id']}"
            attempts = sorted(task_root.glob("attempt-*"))
            journals = [path for path in attempts if (path / "promotion.json").is_file()]
            if len(journals) > 1:
                raise ValueError("multiple durable semantic promotions at one position")
            if journals:
                promotion = read_json(journals[0] / "promotion.json")
                _promote(root, journals[0], promotion, plan, task, reset=reset,
                         test_injected=injected, previous=checkpoint)
            else:
                attempt_identity = {"plan_sha256": digest(plan), "event_id": task["event_id"],
                                    "position": task["position"], "reset_control": reset,
                                    "test_injected": injected, "previous_checkpoint_sha256": digest(checkpoint)}
                recovered = None
                for prior in attempts:
                    if read_json(prior / "attempt.json") != attempt_identity:
                        raise ValueError("prior attempt identity differs")
                    evidence = _completed_evidence(prior)
                    if evidence is not None:
                        if recovered is not None:
                            raise ValueError("multiple semantic results at one position")
                        recovered = (prior, evidence)
                    elif (prior / "result.json").is_file():
                        raise RuntimeError("semantic evidence exists without a complete journal; do not rerun the model")
                if attempts and recovered is None and not retry_infrastructure:
                    raise RuntimeError("prior infrastructure attempt requires explicit retry")
                attempt = recovered[0] if recovered else task_root / f"attempt-{len(attempts) + 1:04d}"
                if recovered is None:
                    attempt.mkdir(parents=True)
                    write_json(attempt / "attempt.json", attempt_identity)
                world = driver.create_world()
                try:
                    world.restore(deepcopy(state))
                    if digest(world.snapshot()) != digest(state):
                        raise ValueError("checkpoint restoration changed world state")
                    if recovered:
                        result, terminal_state = recovered[1]
                        if read_json(attempt / "assets-before.json") != assets or read_json(attempt / "history-before.json") != episodes:
                            raise ValueError("recovered episode entry carriers differ")
                        boundary = driver.begin(world, task)
                        if boundary.get("status") != "success":
                            raise ValueError("recovered task boundary failed")
                        if (boundary != read_json(attempt / "boundary-private.json")
                                or digest(world.snapshot()) != digest(read_json(attempt / "entry-snapshot.json"))):
                            raise ValueError("reconstructed task entry differs from completed evidence")
                        # Restore the closed terminal state after rebuilding ephemeral grading context.
                        world.restore(deepcopy(terminal_state))
                        if digest(world.snapshot()) != digest(terminal_state):
                            raise ValueError("recovered terminal state differs")
                    else:
                        boundary = driver.begin(world, task)
                        if boundary.get("status") != "success":
                            raise ValueError("task boundary failed")
                        write_json(attempt / "boundary-private.json", boundary)
                        write_json(attempt / "entry-snapshot.json", world.snapshot())
                        write_json(attempt / "assets-before.json", assets)
                        write_json(attempt / "history-before.json", episodes)
                        archive = history.HistoryArchive(driver.tools, episodes)
                        handler = archive.session(assets)
                        facade = Facade(driver, world, task)
                        context = factory(attempt / "provider")
                        if not hasattr(context, "__enter__"):
                            context = nullcontext(context)
                        with context as provider:
                            result = execute(attempt / "episode", public_task(task), facade, deepcopy(assets), provider,
                                             history_handler=handler, condition="native_memory_skill")
                    promotion = _journal_result(driver, world, task, result, assets, episodes, attempt, plan, checkpoint,
                                                reset=reset, injected=injected)
                    _promote(root, attempt, promotion, plan, task, reset=reset,
                             test_injected=injected, previous=checkpoint)
                except BaseException as error:
                    write_json(attempt / "failure.json", {"status": "infrastructure_error", "error_type": type(error).__name__,
                               "error": native_runtime._safe_error(error, None), "state_promoted": False})
                    raise
                finally:
                    driver.close(world)
            state, assets, episodes, checkpoint = current_checkpoint(root, plan, reset=reset)
            completed += 1
            write_json(root / "summary.json", {"schema": SCHEMA + "/arm-summary", "arm": arm, "position": checkpoint["position"],
                       "complete": checkpoint["position"] == len(plan["tasks"]), "model": plan["model"], "plan_sha256": digest(plan),
                       "execution_mode": plan["execution_mode"], "claim_scope": plan["claim_scope"],
                       "formal_launch_authorized": plan["formal_launch_authorized"],
                       "candidate_only": plan["candidate_only"], "test_injected": injected})
        summary = {"schema": SCHEMA + "/arm-summary", "arm": arm, "position": checkpoint["position"],
                   "complete": checkpoint["position"] == len(plan["tasks"]), "model": plan["model"], "plan_sha256": digest(plan),
                   "execution_mode": plan["execution_mode"], "claim_scope": plan["claim_scope"],
                   "formal_launch_authorized": plan["formal_launch_authorized"],
                   "candidate_only": plan["candidate_only"], "test_injected": injected}
        write_json(root / "summary.json", summary)
        return summary
