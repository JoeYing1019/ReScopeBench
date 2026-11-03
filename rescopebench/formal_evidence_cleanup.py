"""Safely remove redundant evidence from promoted formal-stream attempts.

The promotion journal and checkpoint are the durable state transition.  Native
execution also leaves several byte-for-byte or semantically redundant copies of
the same trajectory, carriers, and world state.  This module removes only those
copies after validating the promotion/checkpoint chain.  It intentionally keeps
the scored result (including the full trajectory), private grade, exact model
input, compact runtime receipts, promotion journal, and every checkpoint.
Provider/worker records are archived losslessly first: model-input.json alone
is a pre-assembly packet, not evidence of the actual provider conversation.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable
import uuid

from rescopebench import evidence_archive, metrics, stream_engine


SCHEMA = "rescopebench-promoted-evidence-cleanup/v1"
SUMMARY_SCHEMA = "rescopebench-formal-evidence-cleanup-summary/v1"
RECEIPT_NAME = "evidence-cleanup-v1.json"

# These are redundant after a semantic promotion has been committed.  Unknown
# paths are retained by default so a future runtime cannot silently lose a new
# evidence class merely because this cleanup code predates it.
ATTEMPT_FILES = (
    "entry-snapshot.json",
    "assets-before.json",
    "history-before.json",
)
PROVIDER_PATHS = (
    "provider/wire",
    "provider/requests",
    "provider/responses",
)
EPISODE_PATHS = (
    "episode/hermes-wire",
    "episode/provider-admitted",
    "episode/provider-returned",
    "episode/exchange",
    "episode/events",
    "episode/history-events",
    "episode/before-state-private.json",
    "episode/episode-result-private.json",
    "episode/sources-before.json",
    "episode/sources-after.json",
    "episode/assets.json",
    "episode/grade-private.json",
    "episode/feedback.json",
)


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_record(path: Path, attempt: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"refusing to clean symlinked evidence: {path}")
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    tree = hashlib.sha256()
    total = 0
    for item in files:
        if item.is_symlink():
            raise ValueError(f"refusing to clean tree containing a symlink: {item}")
        relative = item.relative_to(attempt).as_posix()
        size = item.stat().st_size
        checksum = _file_sha256(item)
        tree.update(relative.encode("utf-8") + b"\0")
        tree.update(str(size).encode("ascii") + b"\0")
        tree.update(checksum.encode("ascii") + b"\n")
        total += size
    return {
        "relative_path": path.relative_to(attempt).as_posix(),
        "kind": "file" if path.is_file() else "directory",
        "file_count": len(files),
        "logical_bytes": total,
        "tree_sha256": tree.hexdigest(),
    }


def _safe_candidate(attempt: Path, relative: str) -> Path | None:
    path = attempt / relative
    if not path.exists() and not path.is_symlink():
        return None
    if attempt not in path.resolve().parents:
        raise ValueError(f"cleanup candidate escapes its attempt: {path}")
    return path


def _validate_attempt(run_root: Path, arm: str, attempt: Path) -> tuple[dict[str, Any], Path]:
    if arm not in stream_engine.ARMS:
        raise ValueError(f"unknown arm: {arm}")
    arm_root = run_root / arm
    attempt = attempt.resolve()
    if arm_root.resolve() not in attempt.parents or not attempt.name.startswith("attempt-"):
        raise ValueError(f"attempt is outside the selected arm: {attempt}")

    promotion_path = attempt / "promotion.json"
    result_path = attempt / "result.json"
    grade_path = attempt / "grade-private.json"
    identity_path = attempt / "attempt.json"
    required = (promotion_path, result_path, grade_path, identity_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"promoted attempt is missing durable evidence: {missing}")

    promotion = _read(promotion_path)
    result = _read(result_path)
    task_count = len(_read(run_root / "run-plan.json")["tasks"])
    position = promotion.get("position")
    if (promotion.get("schema") != stream_engine.SCHEMA + "/promotion"
            or type(position) is not int or not 1 <= position <= task_count
            or promotion.get("reset_control") is not (arm == "reset")
            or promotion.get("result", {}).get("status") != "semantic_result"):
        raise ValueError(f"invalid promotion identity: {promotion_path}")
    if result != promotion["result"] or promotion.get("result_sha256") != stream_engine.digest(result):
        raise ValueError(f"result differs from its promotion: {promotion_path}")
    for key in ("snapshot", "assets", "history"):
        if promotion.get(f"{key}_sha256") != stream_engine.digest(promotion.get(key)):
            raise ValueError(f"promotion {key} integrity failed: {promotion_path}")

    checkpoint = arm_root / "checkpoints" / f"{position:04d}"
    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"promotion has no committed checkpoint: {promotion_path}")
    manifest = _read(manifest_path)
    if manifest.get("promotion_sha256") != stream_engine.digest(promotion):
        raise ValueError(f"checkpoint is not bound to promotion: {promotion_path}")
    state, assets, history, _ = stream_engine._load_checkpoint(checkpoint, manifest)
    expected_assets = {} if arm == "reset" else promotion["assets"]
    expected_history = [] if arm == "reset" else promotion["history"]
    if state != promotion["snapshot"] or assets != expected_assets or history != expected_history:
        raise ValueError(f"checkpoint differs from promoted terminal evidence: {promotion_path}")
    return promotion, checkpoint


def _retained_records(attempt: Path, checkpoint: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "attempt_identity": attempt / "attempt.json",
        "promotion": attempt / "promotion.json",
        "result_with_trajectory": attempt / "result.json",
        "private_grade": attempt / "grade-private.json",
        "checkpoint_manifest": checkpoint / "manifest.json",
        "checkpoint_snapshot": checkpoint / "snapshot.json.gz",
        "checkpoint_assets": checkpoint / "assets.json",
        "checkpoint_history": checkpoint / "history.json",
    }
    for name in (
        "boundary-private.json",
        "episode/model-input.json",
        "episode/trace.json",
        "episode/container-receipt.json",
        "episode/runtime-contract.json",
        "episode/bridge-integrity.json",
        "episode/started.json",
        "episode/worker.log",
        "episode/public/packet.json",
        "episode/postprocessing-amendment.json",
        "episode/failure.json",
        evidence_archive.ARCHIVE,
    ):
        path = attempt / name
        if path.is_file():
            paths[name] = path
    return {
        name: {
            "relative_path": path.relative_to(attempt).as_posix()
            if attempt in path.parents else str(path),
            "logical_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for name, path in paths.items()
        if path.is_file()
    }


def cleanup_promoted_attempt(
    run_root: Path,
    arm: str,
    attempt: Path,
    *,
    apply: bool,
) -> dict[str, Any]:
    """Validate one promotion and remove only its known redundant copies."""

    run_root = Path(run_root).expanduser().resolve()
    attempt = Path(attempt).expanduser().resolve()
    receipt_path = attempt / RECEIPT_NAME
    promotion, checkpoint = _validate_attempt(run_root, arm, attempt)
    removable = [*ATTEMPT_FILES, *PROVIDER_PATHS, *EPISODE_PATHS]
    if promotion["result"].get("postprocessing_amendment") is None:
        removable.append("episode/final-state-private.json")

    if receipt_path.is_file():
        receipt = _read(receipt_path)
        if (receipt.get("schema") != SCHEMA
                or receipt.get("promotion_sha256") != stream_engine.digest(promotion)
                or receipt.get("status") != "removed"):
            raise ValueError(f"invalid existing cleanup receipt: {receipt_path}")
        reappeared = [relative for relative in removable if (attempt / relative).exists()]
        if reappeared:
            raise ValueError(f"cleaned evidence paths reappeared: {reappeared}")
        return receipt

    candidates = [path for relative in removable if (path := _safe_candidate(attempt, relative)) is not None]
    records = [_tree_record(path, attempt) for path in candidates]
    archived = evidence_archive.archive_attempt(attempt) if apply else {"status": "would_archive_before_removal"}
    retained = _retained_records(attempt, checkpoint)
    receipt = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if not apply else "removed",
        "run_root": str(run_root),
        "arm": arm,
        "position": promotion["position"],
        "event_id": promotion["event_id"],
        "promotion_sha256": stream_engine.digest(promotion),
        "checkpoint_manifest_sha256": _file_sha256(checkpoint / "manifest.json"),
        "removed_logical_bytes": sum(row["logical_bytes"] for row in records),
        "removed_file_count": sum(row["file_count"] for row in records),
        "removed": records,
        "retained": retained,
        "regrade_inputs_retained": True,
        "full_trajectory_retained_in_result": isinstance(promotion["result"].get("trajectory"), list),
        "original_environment_preserved": True,
        "exact_model_evidence": archived,
    }
    if not apply:
        return receipt

    for path in candidates:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    _write_atomic(receipt_path, receipt)
    return receipt


def _promotion_attempts(run_root: Path, arm: str) -> Iterable[Path]:
    yield from sorted((run_root / arm).glob("[0-9][0-9][0-9][0-9]-*/attempt-*/promotion.json"))


def cleanup_run(run_root: Path, *, apply: bool, require_metrics: bool = True) -> dict[str, Any]:
    """Clean promoted attempts in one case run while both arms are idle."""

    run_root = Path(run_root).expanduser().resolve()
    plan_path = run_root / "run-plan.json"
    if not plan_path.is_file():
        raise ValueError(f"not a formal run root: {run_root}")
    metric_path = run_root / "metrics.json"
    if require_metrics:
        if not metric_path.is_file():
            raise ValueError("automatic evidence cleanup requires a completed metrics.json")
        stored_metrics = _read(metric_path)
        recomputed = metrics.analyze_run(run_root)
        if stored_metrics != recomputed:
            raise ValueError("stored metrics differ from a fresh pre-cleanup regrade")

    with ExitStack() as stack:
        for arm in stream_engine.ARMS:
            stack.enter_context(stream_engine._arm_lock(run_root / arm))
        rows = []
        for arm in stream_engine.ARMS:
            for promotion_path in _promotion_attempts(run_root, arm):
                rows.append(cleanup_promoted_attempt(
                    run_root, arm, promotion_path.parent, apply=apply,
                ))

    summary = {
        "schema": SUMMARY_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if not apply else "removed",
        "run_root": str(run_root),
        "metrics_required": require_metrics,
        "promoted_attempts": len(rows),
        "removed_file_count": sum(row.get("removed_file_count", 0) for row in rows),
        "removed_logical_bytes": sum(row.get("removed_logical_bytes", 0) for row in rows),
        "attempts": [{key: row[key] for key in (
            "arm", "position", "event_id", "status", "removed_file_count", "removed_logical_bytes"
        )} for row in rows],
    }
    if apply:
        _write_atomic(run_root / "maintenance/evidence-cleanup-v1.json", summary)
    return summary


def cleanup_tree(root: Path, *, apply: bool, require_metrics: bool = True) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    runs = sorted(path.parent for path in root.rglob("run-plan.json"))
    rows, retained = [], []
    for run in runs:
        if require_metrics and not (run / "metrics.json").is_file():
            continue
        try:
            rows.append(cleanup_run(run, apply=apply, require_metrics=require_metrics))
        except (OSError, ValueError, RuntimeError) as error:
            retained.append({
                "run_root": str(run),
                "error_type": type(error).__name__,
                "reason": str(error)[:1000],
            })
    return {
        "schema": SUMMARY_SCHEMA + "/tree",
        "status": "partial" if retained else ("dry_run" if not apply else "removed"),
        "root": str(root),
        "run_count": len(rows),
        "retained_run_count": len(retained),
        "promoted_attempts": sum(row["promoted_attempts"] for row in rows),
        "removed_file_count": sum(row["removed_file_count"] for row in rows),
        "removed_logical_bytes": sum(row["removed_logical_bytes"] for row in rows),
        "runs": rows,
        "retained_runs": retained,
    }


def finalize_completed_run(run_root: Path) -> dict[str, Any]:
    """Score and clean a dual-arm run once its second arm completes.

    Every launcher may call this function.  A run-local lock makes the operation
    idempotent when both arms or a batch controller notice completion together.
    """

    run_root = Path(run_root).expanduser().resolve()
    maintenance = run_root / "maintenance"
    maintenance.mkdir(parents=True, exist_ok=True)
    with (maintenance / "finalize.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        summaries = []
        for arm in stream_engine.ARMS:
            path = run_root / arm / "summary.json"
            if not path.is_file():
                return {"status": "waiting_for_other_arm", "arm": arm}
            summary = _read(path)
            if summary.get("complete") is not True:
                return {"status": "waiting_for_other_arm", "arm": arm}
            summaries.append(summary)

        metric_path = run_root / "metrics.json"
        try:
            task_count = len(_read(run_root / "run-plan.json")["tasks"])
            if any(summary.get("position") != task_count for summary in summaries):
                raise ValueError("completed arm position differs from the frozen task count")
            result = metrics.analyze_run(run_root)
            if metric_path.is_file() and _read(metric_path) != result:
                raise ValueError("existing metrics differ from a fresh completed-run regrade")
            if not metric_path.is_file():
                _write_atomic(metric_path, result)
        except Exception as error:
            # Aggregation cannot undo committed episodes or justify a model
            # rerun. Keep every input for an offline repair; do not clean or
            # publish metrics when their validation has not succeeded.
            failure = {
                "schema": SUMMARY_SCHEMA + "/failure",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "completed_metrics_pending",
                "error_type": type(error).__name__,
                "error": str(error)[:1800],
                "metrics_validated": False,
                "evidence_retained": True,
                "model_rerun_required": False,
            }
            try:
                _write_atomic(maintenance / "finalization-pending.json", failure)
            except OSError as receipt_error:
                failure["receipt_write_error"] = str(receipt_error)[:800]
            return failure
        pending = maintenance / "finalization-pending.json"
        if pending.is_file():
            previous_failure = _read(pending)
            if previous_failure.get("status") == "completed_metrics_pending":
                _write_atomic(pending, {
                    **previous_failure, "status": "resolved_offline",
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                    "metrics_validated": True,
                })
        try:
            cleanup = cleanup_run(run_root, apply=True, require_metrics=True)
        except Exception as error:
            failure = {
                "schema": SUMMARY_SCHEMA + "/failure",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "scored_cleanup_failed",
                "error_type": type(error).__name__,
                "error": str(error)[:1800],
                "metrics": str(metric_path),
                "evidence_retained": True,
            }
            _write_atomic(maintenance / "evidence-cleanup-failure.json", failure)
            return failure
        return {
            "status": "scored_and_cleaned",
            "metrics": str(metric_path),
            "promoted_attempts": cleanup["promoted_attempts"],
            "removed_logical_bytes": cleanup["removed_logical_bytes"],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Clean individually committed promotions without requiring final metrics.",
    )
    args = parser.parse_args()
    if (args.run is None) == (args.tree is None):
        parser.error("provide exactly one of --run or --tree")
    function = cleanup_run if args.run is not None else cleanup_tree
    target = args.run if args.run is not None else args.tree
    result = function(target, apply=args.apply, require_metrics=not args.allow_incomplete)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
