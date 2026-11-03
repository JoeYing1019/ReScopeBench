"""Read committed paired runs without running tools, learners, or containers."""

from pathlib import Path, PurePosixPath

from .common import digest, load_gold, read, require, task_signature
from .scores import numbers

SYSTEM_ASSETS = {"skills/.usage.json", "skills/.curator_ledger.jsonl"}


def documents(assets):
    content = {}
    for name, item in sorted(assets.items()):
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe saved asset name")
        if name in SYSTEM_ASSETS:
            continue
        require(path.parts[0] in ("skills", "memories"), "Unknown experience asset scope")
        require(isinstance(item, dict) and isinstance(item.get("content"), str), "Invalid saved asset")
        content[name] = item["content"]
    require(sum(len(s.encode()) for s in content.values()) < 2 * 1024 ** 2,
            "Large saved corpus needs explicit non-truncating handling")
    return [{"id": f"d{i}", "lines": text.splitlines()} for i, text in enumerate(content.values(), 1)]


def from_run(path):
    root = Path(path).resolve()
    plan = read(root / "run-plan.json")
    cid = plan["case_id"]
    gold = load_gold(cid)
    require(plan["case_version"] == gold["case_version"], "Run uses another case version")
    require(task_signature(plan["tasks"]) == gold["task_signature"], "Run uses different tasks or budgets")
    drifts = [e["before_task"] for e in plan["lifecycle_schedule"]["events"]]
    require(drifts == gold["drifts"], "Run uses a different drift schedule")
    plan_hash = digest(plan)
    arms, corpora, receipts = {}, {}, []
    for arm in ("native", "reset"):
        summary = read(root / arm / "summary.json")
        require(summary["complete"] is True and summary["position"] == gold["task_count"], "Both streams must be complete")
        arms[arm] = []
        for task in plan["tasks"]:
            position, event = task["position"], task["event_id"]
            require("/" not in event and "\\" not in event and event not in (".", ".."), "Invalid event ID")
            task_root = root / arm / f"{position:04d}-{event}"
            candidates = list(task_root.glob("attempt-*/promotion.json"))
            require(len(candidates) == 1, "Expected exactly one committed semantic result per task")
            promotion = read(candidates[0])
            result = read(candidates[0].with_name("result.json"))
            require(promotion["position"] == position and promotion["event_id"] == event
                    and promotion["plan_sha256"] == plan_hash and promotion["reset_control"] is (arm == "reset"),
                    "Promotion belongs to another task, plan or setting")
            require(promotion.get("test_injected") is False and result.get("test_injected") is False,
                    "Injected development results are not benchmark scores")
            require(result["status"] == "semantic_result" and promotion["result"] == result
                    and promotion["result_sha256"] == digest(result), "Result not bound to committed promotion")
            require(promotion["assets_sha256"] == digest(promotion["assets"]), "Saved experience changed")
            checkpoint = root / arm / "checkpoints" / f"{position:04d}"
            manifest = read(checkpoint / "manifest.json")
            require(manifest["plan_sha256"] == plan_hash and manifest["promotion_sha256"] == digest(promotion),
                    "Promotion is not committed in checkpoint")
            value = numbers([result["score"]])[0]
            arms[arm].append(value)
            if arm == "native":
                require(result["assets"] == promotion["assets"], "Experience differs between result and promotion")
                corpora[position] = documents(promotion["assets"])
            receipts.append({"arm": arm, "position": position, "promotion_sha256": digest(promotion),
                             "result_sha256": digest(result), "checkpoint_sha256": digest(manifest)})
    pair = {"schema": "rescope-metric-pair/v1", "model": plan["model"], "case_id": cid,
            "case_version": gold["case_version"], "task_signature": gold["task_signature"], "drifts": drifts,
            "plan_sha256": plan_hash, "receipts": receipts,
            "positions": [{"position": i + 1, "w_exp": a, "wo_exp": b}
                          for i, (a, b) in enumerate(zip(arms["native"], arms["reset"]))]}
    return pair, corpora
