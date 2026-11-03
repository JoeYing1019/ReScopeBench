"""Read planned delivery denominators from frozen public tasks, not their scores."""

from __future__ import annotations

import hashlib
import json
import re


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def public_targets(task, case_id):
    query = task["query"]
    if query.lstrip().startswith("{"):
        packet = json.loads(query)
        if packet.get("event_id") != task["event_id"]:
            raise ValueError("public packet event differs from task")
        targets, method = packet["targets"], "public_json_packet.targets"
    elif "Requested artifacts:" in query:
        if query.count("Requested artifacts:") != 1:
            raise ValueError("ambiguous public target block")
        targets, _ = json.JSONDecoder().raw_decode(query.split("Requested artifacts:", 1)[1].lstrip())
        method = "public_requested_artifacts_json"
    elif case_id == "robot":
        section = query.split("queued work orders:\n", 1)[1].split("\n\n", 1)[0]
        lines = section.splitlines()
        if not lines or not all(re.fullmatch(r"- WO-[A-Za-z0-9-]+", line) for line in lines):
            raise ValueError("unrecognized public work-order list")
        targets = [line[2:] for line in lines]
        if targets != task["work_orders"]:
            raise ValueError("public work-order list differs from declared task metadata")
        method = "public_work_order_list_matched_to_metadata"
    elif case_id == "filesystem_c_14_01_54":
        headers = ("Active requests in priority order:\n", "Independent active requests:\n")
        if sum(query.count(header) for header in headers) != 1:
            raise ValueError("missing or ambiguous public manifest-request block")
        header = next(header for header in headers if header in query)
        section = query.split(header, 1)[1].split("\n\n", 1)[0]
        matches = [re.fullmatch(r"(\d+)\. ([^\s:]+): workspace=.+; manifest=.+", line)
                   for line in section.splitlines()]
        if not matches or not all(matches) or [int(m[1]) for m in matches] != list(range(1, len(matches) + 1)):
            raise ValueError("unrecognized public manifest-request list")
        targets = [m[2] for m in matches]
        if targets != task["target_ids"]:
            raise ValueError("public manifest-request list differs from declared task metadata")
        method = "public_manifest_request_list_matched_to_metadata"
    else:
        raise ValueError(f"no reviewed public denominator parser for {case_id}")
    if not isinstance(targets, list) or not targets or any(
        not isinstance(t, (dict, str)) or not t for t in targets
    ):
        raise ValueError("public targets must be a nonempty list of distinct records or IDs")
    signatures = [_digest(t) for t in targets]
    if len(signatures) != len(set(signatures)):
        raise ValueError("duplicate public target")
    return targets, method


def planned_schedule(plan):
    tasks = plan["tasks"]
    if [t["position"] for t in tasks] != list(range(1, len(tasks) + 1)):
        raise ValueError("planned positions must be unique, ordered and contiguous")
    rows = []
    for task in tasks:
        targets, method = public_targets(task, plan["case_id"])
        rows.append({"position": task["position"], "event_id": task["event_id"],
                     "planned_artifacts": len(targets), "method": method,
                     "public_query_sha256": hashlib.sha256(task["query"].encode()).hexdigest(),
                     "public_target_sha256": [_digest(t) for t in targets]})
    return {"schema": "rescopebench-public-artifact-denominator/v1", "positions": rows,
            "scope": "Publicly requested business artifacts, not rule counts, substeps, or score divisions."}


def delivery_rates(completed, planned, windows):
    if len(completed) != len(planned) or not planned or any(type(n) is not int or n < 1 for n in planned):
        raise ValueError("invalid planned artifact counts")
    for count, total in zip(completed, planned):
        if count is not None and (type(count) is not int or not 0 <= count <= total):
            raise ValueError("completed artifacts must be null or an integer within the planned denominator")
    rates = {}
    for name, positions in windows.items():
        if list(positions) != sorted(set(positions)) or any(not 1 <= p <= len(planned) for p in positions):
            raise ValueError("invalid artifact window positions")
        total = sum(planned[p - 1] for p in positions)
        observed_total = sum(planned[p - 1] for p in positions if completed[p - 1] is not None)
        success = sum(completed[p - 1] for p in positions if completed[p - 1] is not None)
        rates[name] = {"planned_artifacts": total, "observed_artifact_opportunities": observed_total,
                       "completed_artifacts_observed": success,
                       "rate": success / total if total and observed_total == total else None,
                       "bounds": [success / total, (success + total - observed_total) / total] if total else None}
    return {"unit": "fraction", "planned_counts": planned, "completed_counts": completed,
            "per_position": [count / total if count is not None else None for count, total in zip(completed, planned)],
            "windows": rates}
