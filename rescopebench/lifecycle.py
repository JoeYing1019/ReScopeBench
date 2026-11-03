"""Approved per-environment schedules, not a factorial experiment matrix."""

from copy import deepcopy


SCHEMA = "rescopebench-lifecycle-schedule/v1"
_LAYOUT = {
    "api_reliability": (16, (9,)),
    "backup_recovery": (16, (5,)),
    "carbon_credits": (32, (9, 17, 25)),
    "cinema_c_20_04_03": (24, (9, 17)),
    "cloud_data_warehouse": (24, (9, 17)),
    "course_registration": (16, (9,)),
    "emergency_response": (16, (9,)),
    "event_ticketing": (24, (9, 17)),
    "filesystem_c_14_01_54": (16, (7,)),
    "k8s_operations": (24, (9, 17)),
    "medical_appointment": (16, (11,)),
    "mes_order_closeout": (16, (7,)),
    "oracle_clinical_one_data_collection": (16, (11,)),
    "railway_ticketing_c_04_06_01": (32, (9, 17, 25)),
    "redcap": (16, (9,)),
    "robot": (16, (9,)),
    "sap_s4hana": (16, (9,)),
    "supply_chain_finance_c_05_10_53": (16, (13,)),
}


def approved_schedule(case_id):
    count, positions = _LAYOUT[case_id]
    return {"schema": SCHEMA, "task_count": count, "events": [
        {"event_id": f"E{index}", "before_task": position}
        for index, position in enumerate(positions, 1)
    ]}


def portfolio():
    return {case: approved_schedule(case) for case in _LAYOUT}


def validate_schedule(schedule, task_count):
    if not isinstance(schedule, dict) or schedule.get("schema") != SCHEMA:
        raise ValueError("unsupported lifecycle schedule")
    if type(task_count) is not int or task_count not in (16, 24, 32):
        raise ValueError("lifecycle stream length must be 16, 24, or 32")
    if schedule.get("task_count") != task_count:
        raise ValueError("schedule and task count differ")
    events = schedule.get("events")
    if not isinstance(events, list) or len(events) != task_count // 8 - 1:
        raise ValueError("event count differs from the declared stream length")
    positions = [event.get("before_task") for event in events]
    ids = [event.get("event_id") for event in events]
    if (any(type(p) is not int or not 1 < p < task_count for p in positions)
            or sorted(set(positions)) != positions
            or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids)):
        raise ValueError("invalid event identities or positions")
    if task_count > 16 and positions != list(range(9, task_count, 8)):
        raise ValueError("multi-event streams use fixed eight-task intervals")
    return deepcopy(schedule)


def event_window(schedule, event_id):
    """Disjoint event-relative windows; later events censor the previous event."""
    validate_schedule(schedule, schedule["task_count"])
    events = schedule["events"]
    index = next(i for i, event in enumerate(events) if event["event_id"] == event_id)
    onset = events[index]["before_task"]
    previous = events[index - 1]["before_task"] if index else 1
    end = events[index + 1]["before_task"] - 1 if index + 1 < len(events) else schedule["task_count"]
    return {"pre": list(range(max(previous, onset - 4), onset)),
            "onset": [onset], "post_four": list(range(onset, min(onset + 3, end) + 1)),
            "post": list(range(onset, end + 1)), "censored_at": end,
            "censor_reason": "next_event" if index + 1 < len(events) else "stream_end"}
