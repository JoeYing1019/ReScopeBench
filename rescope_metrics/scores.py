"""Deterministic B/W/D/S/R/U/H, with experience-first aggregation."""

from collections import defaultdict
import math
from statistics import fmean

from .common import digest, epoch_at, require

METRICS = ("B", "W", "D", "S", "R", "U", "H")


def numbers(values, binary=False):
    require(bool(values), "An empty set is not a zero score")
    if binary:
        require(all(type(v) is int and v in (0, 1) for v in values), "Claims must be binary integers")
    else:
        require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 100 for v in values),
                "Scores must be finite in [0,100]; missing is not zero")
    return values


def average(values):
    require(set(values) == set(METRICS), "Expected exactly B/W/D/S/R/U/H")
    present = [v for v in values.values() if v is not None]
    if present:
        numbers(present)
    return {**values, "AVG": fmean(values.values()) if len(present) == len(METRICS) else None}


def performance(w_exp, wo_exp, drifts):
    numbers(w_exp)
    numbers(wo_exp)
    require(len(w_exp) == len(wo_exp), "Paired streams have different lengths")
    require(drifts and drifts == sorted(set(drifts)) and
            all(type(d) is int and 2 <= d <= len(w_exp) for d in drifts), "Invalid drift positions")
    return {"B": fmean(wo_exp), "W": fmean(w_exp), "D": fmean(w_exp[d - 1] for d in drifts)}


def aggregate(rows):
    require(bool(rows), "No environments to aggregate")
    return average({k: fmean(r[k] for r in rows) if all(r[k] is not None for r in rows) else None
                    for k in METRICS})


def position_score(epoch, grades):
    targets = [r["claim"] for u in epoch["units"] for r in u["targets"]]
    ids = [c["id"] for c in targets]
    require(len(ids) == len(set(ids)) and set(grades) == set(ids), "Missing, extra or duplicate claims")
    numbers(list(grades.values()), binary=True)
    units = {u["id"]: 100 * fmean(grades[r["claim"]["id"]] for r in u["targets"]) for u in epoch["units"]}
    require(len(units) == len(epoch["units"]), "Duplicate experience identity")
    return {"units": units, **{typ: fmean(units[u["id"]] for u in epoch["units"] if u["type"] == typ)
                               for typ in ("S", "R")}}


def lifecycle(gold, credits):
    events = []
    for event in gold["transitions"]:
        start, end = event["start"], event["end"]
        baseline = credits[event["baseline_position"]]
        units = []
        for unit in event["units"]:
            changed = unit["update_claim_ids"]
            if not changed:
                continue
            retained = [link["new_id"] for link in unit["preserved_links"] if baseline[link["old_id"]] == 1]
            targets = changed + retained
            require(len(targets) == len(set(targets)), "Repeated recovery targets")
            first = next((p for p in range(start, end + 1) if all(credits[p][c] == 1 for c in targets)), None)
            units.append({"id": unit["id"], "type": unit["type"],
                          "U": 100 * sum(credits[start][c] for c in changed) / len(changed),
                          "H": 100 * (end - first + 1) / (end - start + 1) if first is not None else 0.0,
                          "update_claim_ids": changed, "retained_claim_ids": retained,
                          "first_joint_task": first, "never_jointly_complete": first is None})
        events.append({"start": start, "end": end, "units": units,
                       **{k: fmean(u[k] for u in units) if units else None for k in ("U", "H")}})
    return {"events": events, **{k: fmean(e[k] for e in events if e[k] is not None)
                                if any(e[k] is not None for e in events) else None for k in ("U", "H")}}


def score_pair(pair, gold, credits=None):
    require(pair["case_id"] == gold["case_id"] and pair["case_version"] == gold["case_version"],
            "Business run and experience inventory use different environment versions")
    require(pair["task_signature"] == gold["task_signature"], "Task requests/budgets differ from evaluated version")
    n = gold["task_count"]
    require([p["position"] for p in pair["positions"]] == list(range(1, n + 1)), "Incomplete or reordered task stream")
    require(pair["drifts"] == gold["drifts"], "Drift schedule mismatch")
    rows = [{"position": r["position"], "w_exp": r["w_exp"], "wo_exp": r["wo_exp"], "S": None, "R": None}
            for r in pair["positions"]]
    values = performance([r["w_exp"] for r in rows], [r["wo_exp"] for r in rows], gold["drifts"])
    values.update({k: None for k in ("S", "R", "U", "H")})
    events, experience_positions = [], []
    if credits is not None:
        normalized = {int(p): v for p, v in credits.items()}
        require(len(normalized) == len(credits) and set(normalized) == set(range(1, n + 1)), "Incomplete claim snapshots")
        for row in rows:
            scored = position_score(epoch_at(gold, row["position"]), normalized[row["position"]])
            row.update({k: scored[k] for k in ("S", "R")})
            experience_positions.append({"position": row["position"], "units": scored["units"]})
        values.update({k: fmean(r[k] for r in rows) for k in ("S", "R")})
        measured = lifecycle(gold, normalized)
        values.update({k: measured[k] for k in ("U", "H")})
        events = measured["events"]
    return {"schema": "rescope-seven-metrics/v1", "model": pair["model"], "case_id": gold["case_id"],
            "case_version": gold["case_version"], "task_count": n, "drifts": gold["drifts"],
            "gold_sha256": digest(gold), "pair_sha256": digest(pair),
            "claim_scores_sha256": digest(credits) if credits is not None else None,
            "complete_experience": credits is not None, "metrics": average(values), "positions": rows,
            "experience_positions": experience_positions, "events": events}


def cohort(pairs):
    by_model = defaultdict(dict)
    versions = {}
    for pair in pairs:
        model, cid = pair["model"], pair["case_id"]
        require(cid not in by_model[model], "Duplicate model/environment result")
        identity = (pair["case_version"], pair["gold_sha256"])
        require(cid not in versions or versions[cid] == identity, "Cannot compare different environment/golden versions")
        versions[cid] = identity
        by_model[model][cid] = pair
    require(bool(by_model), "No results")
    expected = set(next(iter(by_model.values())))
    require(all(set(rows) == expected for rows in by_model.values()), "Models need the same environment set; no silent subset pooling")
    return {"schema": "rescope-seven-metrics-cohort/v1", "environments": sorted(expected),
            "models": [{"model": m, "metrics": aggregate([rows[c]["metrics"] for c in sorted(expected)])}
                       for m, rows in sorted(by_model.items())], "pairs": pairs}
