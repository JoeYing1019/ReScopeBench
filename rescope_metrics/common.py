"""Small I/O helpers; no dependency on author-side runs or provider settings."""

import hashlib
import json
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).with_name("data")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_once(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, "Frozen input changed: " + path.name)
    else:
        write(path, value)


def load_gold(case_id, data=DATA):
    manifest = read(Path(data) / "manifest.json")
    require(case_id in manifest["cases"], "Unknown environment")
    row = manifest["cases"][case_id]
    gold = read(Path(data) / (case_id + ".json"))
    require(digest(gold) == row["sha256"], "Golden/rubric package differs from manifest")
    return gold


def task_signature(tasks):
    keys = ("position", "event_id", "query", "business_budget", "completion")
    return digest([{k: t.get(k) for k in keys} for t in tasks])


def epoch_at(gold, position):
    matches = [e for e in gold["epochs"] if e["start"] <= position <= e["end"]]
    require(len(matches) == 1, "Position has no unique golden epoch")
    return matches[0]
