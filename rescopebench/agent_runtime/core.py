"""Deterministic factual views, exact source records, and transactional assets.

No evaluator, model, filesystem, executable learner code, or semantic summarizer
belongs in this module. The caller admits completed own episodes and public data.
"""

from copy import deepcopy
import difflib
import hashlib
import json
import math
from pathlib import PurePosixPath

import jsonschema


VERSION = "consolidation-scaffold/v1"
MODES = ("none", "score", "components")
RETRIEVAL = {"history_search", "history_read", "evidence_search", "evidence_read",
             "episode_list", "episode_read", "record_search", "record_read"}
REVIEW_PREFIX = "Review the conversation above"


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def asset_path(path):
    if not isinstance(path, str) or not path or "\\" in path:
        raise ValueError("Invalid asset path")
    parts = PurePosixPath(path).parts
    if (PurePosixPath(path).is_absolute() or PurePosixPath(path).as_posix() != path
            or len(parts) < 2 or parts[0] not in {"skills", "memories"}
            or any(part.startswith(".") for part in parts)):
        raise ValueError("Use a relative, non-hidden path under skills/ or memories/")
    return path


def public_assets(assets):
    result = {}
    for path, item in assets.items():
        # Harness bookkeeping is not learner-authored experience.
        if any(part.startswith(".") for part in PurePosixPath(path).parts):
            continue
        asset_path(path)
        text = item["content"]
        if not isinstance(text, str) or text_hash(text) != item["sha256"]:
            raise ValueError("Asset content/hash mismatch")
        result[path] = dict(content=text, sha256=text_hash(text))
    return result


def outcome(supplied, mode):
    allowed = {"score", "components"} if mode == "components" else {"score"}
    if set(supplied) - allowed or "score" not in supplied:
        raise ValueError("Unexpected outcome fields")
    score = supplied["score"]
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("Expected a finite score from 0 to 100")
    if mode == "components":
        counts = supplied.get("components")
        if not isinstance(counts, dict) or not counts:
            raise ValueError("Expected aggregate component counts")
        for field, count in counts.items():
            if (not isinstance(field, str) or not isinstance(count, dict) or set(count) != {"correct", "total"}
                    or any(type(count[key]) is not int for key in count)
                    or not 0 <= count["correct"] <= count["total"] or count["total"] < 1):
                raise ValueError("Only aggregate correct/total counts are allowed")
    return deepcopy(supplied)


def build(episodes, documents, assets, feedback_mode="none", feedback=None, delivery_tools=()):
    """Only copy facts; do not infer important decisions, errors, or business rules.

    A no-feedback bundle cannot accept historical score-bearing review prompts
    or retrieval echoes. The adapter must supply feedback-free source episodes;
    quietly deleting scalar fields cannot undo feedback already in a transcript.
    """
    if feedback_mode not in MODES:
        raise ValueError("Unknown feedback mode")
    feedback = feedback or {}
    if feedback_mode == "none" and feedback:
        raise ValueError("No-feedback mode cannot accept evaluation receipts")
    records, rows, ids = {}, [], set()

    def record(text, kind, origin):
        if not isinstance(text, str):
            text = encoded(text)
        key = "r-" + text_hash(text)
        item = records.setdefault(key, dict(text=text, sha256=text_hash(text), characters=len(text), origins=[]))
        source = dict(kind=kind, **origin)
        if source not in item["origins"]:
            item["origins"].append(source)
        return key

    for episode in episodes:
        eid = episode["episode_id"]
        if eid in ids:
            raise ValueError("Duplicate episode ID")
        ids.add(eid)
        row = dict(episode_id=eid, task=episode["task"], events=[], skill_changes=[], deliveries=[])
        if feedback_mode != "none":
            supplied = feedback.get(eid)
            if supplied is None:
                raise ValueError("Missing declared outcome receipt")
            row["feedback"] = outcome(supplied, feedback_mode)
        for section in ("trajectory", "review"):
            names = {}
            for index, message in enumerate(episode.get(section) or []):
                role = message.get("role")
                if role not in {"user", "assistant", "tool"}:
                    continue
                calls = message.get("tool_calls") or []
                for call in calls:
                    names[call.get("id")] = call["function"]["name"]
                name = message.get("name") or names.get(message.get("tool_call_id"))
                text = message.get("content")
                origin = dict(episode_id=eid, section=section, message_index=index)
                if feedback_mode == "none" and section == "review" and (
                        role == "user" or name in RETRIEVAL):
                    raise ValueError("Admit feedback-free episodes, not a sanitized feedback-bearing review")
                event = dict(section=section, message_index=index, role=role)
                if name:
                    event["tool"] = name
                if role == "tool" and message.get("tool_call_id"):
                    event["call_id"] = message["tool_call_id"]
                if text:
                    kind = ("retrieval_echo" if role == "tool" and name in RETRIEVAL else
                            "review_instruction" if role == "user" and text.startswith(REVIEW_PREFIX) else
                            "tool_observation" if role == "tool" else "own_text")
                    event["record_id"] = record(text, kind, origin)
                    event["kind"] = kind
                if calls:
                    event["calls"] = []
                    for call in calls:
                        fn = call["function"]
                        call_ref = dict(name=fn["name"], call_id=call.get("id"),
                            arguments_record=record(fn["arguments"], "own_call", origin))
                        event["calls"].append(call_ref)
                        if fn["name"] in delivery_tools:
                            row["deliveries"].append(dict(section=section, message_index=index, **call_ref))
                # Exact echoes remain addressable, but are not injected into the view.
                row["events"].append(event)
        before, after = episode.get("skills_before") or {}, episode.get("skills_after")
        row["review_completed"] = after is not None
        if after is not None:
            for path in sorted(set(before) | set(after)):
                if before.get(path) == after.get(path):
                    continue
                change = dict(path=path, action="create" if path not in before else "delete" if path not in after else "modify")
                for version, data in (("before", before), ("after", after)):
                    if path in data:
                        change[version] = record(data[path], "own_skill", dict(episode_id=eid, version=version, path=path))
                diff = "".join(difflib.unified_diff(before.get(path, "").splitlines(True),
                    after.get(path, "").splitlines(True), fromfile="before/" + path, tofile="after/" + path))
                change["diff"] = record(diff, "literal_diff", dict(episode_id=eid, path=path))
                row["skill_changes"].append(change)
        rows.append(row)
    public = []
    for document in documents:
        meta = {key: document[key] for key in ("workbook_id", "document_id", "title")}
        public.append(dict(**meta, record_id=record(document["body"], "public_document", meta)))
    value = dict(schema=VERSION, feedback_mode=feedback_mode, episodes=rows, documents=public,
                 records=records, assets=public_assets(assets))
    value["bundle_sha256"] = digest(value)
    return value


def schema(name, description, properties, required=()):
    return dict(type="function", function=dict(name=name, description=description,
        parameters=dict(type="object", properties=properties, required=list(required), additionalProperties=False)))


TEXT = {"type": "string"}
OFFSET = dict(type="integer", minimum=0, description="Zero-based offset; use the previous response's next_offset.")
LIMIT = dict(type="integer", minimum=1, maximum=20, description="Number of items; default 8, at most 20.")
SCHEMAS = [
    schema("episode_list", "List completed own episodes with task, declared feedback, delivery references and skill changes. No inferred explanation. Returns episodes and next_offset (null at end).", dict(offset=OFFSET, limit=LIMIT)),
    schema("episode_read", "Read an episode's ordered event index, without embedding previous retrieval payloads. Each record_id can be opened with record_read. Returns events and next_offset; offsets count events, not characters.", dict(episode_id=TEXT, offset=OFFSET, limit=LIMIT), ("episode_id",)),
    schema("record_search", "Case-insensitive AND of whitespace-separated literal terms. Empty query lists source records. Default scope sources excludes historical retrieval echoes and review instructions, not original observations. Returns record IDs, source locators and one exact snippet each. Snippets are not full records; open record_read. Offsets count records.", dict(query=TEXT, scope=dict(type="string", enum=["sources", "all"]), offset=OFFSET, limit=LIMIT)),
    schema("record_read", "Read exact text by record_id obtained from an index/search. Offset and limit count Unicode characters (not bytes). Returns text, sha256, total_characters and next_offset; null means end. Unknown IDs return an explicit error.", dict(record_id=TEXT, offset=OFFSET, limit=dict(type="integer", minimum=1, maximum=24000, description="Characters to return; default 12000, at most 24000.")), ("record_id",)),
    schema("asset_list", "List current editable experience paths and hashes. These assets, not history records, are transferred to the next task. Returns assets and next_offset.", dict(offset=OFFSET, limit=LIMIT)),
    schema("asset_read", "Read the complete current text and sha256 of an experience path. Read the current version before replacing an existing asset.", dict(path=TEXT), ("path",)),
    schema("asset_write", "Stage a complete UTF-8 text asset under skills/ or memories/. No execution. Supply the current sha256 from asset_read; use empty string for a new path. Stale hashes fail without mutation. SKILL.md requires YAML name and description. Nothing transfers until the host commits a successfully completed review. Maximum 2 MB per asset.", dict(path=TEXT, content=TEXT, expected_sha256=TEXT), ("path", "content", "expected_sha256")),
]
NAMES = {item["function"]["name"] for item in SCHEMAS}


class Session:
    def __init__(self, bundle):
        data = deepcopy(bundle)
        expected = data.pop("bundle_sha256")
        if data.get("schema") != VERSION or digest(data) != expected:
            raise ValueError("Bundle integrity failure")
        self.bundle, self.assets = data, public_assets(data["assets"])
        self.initial_hash = digest(self.assets)
        self.read_versions, self.events = {}, []

    def call(self, name, arguments):
        declaration = next((item["function"]["parameters"] for item in SCHEMAS if item["function"]["name"] == name), None)
        try:
            if declaration is None:
                raise ValueError("Tool unavailable in consolidation; use the listed read-only record and experience tools")
            jsonschema.validate(arguments, declaration)
            result = self._call(name, arguments)
        except (ValueError, KeyError, jsonschema.ValidationError) as error:
            result = dict(error=type(error).__name__, message=str(error).split("\n")[0])
        self.events.append(dict(name=name, arguments=deepcopy(arguments), result=deepcopy(result)))
        return result

    def _page(self, values, arguments, key):
        start, limit = arguments.get("offset", 0), arguments.get("limit", 8)
        return {key: deepcopy(values[start:start + limit]), "total": len(values),
                "next_offset": start + limit if start + limit < len(values) else None}

    def _call(self, name, a):
        if name == "episode_list":
            rows = [{key: value for key, value in row.items() if key != "events"} for row in self.bundle["episodes"]]
            return self._page(rows, a, "episodes")
        if name == "episode_read":
            row = next((row for row in self.bundle["episodes"] if row["episode_id"] == a["episode_id"]), None)
            if row is None:
                raise ValueError("Unknown episode_id; obtain it from episode_list")
            return self._page(row["events"], a, "events")
        if name == "record_search":
            terms = a.get("query", "").casefold().split()
            found = []
            for rid, row in self.bundle["records"].items():
                origins = row["origins"]
                if a.get("scope", "sources") == "sources" and all(
                        source["kind"] in {"retrieval_echo", "review_instruction"} for source in origins):
                    continue
                text = row["text"]
                if all(term in text.casefold() for term in terms):
                    # No semantic ranking or selective source summarization.
                    found.append(dict(record_id=rid, origins=origins, characters=len(text), snippet=text[:240]))
            return self._page(found, a, "records")
        if name == "record_read":
            row = self.bundle["records"].get(a["record_id"])
            if row is None:
                raise ValueError("Unknown record_id; obtain it from an index/search")
            offset, limit = a.get("offset", 0), a.get("limit", 12000)
            return dict(text=row["text"][offset:offset + limit], sha256=row["sha256"], origins=deepcopy(row["origins"]),
                total_characters=row["characters"], next_offset=offset + limit if offset + limit < row["characters"] else None)
        if name == "asset_list":
            return self._page([dict(path=path, sha256=item["sha256"]) for path, item in sorted(self.assets.items())], a, "assets")
        path = asset_path(a["path"])
        current = self.assets.get(path)
        if name == "asset_read":
            if current is None:
                raise ValueError("Unknown asset path; obtain existing paths from asset_list")
            self.read_versions[path] = current["sha256"]
            return dict(path=path, **current)
        expected = current["sha256"] if current else ""
        if a["expected_sha256"] != expected or (current and self.read_versions.get(path) != expected):
            raise ValueError("STALE_OR_UNREAD: asset_read the current version and retry with its sha256")
        content = a["content"]
        if len(content.encode()) > 2_000_000:
            raise ValueError("Asset exceeds 2 MB")
        if path.endswith("/SKILL.md"):
            import yaml
            parts = content.split("---", 2)
            if len(parts) != 3 or parts[0].strip():
                raise ValueError("SKILL.md needs YAML frontmatter with name and description")
            try:
                header = yaml.safe_load(parts[1])
            except yaml.YAMLError as error:
                raise ValueError("Invalid YAML frontmatter") from error
            if not isinstance(header, dict) or not all(isinstance(header.get(key), str) and header[key].strip() for key in ("name", "description")):
                raise ValueError("SKILL.md needs nonempty string name and description")
        self.assets[path] = dict(content=content, sha256=text_hash(content))
        return dict(path=path, sha256=self.assets[path]["sha256"], staged=True)

    def finish(self, completed):
        return dict(schema=VERSION, completed=bool(completed), transferable=bool(completed),
            initial_assets_sha256=self.initial_hash,
            assets=deepcopy(self.assets) if completed else None,
            candidate_assets=deepcopy(self.assets), events=deepcopy(self.events))
