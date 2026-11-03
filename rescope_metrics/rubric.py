"""The frozen item-AND contract, with complete-corpus citation validation."""

from copy import deepcopy
import json

from .common import DATA, digest, read, require


def result_schema(targets):
    citation = {"type": "object", "properties": {
        "document_id": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1},
        "end_line": {"type": "integer", "minimum": 1}},
        "required": ["document_id", "start_line", "end_line"], "additionalProperties": False}
    claims = {}
    for target in targets:
        items = {}
        for item in target["items"]:
            allowed = ["met", "not_met"] + (["not_applicable"] if item["role"] == "conditional" else [])
            items[item["id"]] = {"type": "object", "properties": {
                "finding": {"type": "string", "enum": allowed}, "reason": {"type": "string"},
                "support": {"type": "array", "items": citation},
                "counterevidence": {"type": "array", "items": citation}},
                "required": ["finding", "reason", "support", "counterevidence"], "additionalProperties": False}
        claims[target["claim"]["id"]] = {"type": "object", "properties": items,
                                          "required": list(items), "additionalProperties": False}
    return {"type": "object", "properties": claims, "required": list(claims), "additionalProperties": False}


def payload(targets, documents):
    catalog = read(DATA / "manifest.json")
    require(digest(catalog["instructions"]) == catalog["instructions_sha256"], "Judge instructions changed")
    require(targets and len({t["claim"]["id"] for t in targets}) == len(targets), "Duplicate claim")
    require(len({(t["claim"]["type"], t["claim"]["unit_id"]) for t in targets}) == 1,
            "One request must judge one complete experience")
    return {"schema": catalog["scoring_schema"], "instructions": catalog["instructions"],
            "targets": deepcopy(targets), "documents": deepcopy(documents), "response_schema": result_schema(targets)}


def request(body, config):
    visible = {k: deepcopy(v) for k, v in body.items() if k not in ("instructions", "response_schema", "documents")}
    visible["documents"] = [{"id": d["id"], "numbered_text": "\n".join(
        f"{i}: {line}" for i, line in enumerate(d["lines"], 1))} for d in body["documents"]]
    messages = [{"role": "system", "content": body["instructions"]},
                {"role": "user", "content": json.dumps(visible, ensure_ascii=True)}]
    schema = {"name": "claim_rubric_items", "strict": True, "schema": deepcopy(body["response_schema"])}
    if config["api"] == "responses":
        result = {"model": config["model"], "store": False, "stream": False,
                  "max_output_tokens": config["max_output_tokens"], "input": messages,
                  "text": {"format": {"type": "json_schema", **schema}}}
        if config["reasoning_effort"]:
            result["reasoning"] = {"effort": config["reasoning_effort"]}
        return result
    result = {"model": config["model"], "store": False, "stream": False,
              "messages": messages, "max_completion_tokens": config["max_output_tokens"],
              "response_format": {"type": "json_schema", "json_schema": schema}}
    if config["reasoning_effort"]:
        result["reasoning_effort"] = config["reasoning_effort"]
    return result


def grade(body, response):
    targets = {t["claim"]["id"]: t for t in body["targets"]}
    require(isinstance(response, dict) and set(response) == set(targets), "Incomplete claim judgments")
    documents = {d["id"]: d["lines"] for d in body["documents"]}
    require(len(documents) == len(body["documents"]), "Duplicate document identity")
    grades = []
    for cid, target in targets.items():
        rules, rows = {r["id"]: r for r in target["items"]}, response[cid]
        require(isinstance(rows, dict) and set(rows) == set(rules), "Incomplete item judgments")
        failed = []
        for item_id, rule in rules.items():
            row = rows[item_id]
            require(isinstance(row, dict) and set(row) == {"finding", "reason", "support", "counterevidence"},
                    "Unexpected item result fields")
            allowed = {"met", "not_met"} | ({"not_applicable"} if rule["role"] == "conditional" else set())
            require(row["finding"] in allowed, "Invalid required or conditional item judgment")
            require(isinstance(row["reason"], str) and row["reason"].strip(), "Missing item rationale")
            for kind in ("support", "counterevidence"):
                require(isinstance(row[kind], list), "Citations must be a list")
                for citation in row[kind]:
                    require(isinstance(citation, dict) and set(citation) == {"document_id", "start_line", "end_line"},
                            "Unexpected citation fields")
                    doc, start, end = citation["document_id"], citation["start_line"], citation["end_line"]
                    require(isinstance(doc, str) and doc in documents, "Unknown cited document")
                    # Line-order normalization is formatting only; it changes no findings.
                    require(type(start) is int and type(end) is int and
                            1 <= min(start, end) <= max(start, end) <= len(documents[doc]), "Citation outside original document")
            require(row["finding"] != "met" or row["support"], "A met item needs saved-text support")
            if row["finding"] == "not_met":
                failed.append(item_id)
        grades.append({"claim_id": cid, "credit": int(not failed), "failed_items": failed,
                       "item_judgments": deepcopy(rows)})
    return grades


def parsed_response(raw, api):
    if api == "responses":
        require(raw.get("status") == "completed", "Judge response did not complete")
        blocks = [c for output in raw.get("output", []) for c in output.get("content", [])]
        require(not any(c.get("type") == "refusal" for c in blocks), "Judge refused")
        text = "".join(c["text"] for c in blocks if c.get("type") == "output_text")
    else:
        choices = raw.get("choices", [])
        require(len(choices) == 1 and choices[0].get("finish_reason") == "stop", "Judge response truncated or incomplete")
        require(not choices[0]["message"].get("refusal"), "Judge refused")
        text = choices[0]["message"].get("content")
    require(isinstance(text, str) and bool(text.strip()), "Judge returned no structured output")
    return json.loads(text)
