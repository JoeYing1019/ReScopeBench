"""Single-pass, resumable unit judgments; retries never turn failures into scores."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from .common import digest, epoch_at, load_gold, read, require, write, write_once
from .extract import from_run
from .rubric import grade, parsed_response, payload, request
from .scores import score_pair


def configuration(model, base_url, api="responses", reasoning_effort="medium", max_output_tokens=14000):
    parsed = urlsplit(base_url)
    require(parsed.scheme in ("http", "https") and bool(parsed.hostname) and
            not (parsed.username or parsed.password or parsed.query or parsed.fragment),
            "Use a base URL without credentials, query secrets or fragment")
    require(api in ("responses", "chat_completions"), "Unknown judge API")
    require(isinstance(model, str) and bool(model.strip()), "Provide an accessible judge model")
    require(type(max_output_tokens) is int and max_output_tokens > 0, "Invalid output limit")
    return {"model": model, "base_url": base_url.rstrip("/"), "api": api,
            "reasoning_effort": reasoning_effort, "max_output_tokens": max_output_tokens,
            "timeout_seconds": 600, "transport_attempts": 8, "semantic_attempts": 1}


def prepare(run, output, config):
    pair, corpora = from_run(run)
    return prepare_pair(pair, corpora, output, config)


def prepare_pair(pair, corpora, output, config):
    output = Path(output)
    gold = load_gold(pair["case_id"])
    score_pair(pair, gold)
    require(set(corpora) == set(range(1, gold["task_count"] + 1)), "Missing saved experience checkpoints")
    jobs, positions = {}, []
    for row in pair["positions"]:
        p = row["position"]
        docs = corpora[p]
        corpus_hash = digest(docs)
        write_once(output / "corpora" / (corpus_hash + ".json"), docs)
        ids = []
        for unit in epoch_at(gold, p)["units"]:
            body = payload(unit["targets"], docs)
            wire = request(body, config)
            jid = digest([digest(wire), config])
            ref = {"id": jid, "request_sha256": digest(wire), "corpus_sha256": corpus_hash,
                   "epoch": epoch_at(gold, p)["epoch"], "unit_id": unit["id"]}
            if jid in jobs:
                require(jobs[jid]["request_sha256"] == ref["request_sha256"], "Request hash collision")
            else:
                jobs[jid] = ref
            ids.append(jid)
        positions.append({"position": p, "jobs": ids})
    manifest = {"schema": "rescope-experience-judge/v1", "pair_sha256": digest(pair),
                "gold_sha256": digest(gold), "config": config, "jobs": list(jobs.values()),
                "positions": positions, "case_id": pair["case_id"], "model": pair["model"]}
    write_once(output / "pair.json", pair)
    write_once(output / "manifest.json", manifest)
    return {"prepared_requests": len(jobs), "positions": len(positions), "case_id": pair["case_id"]}


def materialize(output, manifest, ref):
    gold = load_gold(manifest["case_id"])
    require(digest(gold) == manifest["gold_sha256"], "Prepared golden changed")
    epoch = next(e for e in gold["epochs"] if e["epoch"] == ref["epoch"])
    unit = next(u for u in epoch["units"] if u["id"] == ref["unit_id"])
    docs = read(output / "corpora" / (ref["corpus_sha256"] + ".json"))
    require(digest(docs) == ref["corpus_sha256"], "Prepared experience corpus changed")
    body = payload(unit["targets"], docs)
    wire = request(body, manifest["config"])
    require(digest(wire) == ref["request_sha256"], "Reconstructed request differs from frozen input")
    return body, wire


def _one(output, manifest, ref, client, secret, sleep=time.sleep):
    jid = ref["id"]
    terminal = output / "judgments" / (jid + ".json")
    if terminal.exists():
        saved = read(terminal)
        require(saved["request_sha256"] == ref["request_sha256"], "Stored judgment input differs")
        if saved["status"] != "transport_error":
            return saved["status"]
    body, wire = materialize(output, manifest, ref)
    config = manifest["config"]
    endpoint = config["base_url"] + ("/responses" if config["api"] == "responses" else "/chat/completions")
    attempts = output / "calls" / jid
    attempts.mkdir(parents=True, exist_ok=True)
    previous = sorted(attempts.glob("*.json"))
    # A crash after persisting a reply must not trigger a new semantic judgment.
    if previous:
        receipt = read(previous[-1])
        require(receipt["request_sha256"] == ref["request_sha256"], "Prior attempt input differs")
        if receipt["status"] != "transport_error":
            record = {"id": jid, "request_sha256": ref["request_sha256"], "status": receipt["status"],
                      "attempt_path": str(previous[-1].relative_to(output)), "attempt_sha256": digest(receipt)}
            if receipt["status"] == "completed":
                record["assessments"] = grade(body, parsed_response(receipt["response"], config["api"]))
                require(record["assessments"] == receipt["assessments"], "Persisted judgment does not replay")
            write(terminal, record)
            return receipt["status"]
    offset = len(previous)
    for i in range(config["transport_attempts"]):
        receipt = {"request_sha256": ref["request_sha256"], "at": datetime.now(timezone.utc).isoformat()}
        response, retry = None, False
        try:
            response = client.post(endpoint, json=wire, headers={"Authorization": "Bearer " + secret},
                                   timeout=config["timeout_seconds"])
            receipt["http_status"] = response.status_code
            if response.status_code == 429 or response.status_code >= 500 or response.status_code in (408, 409):
                retry = True
            elif response.status_code >= 400:
                receipt["status"] = "configuration_error"
            else:
                raw = response.json()
                receipt["response"] = raw
                try:
                    assessments = grade(body, parsed_response(raw, config["api"]))
                    receipt.update(status="completed", assessments=assessments)
                except (ValueError, TypeError, KeyError, IndexError) as error:
                    receipt.update(status="invalid_judgment", error_type=type(error).__name__)
        except Exception as error:
            import httpx
            if not isinstance(error, httpx.TransportError):
                # A successful response with invalid JSON is not semantically resampled.
                if response is not None and response.status_code < 400 and isinstance(error, ValueError):
                    receipt.update(status="invalid_judgment", error_type=type(error).__name__, raw_text=response.text)
                else:
                    raise
            else:
                retry = True
                receipt["error_type"] = type(error).__name__
        if retry:
            receipt["status"] = "transport_error"
        attempt_path = attempts / f"{offset + i + 1:04d}.json"
        write_once(attempt_path, receipt)
        record = {"id": jid, "request_sha256": ref["request_sha256"], "status": receipt["status"],
                  "attempt_path": str(attempt_path.relative_to(output)), "attempt_sha256": digest(receipt)}
        if receipt["status"] == "completed":
            record["assessments"] = receipt["assessments"]
        write(terminal, record)
        if not retry or i + 1 == config["transport_attempts"]:
            return receipt["status"]
        delay = min(120.0, 2.0 ** min(i + 1, 7))
        if response is not None:
            try:
                delay = max(delay, min(120.0, float(response.headers.get("Retry-After", 0))))
            except ValueError:
                pass
        sleep(delay)


def run(output, workers=16):
    import httpx
    output = Path(output)
    require(type(workers) is int and 1 <= workers <= 256, "Workers must be between 1 and 256")
    secret = os.environ.get("RESCOPE_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    require(bool(secret), "Inject RESCOPE_JUDGE_API_KEY or OPENAI_API_KEY; keys are not saved in the manifest")
    manifest = read(output / "manifest.json")
    states = {}
    with (output / "judge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with httpx.Client(limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers),
                          follow_redirects=False) as client, ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one, output, manifest, ref, client, secret) for ref in manifest["jobs"]]
            for i, future in enumerate(as_completed(futures), 1):
                state = future.result()
                states[state] = states.get(state, 0) + 1
                if i % 25 == 0 or i == len(futures):
                    print(json.dumps({"processed": i, "total": len(futures), "states": states}), flush=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return states


def score(output):
    output = Path(output)
    manifest, pair = read(output / "manifest.json"), read(output / "pair.json")
    require(digest(pair) == manifest["pair_sha256"], "Prepared business results changed")
    grades = {}
    for ref in manifest["jobs"]:
        record = read(output / "judgments" / (ref["id"] + ".json"))
        require(record["status"] == "completed" and record["request_sha256"] == ref["request_sha256"],
                "Scoring incomplete: missing or failed judgment is not zero")
        path = (output / record["attempt_path"]).resolve()
        require(path.is_relative_to(output.resolve()), "Unsafe judgment receipt path")
        receipt = read(path)
        require(digest(receipt) == record["attempt_sha256"], "Judge response receipt changed")
        body, _ = materialize(output, manifest, ref)
        replayed = grade(body, parsed_response(receipt["response"], manifest["config"]["api"]))
        require(replayed == record["assessments"], "Scores do not replay from raw judge response")
        grades[ref["id"]] = {r["claim_id"]: r["credit"] for r in replayed}
    credits = {}
    for pos in manifest["positions"]:
        values = {}
        for jid in pos["jobs"]:
            require(not (set(values) & set(grades[jid])), "Duplicate claim within snapshot")
            values.update(grades[jid])
        credits[pos["position"]] = values
    result = score_pair(pair, load_gold(pair["case_id"]), credits)
    result["judge_config"] = {k: v for k, v in manifest["config"].items() if k != "base_url"}
    result["judge_endpoint_sha256"] = digest(manifest["config"]["base_url"])
    write(output / "claim_scores.json", {str(p): values for p, values in credits.items()})
    write(output / "metrics.json", result)
    return result["metrics"]
