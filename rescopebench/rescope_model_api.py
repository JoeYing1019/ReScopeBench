"""Native model APIs behind the frozen chat-completions bridge.

Added by the rescope harness; not part of the original release.

Hermes (inside the container) always speaks chat completions to the host. The
host provider converts at this boundary only:

    openai-chat              /v1/chat/completions; optional extra fields such as
                             reasoning_effort; reasoning_content replayed per turn
    openai-responses         /v1/responses via rescopebench's own provider;
                             reasoning items with encrypted_content replayed
    anthropic-messages       /v1/messages; thinking blocks with signatures replayed
    gemini-generate-content  generateContent on the Gemini API (default) or Vertex AI,
                             see RESCOPE_GEMINI_URL; raw parts with thoughtSignature
                             replayed

The selection is frozen inside plan["transport"]["rescope_model_api"], so it is
covered by the plan hash and cannot change between prepare and run-arm.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

import time

import httpx

from rescopebench.hermes_arg_coercion import normalize_tool_arguments

SCHEMA = "rescope-model-api/v1"
MODEL_APIS = ("openai-chat", "openai-responses", "anthropic-messages", "gemini-generate-content")
DEFAULT_PARAMS = {
    "openai-chat": {},
    "openai-responses": {},
    "anthropic-messages": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}, "stream": True},
    "gemini-generate-content": {"thinking_level": "high", "stream": True},
}
DEFAULT_TIMEOUT = {"openai-chat": 60.0, "openai-responses": 600.0,
                   "anthropic-messages": 900.0, "gemini-generate-content": 900.0}
ANTHROPIC_VERSION = "2023-06-01"
# Gemini endpoint. {root} is OPENAI_BASE_URL without its trailing /v1, {model} the model
# name, {method} generateContent or streamGenerateContent?alt=sse. The default is the
# Gemini API; for Vertex AI set RESCOPE_GEMINI_URL to
#   https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:{method}
# with your project and region written in, and RESCOPE_GEMINI_AUTH=bearer with an access token as the key.
GEMINI_URL = "{root}/v1beta/models/{model}:{method}"
GEMINI_AUTH = ("api-key", "bearer")
CONTENT_FILTER_RETRIES = 3
# The pinned episode executor provides a 10800-second wall-clock deadline.
EPISODE_DEADLINE_SECONDS = 10800.0
TRANSIENT_RETRIES = 10000
# A stream cut before its finish reason is an upstream failure: resend it unchanged.
MODEL_RESAMPLES = 3

# A turn the frozen host would reject (a call to an undeclared tool) or that Hermes
# would answer with its own hidden retry (arguments that are not JSON, no visible
# reply) breaks the 1:1 binding between provider responses and Hermes' transcript,
# so it never reaches Hermes. Instead the model is told what went wrong on the same
# conversation and answers again:
#   undeclared tool              tool result "Invalid function name {name}"
#   arguments that are not JSON  tool result: pydantic's json_invalid error
#   no tool call and no visible text (empty, refusal, cut at max_tokens, a tool call
#   left in the text)            user "Please continue the task"
# The exchange is sent upstream only; Hermes receives the answer that follows it.
# A reply with visible text and no tool call is the delivery and passes unchanged:
# ReScopeBench has no finish tool, so that is how a task ends. After FEEDBACK_ROUNDS
# such rounds in one turn the turn becomes a final answer with whatever visible text
# the last reply had, like Hermes' own terminal "(empty)", and the task is graded.
# Every rejected reply and the feedback it got is kept under feedback/.
FEEDBACK_ROUNDS = 3
CONTINUE_MESSAGE = "Please continue the task"
EMPTY_SENTINEL = "(empty)"
# Finish reasons across the four APIs, used before any look at the text.
TRUNCATED = {"length", "max_tokens", "MAX_TOKENS", "max_output_tokens", "model_context_window_exceeded"}
REFUSED = {"refusal", "SAFETY", "PROHIBITED_CONTENT", "SPII", "BLOCKLIST", "RECITATION", "IMAGE_SAFETY"}
MALFORMED_CALL = {"tool_calls", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"}
# A tool call the provider left as text instead of parsing it into tool_calls:
# MiniMax's <invoke> markup inside <think>, or MiMo's <tool_call><function=...>.
UNPARSED_TOOL_CALL = re.compile(r"<tool_call|<invoke\b|<function=|\]<\]minimax\[>\[")

# Hermes' visible-text test (agent/agent_runtime_helpers.strip_think_blocks in the
# pinned image): a reply is empty when nothing is left after these are removed.
THINK_TAG_NAMES = ("think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD")
TOOL_CALL_TAG_NAMES = ("tool_call", "tool_calls", "tool_result", "function_call", "function_calls")
STRIP_PATTERNS = (
    *(re.compile(rf"<{name}>.*?</{name}>", re.DOTALL | re.IGNORECASE) for name in THINK_TAG_NAMES),
    *(re.compile(rf"<{name}\b[^>]*>.*?</{name}>", re.DOTALL | re.IGNORECASE) for name in TOOL_CALL_TAG_NAMES),
    re.compile(r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*<function\b[^>]*\bname\s*=[^>]*>'
               r'(?:(?:(?!</function>).)*)</function>', re.DOTALL | re.IGNORECASE),
    re.compile(rf'(?:^|\n)[ \t]*<(?:{"|".join(THINK_TAG_NAMES)})\b[^>]*>.*$', re.DOTALL | re.IGNORECASE),
    re.compile(rf'</?(?:{"|".join(THINK_TAG_NAMES)})>\s*', re.IGNORECASE),
    re.compile(rf'</(?:{"|".join(TOOL_CALL_TAG_NAMES)}|function)>\s*', re.IGNORECASE),
    re.compile(rf'(?:^|\n)[ \t]*<(?:{"|".join(TOOL_CALL_TAG_NAMES)})\b[^>]*>.*$'
               r'|(?:^|\n)[^\n<]*</?arg_(?:key|value)\b.*$', re.DOTALL | re.IGNORECASE),
)


def visible_text(content) -> str:
    text = text_of(content)
    for pattern in STRIP_PATTERNS if text else ():
        text = pattern.sub("", text)
    return text.strip()


def json_error(name: str, arguments) -> str | None:
    """pydantic's error text for arguments that are not JSON, else None."""
    if not isinstance(arguments, str) or not arguments.strip():
        return None   # Hermes turns "" into {}; normalize_tool_arguments does the same
    try:
        json.loads(arguments)
        return None
    except ValueError:
        pass
    import pydantic
    model = pydantic.create_model(name or "arguments", __config__=pydantic.ConfigDict(extra="allow"))
    try:
        model.model_validate_json(arguments, strict=True)
    except pydantic.ValidationError as error:
        return str(error)
    return "Invalid JSON arguments"


def feedback_for(body: dict, message: dict, finish) -> tuple[str, list[dict]] | None:
    """Why this reply must be answered again, and the messages that tell the model; None to accept it."""
    calls = message.get("tool_calls") or []
    content = text_of(message.get("content"))
    if calls and finish not in TRUNCATED:
        declared = {(tool.get("function") or {}).get("name") for tool in body.get("tools") or []}

        def problem(call):
            function = call.get("function") or {}
            name = function.get("name")
            if name not in declared:
                return f"undeclared_tool_call: {name}", f"Invalid function name {name}"
            error = json_error(name, function.get("arguments"))
            return (f"invalid_json_arguments: {name}", error) if error else (None, None)

        # Every call counts: the batched host keeps them all, the frozen check rejects
        # any undeclared name, and Hermes re-requests the turn if any is not JSON.
        reason = next((found for found, _ in map(problem, calls) if found), None)
        if reason is None:
            return None
        assistant = {key: deepcopy(value) for key, value in message.items()
                     if key in ("role", "content", "tool_calls", "reasoning_content")}
        results = []
        for call in calls:
            _, text = problem(call)
            results.append({"role": "tool", "tool_call_id": call.get("id"), "name": (call.get("function") or {}).get("name"),
                            "content": text or "Not executed: another tool call in this turn was invalid."})
        return reason, [assistant, *results]
    if finish in TRUNCATED:
        reason = f"truncated: {finish}"
    elif finish in REFUSED:
        reason = f"refusal: {finish}"
    elif finish in MALFORMED_CALL or UNPARSED_TOOL_CALL.search(content):
        reason = f"unparsed_tool_call: {finish}"
    elif not visible_text(content):
        reason = f"empty_response: {finish}"
    else:
        return None
    return reason, [{"role": "assistant", "content": content if content.strip() else EMPTY_SENTINEL},
                    {"role": "user", "content": CONTINUE_MESSAGE}]


def final_answer(response: dict) -> dict:
    """The turn that ends the task once feedback is used up: the last visible text, no call."""
    final = deepcopy(response)
    message = final["choices"][0]["message"]
    message.pop("tool_calls", None)
    message["content"] = visible_text(message.get("content")) or EMPTY_SENTINEL
    final["choices"][0]["finish_reason"] = "stop"
    return final


def http_status(error) -> int | None:
    for item in (error, getattr(error, "__cause__", None), getattr(error, "last_error", None)):
        status = getattr(item, "status_code", None) or getattr(getattr(item, "response", None), "status_code", None)
        if status:
            return status
    return None


def answer_with_feedback(body: dict, ask, record) -> tuple[dict, list[dict]]:
    """Ask until the reply may reach Hermes; return it and every rejected reply.

    ask(body) -> (chat response, native finish reason, raw payload).
    record(round, row) keeps the evidence for one rejected reply.
    """
    asked, rejected = body, []
    for round_ in range(1, FEEDBACK_ROUNDS + 2):
        try:
            response, finish, raw = ask(asked)
        except Exception as error:
            # A provider that refuses the feedback history (HTTP 400) still owes this
            # turn an answer: ask the original request again without the feedback.
            if asked is body or http_status(error) != 400:
                raise
            record(round_, {"reason": "feedback_rejected_by_api", "error": str(error)[:2000]})
            asked = body
            response, finish, raw = ask(asked)
        verdict = feedback_for(body, response["choices"][0]["message"], finish)
        if verdict is None:
            return response, rejected
        reason, messages = verdict
        rejected.append(response)
        if round_ > FEEDBACK_ROUNDS:
            final = final_answer(response)
            record(round_, {"reason": reason, "discarded_response": raw, "feedback_exhausted": True,
                            "returned_instead": final})
            return final, rejected
        record(round_, {"reason": reason, "discarded_response": raw, "feedback": messages})
        asked = {**asked, "messages": [*asked["messages"], *messages]}
    raise AssertionError("unreachable")


def normalize_arguments(directory, logical_turn: int, body: dict, response: dict) -> list:
    """Apply Hermes' argument repairs in place and keep a record of each rewrite."""
    changes = normalize_tool_arguments(body, response)
    if changes:
        from rescopebench.agent_runtime import model_api
        model_api.provider_base.write(directory / "normalization-arguments" / f"{logical_turn:04d}.json",
                                      {"changes": changes})
    return changes


def feedback_recorder(provider):
    """Rows land beside the rejected reply's own requests/ and responses/ number."""
    from rescopebench.agent_runtime import model_api

    def record(round_, row):
        model_api.provider_base.write(provider.directory / "feedback" / f"{provider.count:04d}.json",
                                      {"round": round_, **row})
    return record


def record_resample(directory, logical_turn: int, resample: int, reason: str, raw) -> None:
    from rescopebench.agent_runtime import model_api
    model_api.provider_base.write(directory / "resamples" / f"{logical_turn:04d}-{resample:02d}.json",
                                  {"reason": reason, "discarded_response": raw})


# ---------------------------------------------------------------- selection

def implementation_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), Path(__file__).with_name("hermes_arg_coercion.py")]
    paths.extend((root / "rescope_harness").rglob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def selection_from_environment(provider_api: str, environ=None) -> dict:
    env = os.environ if environ is None else environ
    api = env.get("RESCOPE_MODEL_API") or (
        "openai-responses" if provider_api == "responses" else "openai-chat")
    if api not in MODEL_APIS:
        raise ValueError(f"RESCOPE_MODEL_API must be one of {MODEL_APIS}")
    if (api == "openai-responses") != (provider_api == "responses"):
        raise ValueError("openai-responses requires --provider-api responses, and only it")
    params = deepcopy(DEFAULT_PARAMS[api])
    extra = json.loads(env.get("RESCOPE_MODEL_PARAMS", "{}"))
    if not isinstance(extra, dict):
        raise ValueError("RESCOPE_MODEL_PARAMS must be a JSON object")
    params.update(extra)
    timeout = float(env.get("RESCOPE_REQUEST_TIMEOUT_SECONDS", DEFAULT_TIMEOUT[api]))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("RESCOPE_REQUEST_TIMEOUT_SECONDS must be positive")
    selection = {"schema": SCHEMA, "model_api": api, "params": params, "request_timeout_seconds": timeout,
                 "implementation_sha256s": implementation_hashes()}
    if api == "gemini-generate-content":
        auth = env.get("RESCOPE_GEMINI_AUTH", GEMINI_AUTH[0])
        if auth not in GEMINI_AUTH:
            raise ValueError(f"RESCOPE_GEMINI_AUTH must be one of {GEMINI_AUTH}")
        selection.update(gemini_url=env.get("RESCOPE_GEMINI_URL", GEMINI_URL), gemini_auth=auth)
    return selection


def api_root(endpoint: str) -> str:
    """https://host/v1 -> https://host (native APIs carry their own path)."""
    parts = urlsplit(endpoint.rstrip("/"))
    path = parts.path[:-3] if parts.path.endswith("/v1") else parts.path
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


# ---------------------------------------------------------------- shared helpers

def assistant_key(message: dict) -> str:
    """Same identity as rescopebench.responses_provider.assistant_key."""
    value = {"content": message.get("content") or "", "calls": []}
    for call in message.get("tool_calls") or []:
        fn = call["function"]
        arguments = fn["arguments"]
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            pass
        value["calls"].append([call["id"], fn["name"], arguments])
    return json.dumps(value, sort_keys=True)


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text") or "" for part in content
                         if isinstance(part, dict) and part.get("type") == "text")
    return "" if content is None else str(content)


def arguments_object(arguments) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def chat_completion(response_id, model, content, calls, finish, usage) -> dict:
    message = {"role": "assistant", "content": content or None}
    if calls:
        message["tool_calls"] = calls
    return {"id": response_id, "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage}


class Replay:
    """Native assistant turns keyed by the chat message Hermes sends back."""

    def __init__(self):
        self.turns = defaultdict(list)

    def remember(self, chat_message, native):
        self.turns[assistant_key(chat_message)].append(deepcopy(native))

    def forget(self, chat_message):
        stored = self.turns.get(assistant_key(chat_message))
        if stored:
            stored.pop()

    def lookup(self, message, counts):
        key = assistant_key(message)
        index = counts[key]
        counts[key] += 1
        stored = self.turns.get(key) or []
        return deepcopy(stored[index]) if index < len(stored) else None


# ---------------------------------------------------------------- anthropic

def anthropic_request(body: dict, params: dict, replay: Replay) -> tuple[dict, dict]:
    system, messages, counts = [], [], defaultdict(int)
    replayed = synthetic = 0

    def push(role, blocks):
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    for message in body["messages"]:
        role = message["role"]
        if role in ("system", "developer"):
            system.append(text_of(message.get("content")))
        elif role == "user":
            push("user", [{"type": "text", "text": text_of(message.get("content"))}])
        elif role == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": message["tool_call_id"],
                           "content": text_of(message.get("content"))}])
        elif role == "assistant":
            native = replay.lookup(message, counts)
            if native is not None:
                replayed += 1
                push("assistant", native)
                continue
            synthetic += 1
            blocks = []
            if text_of(message.get("content")):
                blocks.append({"type": "text", "text": text_of(message.get("content"))})
            for call in message.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                               "input": arguments_object(call["function"]["arguments"])})
            push("assistant", blocks)
    request = {"model": body["model"], "max_tokens": body["max_tokens"], "messages": messages}
    tools = [{"name": tool["function"]["name"], "description": tool["function"].get("description", ""),
              "input_schema": tool["function"].get("parameters") or {"type": "object", "properties": {}}}
             for tool in body.get("tools") or []]
    if tools:
        request["tools"] = tools
    if system:
        request["system"] = "\n\n".join(item for item in system if item)
    if not body.get("parallel_tool_calls", True):
        request["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
    for key in ("thinking", "output_config", "temperature", "top_p", "top_k"):
        if key in params:
            request[key] = deepcopy(params[key])
    if params.get("stream"):
        request["stream"] = True
    return request, {"replayed_assistant_turns": replayed, "synthetic_assistant_turns": synthetic,
                     "seed_omitted": body.get("seed"), "all_tool_calls_preserved": True}


def anthropic_assemble(events) -> dict:
    """Rebuild a non-streaming Messages response from SSE events, signatures included."""
    message, blocks, partial = None, {}, {}
    for event in events:
        kind = event.get("type")
        if kind == "error":
            raise RuntimeError(f"anthropic stream error: {event.get('error')}")
        if kind == "message_start":
            message = deepcopy(event["message"])
        elif kind == "content_block_start":
            blocks[event["index"]] = deepcopy(event["content_block"])
            partial[event["index"]] = ""
        elif kind == "content_block_delta":
            block, delta = blocks[event["index"]], event["delta"]
            if delta["type"] == "text_delta":
                block["text"] = block.get("text", "") + delta["text"]
            elif delta["type"] == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta["thinking"]
            elif delta["type"] == "signature_delta":
                block["signature"] = block.get("signature", "") + delta["signature"]
            elif delta["type"] == "input_json_delta":
                partial[event["index"]] += delta["partial_json"]
        elif kind == "content_block_stop":
            block = blocks[event["index"]]
            if block.get("type") == "tool_use" and partial[event["index"]]:
                block["input"] = json.loads(partial[event["index"]])
        elif kind == "message_delta":
            message.update({key: value for key, value in event["delta"].items() if value is not None})
            message["usage"] = {**(message.get("usage") or {}), **(event.get("usage") or {})}
    if message is None:
        raise RuntimeError("anthropic stream ended without message_start")
    message["content"] = [blocks[index] for index in sorted(blocks)]
    return message


def anthropic_chat(raw: dict) -> tuple[dict, list]:
    text, calls = [], []
    for block in raw.get("content") or []:
        if block.get("type") == "text":
            text.append(block.get("text") or "")
        elif block.get("type") == "tool_use":
            calls.append({"id": block["id"], "type": "function",
                          "function": {"name": block["name"],
                                       "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False)}})
    stop = raw.get("stop_reason")
    if stop == "max_tokens":
        finish = "length"
    else:
        finish = "tool_calls" if calls else "stop"
    usage = raw.get("usage") or {}
    prompt = sum(usage.get(key) or 0 for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    response = chat_completion(raw.get("id"), raw.get("model"), "\n".join(text), calls, finish,
                               {"prompt_tokens": prompt, "completion_tokens": usage.get("output_tokens", 0),
                                "total_tokens": prompt + (usage.get("output_tokens") or 0)})
    return response, deepcopy(raw.get("content") or [])


# ---------------------------------------------------------------- gemini

def _gemini_tool_result(content) -> dict:
    text = text_of(content)
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return {"result": text}
    return value if isinstance(value, dict) else {"result": value}


def gemini_request(body: dict, params: dict, replay: Replay) -> tuple[dict, dict]:
    system, contents, counts, names = [], [], defaultdict(int), {}
    replayed = synthetic = 0

    def push(role, parts):
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    for message in body["messages"]:
        for call in message.get("tool_calls") or []:
            names[call["id"]] = call["function"]["name"]
    for message in body["messages"]:
        role = message["role"]
        if role in ("system", "developer"):
            system.append(text_of(message.get("content")))
        elif role == "user":
            push("user", [{"text": text_of(message.get("content"))}])
        elif role == "tool":
            response = {"name": names.get(message["tool_call_id"]),
                        "response": _gemini_tool_result(message.get("content"))}
            push("user", [{"functionResponse": response}])
        elif role == "assistant":
            native = replay.lookup(message, counts)
            if native is not None:
                replayed += 1
                push("model", native)
                continue
            synthetic += 1
            parts = [{"text": text_of(message.get("content"))}] if text_of(message.get("content")) else []
            parts += [{"functionCall": {"name": call["function"]["name"],
                                        "args": arguments_object(call["function"]["arguments"])}}
                      for call in message.get("tool_calls") or []]
            push("model", parts)
    config = {"maxOutputTokens": body["max_tokens"]}
    if params.get("thinking_level"):
        config["thinkingConfig"] = {"thinkingLevel": params["thinking_level"], "includeThoughts": True}
    for source, target in (("temperature", "temperature"), ("top_p", "topP")):
        if source in params:
            config[target] = params[source]
    request = {"contents": contents, "generationConfig": config}
    if system:
        request["systemInstruction"] = {"parts": [{"text": "\n\n".join(item for item in system if item)}]}
    # parametersJsonSchema takes the frozen JSON Schema verbatim; the OpenAPI-subset
    # "parameters" field rejects maxItems/minItems ints and list-valued "type".
    declarations = [{"name": tool["function"]["name"], "description": tool["function"].get("description", ""),
                     "parametersJsonSchema": tool["function"].get("parameters") or {"type": "object", "properties": {}}}
                    for tool in body.get("tools") or []]
    if declarations:
        request["tools"] = [{"functionDeclarations": declarations}]
    return request, {"replayed_assistant_turns": replayed, "synthetic_assistant_turns": synthetic,
                     "seed_omitted": body.get("seed"), "parallel_tool_calls_not_configurable": True,
                     "all_tool_calls_preserved": True}


def gemini_assemble(chunks) -> dict:
    """Merge streamed GenerateContentResponses; text runs merge, signatures stay on their part."""
    parts, usage, finish, version = [], None, None, None
    for chunk in chunks:
        if chunk.get("error"):
            raise RuntimeError(f"gemini stream error: {chunk['error']}")
        version = chunk.get("modelVersion") or version
        usage = chunk.get("usageMetadata") or usage
        for candidate in chunk.get("candidates") or []:
            finish = candidate.get("finishReason") or finish
            for part in (candidate.get("content") or {}).get("parts") or []:
                last = parts[-1] if parts else None
                mergeable = ("text" in part and last is not None and "text" in last
                             and bool(last.get("thought")) == bool(part.get("thought"))
                             and "thoughtSignature" not in last)
                if mergeable:
                    last["text"] += part["text"]
                    if "thoughtSignature" in part:
                        last["thoughtSignature"] = part["thoughtSignature"]
                else:
                    parts.append(deepcopy(part))
    # Streams can end with {"text": ""}; an unsigned empty part is dropped before it is
    # replayed. Signed empty parts carry state and are kept.
    parts = [part for part in parts
             if not (set(part) <= {"text", "thought"} and not part.get("text"))]
    if not parts and finish is None:
        raise RuntimeError("gemini stream ended without candidates")
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
            "usageMetadata": usage or {}, "modelVersion": version}


def gemini_chat(raw: dict, turn: int, prefix: str = "") -> tuple[dict, list]:
    candidate = (raw.get("candidates") or [{}])[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text, calls = [], []
    for part in parts:
        if "functionCall" in part:
            call = part["functionCall"]
            # The id is always synthesized. Some Gemini models return their own ids and
            # repeat them within an episode, and the robot case refuses any episode with
            # a repeated id when it packages past episodes as experience. The prefix keeps
            # ids unique when one episode uses more than one provider instance (task, then
            # review); functionResponse parts are matched by name, and the replayed native
            # parts keep the upstream id.
            calls.append({"id": f"call_{prefix}{turn}_{len(calls)}", "type": "function",
                          "function": {"name": call["name"],
                                       "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False)}})
        elif "text" in part and not part.get("thought"):
            text.append(part["text"])
    if candidate.get("finishReason") == "MAX_TOKENS":
        finish = "length"
    else:
        finish = "tool_calls" if calls else "stop"
    usage = raw.get("usageMetadata") or {}
    prompt = usage.get("promptTokenCount") or 0
    completion = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
    response = chat_completion(f"gemini-{turn}", raw.get("modelVersion"), "".join(text), calls, finish,
                               {"prompt_tokens": prompt, "completion_tokens": completion,
                                "total_tokens": prompt + completion})
    return response, deepcopy(parts)


# ---------------------------------------------------------------- providers

def native_provider_class():
    from rescopebench.agent_runtime import model_api

    class NativeProvider(model_api.provider_base.Provider):
        """Converts audited chat bodies to one native API; same retry, slots and evidence layout."""

        def __init__(self, config, pool, directory, *, selection, policy):
            if config.api != "chat_completions" or config.protocol != "openai":
                raise ValueError("Native APIs sit behind the chat_completions/openai bridge")
            super().__init__(pool.keys, directory, base_url=config.endpoint, policy=policy)
            self.selection = selection
            self.api = selection["model_api"]
            self.params = selection["params"]
            self.root = api_root(config.endpoint)
            self.replay = Replay()
            self.id_prefix = os.urandom(3).hex() + "_"
            self.http = httpx.AsyncClient(timeout=None)
            model_api.provider_base.write(self.directory / "native-model-api.json", {
                **selection, "endpoint_root": self.root, "credentials": pool.public_record()})

        def _url_and_headers(self, key, body):
            if self.api == "anthropic-messages":
                return (f"{self.root}/v1/messages",
                        {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                         "content-type": "application/json"})
            method = "streamGenerateContent?alt=sse" if self.params.get("stream") else "generateContent"
            url = self.selection.get("gemini_url", GEMINI_URL).format(
                root=self.root, model=self.expected_model, method=method)
            auth = ({"Authorization": f"Bearer {key}"} if self.selection.get("gemini_auth") == "bearer"
                    else {"x-goog-api-key": key})
            return url, {**auth, "content-type": "application/json"}

        @staticmethod
        def _raise_for_status(response):
            if response.status_code == 529:  # Anthropic "overloaded"
                # The frozen retry taxonomy only retries 5xx gateway codes.
                response = httpx.Response(503, content=response.content, headers=response.headers,
                                          request=response.request)
            response.raise_for_status()

        async def _post(self, url, headers, body, stream):
            if not stream:
                response = await self.http.post(url, headers=headers, json=body)
                self._raise_for_status(response)
                return response.json()
            events = []
            async with self.http.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data and data != "[DONE]":
                            events.append(json.loads(data))
                request = response.request
            try:
                return anthropic_assemble(events) if self.api == "anthropic-messages" else gemini_assemble(events)
            except RuntimeError as error:
                # An error event inside a 200 stream (Anthropic overloaded_error /
                # api_error) or a stream cut before its first message is an upstream
                # failure. The frozen retry taxonomy only knows HTTP statuses, so
                # surface it as the 503 it would have been; request errors stay fatal.
                if "invalid_request" in str(error) or "authentication" in str(error):
                    raise
                raise httpx.HTTPStatusError(str(error), request=request,
                                            response=httpx.Response(503, text=str(error), request=request)) from error

        async def _attempt(self, body, timeout_seconds):
            slot = ((self.count - 1) + self._physical_attempt) % len(self.keys)
            self._physical_attempt += 1
            if self._record is not None:
                self._record({"kind": "transport_credential_selected", "logical_turn": self.count,
                              "attempt": self._physical_attempt, "credential_slot": f"slot-{slot + 1}",
                              "credential_slots": len(self.keys)})
            url, headers = self._url_and_headers(self.keys[slot], body)
            self.wires += 1
            model_api.provider_base.write(self.directory / "wire" / f"{self.wires:04d}.json", body)
            stream = bool(self.params.get("stream"))
            return await asyncio.wait_for(self._post(url, headers, body, stream), timeout_seconds)

        def __call__(self, body):
            def ask(asked):
                response, raw = self._call_once(asked)
                if self.api == "anthropic-messages":
                    finish = raw.get("stop_reason")
                else:
                    finish = ((raw.get("candidates") or [{}])[0]).get("finishReason")
                # A rejected call is replayed with its signed thinking in the feedback
                # round; a reply cut at max_tokens may hold unsigned thinking and is not.
                if finish not in TRUNCATED:
                    self.replay.remember(response["choices"][0]["message"], self._native)
                    natives[id(response)] = self._native
                return response, finish, raw

            natives = {}
            response, rejected = answer_with_feedback(body, ask, feedback_recorder(self))
            for item in [*rejected, response]:
                if id(item) in natives:
                    self.replay.forget(item["choices"][0]["message"])
            normalize_arguments(self.directory, self.count, body, response)
            if response["choices"][0]["finish_reason"] not in {"stop", "tool_calls"}:
                raise RuntimeError("Model completion malformed or truncated; not a transport retry")
            if id(response) in natives:
                self.replay.remember(response["choices"][0]["message"], natives[id(response)])
            return response

        def _call_once(self, body):
            self._require_sync_thread()
            with self.lock:
                if self.closed:
                    raise RuntimeError("provider is closed")
                if body.get("stream", False):
                    raise ValueError("Audited evaluation requests must be non-streaming")
                self.count += 1
                self.expected_model = body["model"]
                convert = anthropic_request if self.api == "anthropic-messages" else gemini_request
                self.expected, conversion = convert(body, self.params, self.replay)
                self._physical_attempt = 0
                write = model_api.provider_base.write
                write(self.directory / "requests" / f"{self.count:04d}.json", self.expected)
                write(self.directory / "conversion" / f"{self.count:04d}.json",
                      {"model_api": self.api, **conversion})
                events = []

                def record(row):
                    encoded = json.dumps(row)
                    for secret in self.keys:
                        encoded = encoded.replace(secret, "[REDACTED]")
                    row = json.loads(encoded)
                    events.append(row)
                    write(self.directory / "transport" / f"{self.count:04d}-{len(events):02d}.json", row)

                self._record = record
                try:
                    raw = model_api.retry.request_with_transient_retry(
                        create=self._create, body=self.expected, logical_turn=self.count,
                        policy=self.policy, record=record)
                finally:
                    self._record = None
                write(self.directory / "responses" / f"{self.count:04d}.json", raw)
                if self.api == "anthropic-messages":
                    response, native = anthropic_chat(raw)
                else:
                    response, native = gemini_chat(raw, self.count, self.id_prefix)
                self._native = native
                return response, raw

        def close(self):
            self._require_sync_thread()
            if not self.closed:
                self.loop.run_until_complete(self.http.aclose())
            super().close()

    return NativeProvider


def chat_provider_class():
    """openai-chat with frozen extra fields; reasoning_content replayed per turn unless drop_thinking=always."""
    from rescopebench.portable_provider_runtime import PortableChatProvider

    class ExtendedChatProvider(PortableChatProvider):
        def __init__(self, config, pool, directory, *, selection, policy):
            super().__init__(config, pool, directory, policy=policy)
            params = selection["params"]
            self.drop_thinking = params.get("drop_thinking", "never")
            if self.drop_thinking not in {"never", "always"}:
                raise ValueError("drop_thinking must be never or always")
            self.extra = {key: value for key, value in params.items() if key not in {"stream", "drop_thinking"}}
            self.reasoning = Replay()
            from rescopebench.agent_runtime import model_api
            model_api.provider_base.write(self.directory / "native-model-api.json", selection)

        def __call__(self, body):
            body = deepcopy(body)
            counts = defaultdict(int)
            for message in body["messages"]:
                if message.get("role") != "assistant":
                    continue
                if self.drop_thinking == "always":
                    message.pop("reasoning_content", None)
                elif "reasoning_content" not in message:
                    stored = self.reasoning.lookup(message, counts)
                    if stored:
                        message["reasoning_content"] = stored
            body.update(deepcopy(self.extra))

            def ask(asked):
                filtered = missing = 0
                while True:
                    try:
                        raw = super(ExtendedChatProvider, self).__call__(asked)
                    except RuntimeError as error:
                        # The frozen provider raises on any finish_reason other than
                        # stop/tool_calls, after writing the raw response.
                        if "Model completion malformed or truncated" not in str(error):
                            raise
                        raw = json.loads((self.directory / "responses" / f"{self.count:04d}.json").read_text())
                    choices = raw.get("choices") or []
                    finish = choices[0].get("finish_reason") if len(choices) == 1 else None
                    if finish == "content_filter":
                        # Upstream moderation occasionally rejects a turn that succeeds on resend.
                        if filtered == CONTENT_FILTER_RETRIES:
                            raise RuntimeError("Upstream moderation rejected the turn after retries")
                        filtered += 1
                        time.sleep(2.0 * filtered)
                        continue
                    if not finish:
                        # A stream cut before its finish reason: resend unchanged.
                        if missing == MODEL_RESAMPLES:
                            raise RuntimeError("Model completion malformed or truncated; not a transport retry")
                        missing += 1
                        record_resample(self.directory, self.count, missing, "missing_finish_reason", raw)
                        continue
                    # reasoning_content is already on the message, so a rejected call is
                    # replayed with its reasoning in the feedback round.
                    return raw, finish, raw

            raw, _ = answer_with_feedback(body, ask, feedback_recorder(self))
            normalize_arguments(self.directory, self.count, body, raw)
            message = raw["choices"][0]["message"]
            # vLLM 0.23 names the field "reasoning"; Qwen chat templates read reasoning_content.
            self.reasoning.remember(message, message.get("reasoning_content") or message.get("reasoning"))
            return raw

    return ExtendedChatProvider


class ResponsesGuard:
    """The frozen Responses provider with the same response checks as the other APIs.

    responses_provider.py is hashed into the plan and cannot change, so this wraps the
    instance instead. A resampled response is also removed from the provider's replay
    memory, so a discarded turn can never be replayed into a later request.
    """

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __enter__(self):
        self.inner.__enter__()
        return self

    def __exit__(self, *args):
        return self.inner.__exit__(*args)

    def _raw(self):
        """The raw Responses payload the provider just wrote for this logical turn."""
        return json.loads((self.inner.directory / "responses" / f"{self.inner.count:04d}.json").read_text())

    def _forget(self, response):
        from rescopebench import responses_provider
        key = responses_provider.assistant_key(response["choices"][0]["message"])
        outputs = self.inner.conversation.outputs.get(key)
        if outputs:
            outputs.pop()

    def __call__(self, body):
        from rescopebench import responses_provider

        def ask(asked):
            filtered = 0
            while True:
                try:
                    response = self.inner(asked)
                except RuntimeError as error:
                    # chat_response() refuses status != completed, before the reply is
                    # remembered. Moderation (content_filter) succeeds on resend; a reply
                    # cut at max_output_tokens is the model's and gets feedback.
                    if "Responses completion not complete" not in str(error):
                        raise
                    raw = self._raw()
                    reason = (raw.get("incomplete_details") or {}).get("reason")
                    if reason == "content_filter" and filtered < CONTENT_FILTER_RETRIES:
                        filtered += 1
                        time.sleep(2.0 * filtered)
                        continue
                    if reason != "max_output_tokens":
                        raise
                    cut = responses_provider.chat_response({**raw, "status": "completed"})
                    return cut, reason, raw
                remembered.add(id(response))
                return response, response["choices"][0]["finish_reason"], self._raw()

        remembered = set()
        response, rejected = answer_with_feedback(body, ask, feedback_recorder(self.inner))
        # Rejected replies leave the replay memory: Hermes never sends them back.
        for item in rejected:
            if id(item) in remembered:
                self._forget(item)
        before = deepcopy(response)
        if normalize_arguments(self.inner.directory, self.inner.count, body, response):
            # Re-key the replay memory under the arguments Hermes will send back.
            self._forget(before)
            self.inner.conversation.remember(self._raw(), response)
        return response


# ---------------------------------------------------------------- installation

def install() -> None:
    """Freeze the selection inside plan["transport"] and route providers by it."""
    from rescopebench import credentials
    from rescopebench import portable_provider_runtime as runtime
    from rescopebench.portable_provider_config import ProviderConfig, load_credentials

    if getattr(runtime, "_rescope_model_api_installed", False):
        return
    original_from_env = runtime.transport_from_environment
    original_validate = runtime.validate_transport
    original_provider = runtime.provider_from_transport

    def transport_from_environment(api="chat_completions", *, environ=None):
        record = original_from_env(api, environ=environ)
        selection = selection_from_environment(api, environ=environ)
        policy = dict(record["retry_policy"])
        policy["request_timeout_seconds"] = selection["request_timeout_seconds"]
        # 429, 5xx and timeouts are the provider's problem, not the model's: keep
        # retrying one turn until it gets an answer. The frozen policy needs finite
        # values, so the bound is the host's own 3h episode deadline; the counts are
        # set past what that window can hold so they never trip first.
        policy.update(rate_limit_retries=TRANSIENT_RETRIES, server_retries=TRANSIENT_RETRIES,
                      connection_retries=TRANSIENT_RETRIES, backoff_cap_seconds=120.0,
                      max_elapsed_seconds=EPISODE_DEADLINE_SECONDS)
        record = original_validate({**record, "retry_policy": policy})
        return {**record, "rescope_model_api": selection}

    def validate_transport(record):
        selection = record.get("rescope_model_api")
        expected = original_validate({key: value for key, value in record.items() if key != "rescope_model_api"})
        if selection is not None:
            if selection.get("schema") != SCHEMA or selection.get("model_api") not in MODEL_APIS:
                raise ValueError("Invalid frozen rescope_model_api")
            if (selection.get("implementation_sha256s") is not None
                    and selection["implementation_sha256s"] != implementation_hashes()):
                raise ValueError("Harness implementation changed; do not resume an old plan with different code")
            expected = {**expected, "rescope_model_api": deepcopy(selection)}
        return expected

    def provider_from_transport(directory, record, *, pool=None, credentials_file=None,
                                environ=None, client=None):
        frozen = validate_transport(record)
        selection = frozen.get("rescope_model_api")
        bare = {key: value for key, value in frozen.items() if key != "rescope_model_api"}
        if selection is None:
            return original_provider(directory, bare, pool=pool, credentials_file=credentials_file,
                                     environ=environ, client=client)
        if selection["model_api"] == "openai-responses":
            return ResponsesGuard(original_provider(directory, bare, pool=pool, credentials_file=credentials_file,
                                                    environ=environ, client=client))
        config = ProviderConfig.from_record(bare)
        if pool is None:
            pool = load_credentials(config, path=credentials_file, environ=environ)
        factory = chat_provider_class() if selection["model_api"] == "openai-chat" else native_provider_class()
        return factory(config, pool, directory, selection=selection, policy=bare["retry_policy"])

    runtime.transport_from_environment = transport_from_environment
    runtime.validate_transport = validate_transport
    runtime.provider_from_transport = provider_from_transport
    credentials.validate_transport = validate_transport
    credentials.provider_from_transport = provider_from_transport
    runtime._rescope_model_api_installed = True
