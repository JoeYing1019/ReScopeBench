"""Stateless Responses transport for the frozen native Hermes chat bridge.

All provider output items are retained and replayed when the corresponding
assistant message survives in Hermes context, including opaque reasoning items.
"""

import asyncio
from collections import defaultdict
from copy import deepcopy
import json
import os

import httpx

from rescopebench.agent_runtime import model_api
from rescopebench import credentials




def transport(api="chat_completions"):
    from rescopebench.portable_provider_runtime import transport_from_environment
    return transport_from_environment(api)


def validate_transport(value):
    from rescopebench.portable_provider_runtime import validate_transport as validate
    return validate(value)


def assistant_key(message):
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


def chat_response(raw):
    if raw.get("status") != "completed":
        raise RuntimeError("Responses completion not complete; raw response retained")
    calls, text = [], []
    for item in raw.get("output", []):
        if item.get("type") == "function_call":
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item["name"], "arguments": item["arguments"]}})
        elif item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text.append(part["text"])
                elif part.get("type") == "refusal":
                    text.append(part["refusal"])
    message = {"role": "assistant", "content": "\n".join(text) or None}
    if calls:
        message["tool_calls"] = calls
    usage = raw.get("usage") or {}
    return {"id": raw["id"], "object": "chat.completion", "model": raw["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": usage.get("input_tokens", 0),
                      "completion_tokens": usage.get("output_tokens", 0),
                      "total_tokens": usage.get("total_tokens", 0),
                      "completion_tokens_details": usage.get("output_tokens_details", {})}}


class Conversation:
    def __init__(self):
        self.outputs = defaultdict(list)

    def remember(self, raw, response):
        key = assistant_key(response["choices"][0]["message"])
        self.outputs[key].append(deepcopy(raw["output"]))

    def request(self, body, config):
        items, counts, replayed, synthetic = [], defaultdict(int), 0, 0
        for message in body["messages"]:
            role = message["role"]
            if role == "tool":
                items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                              "output": message.get("content") or ""})
            elif role == "assistant":
                key = assistant_key(message)
                index = counts[key]
                counts[key] += 1
                if index < len(self.outputs[key]):
                    items.extend(deepcopy(self.outputs[key][index]))
                    replayed += 1
                else:
                    # Hermes may insert or compact assistant context itself.
                    # Do not invent missing model reasoning for those messages.
                    synthetic += 1
                    if message.get("content"):
                        items.append({"role": "assistant", "content": message["content"]})
                    for call in message.get("tool_calls") or []:
                        items.append({"type": "function_call", "call_id": call["id"],
                                      **deepcopy(call["function"])})
            else:
                items.append({"role": role, "content": deepcopy(message.get("content") or "")})
        request = {"model": body["model"], "input": items, "stream": False,
                   "store": config["store"], "reasoning": deepcopy(config["reasoning"]),
                   "max_output_tokens": body["max_tokens"],
                   "parallel_tool_calls": body["parallel_tool_calls"],
                   "tools": [{"type": "function", **deepcopy(tool["function"]),
                              "strict": config["tool_schema_strict"]} for tool in body["tools"]]}
        return request, {"replayed_assistant_outputs": replayed, "synthetic_or_compacted_assistants": synthetic,
                         "seed_omitted": body.get("seed"), "all_function_calls_preserved": True}


class ResponsesProvider(model_api.provider_base.Provider):
    def __init__(self, key, directory, *, config=None, client=None):
        self.config = validate_transport(config or transport("responses"))
        self.conversation = Conversation()
        super().__init__(key, directory, base_url=self.config["endpoint"],
                         policy=self.config["retry_policy"], client=client)

    async def _attempt(self, body, timeout_seconds):
        self._physical_attempt += 1
        timeout = httpx.Timeout(timeout_seconds, connect=min(30, timeout_seconds))
        return await asyncio.wait_for(self.client.with_options(timeout=timeout, max_retries=0)
                                      .responses.create(**body), timeout_seconds)

    def __call__(self, body):
        self._require_sync_thread()
        with self.lock:
            if self.closed:
                raise RuntimeError("provider is closed")
            self.count += 1
            self.expected, conversion = self.conversation.request(body, self.config)
            self._physical_attempt = 0
            write = model_api.provider_base.write
            write(self.directory / "requests" / f"{self.count:04d}.json", self.expected)
            write(self.directory / "conversion" / f"{self.count:04d}.json", conversion)
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
                value = model_api.retry.request_with_transient_retry(
                    create=self._create, body=self.expected, logical_turn=self.count,
                    policy=self.policy, record=record)
            finally:
                self._record = None
            raw = value.model_dump(mode="json", exclude_none=True)
            write(self.directory / "responses" / f"{self.count:04d}.json", raw)
            response = chat_response(raw)
            self.conversation.remember(raw, response)
            return response


def provider(directory, config):
    return credentials.provider(directory, config)
