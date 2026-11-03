"""Minimal model/deployment availability check with secret-free receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescopebench.credentials import load_credentials
from rescopebench import responses_provider
from rescopebench.stream_engine import _provider, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--provider-api", choices=("chat_completions", "responses"), default="chat_completions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    transport = responses_provider.transport(args.provider_api)
    slots = load_credentials(args.credentials_file, transport=transport)
    args.output.mkdir(parents=True, exist_ok=False)
    provider = _provider(args.output / "provider", transport)
    try:
        response = provider({"model": args.model, "messages": [{"role": "user", "content": "What is 1+1? Reply with the number only."}],
                             "max_tokens": 256, "stream": False, "parallel_tool_calls": True, "tools": []})
        message = response.get("choices", [{}])[0].get("message", {})
        result = {"status": "passed", "requested_model": args.model, "returned_model": response.get("model"),
                  "credential_slots": slots, "content": message.get("content"), "secrets_persisted": False}
        write_json(args.output / "result.json", result)
        print(json.dumps(result, ensure_ascii=True))
    finally:
        close = getattr(provider, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
