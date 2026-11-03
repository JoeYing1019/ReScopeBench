"""Apply the harness changes to Hermes inside the release image.

    python3 apply_hermes_patches.py /opt/hermes context_lengths.json

Each change replaces one exact snippet of the Hermes commit pinned by the release
(hermes_commit in rescopebench/vendor/image-manifest.json) and fails loudly if
that snippet is missing or repeated, so a different Hermes is never patched blindly.

1. Timeouts: Hermes' request timeout and its stale-call timeout default to 11000s.
   The host owns every wait: one model request may run RESCOPE_REQUEST_TIMEOUT_SECONDS,
   and 429 / 5xx / timeouts are retried until the release's 10800s episode deadline.
   Hermes' own table would abandon a silent reasoning call after 180-240s and resend
   it, so its defaults sit past the episode deadline. The worker starts under
   `env -i`, so HERMES_API_TIMEOUT in the environment would never reach it.
2. Kimi tool schemas: Hermes rewrites the tool schemas for any model whose name looks
   like Kimi/Moonshot. The release freezes the tool surface Hermes advertises and
   rejects the request as "Hermes changed the reviewed tool surface", so every Kimi
   task failed. The rewrite is switched off; Kimi sees the same tools as every model.
3. Context windows: Hermes sizes the context from a built-in table keyed by model
   name and falls back to a small window for names it does not know (the bridge has
   no /v1/models). context_lengths.json maps your model names to their windows; Hermes
   matches the longest key that is a substring of the model name.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

TIMEOUT = "11000"

EDITS = {
    "run_agent.py": [
        ("from agent.model_metadata import is_local_endpoint\n",
         "from agent.model_metadata import is_local_endpoint\n\n"
         "# rescope-harness: see _resolved_api_call_stale_timeout_base.\n"
         f"RESCOPE_STALE_TIMEOUT_SECONDS = \"{TIMEOUT}\"\n"),
        ('return cfg if cfg is not None else env_float("HERMES_API_TIMEOUT", 1800.0)\n',
         f'return cfg if cfg is not None else env_float("HERMES_API_TIMEOUT", {TIMEOUT}.0)'
         "  # rescope-harness: the host owns the wait\n"),
        ('        env_timeout = os.getenv("HERMES_API_CALL_STALE_TIMEOUT")\n',
         "        # rescope-harness: the host applies its own request timeout and retries 429/5xx\n"
         "        # until the 10800s episode deadline. Default the stale timeout past that window\n"
         "        # so Hermes never abandons and resends a call the host is still serving. The\n"
         "        # reasoning-model table stays untouched: the error classifier still reads it.\n"
         '        env_timeout = os.getenv("HERMES_API_CALL_STALE_TIMEOUT", RESCOPE_STALE_TIMEOUT_SECONDS)\n'),
        ('                or os.getenv("HERMES_API_CALL_STALE_TIMEOUT") is not None)\n',
         '                or os.getenv("HERMES_API_CALL_STALE_TIMEOUT", RESCOPE_STALE_TIMEOUT_SECONDS) is not None)\n'),
    ],
    "agent/chat_completion_helpers.py": [
        ('base = cfg if cfg is not None else env_float("HERMES_API_TIMEOUT", 1800.0)\n',
         f'base = cfg if cfg is not None else env_float("HERMES_API_TIMEOUT", {TIMEOUT}.0)'
         "  # rescope-harness: the host owns the wait\n"),
    ],
    "agent/transports/chat_completions.py": [
        ('        api_kwargs["tools"] = sanitize_moonshot_tools(tools) if is_moonshot_model(model) else tools\n',
         "        # rescope-harness: the benchmark freezes the tool schemas Hermes advertises;\n"
         "        # rewriting them for Kimi makes the worker reject the request as a changed tool surface.\n"
         '        api_kwargs["tools"] = tools\n'),
    ],
}


def load_context_lengths(context_lengths: Path) -> dict[str, int]:
    values = json.loads(context_lengths.read_text())
    if not isinstance(values, dict):
        raise SystemExit(f"{context_lengths}: expected {{\"model-name\": tokens, ...}}")
    # Underscore-prefixed entries are documentation, not model configurations.
    windows = {key.lower(): value for key, value in values.items() if not key.startswith("_")}
    if not all(key.strip() and type(value) is int and value > 0 for key, value in windows.items()):
        raise SystemExit(f"{context_lengths}: model windows must be positive integers")
    return windows


def main(hermes: Path, context_lengths: Path) -> None:
    windows = load_context_lengths(context_lengths)
    for relative, edits in EDITS.items():
        path = hermes / relative
        text = path.read_text()
        for old, new in edits:
            if text.count(old) != 1:
                raise SystemExit(f"{path}: expected exactly one match of {old.strip()[:80]!r}, found {text.count(old)}")
            text = text.replace(old, new)
        path.write_text(text)
        print(f"patched {relative}: {len(edits)} edit(s)")
    metadata = hermes / "agent" / "model_metadata.py"
    text = metadata.read_text()
    if "DEFAULT_CONTEXT_LENGTHS" not in text:
        raise SystemExit(f"{metadata}: DEFAULT_CONTEXT_LENGTHS not found")
    if windows:
        text += ("\n\n# rescope-harness: context windows of the evaluated models. Full names win\n"
                 "# Hermes' longest-key match over the built-in entries.\n"
                 f"DEFAULT_CONTEXT_LENGTHS.update({json.dumps(windows, sort_keys=True)})\n")
        metadata.write_text(text)
    print(f"context windows: {len(windows)} model name(s)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
