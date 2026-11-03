"""Pinned official Hermes schemas; business/history tools remain benchmark-owned."""

from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath


HISTORY_NAMES = {"episode_list", "episode_read", "record_search", "record_read"}
NATIVE_NAMES = {"skills_list", "skill_view", "skill_manage", "memory"}
UPSTREAM = "345cd2b057a452236de401d3534b8502a7465e8d"


def function(name, description, properties, required=()):
    params = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        params["required"] = list(required)
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": params}}


# BEGIN GENERATED UPSTREAM SCHEMAS
NATIVE_TOOLS = json.loads(r'''[
  {
    "type": "function",
    "function": {
      "name": "skills_list",
      "description": "List available skills (name + description). Use skill_view(name) to load full content.",
      "parameters": {
        "type": "object",
        "properties": {
          "category": {
            "type": "string",
            "description": "Optional category filter to narrow results"
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "skill_view",
      "description": "Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter.",
      "parameters": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "The skill name (use skills_list to see available skills). For plugin-provided skills, use the qualified form 'plugin:skill' (e.g. 'superpowers:writing-plans')."
          },
          "file_path": {
            "type": "string",
            "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content."
          }
        },
        "required": [
          "name"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "skill_manage",
      "description": "Create, update, or delete skills \u2014 your procedural memory for recurring task types. The call is an operations array (a single edit is a list of one); it applies atomically \u2014 any failure rolls every touched skill back. Ops: create (full SKILL.md; lands in the profile's skills.create_dir; must precede that skill's other ops), patch (targeted old_string/new_string fix \u2014 preferred; content alone REPLACES the whole file, read it via skill_view() first), write_file/remove_file (supporting files), delete (sole op only). Existing skills are modified wherever they live. Keep the description's first 57 chars a self-contained trigger: 'Use when <trigger>. <one-line behavior>.' Write lessons, not logs: imperative rule + why, no PR numbers/dates/incident narration, one rule per lesson, references/ named by topic (extend before adding). skill_view() shows format conventions.",
      "parameters": {
        "type": "object",
        "properties": {
          "operations": {
            "type": "array",
            "description": "Ordered ops; each names its target skill.",
            "items": {
              "type": "object",
              "properties": {
                "name": {
                  "type": "string",
                  "description": "Skill name (lowercase, hyphens/underscores, max 64 chars); an existing skill's name unless creating."
                },
                "action": {
                  "type": "string",
                  "enum": [
                    "create",
                    "patch",
                    "delete",
                    "write_file",
                    "remove_file"
                  ]
                },
                "content": {
                  "type": "string",
                  "description": "Full SKILL.md text (YAML frontmatter + markdown body) for create, or a full rewrite on patch."
                },
                "category": {
                  "type": "string",
                  "description": "Optional category subdir for create (e.g. 'devops')."
                },
                "old_string": {
                  "type": "string",
                  "description": "Text to find (patch; same matching semantics as the patch tool)."
                },
                "new_string": {
                  "type": "string",
                  "description": "Replacement (patch); empty string deletes the match."
                },
                "replace_all": {
                  "type": "boolean",
                  "description": "patch: replace all occurrences (default false)."
                },
                "file_path": {
                  "type": "string",
                  "description": "Path RELATIVE to the skill's own directory, e.g. 'references/api.md' \u2014 no leading slash, never absolute. write_file/remove_file: required; first segment references/, templates/, scripts/, or assets/. patch: optional (default SKILL.md)."
                },
                "file_content": {
                  "type": "string",
                  "description": "Content for write_file."
                }
              },
              "required": [
                "name",
                "action"
              ]
            }
          }
        },
        "required": [
          "operations"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "memory",
      "description": "Save durable facts to persistent memory that survive across sessions. Memory is injected into every future turn, so keep entries compact and high-signal.\n\nHOW: make ALL your changes in ONE call via an 'operations' array (each item: {action, content?, old_text?}). The batch applies atomically and the char limit is checked only on the FINAL result \u2014 so a single call can remove/replace stale entries to free room AND add new ones, even when an add alone would overflow. The response reports current/limit chars and confirms completion; one batch call finishes the update, so don't repeat it. Use the bare action/content/old_text fields only for a single lone change.\n\nWHEN: only for facts that apply to EVERY session regardless of task: who the user is, stable environment facts, standing conventions with no task home. Anything learned while doing a task (procedures, pitfalls, and the user's preferences and corrections for that kind of work) belongs in the task's skill via skill_manage, where it loads only when relevant; memory is injected into every turn and must stay small.\n\nIF FULL: an add is rejected with the current entries shown. Reissue as ONE batch that removes or shortens enough stale entries and adds the new one together.\n\nTARGETS: 'user' = who the user is (name, role, preferences, style). 'memory' = your notes (environment, conventions, tool quirks, lessons).\n\nSKIP: trivial/obvious info, easily re-discovered facts, raw data dumps, task progress, completed-work logs, temporary TODO state (use session_search for those). Reusable procedures belong in a skill, not memory.",
      "parameters": {
        "type": "object",
        "properties": {
          "action": {
            "type": "string",
            "enum": [
              "add",
              "replace",
              "remove"
            ],
            "description": "The action to perform (single-op shape). Omit when using 'operations'."
          },
          "target": {
            "type": "string",
            "enum": [
              "memory",
              "user"
            ],
            "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
          },
          "content": {
            "type": "string",
            "description": "The entry content. Required for 'add' and 'replace' (single-op shape). Alias: 'new_text' is also accepted (mirrors old_text)."
          },
          "old_text": {
            "type": "string",
            "description": "REQUIRED for 'replace' and 'remove' (single-op shape): a short unique substring identifying the existing entry to modify. Omit only for 'add'."
          },
          "new_text": {
            "type": "string",
            "description": "Alias for 'content' (single-op shape). Provided so the replace/remove old_text/new_text pairing works; if both are set, 'content' wins."
          },
          "operations": {
            "type": "array",
            "description": "Batch shape: a list of operations applied atomically in one call against the final char budget. Preferred when making multiple changes or consolidating to make room. Each item is {action, content?, old_text?}.",
            "items": {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "add",
                    "replace",
                    "remove"
                  ]
                },
                "content": {
                  "type": "string",
                  "description": "Entry content for add/replace. Alias: 'new_text'."
                },
                "new_text": {
                  "type": "string",
                  "description": "Alias for 'content' in a batch op."
                },
                "old_text": {
                  "type": "string",
                  "description": "Substring identifying the entry for replace/remove."
                }
              },
              "required": [
                "action"
              ]
            }
          }
        },
        "required": [
          "target"
        ]
      }
    }
  }
]''')
# END GENERATED UPSTREAM SCHEMAS

def validate_assets(assets):
    """Integrity/path checks, not a new content-quality gate on native writes."""
    if not isinstance(assets, dict):
        raise ValueError("Assets must be a mapping")
    for name, item in assets.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or str(path) != name or "\\" in name or ".." in path.parts
                or len(path.parts) < 2 or path.parts[0] not in {"skills", "memories"}):
            raise ValueError("Unsafe native asset path")
        if (not isinstance(item, dict) or set(item) != {"content", "sha256"}
                or not isinstance(item["content"], str)
                or hashlib.sha256(item["content"].encode()).hexdigest() != item["sha256"]):
            raise ValueError("Native asset content/hash mismatch")
    return deepcopy(assets)


def project(tools):
    result = deepcopy(tools)
    for tool in result:
        if tool["function"]["name"] in NATIVE_NAMES:
            continue
        params = tool["function"]["parameters"]
        if params.get("required") == []:
            params.pop("required")
    return result
