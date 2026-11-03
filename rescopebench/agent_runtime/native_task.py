"""Sandbox bridge validation and exact pinned worker mounts."""

from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import re
import jsonschema
from . import surface
from rescopebench import pathing

ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "workers/worker.py"
NATIVE_BASE = ROOT / "workers/worker_native.py"
WORKER_BASE = ROOT / "workers/worker_base.py"
ADOPTION_FILE = ROOT / "workers/adoption.py"
SEMANTIC_LIMIT = ROOT / "workers/semantic_limit.py"
SURFACE_FILE = ROOT / "surface.py"
TOOL_EXECUTION = pathing.FINAL_ROOT / "tool_execution.py"
COMPLETION_EVIDENCE = pathing.FINAL_ROOT / "completion_evidence.py"

CYCLE_NOTE = re.compile('\\n\\n\\[hermes note: the last (\\d+) rounds repeated the same cycle of (\\d+) tool calls \\(ending with ([A-Za-z_][A-Za-z0-9_]*)\\) with identical arguments and identical results\\. Do not repeat the batch \\u2014 change arguments, use a different tool, or proceed with what you have\\.\\]')
SAME_TOOL_FAILURE_NOTE = re.compile('\\n\\n\\[Tool loop warning: same_tool_failure_warning; count=(\\d+); ([A-Za-z_][A-Za-z0-9_]*) has failed (\\d+) times this turn\\. This looks like a loop\\. Do not switch to text-only replies; keep using tools, but diagnose before retrying\\. First inspect the latest error/output and verify your assumptions\\. Try different arguments, a narrower query/path, an absolute path when relevant, or a different tool that can make progress\\. If the blocker is external, report the blocker after one diagnostic attempt instead of repeating the same failing path\\.\\]')

def called_tool_names(message):
    result = set()
    direct_name = (message or {}).get('name')
    if (message or {}).get('role') in {'tool', 'function'} and isinstance(direct_name, str):
        result.add(direct_name)
    legacy = (message or {}).get('function_call') or {}
    legacy_name = legacy.get('name')
    if isinstance(legacy_name, str):
        result.add(legacy_name)
    for call in (message or {}).get('tool_calls') or []:
        function = call.get('function') or {}
        name = function.get('name')
        if isinstance(name, str):
            result.add(name)
    return result

def provider_response_tool_names(response):
    result = set()
    for choice in response.get('choices') or []:
        result.update(called_tool_names(choice.get('message') or {}))
    return result

def validate_provider_response(response, expected_tools):
    allowed = {schema['function']['name'] for schema in expected_tools}
    unexpected = provider_response_tool_names(response) - allowed
    if unexpected:
        raise ValueError('Provider response attempted an undeclared tool call: ' + ', '.join(sorted(unexpected)))
    choices = response.get('choices') or []
    if len(choices) != 1 or choices[0].get('finish_reason') not in {'stop', 'tool_calls'}:
        raise ValueError('Provider response has an unaccepted finish reason')
    finish_reason = choices[0]['finish_reason']
    calls = provider_response_tool_names(response)
    if (finish_reason == 'tool_calls') != bool(calls):
        raise ValueError('Provider finish reason disagrees with its tool calls')

def validate_exchange_complete(exchange_dir, seen):
    exchange_dir = Path(exchange_dir)
    requests = {path.name for path in (exchange_dir / 'requests').glob('*.json')}
    responses = {path.name for path in (exchange_dir / 'responses').glob('*.json')}
    temporary = sorted((str(path.relative_to(exchange_dir)) for path in exchange_dir.rglob('*.tmp')))
    if temporary:
        raise ValueError('Read-only bridge retained temporary exchange files')
    if requests != set(seen) or responses != set(seen):
        raise ValueError('Read-only bridge has an unaccounted request or response')
    return {'request_count': len(requests), 'response_count': len(responses)}

def _tool_names(tools):
    names = []
    for schema in tools:
        if not isinstance(schema, dict) or schema.get('type') != 'function':
            raise ValueError('Learning tool schema must be a function declaration')
        function = schema.get('function')
        name = function.get('name') if isinstance(function, dict) else None
        parameters = function.get('parameters') if isinstance(function, dict) else None
        if not isinstance(name, str) or not name or (not isinstance(parameters, dict)):
            raise ValueError('Learning tool schema is incomplete')
        jsonschema.Draft7Validator.check_schema(parameters)
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError('Learning tool schemas contain duplicate names')
    return names

def validate_provider_request(body, expected_tools):
    if body.get('tools') != expected_tools:
        raise ValueError('Provider request changed the learning tool surface')
    declared = set(_tool_names(expected_tools))
    historical = set()
    for message in body.get('messages') or []:
        historical.update(called_tool_names(message))
    unexpected = historical - declared
    if unexpected:
        raise ValueError('Provider history contains an undeclared tool call: ' + ', '.join(sorted(unexpected)))

def decode_business_result(content, name):
    if not isinstance(content, str):
        raise ValueError('Hermes business result must contain JSON')
    try:
        content = content.lstrip()
        (value, end) = json.JSONDecoder().raw_decode(content)
    except json.JSONDecodeError as error:
        raise ValueError('Hermes business result must contain JSON') from error
    return (value, bool(content[end:].strip()))

def docker_command(public, exchange, name):
    return [
        'docker',
        'run',
        '--rm',
        '--name',
        name,
        '--network=none',
        '--read-only',
        '--memory=512m',
        '--cpus=0.5',
        '--pids-limit=96',
        '--cap-drop=ALL',
        '--security-opt=no-new-privileges',
        '--user=1000:1000',
        '--tmpfs',
        '/tmp:rw,nosuid,nodev,size=64m,mode=1777',
        '--tmpfs',
        '/work:rw,nosuid,nodev,size=128m,mode=1777',
        '-e',
        'PYTHONDONTWRITEBYTECODE=1',
        '-e',
        'PYTHONUNBUFFERED=1',
        '-v',
        str(public) + ':/public:ro',
        '-v',
        str(exchange) + ':/exchange:rw',
        '-v',
        str(ROOT / 'workers/worker_base.py') + ':/driver_base.py:ro',
        '-v',
        str(ROOT / 'workers/worker.py') + ':/driver.py:ro',
        '-v',
        str(ROOT / 'surface.py') + ':/driver_surface.py:ro',
        '-v',
        str(ROOT / 'workers/adoption.py') + ':/driver_adoption.py:ro',
        '-e',
        'HOME=/work',
        '-e',
        'HERMES_HOME=/work/profile',
        '-v',
        str(ROOT / 'workers/worker_native.py') + ':/driver_native_base.py:ro',
        '-v',
        str(ROOT / 'workers/semantic_limit.py') + ':/driver_semantic_limit.py:ro',
        '-v',
        str(TOOL_EXECUTION) + ':/driver_tool_execution.py:ro',
        '-v',
        str(COMPLETION_EVIDENCE) + ':/driver_completion_evidence.py:ro',
        '--entrypoint',
        '/opt/hermes/.venv/bin/python',
        'sha256:596bcba766f9ab30aaab081a5d513d6deda96ba25342b337d5145a0e875b217b',
        '/driver.py'
    ]
