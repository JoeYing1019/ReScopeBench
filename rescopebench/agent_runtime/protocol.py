"""Provider/trajectory contracts and the selected host episode loop."""

from __future__ import annotations
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
import uuid
import jsonschema
from rescopebench import tool_execution
from . import core, surface, native_task, history as native_history
from .profile import MODEL, SEED, MAX_TOKENS, PROTOCOL, ADOPTION_ALLOWLIST, MAX_ITERATIONS, PARALLEL_TOOL_CALLS, PROFILE, SYSTEM, PHASES, TASK_PREFIX, SAVE_PROMPT, REFLECT_PREFIX


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write('\n')


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def tool_parameters(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(tools, list) or not tools:
        raise ValueError('Adapter must declare a nonempty tool-schema list')
    result = {}
    for tool in tools:
        function = tool.get('function', {}) if isinstance(tool, dict) else {}
        (name, params) = (function.get('name'), function.get('parameters'))
        if not isinstance(tool, dict) or tool.get('type') != 'function' or (not isinstance(name, str)) or (not re.fullmatch('[A-Za-z_][A-Za-z0-9_-]{0,63}', name)) or (name in result) or (not isinstance(params, dict)) or (not isinstance(function.get('description'), str)) or (not function['description'].strip()):
            raise ValueError('Invalid, duplicate, or incomplete public tool schema')
        jsonschema.Draft7Validator.check_schema(params)
        result[name] = deepcopy(params)
    return result


def validate_arguments(parameters: dict[str, Any], arguments: Any) -> tuple[bool, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    try:
        jsonschema.validate(arguments, parameters)
    except jsonschema.ValidationError:
        return (False, _json_copy(arguments))
    return (True, _json_copy(arguments))


def public_task_text(task: str | dict[str, Any]) -> str:
    query = task if isinstance(task, str) else task.get('query') if isinstance(task, dict) else None
    if not isinstance(query, str) or not query.strip():
        raise ValueError('task must be a nonempty string or a mapping with a nonempty query')
    return query


def history_count(handler: Any) -> int | None:
    if handler is None:
        return 0
    count = getattr(handler, 'episode_count', None)
    if type(count) is int and count >= 0:
        return count
    bundle = getattr(handler, 'bundle', None)
    if isinstance(bundle, dict) and isinstance(bundle.get('episodes'), list):
        return len(bundle['episodes'])
    return None


def expected_tools(packet: dict[str, Any]) -> list[dict[str, Any]]:
    return surface.project(packet['business_tools'] + packet['history_tools'] + surface.NATIVE_TOOLS)


def validate_surface(actual: dict[str, Any], packet: dict[str, Any]) -> None:
    expected = {'tools': expected_tools(packet), 'entry_assets': packet['assets'], 'upstream': surface.UPSTREAM, 'skip_memory': False, 'skip_background_review': False, 'native_handlers': True, 'skill_nudge_interval': 10, 'curator_adoption_allowlist': ADOPTION_ALLOWLIST}
    if any((actual.get(key) != value for (key, value) in expected.items())):
        raise ValueError('Native Hermes surface, entry assets, or runtime flags changed')


def provider_body(raw: dict[str, Any], tools: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Apply only the declared provider token-field compatibility conversion."""
    allowed = {'model', 'messages', 'tools', 'seed', 'max_tokens', 'max_completion_tokens', 'stream', 'parallel_tool_calls', 'tool_choice'}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError('Unreviewed provider fields; refusing silent configuration overrides')
    if raw.get('tools') != tools or raw.get('model') != MODEL or raw.get('seed') != SEED:
        raise ValueError('Provider model, seed, or compact tools differ from the frozen contract')
    token_fields = [key for key in ('max_tokens', 'max_completion_tokens') if key in raw]
    if not token_fields or any((raw[key] != MAX_TOKENS for key in token_fields)):
        raise ValueError('Provider token budget differs from the frozen contract')
    if raw.get('stream', False) is not False:
        raise ValueError('Streaming is not part of the frozen evaluation contract')
    if raw.get('parallel_tool_calls') is not PARALLEL_TOOL_CALLS:
        raise ValueError('parallel_tool_calls must match the frozen execution mode')
    if raw.get('tool_choice', 'auto') != 'auto':
        raise ValueError('Forced tool choice is not part of the frozen evaluation contract')
    native_task.validate_provider_request(raw, tools)
    (result, changes) = (deepcopy(raw), [])
    if 'max_completion_tokens' in result:
        result.pop('max_completion_tokens')
        result['max_tokens'] = MAX_TOKENS
        changes.append('max_completion_tokens_to_max_tokens')
    if 'stream' not in result:
        result['stream'] = False
        changes.append('explicit_non_streaming_default')
    return (result, changes)


def normalize_serial_tool_response(response: dict[str, Any], *, parallel: bool=False) -> tuple[dict[str, Any], int]:
    """Retain all batched calls; preserve the historical serial fallback."""
    normalized = deepcopy(response)
    if parallel:
        return (normalized, 0)
    choices = normalized.get('choices') if isinstance(normalized, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        return (normalized, 0)
    message = choices[0].get('message')
    tool_calls = message.get('tool_calls') if isinstance(message, dict) else None
    if not isinstance(tool_calls, list) or len(tool_calls) <= 1:
        return (normalized, 0)
    message['tool_calls'] = tool_calls[:1]
    return (normalized, len(tool_calls) - 1)


def _assistant(message: dict[str, Any]) -> dict[str, Any]:
    value = {'role': message.get('role'), 'content': message.get('content') or ''}
    calls = []
    for call in message.get('tool_calls') or []:
        raw_function = call.get('function') or {}
        function = {key: raw_function.get(key) for key in ('name', 'arguments')}
        arguments = function.get('arguments')
        if isinstance(arguments, str):
            try:
                function['arguments'] = core.encoded(json.loads(arguments))
            except json.JSONDecodeError:
                pass
        calls.append({'id': call.get('id'), 'type': call.get('type'), 'function': function})
    if calls:
        value['tool_calls'] = calls
    return value


def validate_trajectory_binding(messages: list[dict[str, Any]], events: list[dict[str, Any]], responses: list[dict[str, Any]], tools: list[dict[str, Any]], *, decoder=None) -> dict[str, Any]:
    decoder = native_task.decode_business_result if decoder is None else decoder
    parameters = tool_parameters(tools)
    actual = [_assistant(message) for message in messages if message.get('role') == 'assistant']
    expected = []
    for response in responses:
        choices = response.get('choices') or []
        if len(choices) != 1:
            raise ValueError('Provider response must have one choice')
        expected.append(_assistant(choices[0].get('message') or {}))
    actual_actions = [{key: value for (key, value) in row.items() if key != 'content'} for row in actual]
    expected_actions = [{key: value for (key, value) in row.items() if key != 'content'} for row in expected]
    if actual_actions != expected_actions:
        raise ValueError('Foreground assistant messages differ from actual provider responses')
    (pending, bound, warnings, seen_ids, pre_dispatch) = ({}, [], 0, set(), [])
    for message in messages:
        for call in message.get('tool_calls') or []:
            function = call.get('function') or {}
            name = function.get('name')
            if name not in parameters:
                continue
            call_id = call.get('id')
            if not isinstance(call_id, str) or call_id in seen_ids:
                raise ValueError('Invalid or duplicate business tool call ID')
            seen_ids.add(call_id)
            (_, arguments) = validate_arguments(parameters[name], function.get('arguments'))
            pending[call_id] = {'name': name, 'arguments': arguments}
        if message.get('role') != 'tool':
            continue
        call_id = message.get('tool_call_id')
        if call_id not in pending:
            if message.get('name') in parameters:
                raise ValueError('Business tool result has no matching call')
            continue
        call = pending.pop(call_id)
        if message.get('name') != call['name']:
            raise ValueError('Business tool result name differs from its call')
        if not isinstance(call['arguments'], dict):
            expected_error = {'error': 'Invalid tool arguments', 'message': 'Tool arguments must be a valid JSON object; tool was not executed.'}
            try:
                parsed = json.loads(message.get('content', ''))
            except (TypeError, ValueError):
                parsed = None
            if parsed != expected_error:
                raise ValueError('Invalid-argument result differs from pinned Hermes rejection')
            pre_dispatch.append({'tool_call_id': call_id, **call, 'response': parsed})
            continue
        (result, warned) = decoder(message.get('content'), call['name'])
        warnings += int(warned)
        bound.append({**call, 'response': result})
    if pending:
        raise ValueError('Business call has no returned result')
    normalized = []
    for (index, event) in enumerate(events, 1):
        if event.get('event_id') != f'e{index:04d}':
            raise ValueError('Business event identifiers are not consecutive')
        normalized.append({key: event.get(key) for key in ('name', 'arguments', 'response')})
    if normalized != bound:
        raise ValueError('Foreground business evidence differs from the host event trace')
    return {'provider_responses_bound': True, 'business_events_bound': True, 'assistant_messages': len(actual), 'business_events': len(bound), 'tool_loop_warnings': warnings, 'trajectory_sha256': core.digest(messages), 'business_trace_sha256': core.digest(events), 'pre_dispatch_rejections': pre_dispatch}


def execution_records(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain new business schemas without importing the old six-tool filter."""
    (names, pending, result) = (set(tool_parameters(tools)), set(), [])
    for message in messages:
        if message.get('role') == 'assistant':
            calls = [deepcopy(call) for call in message.get('tool_calls') or [] if call.get('function', {}).get('name') in names]
            pending.update((call['id'] for call in calls))
            row = {'role': 'assistant'}
            if message.get('content'):
                row['content'] = message['content']
            if calls:
                row['tool_calls'] = calls
            if len(row) > 1:
                result.append(row)
        elif message.get('role') == 'tool' and message.get('tool_call_id') in pending:
            result.append(deepcopy(message))
            pending.remove(message['tool_call_id'])
    return result


def public_feedback(grade: dict[str, Any]) -> dict[str, Any]:
    """Expose outcome predicates, never private targets or corrective answers."""
    score = grade.get('score')
    if type(score) not in (int, float) or not math.isfinite(score) or (not 0 <= score <= 100):
        raise ValueError('Grade must contain a finite score from 0 to 100')
    if 'public_components' in grade:
        return core.outcome({'score': score, 'components': grade['public_components']}, 'components')
    components = {}
    for (index, row) in enumerate(grade.get('orders', grade.get('items', [])), 1):
        if not isinstance(row, dict):
            raise ValueError('Grade order/items entries must be mappings')
        for (public_key, source_key) in (('physical_delivery', 'physical_delivery'), ('workflow_completed', 'workflow_completed' if 'workflow_completed' in row else 'closed')):
            if source_key in row:
                if type(row[source_key]) is not bool:
                    raise ValueError('Public artifact predicates must be boolean')
                components[f'item_{index}_{public_key}'] = {'correct': int(row[source_key]), 'total': 1}
    if not components:
        raise ValueError('Grade must declare public_components or per-item physical_delivery/workflow_completed predicates')
    return core.outcome({'score': score, 'components': components}, 'components')


def _history_call(handler: Any, name: str, arguments: dict[str, Any]) -> Any:
    if handler is not None:
        callback = handler.call if hasattr(handler, 'call') else handler
        return _json_copy(callback(name, deepcopy(arguments)))
    if name == 'episode_list':
        return {'episodes': [], 'next_offset': None, 'total': 0}
    if name == 'record_search':
        return {'records': [], 'next_offset': None, 'total': 0}
    return {'error': 'NOT_FOUND', 'message': 'No completed episode records are available in this run.'}


def source_snapshot(adapter: Any, provider: Any=None) -> dict[str, Any]:
    paths = {Path(__file__), Path(native_task.__file__), Path(surface.__file__), Path(native_history.__file__), native_task.WORKER, native_task.WORKER_BASE, native_task.ADOPTION_FILE}
    for value in (adapter, provider):
        try:
            filename = inspect.getsourcefile(type(value))
        except (TypeError, OSError):
            filename = None
        if filename:
            paths.add(Path(filename))
    dependencies = getattr(adapter, 'source_files', [])
    dependencies = dependencies() if callable(dependencies) else dependencies
    for path in dependencies:
        paths.add(Path(path))
    files = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths) if path.is_file()}
    return {'files': files, 'sha256': core.digest(files)}


def _safe_error(error: BaseException, provider: Any) -> str:
    message = str(error)
    candidates = [os.environ.get('RESCOPE_PROVIDER_API_KEY'), os.environ.get('RESCOPE_PROVIDER_API_KEY_SECONDARY'), getattr(getattr(provider, 'client', None), 'api_key', None)]
    if provider is not None and callable(getattr(provider, 'secret_values', None)):
        candidates.extend(provider.secret_values())
    for secret in candidates:
        if isinstance(secret, str) and secret:
            message = message.replace(secret, '[redacted]')
    return message[:1800]


def execute_episode(directory: Path, task: str | dict[str, Any], adapter: Any, assets: dict[str, Any], provider: Any, *, history_handler: Any=None, condition: str='native_categorized_skills_reflect') -> dict[str, Any]:
    """Run against an isolated attempt adapter; the caller owns checkpoint promotion.

    Adapter errors and exhausted resource responses are task observations. Raised
    infrastructure/configuration errors preserve evidence and are never scores.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (public, exchange) = (directory / 'public', directory / 'exchange')
    public.mkdir()
    exchange.mkdir(mode=511)
    exchange.chmod(511)
    tools = deepcopy(adapter.tools)
    parameters = tool_parameters(tools)
    packet = packet_for(task, tools, assets, history_handler=history_handler, condition=condition)
    expected = expected_tools(packet)
    history_parameters = tool_parameters(packet['history_tools'])
    _write(public / 'packet.json', packet)
    _write(directory / 'model-input.json', {'kind': 'public_packet_before_hermes_assembly', 'actual_provider_input': 'hermes-wire/0001.json', 'system': packet['system'], 'user': packet['request'], 'tools': expected})
    sources_before = source_snapshot(adapter, provider)
    _write(directory / 'sources-before.json', sources_before)
    before = _json_copy(adapter.snapshot())
    _write(directory / 'before-state-private.json', before)
    _write(directory / 'runtime-contract.json', {'schema': PROTOCOL, 'model': MODEL, 'seed': SEED, 'max_tokens': MAX_TOKENS, 'temperature': 'omitted', 'agent_max_iterations': MAX_ITERATIONS, 'agent_run_budget_seconds': 3600, 'host_deadline_seconds': 10800, 'business_budget_authority': 'adapter', 'native_review_enabled': True, 'parallel_tool_calls': PARALLEL_TOOL_CALLS, 'tool_execution_order': 'response_call_index', 'curator_adoption_allowlist': ADOPTION_ALLOWLIST, 'entry_history_count': history_count(history_handler), 'condition': condition, 'business_tools_sha256': core.digest(tools), 'input_sha256': core.digest(packet), 'state_sha256': core.digest(before), 'source_sha256': sources_before['sha256'], 'provider_conversion': f'max_completion_tokens -> max_tokens; default stream=false only; parallel_tool_calls={str(PARALLEL_TOOL_CALLS).lower()} required'})
    container = 'rescope-state-action-' + uuid.uuid4().hex[:10]
    (events, responses, phases, history_events) = ([], [], [], [])
    costs = []
    (review_responses, seen, config) = ([], set(), None)
    (process, foreground, completed) = (None, None, None)
    try:
        with (directory / 'worker.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(native_task.docker_command(public, exchange, container), stdout=log, stderr=subprocess.STDOUT)
            _write(directory / 'started.json', {'container': container, 'schema': PROTOCOL})
            deadline = time.monotonic() + 10800
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError('State-action episode deadline exceeded')
                pending = sorted((exchange / 'requests').glob('*.json'), key=lambda p: p.stat().st_mtime_ns)
                for path in pending:
                    if path.name in seen:
                        continue
                    item = _read(path)
                    if set(item) != {'request_id', 'kind', 'body'} or item['request_id'] != path.stem or item['kind'] not in {'native', 'history', 'model'} or (not isinstance(item['body'], dict)):
                        raise ValueError('Invalid worker bridge request')
                    (body, kind) = (item['body'], item['kind'])
                    phase = body.get('phase')
                    if phase not in PHASES or (exchange / 'completed.json').exists():
                        raise ValueError('Bridge request outside a declared open phase')
                    if kind == 'native':
                        if phase != 'task' or (exchange / 'foreground.json').exists() or (exchange / 'before-native-review.json').exists():
                            raise ValueError('Post-task business actions are forbidden')
                        name = body.get('name')
                        if name not in parameters:
                            raise ValueError('Undeclared business action')
                        (valid, arguments) = validate_arguments(parameters[name], body.get('arguments'))
                        resource_view = getattr(adapter, 'resource_state', lambda : {})
                        resource_before = resource_view()
                        response = _json_copy(adapter.call(name, arguments)) if valid else {'status': 'error', 'data': {}, 'message': 'INVALID_ARGUMENTS: does not match the declared schema'}
                        event = {'event_id': f'e{len(events) + 1:04d}', 'name': name, 'arguments': arguments, 'response': response, 'request_id': item['request_id'], 'cost': tool_execution.cost_receipt(resource_before, resource_view(), dispatched=valid)}
                        events.append(event)
                        costs.append({'kind': 'business', 'event_id': event['event_id'], 'request_id': item['request_id'], 'name': name, **event['cost']})
                        _write(directory / 'events' / f'{len(events):04d}.json', event)
                    elif kind == 'history':
                        name = body.get('name')
                        if name not in history_parameters:
                            raise ValueError('Undeclared history tool')
                        (valid, arguments) = validate_arguments(history_parameters[name], body.get('arguments') or {})
                        used_before = getattr(history_handler, 'operations_used', 0)
                        metered_history = hasattr(history_handler, 'operations_used')
                        response = _history_call(history_handler, name, arguments) if valid or metered_history else {'error': 'INVALID_ARGUMENTS', 'message': 'Arguments do not match the declared schema.'}
                        history_events.append({**body, 'response': response})
                        costs.append({'kind': 'history', 'request_id': item['request_id'], 'name': name, **tool_execution.cost_receipt({'history_operations': {'value': used_before, 'direction': 'used'}}, {'history_operations': {'value': getattr(history_handler, 'operations_used', 0), 'direction': 'used'}}, dispatched=valid or metered_history)})
                        _write(directory / 'history-events' / path.name, history_events[-1])
                    else:
                        validate_surface(_read(exchange / 'surface.json'), packet)
                        raw = body.get('request')
                        (admitted, conversions) = provider_body(raw, expected)
                        settings = {key: value for (key, value) in admitted.items() if key not in {'tools', 'messages'}}
                        if config is not None and config != settings:
                            raise ValueError('Provider configuration changed within the episode')
                        config = deepcopy(settings)
                        phases.append(phase)
                        index = len(phases)
                        _write(directory / 'hermes-wire' / f'{index:04d}.json', {'phase': phase, 'raw': raw, 'admitted': admitted, 'declared_conversions': conversions})
                        _write(directory / 'provider-admitted' / 'requests' / f'{index:04d}.json', admitted)
                        returned = _json_copy(provider(admitted))
                        (response, ignored_tool_calls) = normalize_serial_tool_response(returned, parallel=PARALLEL_TOOL_CALLS)
                        native_task.validate_provider_response(response, expected)
                        _write(directory / 'provider-returned' / 'responses' / f'{index:04d}.json', returned)
                        if ignored_tool_calls:
                            _write(directory / 'provider-normalization' / f'{index:04d}.json', {'kind': 'ignored_additional_tool_calls', 'ignored': ignored_tool_calls, 'kept_index': 0})
                        _write(directory / 'provider-admitted' / 'responses' / f'{index:04d}.json', response)
                        (responses if phase == 'task' else review_responses).append(response)
                    temporary = (exchange / 'responses' / path.name).with_suffix('.tmp')
                    _write(temporary, response)
                    temporary.replace(exchange / 'responses' / path.name)
                    seen.add(path.name)
                if process.poll() is not None and (not {p.name for p in (exchange / 'requests').glob('*.json')} - seen):
                    break
                time.sleep(0.03)
        if process.returncode != 0 or not (exchange / 'completed.json').is_file():
            raise RuntimeError('Native Hermes worker failed; inspect worker.log and exchange evidence')
        (completed, foreground) = (_read(exchange / 'completed.json'), _read(exchange / 'foreground.json'))
        if not all((completed.get(key) is True for key in ('completed', 'provider_server_closed', 'provider_server_joined'))):
            raise ValueError('Worker lifecycle did not close and join')
        validate_surface(_read(exchange / 'surface.json'), packet)
        terminal_assets = surface.validate_assets(completed['terminal_assets'])
        binding = validate_trajectory_binding(foreground['messages'], events, responses, tools)
        bridge = native_task.validate_exchange_complete(exchange, seen)
        if deepcopy(adapter.tools) != tools:
            raise ValueError('Adapter tool schemas changed during execution')
        sources_after = source_snapshot(adapter, provider)
        if sources_after != sources_before:
            raise ValueError('Source files changed during execution; attempt is not comparable')
        after = _json_copy(adapter.snapshot())
        grade = _json_copy(adapter.grade())
        if _json_copy(adapter.snapshot()) != after:
            raise ValueError('Grading modified the environment state')
        feedback = public_feedback(grade)
        _write(directory / 'bridge-integrity.json', {**bridge, 'trajectory_binding': binding, 'model_phases': phases, 'no_post_task_business_actions': True})
        _write(directory / 'grade-private.json', grade)
        _write(directory / 'feedback.json', feedback)
        _write(directory / 'assets.json', terminal_assets)
        task_calls = [message for message in foreground['messages'] if message.get('role') == 'assistant']
        result = {'schema': PROTOCOL, 'status': 'semantic_result', 'score': grade['score'], 'grade': grade, 'feedback': feedback, 'trajectory': foreground['messages'], 'assets': terminal_assets, 'model_calls': len(phases), 'task_model_calls': len(responses), 'native_review_model_calls': len(review_responses), 'agent_tool_rounds': sum((bool(row.get('tool_calls')) for row in task_calls)), 'agent_business_tool_rounds': sum((any((call.get('function', {}).get('name') in parameters for call in row.get('tool_calls') or [])) for row in task_calls)), 'business_tool_calls': len(events), 'history_tool_calls': len(history_events), 'native_experience_tool_calls': len(list((exchange / 'native-tools').glob('*.json'))), 'native_review_triggered': completed['native_review_triggered'], 'curator_adoptions': completed.get('curator_adoptions', []), 'provider_wire_attempts': getattr(provider, 'wires', None), 'source_sha256': sources_before['sha256'], 'sources_unchanged': True, 'entry_state_sha256': core.digest(before), 'terminal_state_sha256': core.digest(after)}
        _write(directory / 'episode-result-private.json', result)
        return result
    except BaseException as error:
        failure = {'schema': PROTOCOL + '/failure', 'status': 'infrastructure_error', 'score': None, 'error_type': type(error).__name__, 'error': _safe_error(error, provider), 'state_promoted': False, 'attempt_evidence_retained': True}
        _write(directory / 'failure.json', failure)
        raise
    finally:
        if process is not None and process.poll() is None:
            subprocess.run(['docker', 'stop', '-t', '1', container], capture_output=True)
            process.wait(timeout=20)
        _write(directory / 'trace.json', {'events': events})
        _write(directory / 'invocation-costs.json', {'schema': 'rescopebench-invocation-ledger/v1', 'calls': costs, 'aggregation': 'sum within a resource only; never sum different units'})
        _write(directory / 'tool-dispatch.json', {'schema': 'rescopebench-tool-batches/v1', 'parallel_tool_calls': PARALLEL_TOOL_CALLS, 'batches': [_read(path) for path in sorted((exchange / 'tool-batches').glob('*.json'))]})
        _write(directory / 'sources-after.json', source_snapshot(adapter, provider))
        _write(directory / 'final-state-private.json', _json_copy(adapter.snapshot()))
