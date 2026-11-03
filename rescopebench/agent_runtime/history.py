"""Business-only own history with an episode-local access budget."""

from __future__ import annotations

from collections import Counter as ext_Counter
from copy import deepcopy as ext_deepcopy
from pathlib import Path
from pathlib import PurePosixPath as ext_PurePosixPath
from rescopebench.public_projection import assert_hard_public_projection as ext_assert_hard_public_projection
from typing import Any as ext_Any
import hashlib as mod_hashlib
import json as mod_json
import jsonschema as mod_jsonschema
import math as mod_math
import re as _re
import re as mod_re
from .history_data import (DATA_HISTORY_TOOLS, DATA_PARAMETERS, DATA_HISTORY_NAMES, DATA_NATIVE_NAMES, DATA_PARAMETERS_2, DATA__PRIVATE_KEYS)

def fn_build_business_only_package(tools, episodes):
    fn__validate_supplied_history(episodes)
    return fn_build_business_only_package_2(tools, episodes)

def fn__validate_supplied_history(episodes):
    for episode in episodes:
        fn_assert_opaque_learner_surface({key: episode.get(key) for key in ('episode_id', 'task', 'grade')}, surface='business history metadata')

def fn__tool_names(tools):
    declared = set(fn_tool_parameters(tools))
    excluded = DATA_HISTORY_NAMES | DATA_NATIVE_NAMES
    business = declared - excluded
    if not business:
        raise ValueError('Business-only history requires at least one business tool')
    return (declared, business)

def fn_public_task_text(task):
    query = task if isinstance(task, str) else task.get('query') if isinstance(task, dict) else None
    if not isinstance(query, str) or not query.strip():
        raise ValueError('task must be a nonempty string or a mapping with a nonempty query')
    return query

def fn_public_feedback(grade):
    """Expose outcome predicates, never private targets or corrective answers."""
    score = grade.get('score')
    if type(score) not in (int, float) or not mod_math.isfinite(score) or (not 0 <= score <= 100):
        raise ValueError('Grade must contain a finite score from 0 to 100')
    if 'public_components' in grade:
        return fn_outcome({'score': score, 'components': grade['public_components']}, 'components')
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
    return fn_outcome({'score': score, 'components': components}, 'components')

def fn_encoded(value):
    return mod_json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def fn_validate_assets(assets):
    """Integrity/path checks, not a new content-quality gate on native writes."""
    if not isinstance(assets, dict):
        raise ValueError('Assets must be a mapping')
    for (name, item) in assets.items():
        path = ext_PurePosixPath(name)
        if path.is_absolute() or str(path) != name or '\\' in name or ('..' in path.parts) or (len(path.parts) < 2) or (path.parts[0] not in {'skills', 'memories'}):
            raise ValueError('Unsafe native asset path')
        if not isinstance(item, dict) or set(item) != {'content', 'sha256'} or (not isinstance(item['content'], str)) or (mod_hashlib.sha256(item['content'].encode()).hexdigest() != item['sha256']):
            raise ValueError('Native asset content/hash mismatch')
    return ext_deepcopy(assets)

def fn_build_business_only_package_2(tools, episodes):
    """Build a signed learner archive from a strict source whitelist."""
    (_, business_names) = fn__tool_names(tools)
    rows: list[dict[str, ext_Any]] = []
    records: dict[str, dict[str, ext_Any]] = {}
    seen_episode_ids: set[str] = set()
    for episode in episodes:
        if not isinstance(episode, dict) or episode.get('status') != 'semantic_result':
            raise ValueError('Only completed semantic episodes may enter history')
        episode_id = episode.get('episode_id')
        task = episode.get('task')
        if not isinstance(episode_id, str) or not episode_id or episode_id in seen_episode_ids or (not isinstance(task, str)) or (not task.strip()):
            raise ValueError('History requires unique episode IDs and nonempty task text')
        seen_episode_ids.add(episode_id)
        feedback = fn_public_feedback(episode.get('grade', {}))
        pairs = fn__business_pairs(episode.get('trajectory', []), business_names)
        task_record = fn__record(records, task, 'task', {'episode_id': episode_id, 'field': 'task_request'})
        (actions, events) = ([], [])
        for pair in pairs:
            provenance = {'episode_id': episode_id, 'section': 'trajectory', 'tool': pair['tool'], 'call_id': pair['call_id'], 'business_event_id': pair['business_event_id']}
            arguments_record = fn__record(records, pair['arguments'], 'own_call', {**provenance, 'message_index': pair['source_call_message_index']})
            result_text = fn__text(pair['result'])
            result_record = fn__record(records, result_text, 'tool_observation', {**provenance, 'message_index': pair['source_result_message_index'], 'direct_result': True})
            action = {'business_event_id': pair['business_event_id'], 'tool': pair['tool'], 'call_id': pair['call_id'], 'arguments_record': arguments_record, 'result_record': result_record, 'source_call_message_index': pair['source_call_message_index'], 'source_result_message_index': pair['source_result_message_index'], **fn__status_fields(result_text)}
            actions.append(action)
            events.extend([{'section': 'trajectory', 'role': 'assistant', 'message_index': pair['source_call_message_index'], 'business_event_id': pair['business_event_id'], 'calls': [{'name': pair['tool'], 'call_id': pair['call_id'], 'arguments_record': arguments_record}]}, {'section': 'trajectory', 'role': 'tool', 'message_index': pair['source_result_message_index'], 'business_event_id': pair['business_event_id'], 'tool': pair['tool'], 'call_id': pair['call_id'], 'record_id': result_record, 'kind': 'tool_observation'}])
        events.sort(key=lambda row: (row['message_index'], row['role'] != 'assistant'))
        summary_actions = [{key: ext_deepcopy(action[key]) for key in ('business_event_id', 'tool', 'call_id', 'arguments_record', 'result_record', 'status', 'error') if key in action} for action in actions]
        rows.append({'episode_id': episode_id, 'task': task, 'task_identity': {'task_id': episode_id, 'task_request': task}, 'task_record': task_record, 'feedback': feedback, 'events': events, 'actions': actions, 'tool_counts': dict(ext_Counter((action['tool'] for action in actions))), 'summary': {'schema': 'robot-business-action-index/v4', 'business_actions': summary_actions, 'policy': 'Exact references to declared business calls and their direct results only; no model prose, experience assets, retrieval output, or inferred rule.'}})
    package = {'schema': 'consolidation-scaffold/v1', 'feedback_mode': 'components', 'episodes': rows, 'documents': [], 'records': records, 'assets': {}, 'history_protocol': {'version': 'robot-experience-business-history/v4', 'only_completed_own_episodes': True, 'business_tools': sorted(business_names), 'learner_visible': ['task_identity', 'business_action_arguments', 'direct_business_results', 'public_artifact_feedback'], 'excluded': ['history_calls_and_results', 'native_skill_or_memory_calls_and_results', 'model_prose_and_final_answers', 'skills_before_after_and_current_assets', 'prior_or_recursive_summaries'], 'record_search_scope': 'business_only_no_override', 'action_result_pairing': 'call_id plus original message indices', 'pagination': 'episode, search, and record reads retain explicit continuation offsets'}}
    package['bundle_sha256'] = fn_digest(package)
    fn__validate_package(package)
    return package

def fn_assert_opaque_learner_surface(value, *, surface):
    hits = fn_leakage_hits(value)
    present = {name: rows for (name, rows) in hits.items() if rows}
    if present:
        raise ValueError(f'{surface} exposes private lifecycle metadata: {present}')
    try:
        ext_assert_hard_public_projection(value)
    except ValueError as error:
        raise ValueError(f'{surface} exposes private lifecycle metadata: {error}') from error

def fn_tool_parameters(tools):
    if not isinstance(tools, list) or not tools:
        raise ValueError('Adapter must declare a nonempty tool-schema list')
    result = {}
    for tool in tools:
        function = tool.get('function', {}) if isinstance(tool, dict) else {}
        (name, params) = (function.get('name'), function.get('parameters'))
        if not isinstance(tool, dict) or tool.get('type') != 'function' or (not isinstance(name, str)) or (not mod_re.fullmatch('[A-Za-z_][A-Za-z0-9_-]{0,63}', name)) or (name in result) or (not isinstance(params, dict)) or (not isinstance(function.get('description'), str)) or (not function['description'].strip()):
            raise ValueError('Invalid, duplicate, or incomplete public tool schema')
        mod_jsonschema.Draft7Validator.check_schema(params)
        result[name] = ext_deepcopy(params)
    return result

def fn_outcome(supplied, mode):
    allowed = {'score', 'components'} if mode == 'components' else {'score'}
    if set(supplied) - allowed or 'score' not in supplied:
        raise ValueError('Unexpected outcome fields')
    score = supplied['score']
    if type(score) not in (int, float) or not mod_math.isfinite(score) or (not 0 <= score <= 100):
        raise ValueError('Expected a finite score from 0 to 100')
    if mode == 'components':
        counts = supplied.get('components')
        if not isinstance(counts, dict) or not counts:
            raise ValueError('Expected aggregate component counts')
        for (field, count) in counts.items():
            if not isinstance(field, str) or not isinstance(count, dict) or set(count) != {'correct', 'total'} or any((type(count[key]) is not int for key in count)) or (not 0 <= count['correct'] <= count['total']) or (count['total'] < 1):
                raise ValueError('Only aggregate correct/total counts are allowed')
    return ext_deepcopy(supplied)

def fn__validate_package(package):
    value = ext_deepcopy(package)
    expected = value.pop('bundle_sha256', None)
    if value.get('schema') != 'consolidation-scaffold/v1' or fn_digest(value) != expected:
        raise ValueError('Bundle integrity failure')
    if set(value) != {'schema', 'feedback_mode', 'episodes', 'documents', 'records', 'assets', 'history_protocol'}:
        raise ValueError('Unexpected business-history package field')
    if value['feedback_mode'] != 'components' or value['documents'] or value['assets']:
        raise ValueError('Business-only history cannot expose documents or assets')
    protocol = value['history_protocol']
    if protocol.get('version') != 'robot-experience-business-history/v4' or protocol.get('record_search_scope') != 'business_only_no_override':
        raise ValueError('Unsupported business-history contract')
    referenced: set[str] = set()
    expected_origins: dict[str, list[dict[str, ext_Any]]] = {}
    episode_ids: set[str] = set()
    business_names = set(protocol.get('business_tools', []))

    def expect_origin(record_id, origin):
        expected_origins.setdefault(record_id, [])
        if origin not in expected_origins[record_id]:
            expected_origins[record_id].append(origin)
    for row in value['episodes']:
        if row.get('episode_id') in episode_ids:
            raise ValueError('Duplicate episode ID')
        episode_ids.add(row.get('episode_id'))
        if set(row) != {'episode_id', 'task', 'task_identity', 'task_record', 'feedback', 'events', 'actions', 'tool_counts', 'summary'}:
            raise ValueError('Unexpected learner-visible episode field')
        if row['task_identity'] != {'task_id': row['episode_id'], 'task_request': row['task']}:
            raise ValueError('Task identity is inconsistent')
        fn_outcome(row['feedback'], 'components')
        referenced.add(row['task_record'])
        expect_origin(row['task_record'], {'kind': 'task', 'episode_id': row['episode_id'], 'field': 'task_request'})
        expected_counts = ext_Counter()
        expected_summary = []
        expected_events = []
        event_ids: set[str] = set()
        call_ids: set[str] = set()
        for (sequence, action) in enumerate(row['actions'], 1):
            required = {'business_event_id', 'tool', 'call_id', 'arguments_record', 'result_record', 'source_call_message_index', 'source_result_message_index'}
            if not required <= set(action) or action['tool'] not in business_names:
                raise ValueError('Malformed or undeclared business action')
            allowed = required | {'status', 'error'}
            if set(action) - allowed:
                raise ValueError('Unexpected business action field')
            if action['business_event_id'] in event_ids:
                raise ValueError('Duplicate business event ID')
            event_ids.add(action['business_event_id'])
            if action['business_event_id'] != f'b{sequence:04d}':
                raise ValueError('Business event IDs are not stable and sequential')
            if action['call_id'] in call_ids:
                raise ValueError('Duplicate business call ID')
            call_ids.add(action['call_id'])
            if action['source_result_message_index'] <= action['source_call_message_index']:
                raise ValueError('Business result precedes its action')
            referenced.update((action['arguments_record'], action['result_record']))
            result_record = value['records'].get(action['result_record'])
            if result_record is None or {key: action[key] for key in ('status', 'error') if key in action} != fn__status_fields(result_record.get('text')):
                raise ValueError('Business action status differs from its direct result')
            provenance = {'episode_id': row['episode_id'], 'section': 'trajectory', 'tool': action['tool'], 'call_id': action['call_id'], 'business_event_id': action['business_event_id']}
            expect_origin(action['arguments_record'], {'kind': 'own_call', **provenance, 'message_index': action['source_call_message_index']})
            expect_origin(action['result_record'], {'kind': 'tool_observation', **provenance, 'message_index': action['source_result_message_index'], 'direct_result': True})
            expected_counts[action['tool']] += 1
            expected_summary.append({key: ext_deepcopy(action[key]) for key in ('business_event_id', 'tool', 'call_id', 'arguments_record', 'result_record', 'status', 'error') if key in action})
            expected_events.extend([{'section': 'trajectory', 'role': 'assistant', 'message_index': action['source_call_message_index'], 'business_event_id': action['business_event_id'], 'calls': [{'name': action['tool'], 'call_id': action['call_id'], 'arguments_record': action['arguments_record']}]}, {'section': 'trajectory', 'role': 'tool', 'message_index': action['source_result_message_index'], 'business_event_id': action['business_event_id'], 'tool': action['tool'], 'call_id': action['call_id'], 'record_id': action['result_record'], 'kind': 'tool_observation'}])
        if row['tool_counts'] != dict(expected_counts):
            raise ValueError('Business tool counts are inconsistent')
        if row['summary'] != {'schema': 'robot-business-action-index/v4', 'business_actions': expected_summary, 'policy': 'Exact references to declared business calls and their direct results only; no model prose, experience assets, retrieval output, or inferred rule.'}:
            raise ValueError('Business summary is not a pure action reference index')
        expected_events.sort(key=lambda event: (event['message_index'], event['role'] != 'assistant'))
        if row['events'] != expected_events:
            raise ValueError('Episode events are not exact business action/result events')
    if referenced != set(value['records']):
        raise ValueError('Archive contains unreferenced or missing records')
    for (record_id, record) in value['records'].items():
        text = record.get('text')
        if not isinstance(text, str) or record_id != 'r-' + fn_text_hash(text) or record.get('sha256') != fn_text_hash(text) or (record.get('characters') != len(text)) or (not isinstance(record.get('origins'), list)) or (not record['origins']) or (record['origins'] != expected_origins.get(record_id)):
            raise ValueError('Invalid business-history record')

def fn_asset_path(path):
    if not isinstance(path, str) or not path or '\\' in path:
        raise ValueError('Invalid asset path')
    parts = ext_PurePosixPath(path).parts
    if ext_PurePosixPath(path).is_absolute() or ext_PurePosixPath(path).as_posix() != path or len(parts) < 2 or (parts[0] not in {'skills', 'memories'}) or any((part.startswith('.') for part in parts)):
        raise ValueError('Use a relative, non-hidden path under skills/ or memories/')
    return path

def fn_text_hash(text):
    return mod_hashlib.sha256(text.encode()).hexdigest()

def fn__business_pairs(trajectory, business_names):
    if not isinstance(trajectory, list):
        raise ValueError('Episode trajectory must be a list')
    all_call_ids: set[str] = set()
    business_calls: dict[str, dict[str, ext_Any]] = {}
    ordered_calls: list[dict[str, ext_Any]] = []
    for (message_index, message) in enumerate(trajectory):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        for (call_position, call) in enumerate(message.get('tool_calls') or []):
            function = call.get('function') if isinstance(call, dict) else None
            call_id = call.get('id') if isinstance(call, dict) else None
            name = function.get('name') if isinstance(function, dict) else None
            if not isinstance(call_id, str) or not call_id:
                raise ValueError('Every tool call requires a nonempty call ID')
            if call_id in all_call_ids:
                raise ValueError('Tool call IDs must be unique within an episode')
            all_call_ids.add(call_id)
            if name not in business_names:
                continue
            if 'arguments' not in function:
                raise ValueError('Business tool calls require explicit arguments')
            row = {'tool': name, 'call_id': call_id, 'arguments': ext_deepcopy(function['arguments']), 'source_call_message_index': message_index, 'source_call_position': call_position}
            business_calls[call_id] = row
            ordered_calls.append(row)
    results: dict[str, dict[str, ext_Any]] = {}
    for (message_index, message) in enumerate(trajectory):
        if not isinstance(message, dict) or message.get('role') != 'tool':
            continue
        call_id = message.get('tool_call_id')
        if call_id not in business_calls:
            continue
        if call_id in results:
            raise ValueError('Business tool calls must have exactly one direct result')
        call = business_calls[call_id]
        name = message.get('name', call['tool'])
        if name != call['tool'] or message_index <= call['source_call_message_index']:
            raise ValueError('Business result does not directly match its declared call')
        content = message.get('content')
        if not isinstance(content, str):
            raise ValueError('Business tool results require exact content')
        try:
            (direct_result, wrapper_warning) = fn_decode_business_result(content, call['tool'])
        except ValueError as error:
            raise ValueError('Business tool result is not a valid direct result') from error
        results[call_id] = {'result': fn_encoded(direct_result) if wrapper_warning else content, 'source_result_message_index': message_index}
    if set(results) != set(business_calls):
        raise ValueError('Every archived business call requires one direct result')
    pairs = []
    for (sequence, call) in enumerate(ordered_calls, 1):
        result = results[call['call_id']]
        pairs.append({**ext_deepcopy(call), **ext_deepcopy(result), 'business_event_id': f'b{sequence:04d}'})
    return pairs

def fn__record(records, text, kind, origin):
    value = fn__text(text)
    digest = fn_text_hash(value)
    record_id = 'r-' + digest
    row = records.setdefault(record_id, {'text': value, 'sha256': digest, 'characters': len(value), 'origins': []})
    source = {'kind': kind, **ext_deepcopy(origin)}
    if source not in row['origins']:
        row['origins'].append(source)
    return record_id

def fn__text(value):
    return value if isinstance(value, str) else fn_encoded(value)

def fn__status_fields(text):
    try:
        body = mod_json.loads(text)
    except (TypeError, ValueError):
        return {}
    if not isinstance(body, dict):
        return {}
    result = {}
    if isinstance(body.get('status'), (str, bool)):
        result['status'] = body['status']
    if isinstance(body.get('error'), (str, bool)):
        result['error'] = str(body['error'])[:160]
    return result

def fn_digest(value):
    return mod_hashlib.sha256(fn_encoded(value).encode()).hexdigest()

def fn_leakage_hits(value):
    """Report private lifecycle labels in a learner/provider-visible value."""
    encoded = mod_json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
    private_keys: set[str] = set()

    def visit(item):
        if isinstance(item, dict):
            private_keys.update((str(key) for key in item if key in DATA__PRIVATE_KEYS))
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
    visit(value)
    return {'private_episode_ids': sorted(set(_re.compile('(?<![A-Za-z0-9])T(?:0[1-9]|1[0-6])(?![A-Za-z0-9]|:[0-5][0-9])', 34).findall(encoded))), 'arm_labels': sorted(set(_re.compile('(?<![A-Za-z0-9])(?:native|reset|reflect)\\d{2}(?![A-Za-z0-9])', 34).findall(encoded))), 'run_paths': sorted(set(_re.compile('(?:benchmark_outputs[/\\\\]robot|robot-final-v\\d+[/\\\\]20\\d{6}-\\d{6})', 34).findall(encoded))), 'private_keys': sorted(private_keys)}

def fn_decode_business_result(content, name):
    if not isinstance(content, str):
        raise ValueError('Hermes business result must contain JSON')
    try:
        content = content.lstrip()
        (value, end) = mod_json.JSONDecoder().raw_decode(content)
    except mod_json.JSONDecodeError as error:
        raise ValueError('Hermes business result must contain JSON') from error
    return (value, bool(content[end:].strip()))

def fn_public_assets(assets):
    result = {}
    for (path, item) in assets.items():
        if any((part.startswith('.') for part in ext_PurePosixPath(path).parts)):
            continue
        fn_asset_path(path)
        text = item['content']
        if not isinstance(text, str) or fn_text_hash(text) != item['sha256']:
            raise ValueError('Asset content/hash mismatch')
        result[path] = dict(content=text, sha256=fn_text_hash(text))
    return result

class HistoryArchive:

    def package(self, assets):
        fn__validate_supplied_history(self.episodes)
        return self._history_archive_package(assets)

    def session(self, assets):
        return Session(self.package(assets))

    def __init__(self, tools, episodes=None):
        self._history_archive_init(tools, episodes)
        (_, self.business_names) = fn__tool_names(self.tools)

    def append(self, episode):
        if episode.get('status') != 'semantic_result':
            raise ValueError('Only completed semantic episodes may enter history')
        episode_id = episode.get('episode_id')
        if not isinstance(episode_id, str) or not episode_id or any((row['episode_id'] == episode_id for row in self.episodes)):
            raise ValueError('History requires unique episode IDs')
        fn_public_task_text(episode.get('task'))
        fn_public_feedback(episode['grade'])
        self.episodes.append(ext_deepcopy(episode))

    def _history_archive_package(self, assets):
        fn_validate_assets(assets)
        return fn_build_business_only_package_2(self.tools, self.episodes)

    def _history_archive_init(self, tools, episodes=None):
        fn_tool_parameters(tools)
        self.tools = ext_deepcopy(tools)
        self.episodes = []
        for episode in episodes or []:
            self.append(episode)

class Session:

    def __init__(self, package, limit=30):
        if type(limit) is not int or limit <= 0:
            raise ValueError('history operation limit must be a positive integer')
        self.operation_limit = limit
        self.operations_used = 0
        self._session_init(package)

    @property
    def operations_remaining(self):
        return max(0, self.operation_limit - self.operations_used)

    def call(self, name, arguments):
        if self.operations_used >= self.operation_limit:
            result = {'error': 'HISTORY_BUDGET_EXHAUSTED', 'message': 'No history operations remain in this episode.', 'history_budget': {'limit': self.operation_limit, 'used': self.operations_used, 'remaining': 0}}
            self.events.append({'name': name, 'arguments': ext_deepcopy(arguments), 'result': ext_deepcopy(result)})
            return result
        self.operations_used += 1
        result = self._session_call(name, arguments)
        if not isinstance(result, dict):
            raise ValueError('history operations must return an object')
        result = {**result, 'history_budget': {'limit': self.operation_limit, 'used': self.operations_used, 'remaining': self.operations_remaining}}
        self.events[-1]['result'] = ext_deepcopy(result)
        return result

    def _call(self, name, arguments):
        if name == 'episode_list':
            terms = arguments.get('query', '').casefold().split()
            cards = []
            for row in reversed(self.bundle['episodes']):
                searchable = row['task'] + '\n' + fn_encoded(row['summary'])
                if not all((term in searchable.casefold() for term in terms)):
                    continue
                card = {'episode_id': row['episode_id'], 'task_identity': ext_deepcopy(row['task_identity']), 'task_record': row['task_record'], 'business_action_count': len(row['actions']), 'business_tool_counts': ext_deepcopy(row['tool_counts']), 'feedback': ext_deepcopy(row['feedback'])}
                cards.append(card)
            return self._page(cards, arguments, 'episodes')
        if name == 'episode_read':
            row = next((item for item in self.bundle['episodes'] if item['episode_id'] == arguments['episode_id']), None)
            if row is None:
                raise ValueError('Unknown episode_id; obtain it from episode_list')
            view = arguments.get('view', 'summary')
            if view == 'summary':
                page = self._page(row['summary']['business_actions'], arguments, 'business_actions')
                return {'episode_id': row['episode_id'], 'task_identity': ext_deepcopy(row['task_identity']), 'summary': {'schema': row['summary']['schema'], 'policy': row['summary']['policy'], **page}, 'feedback': ext_deepcopy(row['feedback'])}
            values = row['actions'] if view == 'calls' else row['events']
            return self._page(values, arguments, 'events')
        if name == 'record_read':
            return Session._session_call_2(self, name, {'limit': 4000, **arguments})
        if name != 'record_search':
            raise ValueError('Only business-history read operations are available')
        kinds = {'tasks': {'task'}, 'actions': {'own_call'}, 'observations': {'tool_observation'}}
        terms = arguments.get('query', '').casefold().split()
        found = []
        for (record_id, row) in self.bundle['records'].items():
            origins = ext_deepcopy(row['origins'])
            if arguments.get('kind'):
                origins = [origin for origin in origins if origin['kind'] in kinds[arguments['kind']]]
            if not origins:
                continue
            haystack = (fn_encoded(origins) + '\n' + row['text']).casefold()
            if not all((term in haystack for term in terms)):
                continue
            folded = row['text'].casefold()
            positions = [folded.find(term) for term in terms]
            position = min((item for item in positions if item >= 0), default=0)
            position = max(0, position - 80)
            found.append({'record_id': record_id, 'origins': origins[:4], 'origin_count': len(origins), 'characters': row['characters'], 'snippet_offset': position, 'snippet': row['text'][position:position + 320]})
        found.sort(key=lambda item: item['record_id'])
        return self._page(found, arguments, 'records')

    def finish(self, completed):
        fn_validate_assets(self.assets)
        return Session._session_finish(self, completed)

    def _page(self, values, arguments, key):
        (start, limit) = (arguments.get('offset', 0), arguments.get('limit', 8))
        return {key: ext_deepcopy(values[start:start + limit]), 'total': len(values), 'next_offset': start + limit if start + limit < len(values) else None}

    def _session_init(self, package):
        fn__validate_package(package)
        self._session_init_2(package)

    def _session_call(self, name, arguments):
        try:
            parameters = DATA_PARAMETERS_2.get(name)
            if parameters is None:
                raise ValueError('Only the declared business-history tools are available')
            mod_jsonschema.validate(arguments, parameters)
            result = self._call(name, arguments)
        except (ValueError, KeyError, mod_jsonschema.ValidationError) as error:
            result = {'error': type(error).__name__, 'message': str(error).split('\n')[0]}
        self.events.append({'name': name, 'arguments': ext_deepcopy(arguments), 'result': ext_deepcopy(result)})
        return result

    def _session_call_2(self, name, a):
        if name == 'episode_list':
            rows = [{key: value for (key, value) in row.items() if key != 'events'} for row in self.bundle['episodes']]
            return self._page(rows, a, 'episodes')
        if name == 'episode_read':
            row = next((row for row in self.bundle['episodes'] if row['episode_id'] == a['episode_id']), None)
            if row is None:
                raise ValueError('Unknown episode_id; obtain it from episode_list')
            return self._page(row['events'], a, 'events')
        if name == 'record_search':
            terms = a.get('query', '').casefold().split()
            found = []
            for (rid, row) in self.bundle['records'].items():
                origins = row['origins']
                if a.get('scope', 'sources') == 'sources' and all((source['kind'] in {'retrieval_echo', 'review_instruction'} for source in origins)):
                    continue
                text = row['text']
                if all((term in text.casefold() for term in terms)):
                    found.append(dict(record_id=rid, origins=origins, characters=len(text), snippet=text[:240]))
            return self._page(found, a, 'records')
        if name == 'record_read':
            row = self.bundle['records'].get(a['record_id'])
            if row is None:
                raise ValueError('Unknown record_id; obtain it from an index/search')
            (offset, limit) = (a.get('offset', 0), a.get('limit', 12000))
            return dict(text=row['text'][offset:offset + limit], sha256=row['sha256'], origins=ext_deepcopy(row['origins']), total_characters=row['characters'], next_offset=offset + limit if offset + limit < row['characters'] else None)
        if name == 'asset_list':
            return self._page([dict(path=path, sha256=item['sha256']) for (path, item) in sorted(self.assets.items())], a, 'assets')
        path = fn_asset_path(a['path'])
        current = self.assets.get(path)
        if name == 'asset_read':
            if current is None:
                raise ValueError('Unknown asset path; obtain existing paths from asset_list')
            self.read_versions[path] = current['sha256']
            return dict(path=path, **current)
        expected = current['sha256'] if current else ''
        if a['expected_sha256'] != expected or (current and self.read_versions.get(path) != expected):
            raise ValueError('STALE_OR_UNREAD: asset_read the current version and retry with its sha256')
        content = a['content']
        if len(content.encode()) > 2000000:
            raise ValueError('Asset exceeds 2 MB')
        if path.endswith('/SKILL.md'):
            import yaml
            parts = content.split('---', 2)
            if len(parts) != 3 or parts[0].strip():
                raise ValueError('SKILL.md needs YAML frontmatter with name and description')
            try:
                header = yaml.safe_load(parts[1])
            except yaml.YAMLError as error:
                raise ValueError('Invalid YAML frontmatter') from error
            if not isinstance(header, dict) or not all((isinstance(header.get(key), str) and header[key].strip() for key in ('name', 'description'))):
                raise ValueError('SKILL.md needs nonempty string name and description')
        self.assets[path] = dict(content=content, sha256=fn_text_hash(content))
        return dict(path=path, sha256=self.assets[path]['sha256'], staged=True)

    def _session_finish(self, completed):
        return dict(schema='consolidation-scaffold/v1', completed=bool(completed), transferable=bool(completed), initial_assets_sha256=self.initial_hash, assets=ext_deepcopy(self.assets) if completed else None, candidate_assets=ext_deepcopy(self.assets), events=ext_deepcopy(self.events))

    def _session_init_2(self, package):
        Session._session_init_3(self, package)
        fn_validate_assets(self.assets)

    def _session_init_3(self, bundle):
        data = ext_deepcopy(bundle)
        expected = data.pop('bundle_sha256')
        if data.get('schema') != 'consolidation-scaffold/v1' or fn_digest(data) != expected:
            raise ValueError('Bundle integrity failure')
        (self.bundle, self.assets) = (data, fn_public_assets(data['assets']))
        self.initial_hash = fn_digest(self.assets)
        (self.read_versions, self.events) = ({}, [])
HISTORY_OPERATION_LIMIT = 30
HISTORY_TOOLS = DATA_HISTORY_TOOLS
METADATA_VERSION = 'robot-experience-business-history/v5'
PARAMETERS = DATA_PARAMETERS
SUMMARY_SCHEMA = 'robot-business-action-index/v4'
build_business_only_package = fn_build_business_only_package
