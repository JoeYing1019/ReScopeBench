"""Isolated upstream Hermes frontend; no environment or provider secret is mounted."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time
import uuid


PUBLIC = Path('/public/packet.json')
EXCHANGE = Path('/exchange')
PROFILE = Path('/work/profile')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, default=str))
    temporary.replace(path)


def exchange(kind, body):
    request_id = uuid.uuid4().hex
    write(EXCHANGE / 'requests' / (request_id + '.json'),
          dict(request_id=request_id, kind=kind, body=body))
    response = EXCHANGE / 'responses' / (request_id + '.json')
    deadline = time.monotonic() + 1800
    while not response.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError('Host bridge did not respond')
        time.sleep(0.1)
    value = json.loads(response.read_text())
    if 'bridge_error' in value:
        raise RuntimeError(value['bridge_error'])
    return value


def snapshot():
    return {
        str(p.relative_to(PROFILE)): dict(content=p.read_text(),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for part in ('skills', 'memories') for p in sorted((PROFILE / part).rglob('*'))
        if p.is_file() and not p.is_symlink()
    }


def install_assets(assets):
    for name, item in assets.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or path.parts[0] not in ('skills', 'memories'):
            raise ValueError('Unsafe asset path')
        content = item['content']
        if hashlib.sha256(content.encode()).hexdigest() != item['sha256']:
            raise ValueError('Asset hash changed')
        dest = PROFILE / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        try:
            if self.path != '/v1/chat/completions':
                raise ValueError('Unexpected provider route')
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            value = exchange('model', body)
            code = 200
        except Exception as error:
            value = {'error': {'message': str(error), 'type': type(error).__name__}}
            code = 500
        payload = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    packet = json.loads(PUBLIC.read_text())
    PROFILE.mkdir(parents=True, exist_ok=True)
    for part in ('skills', 'memories'):
        (PROFILE / part).mkdir(exist_ok=True)
    (PROFILE / 'config.yaml').write_text(packet['profile_config'])
    install_assets(packet['assets'])
    os.environ['HERMES_HOME'] = str(PROFILE)
    os.environ['HOME'] = '/work'
    os.chdir('/work')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port

    def guard(event, values):
        if event == 'socket.connect':
            address = values[1]
            if not isinstance(address, tuple) or address[:2] != ('127.0.0.1', port):
                raise PermissionError('Only the local model bridge is reachable')
        if event in {'subprocess.Popen', 'os.system'}:
            raise PermissionError('Learner subprocesses are disabled')

    sys.addaudithook(guard)
    sys.path.insert(0, '/opt/hermes')
    from tools.registry import registry
    for schema in packet['tools']:
        name = schema['function']['name']
        if registry.get_entry(name) is not None:
            raise ValueError('Tool collision')
        registry.register(name=name, toolset='rescope_robot', schema=schema['function'],
            handler=lambda arguments, _name=name, **_kw: json.dumps(
                exchange('native', dict(name=_name, arguments=arguments))))
    from run_agent import AIAgent
    from hermes_state import SessionDB
    db = SessionDB()
    agent = AIAgent(model=packet['model'], provider='custom', api_mode='chat_completions',
        api_key='local-bridge-placeholder', base_url=f'http://127.0.0.1:{port}/v1',
        enabled_toolsets=['rescope_robot', 'memory', 'skills'], skip_context_files=True,
        load_soul_identity=False, skip_memory=False, skip_background_review=False,
        ephemeral_system_prompt=packet['system'], quiet_mode=True,
        request_overrides={'seed': packet['seed']},
        max_iterations=100, max_tokens=65536, run_budget_seconds=3600,
        session_id=packet['session_id'], session_db=db)
    expected = {s['function']['name'] for s in packet['tools']} | {
        'memory', 'skills_list', 'skill_view', 'skill_manage'}
    if {s['function']['name'] for s in agent.tools} != expected:
        raise RuntimeError('Unexpected Hermes tools')
    write(EXCHANGE / 'surface.json', dict(tools=agent.tools, entry_assets=snapshot(),
        skip_background_review=agent.skip_background_review, upstream='345cd2b057a452236de401d3534b8502a7465e8d'))
    try:
        result = agent.run_conversation(user_message=packet['request'],
            conversation_history=[], task_id=packet['session_id'])
        write(EXCHANGE / 'foreground.json', result)
        deadline = time.monotonic() + 2700
        while any(t.name == 'bg-review' and t.is_alive() for t in threading.enumerate()):
            if time.monotonic() > deadline:
                raise TimeoutError('Hermes native review did not settle')
            time.sleep(0.1)
        if getattr(agent, '_background_review_run', None) is not None:
            raise RuntimeError('Unsettled native review')
        if not result.get('completed') or result.get('failed') or result.get('interrupted'):
            raise RuntimeError('Hermes foreground incomplete')
        write(EXCHANGE / 'completed.json', dict(completed=True, terminal_assets=snapshot()))
    finally:
        agent.close()
        db.close()
        server.shutdown()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write(EXCHANGE / 'worker-failure.json', dict(error_type=type(exc).__name__, error=str(exc)))
        raise
