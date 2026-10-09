import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic

# a small stand in for the Anthropic messages api. it speaks real http to the real sdk and
# says no to the same kind of mistakes the real api says no to, so a bad request in the
# agent shows up as a failing test instead of a surprise later.
#
#   api = FakeAPI([reply(text('hi'))])
#   api.start()
#   client = api.client()
#   ... api.requests has everything that was sent ...


def text(t):
    return {'type': 'text', 'text': t}


def toolcall(name, args, id='toolu_1'):
    return {'type': 'tool_use', 'id': id, 'name': name, 'input': args}


def thinking(signature='sig-1'):
    return {'type': 'thinking', 'thinking': '', 'signature': signature}


def reply(*blocks, stop='end_turn', tokens=(100, 20)):
    return {'id': 'msg_test', 'type': 'message', 'role': 'assistant', 'model': 'test-model',
            'content': list(blocks), 'stop_reason': stop, 'stop_sequence': None,
            'usage': {'input_tokens': tokens[0], 'output_tokens': tokens[1]}}


def error(status, kind, message):
    return ('error', status, kind, message)


def problems(req):
    # the rules of the real api that the agent could break
    out = []
    if not req.get('model'):
        out.append('model: field required')
    if not isinstance(req.get('max_tokens'), int) or req['max_tokens'] < 1:
        out.append('max_tokens: must be a positive integer')
    for k in ('temperature', 'top_p', 'top_k'):
        if k in req:
            out.append(k + ' is not supported on current models')
    if (req.get('tool_choice') or {}).get('type') in ('any', 'tool'):
        out.append('tool_choice: forced tool use is not supported on current models')
    if (req.get('thinking') or {}).get('type') in ('enabled', 'disabled'):
        out.append('thinking: only adaptive thinking is supported on current models')

    for t in req.get('tools') or []:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', t.get('name', '')):
            out.append('tools: bad name {!r}'.format(t.get('name')))
        if not str(t.get('description', '')).strip():
            out.append('tools: {} needs a description'.format(t.get('name')))
        if (t.get('input_schema') or {}).get('type') != 'object':
            out.append('tools: {} input_schema must be an object'.format(t.get('name')))

    msgs = req.get('messages') or []
    if not msgs:
        out.append('messages: at least one message is required')
        return out
    if msgs[0]['role'] != 'user':
        out.append('messages: the first message must be from the user')
    if msgs[-1]['role'] != 'user':
        out.append('messages: the last message must be from the user (assistant prefill is not supported)')

    for i, m in enumerate(msgs):
        if m['role'] not in ('user', 'assistant'):
            out.append('messages.{}: bad role {!r}'.format(i, m['role']))
        if not m.get('content'):
            out.append('messages.{}: content must not be empty'.format(i))
        if m['role'] == 'assistant' and isinstance(m.get('content'), list):
            ids = [b['id'] for b in m['content'] if b.get('type') == 'tool_use']
            if not ids:
                continue
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            blocks = nxt['content'] if nxt and nxt['role'] == 'user' and isinstance(nxt['content'], list) else []
            got = [b.get('tool_use_id') for b in blocks if b.get('type') == 'tool_result']
            if sorted(got) != sorted(ids):
                out.append('messages.{}: every tool_use needs a tool_result in the very next message, all in one message'.format(i))
            if blocks and blocks[0].get('type') != 'tool_result':
                out.append('messages.{}: tool_result blocks must come first'.format(i + 1))
    return out


class FakeAPI:
    def __init__(self, replies, port=0):
        # each reply is a message (see reply()), an error(), or a function of the request that gives one.
        # port 0 picks a free port
        self.replies = list(replies)
        self.requests = []
        self.lock = threading.Lock()
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                req = json.loads(body)
                with api.lock:
                    api.requests.append(dict(path=self.path, json=req,
                                             headers={k.lower(): v for k, v in self.headers.items()}))
                    item = api.replies.pop(0) if api.replies else None
                if self.path != '/v1/messages':
                    return self.send(404, ('error', 404, 'not_found_error', 'no such path'))
                if not self.headers.get('x-api-key'):
                    return self.send(401, ('error', 401, 'authentication_error', 'x-api-key header is required'))
                bad = problems(req)
                if bad:
                    return self.send(400, ('error', 400, 'invalid_request_error', '; '.join(bad)))
                if callable(item):
                    item = item(req)
                if item is None:
                    return self.send(500, ('error', 500, 'api_error', 'the test ran out of scripted replies'))
                self.send(200, item)

            def send(self, status, item):
                if isinstance(item, tuple):
                    status, kind, message = item[1], item[2], item[3]
                    item = {'type': 'error', 'error': {'type': kind, 'message': message}}
                data = json.dumps(item).encode()
                self.send_response(status)
                self.send_header('content-type', 'application/json')
                self.send_header('request-id', 'req_fake')
                self.send_header('content-length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.server.daemon_threads = True

    @property
    def url(self):
        return 'http://127.0.0.1:{}'.format(self.server.server_address[1])

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def client(self, **kw):
        kw.setdefault('max_retries', 0)
        kw.setdefault('timeout', 10.0)
        return anthropic.Anthropic(base_url=self.url, api_key='test-key', **kw)
