import email
import email.policy
import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# stand-ins for the places alerts go, so the real code runs against a real socket. like fakeapi.py, they
# also get used by the browser checks outside the tests


class FakeHook:
    # a webhook. keeps what it was sent, and answers with whatever status you set
    def __init__(self, status=200, token='s3cr3t-token'):
        self.status, self.token, self.posts = status, token, []
        hook = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                hook.posts.append(dict(path=self.path, type=self.headers.get('Content-Type'), json=json.loads(body)))
                self.send_response(hook.status)
                self.send_header('content-length', '2')
                self.end_headers()
                self.wfile.write(b'ok')

            def log_message(self, *args):
                pass

        self.server = HTTPServer(('127.0.0.1', 0), Handler)

    @property
    def url(self):
        return 'http://127.0.0.1:{}/services/{}'.format(self.server.server_port, self.token)

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class FakeSmtp:
    # a mail server with no tls and no login. keeps the mail it is given, already read back into messages
    def __init__(self, refuse=None):
        self.mails, self.refuse = [], refuse
        mail = self

        class Handler(socketserver.StreamRequestHandler):
            def reply(self, line):
                self.wfile.write(line.encode() + b'\r\n')

            def handle(self):
                self.reply('220 fake ready')
                sender, rcpts = None, []
                for raw in self.rfile:
                    line = raw.decode().rstrip('\r\n')
                    cmd = line.upper()
                    if cmd.startswith('EHLO'):
                        self.reply('250-fake')
                        self.reply('250 8BITMIME')
                    elif cmd.startswith('HELO'):
                        self.reply('250 fake')
                    elif cmd.startswith('MAIL FROM'):
                        sender = line.split(':', 1)[1].strip().strip('<>')
                        self.reply('250 ok')
                    elif cmd.startswith('RCPT TO'):
                        if mail.refuse:
                            self.reply('550 ' + mail.refuse)
                        else:
                            rcpts.append(line.split(':', 1)[1].strip().strip('<>'))
                            self.reply('250 ok')
                    elif cmd == 'DATA':
                        self.reply('354 go ahead')
                        data = []
                        for part in self.rfile:
                            if part in (b'.\r\n', b'.\n'):
                                break
                            data.append(part[1:] if part.startswith(b'..') else part)
                        msg = email.message_from_bytes(b''.join(data), policy=email.policy.default)
                        mail.mails.append(dict(sender=sender, to=rcpts, msg=msg))
                        self.reply('250 queued')
                    elif cmd == 'QUIT':
                        self.reply('221 bye')
                        return
                    else:
                        self.reply('250 ok')

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server(('127.0.0.1', 0), Handler)

    @property
    def port(self):
        return self.server.server_address[1]

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
