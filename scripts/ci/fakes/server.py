"""Deterministic forum and OpenAI-compatible API; no upstream network calls."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, value, status=200, content_type="application/json"):
        data = value.encode() if isinstance(value, str) else json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/auth/jaccount":
            return self.respond(
                '<meta name="csrf-token" content="synthetic">', content_type="text/html"
            )
        if self.path == "/session/current.json":
            return self.respond(
                {"current_user": {"username": "wolf_lumine"}},
                401 if Path("/control/deny").exists() else 200,
            )
        if self.path.startswith("/user_actions.json"):
            return self.respond({"user_actions": []})
        return self.respond({})

    def do_POST(self):
        request = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}"
        )
        if self.path.endswith("/embeddings"):
            texts = request["input"]
            if isinstance(texts, str):
                texts = [texts]
            return self.respond(
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": [1.0]
                            + [0.0] * (request.get("dimensions", 1024) - 1),
                        }
                        for i in range(len(texts))
                    ],
                    "model": "synthetic",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            )
        return self.respond({"error": "unexpected request"}, 400)


ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
