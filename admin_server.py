"""
LangGraph Thread Admin — proxy server.
Serves the admin UI at http://localhost:8765
and proxies API calls to the LangGraph server (no CORS issues).

Usage:
    python admin_server.py
Then open http://localhost:8765 in your browser.
"""

import json
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

LANGGRAPH_URL = "http://127.0.0.1:2024"
ADMIN_PORT    = 8765
HTML_FILE     = Path(__file__).parent / "thread_admin.html"


class AdminHandler(BaseHTTPRequestHandler):

    # ------------------------------------------------------------------ GET --
    def do_GET(self):
        if self.path in ("/", "/admin"):
            self._serve_html()
        elif self.path.startswith("/proxy/"):
            self._proxy("GET", self.path[len("/proxy"):])
        else:
            self._not_found()

    # ---------------------------------------------------------------- POST --
    def do_POST(self):
        if self.path.startswith("/proxy/"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            self._proxy("POST", self.path[len("/proxy"):], body)
        else:
            self._not_found()

    # --------------------------------------------------------------- DELETE --
    def do_DELETE(self):
        if self.path.startswith("/proxy/"):
            self._proxy("DELETE", self.path[len("/proxy"):])
        else:
            self._not_found()

    # ---------------------------------------------------------------- utils --
    def _serve_html(self):
        if not HTML_FILE.exists():
            self._error(404, f"thread_admin.html not found at {HTML_FILE}")
            return
        content = HTML_FILE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _proxy(self, method, path, body=None):
        target = LANGGRAPH_URL + path
        req = urllib.request.Request(target, data=body if body else None, method=method)
        req.add_header("Accept", "application/json")
        if body:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self._error(502, f"Could not reach LangGraph server at {LANGGRAPH_URL}: {exc}")

    def _not_found(self):
        self._error(404, "Not found")

    def _error(self, code, msg):
        body = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # suppress default access log spam
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("localhost", ADMIN_PORT), AdminHandler)
    print(f"Admin UI ready: http://localhost:{ADMIN_PORT}")
    print(f"Proxying to LangGraph server: {LANGGRAPH_URL}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAdmin server stopped.")
