from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow.core.experiment_manager import ExperimentManager

APP_HTML = ROOT / "outputs" / "all_results" / "labs" / "log_lab_v2_app.html"
ENTRY_HTML = ROOT / "outputs" / "all_results" / "labs" / "log_lab_v2.html"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WorldSimFlow Log Lab 2.0 local server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--build-static", action="store_true")
    args = parser.parse_args()
    write_entry(args.host, args.port)
    if args.build_static:
        print("log_lab_v2_static=ok")
        print(f"output={ENTRY_HTML}")
        return
    manager = ExperimentManager(ROOT)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path in {"/", "/log_lab_v2"}:
                    self.send_file(APP_HTML)
                elif parsed.path == "/api/scenarios":
                    self.send_json({"scenarios": manager.list_scenarios()})
                elif parsed.path == "/api/scenario":
                    key = parse_qs(parsed.query).get("key", [""])[0]
                    self.send_json(manager.get_scenario(key))
                elif parsed.path == "/api/experiments":
                    self.send_json({"experiments": manager.list_experiments()})
                elif parsed.path.startswith("/artifact/"):
                    self.send_artifact(parsed.path[len("/artifact/"):])
                else:
                    self.send_json({"error": "not found"}, 404)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)

        def do_POST(self):
            try:
                if urlparse(self.path).path != "/api/run":
                    self.send_json({"error": "not found"}, 404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                self.send_json(manager.run(payload))
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)

        def send_json(self, payload, status=200):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def send_file(self, path: Path):
            if not path.exists():
                self.send_json({"error": f"missing file: {path}"}, 404)
                return
            raw = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def send_artifact(self, rel: str):
            path = (ROOT / unquote(rel)).resolve()
            if not str(path).startswith(str(ROOT)) or not path.is_file():
                self.send_json({"error": "artifact not found"}, 404)
                return
            self.send_file(path)

        def log_message(self, fmt, *args):
            print("log_lab_v2:", fmt % args)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("log_lab_v2_server=ok")
    print(f"url=http://{args.host}:{args.port}/")
    print(f"static_entry={ENTRY_HTML}")
    httpd.serve_forever()


def write_entry(host: str, port: int) -> None:
    ENTRY_HTML.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://{host}:{port}/"
    ENTRY_HTML.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>WorldSimFlow Log Lab 2.0</title><style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;background:#f4f6f8;color:#17212f;margin:0}}main{{max-width:900px;margin:0 auto;padding:42px 24px}}.panel{{background:white;border:1px solid #d0d7de;border-radius:8px;padding:24px;box-shadow:0 10px 28px rgba(15,23,42,.08)}}a{{display:inline-block;background:#2457a6;color:white;text-decoration:none;border-radius:6px;padding:10px 14px;font-weight:700}}code{{background:#eef2f6;padding:2px 5px;border-radius:4px}}p{{line-height:1.7;color:#475467}}</style></head><body><main><div class='panel'><h1>WorldSimFlow Log Lab 2.0</h1><p>这是交互式仿真实验台入口。它需要 Python 本地 API server 执行场景转换、B2/B3/B4 仿真和结果写盘。</p><p>请先在项目根目录运行：<code>python scripts\\log_lab_v2_server.py</code></p><p><a href='{url}'>进入 Log Lab 2.0</a></p></div></main></body></html>""", encoding="utf-8")


if __name__ == "__main__":
    main()

