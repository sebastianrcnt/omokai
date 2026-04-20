"""Tiny static server for the omokai web demo.

Serves the `web/` directory and ensures correct MIME types for ONNX/WASM,
plus the COOP/COEP headers needed by some onnxruntime-web features.
Run from the repo root:

    python web/serve.py [--port 8080]

Then open http://localhost:8080 in a browser.
"""
from __future__ import annotations

import argparse
import http.server
import os
from pathlib import Path


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".onnx": "application/octet-stream",
        ".wasm": "application/wasm",
        ".mjs": "application/javascript",
        ".js": "application/javascript",
        ".json": "application/json",
    }

    def end_headers(self):
        # Permissive CORS so the page can also be served from another origin if needed.
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    os.chdir(args.root)
    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"omokai web demo serving {args.root}")
    print(f"  → http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
