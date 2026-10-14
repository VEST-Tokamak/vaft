#!/usr/bin/env python3
"""Local static preview server with byte ranges for precise MP4 seeking.

python docs/scripts/serve_animation_preview.py --directory /tmp/vaft-animation-preview
Binds loopback only; this is a development tool, never a Pages runtime.
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class PreviewHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining = None
        path = Path(self.translate_path(self.path))
        request = self.headers.get("Range")
        if not request or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", request)
        if not match or not any(match.groups()):
            self.send_error(416, "Unsupported byte range")
            return None
        first, last = match.groups()
        start = int(first) if first else max(0, size - int(last))
        end = min(int(last), size - 1) if first and last else size - 1
        if start > end or start >= size or (not first and int(last) == 0):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        stream = path.open("rb")
        stream.seek(start)
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self.remaining))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return stream

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def copyfile(self, source, outputfile):
        if self.remaining is None:
            return super().copyfile(source, outputfile)
        while self.remaining:
            block = source.read(min(64 * 1024, self.remaining))
            if not block:
                break
            outputfile.write(block)
            self.remaining -= len(block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("Preview directory does not exist")
    with ThreadingHTTPServer(
        ("127.0.0.1", args.port), partial(PreviewHandler, directory=str(args.directory))
    ) as server:
        print(f"Preview: http://127.0.0.1:{args.port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
