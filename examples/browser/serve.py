"""Serve the browser pages (register.html) over https, with Neuroglancer on the same origin.

    uv run python examples/browser/serve.py [--port N] [--host 0.0.0.0] [PAGE?QUERY]

Browsers only allow WebGPU and service workers on secure pages (https, or localhost), so a
plain ``http://<machine IP>`` link would not work from another computer. This serves this
directory with chunkmirage's self-signed certificate for the machine's IP and prints a
shareable link. Clicking through the certificate warning is enough for WebGPU but not for
the service worker the viewer needs: each computer trusts the certificate once in its
system (it is served at ``/certificate.crt``). ``--host 127.0.0.1`` serves plain http
instead, for a page opened as ``http://localhost:<port>`` (forward the port if the browser
runs elsewhere), which needs no certificate.

``/ng/`` is the standard Neuroglancer client, relayed from neuroglancer-demo.appspot.com
(fetched once, then kept in memory). It has to be on this origin: the page's service
worker answers the viewer's requests for the registered volume, and a service worker
only sees requests from pages on its own origin. Nothing here computes anything: the
page reads the images from their own URLs and does all the work in the viewer's browser,
so any static https host (GitHub Pages, say, with the client copied under ``ng/``) serves
it just as well.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import re
import ssl
import threading
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEUROGLANCER = "https://neuroglancer-demo.appspot.com"


class Handler(http.server.SimpleHTTPRequestHandler):
    client: dict[str, tuple[bytes, str, str | None]] = {}  # the relayed Neuroglancer files
    lock = threading.Lock()
    certificate: str | None = None

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path == "/ng" or path.startswith("/ng/"):
            return self.neuroglancer(path[len("/ng/") :] or "index.html")
        if path == "/certificate.crt" and self.certificate:  # the public half, to trust it
            body = Path(self.certificate).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-x509-ca-cert")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        return super().do_GET()

    def neuroglancer(self, name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            return self.send_error(404)
        with self.lock:
            hit = self.client.get(name)
        if hit is None:
            try:
                with urllib.request.urlopen(f"{NEUROGLANCER}/{name}", timeout=60) as r:
                    hit = (r.read(), r.headers.get("Content-Type", "application/octet-stream"),
                           r.headers.get("Content-Encoding"))  # fmt: skip
            except urllib.error.HTTPError as e:
                return self.send_error(e.code)
            with self.lock:
                self.client[name] = hit
        body, kind, encoding = hit
        self.send_response(200)
        self.send_header("Content-Type", kind)
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")  # pick up edits to the page and workers
        super().end_headers()

    def log_message(self, *args):  # quiet: the page reports its own progress
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("page", nargs="?", default="register.html", help="page and query to link to")
    ap.add_argument("--host", default="0.0.0.0", help="bind address (default: every interface)")
    ap.add_argument("--port", type=int, help="port (default: 8443 or the next free)")
    args = ap.parse_args()

    from chunkmirage.netutil import free_port, is_loopback, serving_address

    port = args.port or free_port(args.host, 8443)
    https = not is_loopback(args.host)  # localhost is a secure context without it
    address, certs = serving_address(args.host, port, https=https)
    handler = functools.partial(Handler, directory=str(HERE))
    server = http.server.ThreadingHTTPServer((args.host, port), handler)
    if https:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certs["ssl_certfile"], certs["ssl_keyfile"])
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        Handler.certificate = certs["ssl_certfile"]
    print(f"page:  {address}/{args.page}", flush=True)
    if https:
        print(
            "https: a self-signed certificate. The viewer's service worker needs it trusted, "
            "not just clicked through: download\n"
            f"       {address}/certificate.crt and trust it (on a Mac: Keychain Access, "
            "Always Trust).\n"
            "       Or run with --host 127.0.0.1 and open the page through a forwarded port "
            "(localhost needs no certificate).",
            flush=True,
        )
    server.serve_forever()


if __name__ == "__main__":
    main()
