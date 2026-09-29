"""Serve the browser pages (register.html) over https, so WebGPU is allowed on them.

    uv run python examples/browser/serve.py [--port N] [--host 0.0.0.0] [-- register.html?...]

Browsers only allow WebGPU on secure pages (https, or localhost), so a plain
``http://<machine IP>`` link would not work from another computer. This serves this
directory with chunkmirage's self-signed certificate for the machine's IP and prints a
shareable link; each browser trusts the certificate once. Nothing here computes
anything: the page reads the images from their own URLs and fits on the viewer's GPU.
Any static https host (GitHub Pages, say) serves the page just as well.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import ssl
from pathlib import Path

HERE = Path(__file__).resolve().parent


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
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
    server = http.server.ThreadingHTTPServer((args.host, port), handler)
    if https:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certs["ssl_certfile"], certs["ssl_keyfile"])
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
    print(f"page:  {address}/{args.page}", flush=True)
    if https:
        print(
            "https: self-signed certificate: each browser must trust it once (open the link "
            "and accept the warning)",
            flush=True,
        )
    server.serve_forever()


if __name__ == "__main__":
    main()
