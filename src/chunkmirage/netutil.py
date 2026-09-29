"""Small network helpers for building URLs other machines can use."""

from __future__ import annotations

import socket


def lan_ip() -> str:
    """Best-effort address of this machine on its network (no packets are sent).

    Falls back to ``127.0.0.1`` when the machine has no route (e.g. offline laptop).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # unroutable target; only picks the outbound interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def free_port(host: str = "127.0.0.1", start: int = 8000, *, avoid=(), tries: int = 100) -> int:
    """The first port from ``start`` up that a server could bind on ``host`` (skipping
    ``avoid``). It is checked by binding, so another process could still take it before
    the server starts; in practice that race does not happen."""
    for port in range(start, start + tries):
        if port in avoid:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as servers do
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"no free port in {start}..{start + tries - 1} on {host}")


def public_host_for(bind_host: str) -> str:
    """Host other machines should use to reach a server bound to ``bind_host``."""
    if bind_host in ("0.0.0.0", "::", ""):
        return lan_ip()
    if bind_host in ("127.0.0.1", "::1", "localhost"):
        return "localhost"
    return bind_host


def serving_address(
    host: str, port: int, *, https: bool, cert: str | None = None, key: str | None = None
) -> tuple[str, dict]:
    """Base URL that browsers, including other machines', use for a server bound to
    ``host:port`` (the network IP when bound to every interface), and the uvicorn
    ``ssl_*`` arguments. With ``https`` and no ``cert``/``key``, a self-signed certificate
    for that address is generated (or reused)."""
    public_host = public_host_for(host)
    ssl: dict = {}
    if https:
        if not (cert and key):
            cert, key = ensure_self_signed_cert(hosts=[public_host])
        ssl = {"ssl_certfile": cert, "ssl_keyfile": key}
    return f"{'https' if https else 'http'}://{public_host}:{port}", ssl


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost")


def default_cert_dir() -> str:
    import os

    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "chunkmirage")


CERT_DAYS = 397  # browsers and macOS refuse server certificates valid for longer


def _cert_usable(certfile: str) -> bool:
    """Whether a certificate made earlier still meets the rules: for server authentication,
    valid for at most ``CERT_DAYS`` (plus the day it starts early), and not expiring
    within a week. Read with ``cryptography`` when present, else the ``openssl`` CLI."""
    import datetime as dt

    try:
        from cryptography import x509
    except ImportError:
        return _openssl_cert_usable(certfile)
    try:
        with open(certfile, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())
        usage = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    except (ValueError, x509.ExtensionNotFound):
        return False
    after, before = cert.not_valid_after_utc, cert.not_valid_before_utc
    return (
        x509.oid.ExtendedKeyUsageOID.SERVER_AUTH in usage
        and (after - before).days <= CERT_DAYS + 1
        and after > dt.datetime.now(dt.UTC) + dt.timedelta(days=7)
    )


def _openssl_cert_usable(certfile: str) -> bool:
    import datetime as dt
    import subprocess

    week = str(7 * 24 * 3600)
    cmd = ["openssl", "x509", "-in", certfile, "-noout", "-dates", "-ext", "extendedKeyUsage"]
    cmd += ["-checkend", week]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return True  # no openssl either: cannot tell, keep what is there
    if r.returncode != 0:  # expiring within the week, or unreadable
        return False
    fields = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    try:
        before, after = (
            dt.datetime.strptime(fields[k].strip(), "%b %d %H:%M:%S %Y %Z")
            for k in ("notBefore", "notAfter")
        )
    except (KeyError, ValueError):
        return False
    usage = "TLS Web Server Authentication" in r.stdout
    return usage and (after - before).days <= CERT_DAYS + 1


def ensure_self_signed_cert(
    cert_dir: str | None = None, hosts: list[str] | None = None
) -> tuple[str, str]:
    """Return ``(certfile, keyfile)``, generating a self-signed pair on first use.

    Why: the hosted Neuroglancer at ``https://neuroglancer-demo.appspot.com`` may only fetch
    plain ``http`` from ``localhost``. To use it against a server on another machine the
    server must speak https. A self-signed certificate is enough once the browser has been
    told to trust it (visit the server URL once and accept the warning).

    The certificate's subject alternative names include ``hosts`` (default: this machine's
    LAN IP and hostname) plus ``localhost``. It meets the rules browsers and macOS apply
    even to certificates a user chose to trust (at most 398 days, for server
    authentication), so it can be trusted once in the system instead of clicked through;
    a service worker, which a clicked-through certificate does not allow, then works. One
    that does not meet them (older versions made ten-year ones), or has expired, is made
    again. Uses the ``cryptography`` package when present, else the ``openssl`` CLI.
    """
    import datetime as dt
    import os
    import shutil
    import subprocess

    cert_dir = cert_dir or default_cert_dir()
    os.makedirs(cert_dir, exist_ok=True)
    certfile = os.path.join(cert_dir, "chunkmirage.crt")
    keyfile = os.path.join(cert_dir, "chunkmirage.key")
    if os.path.exists(certfile) and os.path.exists(keyfile) and _cert_usable(certfile):
        return certfile, keyfile

    hosts = list(hosts or [lan_ip(), socket.gethostname()])
    hosts = [h for h in dict.fromkeys(hosts + ["localhost", "127.0.0.1"]) if h]

    try:
        import ipaddress

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        if not shutil.which("openssl"):
            raise RuntimeError(
                "need either the 'cryptography' package (pip install chunkmirage[https]) "
                "or the openssl CLI to generate a certificate"
            ) from None
        san = ",".join(f"IP:{h}" if h.replace(".", "").isdigit() else f"DNS:{h}" for h in hosts)
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                str(CERT_DAYS),
                "-keyout",
                keyfile,
                "-out",
                certfile,
                "-subj",
                "/CN=chunkmirage",
                "-addext",
                f"subjectAltName={san}",
                "-addext",
                "extendedKeyUsage=serverAuth",
                "-addext",
                "basicConstraints=critical,CA:FALSE",
            ],
            check=True,
            capture_output=True,
        )
        return certfile, keyfile

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "chunkmirage")])
    sans = []
    for h in hosts:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=CERT_DAYS))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    with open(keyfile, "wb") as f:
        f.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )
    with open(certfile, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return certfile, keyfile
