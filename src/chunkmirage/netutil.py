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


def public_host_for(bind_host: str) -> str:
    """Host other machines should use to reach a server bound to ``bind_host``."""
    if bind_host in ("0.0.0.0", "::", ""):
        return lan_ip()
    if bind_host in ("127.0.0.1", "::1", "localhost"):
        return "localhost"
    return bind_host


def default_cert_dir() -> str:
    import os

    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "chunkmirage")


def ensure_self_signed_cert(
    cert_dir: str | None = None, hosts: list[str] | None = None
) -> tuple[str, str]:
    """Return ``(certfile, keyfile)``, generating a self-signed pair on first use.

    Why: the hosted Neuroglancer at ``https://neuroglancer-demo.appspot.com`` may only fetch
    plain ``http`` from ``localhost``. To use it against a server on another machine the
    server must speak https. A self-signed certificate is enough once the browser has been
    told to trust it (visit the server URL once and accept the warning).

    The certificate's subject alternative names include ``hosts`` (default: this machine's
    LAN IP and hostname) plus ``localhost``. Uses the ``cryptography`` package when present,
    else the ``openssl`` CLI.
    """
    import datetime as dt
    import os
    import shutil
    import subprocess

    cert_dir = cert_dir or default_cert_dir()
    os.makedirs(cert_dir, exist_ok=True)
    certfile = os.path.join(cert_dir, "chunkmirage.crt")
    keyfile = os.path.join(cert_dir, "chunkmirage.key")
    if os.path.exists(certfile) and os.path.exists(keyfile):
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
                "3650",
                "-keyout",
                keyfile,
                "-out",
                certfile,
                "-subj",
                "/CN=chunkmirage",
                "-addext",
                f"subjectAltName={san}",
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
        .not_valid_after(now + dt.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
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
