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
