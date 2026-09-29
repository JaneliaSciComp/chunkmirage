import ipaddress

import pytest

from chunkmirage.netutil import lan_ip, public_host_for


def test_lan_ip_is_an_ipv4_address():
    ipaddress.IPv4Address(lan_ip())  # raises if malformed


def test_public_host_for():
    assert public_host_for("127.0.0.1") == "localhost"
    assert public_host_for("localhost") == "localhost"
    assert public_host_for("10.1.2.3") == "10.1.2.3"
    ipaddress.IPv4Address(public_host_for("0.0.0.0"))


def test_viewer_url_uses_public_host(zarr2_path):
    pytest.importorskip("neuroglancer")
    from chunkmirage.server import DatasetRegistry
    from chunkmirage.viewer import Viewer

    reg = DatasetRegistry()
    reg.add("raw", {"source": zarr2_path, "ops": []})
    v = Viewer(reg, "http://10.9.8.7:8000", public_host="10.9.8.7")
    try:
        assert v.url.startswith("http://10.9.8.7:")
        assert "/v/" in v.url
        assert (
            v.viewer.state.layers["raw"]
            .source[0]
            .url.startswith("zarr3://http://10.9.8.7:8000/raw/@")
        )
    finally:
        v.close()


def test_free_port_skips_ports_in_use_and_avoided():
    import socket

    from chunkmirage.netutil import free_port

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        port = free_port("127.0.0.1", taken, tries=20)
        assert port != taken and taken < port < taken + 20
        assert free_port("127.0.0.1", taken, avoid={port}, tries=20) not in (taken, port)


def test_serving_address_uses_the_network_ip_and_given_certificate(monkeypatch):
    from chunkmirage import netutil

    monkeypatch.setattr(netutil, "lan_ip", lambda: "10.1.2.3")
    assert netutil.serving_address("0.0.0.0", 8005, https=False) == ("http://10.1.2.3:8005", {})
    assert netutil.serving_address("127.0.0.1", 8005, https=False)[0] == "http://localhost:8005"
    url, ssl = netutil.serving_address("0.0.0.0", 8005, https=True, cert="c.pem", key="k.pem")
    assert url == "https://10.1.2.3:8005"
    assert ssl == {"ssl_certfile": "c.pem", "ssl_keyfile": "k.pem"}
    assert netutil.is_loopback("127.0.0.1") and not netutil.is_loopback("0.0.0.0")


def test_certificate_meets_the_rules_for_trusting_it(tmp_path):
    x509 = pytest.importorskip("cryptography.x509")
    from chunkmirage import netutil

    cert_file, _ = netutil.ensure_self_signed_cert(str(tmp_path), hosts=["10.1.2.3"])
    cert = x509.load_pem_x509_certificate(open(cert_file, "rb").read())
    usage = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert x509.oid.ExtendedKeyUsageOID.SERVER_AUTH in usage
    assert (cert.not_valid_after_utc - cert.not_valid_before_utc).days <= 398
    names = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "localhost" in names.get_values_for_type(x509.DNSName)
    # reused while it is good; a ten-year certificate from before is made again
    assert netutil.ensure_self_signed_cert(str(tmp_path))[0] == cert_file
    mtime = (tmp_path / "chunkmirage.crt").stat().st_mtime_ns
    assert netutil._cert_usable(cert_file)
    (tmp_path / "chunkmirage.crt").write_text("not a certificate")
    netutil.ensure_self_signed_cert(str(tmp_path), hosts=["10.1.2.3"])
    assert (
        netutil._cert_usable(cert_file)
        and (tmp_path / "chunkmirage.crt").stat().st_mtime_ns != mtime
    )


def test_without_cryptography_openssl_applies_the_same_rules(tmp_path, monkeypatch):
    import shutil
    import subprocess
    import sys

    if not shutil.which("openssl"):
        pytest.skip("no openssl")
    from chunkmirage import netutil

    monkeypatch.setitem(sys.modules, "cryptography", None)  # as without the https extra
    cert_file, _ = netutil.ensure_self_signed_cert(str(tmp_path), hosts=["10.1.2.3"])
    assert netutil._cert_usable(cert_file)
    old = tmp_path / "old.crt"  # what older versions made: ten years, no server usage
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650"]
        + ["-keyout", str(tmp_path / "old.key"), "-out", str(old), "-subj", "/CN=old"],
        check=True,
        capture_output=True,
    )
    assert not netutil._cert_usable(str(old))
