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
