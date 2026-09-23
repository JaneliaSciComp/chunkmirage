import os
import ssl

import pytest

from chunkmirage.cli import _parse_op, build_registry
from chunkmirage.netutil import ensure_self_signed_cert


def test_parse_op():
    assert _parse_op("threshold:low=120,high=200") == {"op": "threshold", "low": 120, "high": 200}
    assert _parse_op("cast:dtype=uint16") == {"op": "cast", "dtype": "uint16"}
    assert _parse_op('{"op": "gaussian", "sigma": 2}') == {"op": "gaussian", "sigma": 2}


def test_build_registry_with_raw_shares_cache(zarr2_path):
    reg = build_registry(zarr2_path, "thr", ["threshold:low=100"], None, raw=True)
    assert reg.names() == ["thr", "thr-raw"]
    thr, raw = reg.get("thr"), reg.get("thr-raw")
    assert thr.cache is raw.cache
    # stage 0 keys are identical -> the raw chunk is computed once for both
    assert thr.levels[0].info.chunk_shape == raw.levels[0].info.chunk_shape
    thr.chunk(0, (0, 0, 0))
    misses = reg.cache.misses
    raw.chunk(0, (0, 0, 0))
    assert reg.cache.misses == misses


def test_self_signed_cert_roundtrip(tmp_path):
    pytest.importorskip("cryptography")
    certfile, keyfile = ensure_self_signed_cert(str(tmp_path), hosts=["10.1.2.3", "myhost"])
    assert os.path.exists(certfile) and os.path.exists(keyfile)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile, keyfile)  # raises if the pair is inconsistent
    # second call reuses the files
    assert ensure_self_signed_cert(str(tmp_path)) == (certfile, keyfile)
