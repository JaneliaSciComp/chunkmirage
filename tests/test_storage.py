"""Reading stores the way they let one in: credentials only when a public read is refused,
metadata reads that give up, and zarr metadata tensorstore would reject."""

import json

import numpy as np
import pytest
import tensorstore as ts

from chunkmirage import open_source
from chunkmirage.core import Box
from chunkmirage.sources import tensorstore_source as tss


def test_a_zarr_compressor_with_members_tensorstore_rejects_is_read(tmp_path):
    path = tmp_path / "zstd.zarr"
    meta = {"shape": [4, 6], "chunks": [4, 6], "dtype": "<u2", "compressor": {"id": "zstd", "level": 1}}
    spec = {"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}}
    data = np.arange(24, dtype=np.uint16).reshape(4, 6)
    ts.open({**spec, "metadata": meta}, create=True).result().write(data).result()
    zarray = json.loads((path / ".zarray").read_text())
    zarray["compressor"]["checksum"] = False  # as numcodecs >= 0.13 writes it
    (path / ".zarray").write_text(json.dumps(zarray))
    with pytest.raises(ValueError, match="extra members"):
        ts.open(spec, read=True).result()
    src = open_source(str(path))[0]
    np.testing.assert_array_equal(src.read(Box((0, 0), (4, 6))), data)
    assert tss.cleaned_zarray({"compressor": {"id": "gzip", "level": 5}}) is None  # nothing to do


def test_buckets_are_read_anonymously_first_and_gcs_falls_back_to_https():
    s3 = tss.kvstore_specs("s3://bucket-name/a/b")
    assert [s["aws_credentials"]["type"] for s in s3] == ["anonymous", "default"]
    assert {s["path"] for s in s3} == {"a/b/"}
    gs = tss.kvstore_specs("gs://bucket-name/a")
    assert gs[0] == {"driver": "gcs", "bucket": "bucket-name", "path": "a/"}
    assert gs[1] == {"driver": "http", "base_url": "https://storage.googleapis.com", "path": "/bucket-name/a/"}
    assert tss.kvstore_specs("/data/x.zarr") == [{"driver": "file", "path": "/data/x.zarr/"}]


def test_access_moves_on_only_when_refused_and_remembers_the_way_in(monkeypatch):
    monkeypatch.setattr(tss, "_admitted", {})
    specs = tss.kvstore_specs("s3://bucket-name/a")
    tried = []

    def private(spec):
        tried.append(spec["aws_credentials"]["type"])
        if spec["aws_credentials"]["type"] == "anonymous":
            raise ValueError("PERMISSION_DENIED: Access Denied")
        return "data"

    assert tss.with_access(specs, private) == "data" and tried == ["anonymous", "default"]
    tried.clear()
    assert tss.with_access(specs, private) == "data" and tried == ["default"]  # remembered

    def unreachable(spec):
        tried.append(spec)
        raise ValueError("UNAVAILABLE: host not found")

    monkeypatch.setattr(tss, "_admitted", {})
    tried.clear()
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        tss.with_access(specs, unreachable)
    assert len(tried) == 1  # not a refusal: credentials would not help


def test_metadata_reads_give_up_after_a_few_retries():
    for url, resource in [("https://host.example/a", "http_request_retries"), ("s3://bucket-name/a", "s3_request_retries"),
                          ("gs://bucket-name/a", "gcs_request_retries")]:
        spec = tss.metadata_spec(tss.kvstore_specs(url)[0])
        assert spec["context"] == {resource: tss.METADATA_RETRIES}
    assert "context" not in tss.metadata_spec(tss.kvstore_specs("/data/x.zarr")[0])
