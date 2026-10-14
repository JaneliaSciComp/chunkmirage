"""The python op: code carried in the spec, its #halo, #output and #uicontrol lines read
without running it, run only by a server started with --allow-code, and /api changes from
another site's page refused when there is no token."""

import numpy as np
import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from chunkmirage import Pipeline, create_app, open_source
from chunkmirage.ops import op_from_spec, python_op

SRC = "synthetic://blobs?shape=16,16,16&chunk=8,8,8&levels=1"
CODE = """\
#halo 1
#output uint32 label
#uicontrol float low slider(min=0, max=255, default=128)
def apply(block, low):
    from scipy import ndimage
    return ndimage.label(block > low)[0]
"""


def test_the_lines_say_what_it_reads_and_makes_and_the_code_runs_on_a_block():
    op = op_from_spec({"op": "python", "code": CODE, "params": {"low": 100}})
    assert op.halo_for(4) == (1, 1, 1, 1) and op.output_dtype(np.dtype("uint8")) == np.uint32
    assert [(c.name, c.min, c.max, c.default) for c in op.directives.controls] == [("low", 0, 255, 128)]
    block = np.zeros((6, 6, 6), np.uint8)
    block[1:3, 1:3, 1:3] = block[4:, 4:, 4:] = 200
    out = op.apply(block)
    assert out.dtype == np.uint32 and out.max() == 2
    assert op.digest() != op_from_spec({"op": "python", "code": CODE}).digest()  # the slider's value
    renamed = op_from_spec({"op": "python", "code": CODE.replace("low", "lo"), "params": {"low": 100}})
    assert renamed.params == {}  # a slider the code no longer has: its value dropped


@pytest.mark.parametrize("code, says", [
    ("def f(block): return block", "defines no `def apply"),
    ("def apply(block):\n    return (", "line 2"),
    ("#halo -1\ndef apply(block): return block", "line 1: #halo"),
    ("#uicontrol float a slider(max=3)\ndef apply(block, a): return block", "needs min and max"),
    ("#uicontrol float a slider(min=0, max=3)\ndef apply(block): return block", "no parameter 'a'"),
    ("#output complex64\ndef apply(block): return block", "integer or float"),
])
def test_mistakes_are_named_by_line_without_running_anything(code, says):
    with pytest.raises(ValidationError, match=says):
        op_from_spec({"op": "python", "code": code})


def test_code_written_for_the_op_is_a_plugin_with_its_sliders_as_parameters():
    cls = python_op("big_labels", CODE)
    op = cls(low=50)
    assert op.spec() == {"op": "big_labels", "low": 50.0}
    assert cls.model_json_schema()["properties"]["low"]["maximum"] == 255


def client(**kw):
    return TestClient(create_app({"d": Pipeline(open_source(SRC), [])}, **kw))


def test_a_server_runs_code_sent_to_it_only_when_allowed():
    edit = {"source": SRC, "ops": [{"op": "python", "code": CODE}]}
    refused = client().put("/api/datasets/d", json=edit)
    assert refused.status_code == 403 and "--allow-code" in refused.json()["error"]
    assert client().post("/api/datasets", json={"name": "e", "spec": edit}).status_code == 403
    c = client(allow_code=True, token="t")
    r = c.put("/api/datasets/d", json=edit, headers={"Authorization": "Bearer t"})
    assert r.status_code == 200 and r.json()["ops_info"][0]["controls"][0]["name"] == "low"
    assert r.json()["levels"][0]["dtype"] == "uint32"
    assert c.get("/d/zarr3/s0/c/0/0/0").status_code == 200


def test_without_a_token_api_changes_come_only_from_its_own_pages():
    c, edit = client(), {"source": SRC, "ops": []}
    assert c.put("/api/datasets/d", json=edit).status_code == 200  # curl, a notebook
    assert c.put("/api/datasets/d", json=edit, headers={"Origin": "http://testserver"}).status_code == 200  # /ui
    other = c.put("/api/datasets/d", json=edit, headers={"Origin": "https://example.org"})
    assert other.status_code == 403
    assert c.get("/api/ops", headers={"Origin": "https://example.org"}).status_code == 200  # reads stay open
    tokened = client(token="t").put("/api/datasets/d", json=edit,
                                    headers={"Origin": "https://example.org", "Authorization": "Bearer t"})
    assert tokened.status_code == 200


def test_allow_code_generates_a_token_unless_told_not_to(monkeypatch):
    from typer.testing import CliRunner

    import chunkmirage.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "_run", lambda app, *a, **k: seen.setdefault("app", app))
    runner = CliRunner()
    out = runner.invoke(cli.app, ["serve", SRC, "--allow-code", "--port", "0", "--host", "127.0.0.1"])
    assert out.exit_code == 0, out.output
    assert "/ui?token=" in out.output
    out = runner.invoke(cli.app, ["serve", SRC, "--allow-code", "--no-token", "--port", "0", "--host", "127.0.0.1"])
    assert out.exit_code == 0 and "/ui?token=" not in out.output
    assert runner.invoke(cli.app, ["serve", SRC, "--no-token", "--port", "0"]).exit_code != 0


def test_the_command_line_reads_a_python_ops_code_from_a_file(tmp_path):
    from chunkmirage.cli import _parse_op

    (tmp_path / "labels.py").write_text(CODE)
    assert _parse_op(f"python:code=@{tmp_path}/labels.py")["code"] == CODE
    spec = _parse_op(f'{{"op": "python", "code": "@{tmp_path}/labels.py", "params": {{"low": 50}}}}')
    assert spec["code"] == CODE and spec["params"] == {"low": 50}
    import typer

    with pytest.raises(typer.BadParameter, match="no such file"):
        _parse_op("python:code=@not_there.py")
