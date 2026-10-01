"""chunkmirage schema: the one definition of pipelines, ops and register:// parameters that
the browser engine (web/) generates its types from."""

import json
from pathlib import Path

from typer.testing import CliRunner

from chunkmirage.cli import app
from chunkmirage.ops.base import list_ops
from chunkmirage.schema import dumps, spec_schema
from chunkmirage.sources.register import RegisterParams

WEB_COPY = Path(__file__).parents[1] / "web" / "src" / "generated" / "chunkmirage.schema.json"


def test_the_browser_engine_has_the_current_schema():
    # after changing a model: chunkmirage schema --out web/src/generated/chunkmirage.schema.json,
    # then npm run gen in web/ (CI checks the generated TypeScript matches)
    assert WEB_COPY.read_text() == dumps()


def test_it_defines_every_model_once():
    defs = spec_schema()["$defs"]
    assert {"PipelineSpec", "RegisterParams", "OpSpec"} <= set(defs)
    union = {ref["$ref"].rsplit("/", 1)[1] for ref in defs["OpSpec"]["oneOf"]}
    assert {defs[t]["properties"]["op"]["const"] for t in union} == set(list_ops())
    for title in union:  # an op is told apart by its name
        assert defs[title]["required"][0] == "op"


def test_register_defaults_come_from_the_model():
    props = spec_schema()["$defs"]["RegisterParams"]["properties"]
    model = RegisterParams(fixed="f")
    for name in (
        "fixed_channel",
        "moving_channel",
        "iterations",
        "smooth",
        "grid",
        "window",
        "show",
    ):
        assert props[name]["default"] == getattr(model, name)
    assert props["show"]["enum"] == ["image", "pair", "field"]


def test_register_params_read_a_query():
    p = RegisterParams.from_query(
        {"fixed": "a.zarr", "levels": "6,5", "iterations": "50", "smooth": "2"}
    )
    assert (p.levels, p.iterations, p.smooth, p.window) == ([6, 5], [50], 2.0, [7])


def test_the_cli_prints_and_writes_it(tmp_path):
    out = CliRunner().invoke(app, ["schema"])
    assert out.exit_code == 0 and json.loads(out.stdout) == spec_schema()
    path = tmp_path / "schema.json"
    assert CliRunner().invoke(app, ["schema", "--out", str(path)]).exit_code == 0
    assert path.read_text() == dumps()
