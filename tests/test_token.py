"""A server started with a token: its control API answers only with it, its datasets stay
open (viewers send no headers), and a browser's preflight is answered as before."""

from starlette.testclient import TestClient

from chunkmirage import Pipeline, create_app, open_source
from chunkmirage.cli import app as cli_app

SRC = "synthetic://blobs?shape=16,16,16&chunk=8,8,8&levels=1"


def client(token=None):
    return TestClient(create_app({"d": Pipeline(open_source(SRC), [])}, token=token))


def test_the_api_needs_the_token_and_the_data_does_not():
    c = client("s3cret")
    assert c.get("/api/ops").status_code == 401
    assert c.get("/api/ops", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/ops", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/api/datasets/d?token=s3cret").status_code == 200  # as the event stream sends it
    edit = {"source": SRC, "ops": [{"op": "threshold", "low": 100}]}
    assert c.put("/api/datasets/d", json=edit).status_code == 401
    assert c.put("/api/datasets/d", json=edit, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/d/zarr3/s0/zarr.json").status_code == 200  # a viewer's reads
    assert c.get("/d/zarr3/s0/c/0/0/0").status_code == 200
    assert c.get("/ui").status_code == 200 and "TOKEN" in c.get("/ui").text  # the page passes it on
    pre = c.options("/api/ops", headers={"Origin": "https://example.org", "Access-Control-Request-Method": "GET",
                                         "Access-Control-Request-Headers": "authorization"})
    assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] in ("*", "https://example.org")


def test_without_a_token_nothing_changes():
    c = client()
    assert c.get("/api/ops").status_code == 200
    assert c.put("/api/datasets/d", json={"source": SRC, "ops": []}).status_code == 200


def test_the_cli_takes_the_token_from_the_environment():
    from typer.main import get_command

    serve = get_command(cli_app).commands["serve"]
    token = next(p for p in serve.params if p.name == "token")
    assert token.envvar == "CHUNKMIRAGE_TOKEN"
