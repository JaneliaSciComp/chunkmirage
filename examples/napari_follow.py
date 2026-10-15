"""napari follows a chunkmirage server: its datasets opened as layers, each refreshed when
the server says a pipeline changed (a slider on the control page, new code, a PUT), so napari
shows what the python viewer shows, keeping its view and contrast settings.

    uv run --with "napari[all]" --with napari-ome-zarr examples/napari_follow.py \\
        'http://localhost:8004/ui?token=...' raw processed

(or `python examples/napari_follow.py ...` where napari and napari-ome-zarr are installed).
The two arguments are the only things that change from one server to the next:
  - the control UI link the server printed: the port it chose and its token (a server
    restarted has a new token, so a new link);
  - the datasets to open, one layer each: raw and processed for a plain `chunkmirage serve`;
    the --name given, in place of processed; no raw with --no-raw; every name wanted when a
    server has more (a Python registry).
Nothing here is about the pipeline: the same script follows any server, any ops, any
Python, since the server does the work and this only fetches again. Labels open as an image
(the server's zarr does not mark them); right-click the layer to convert it to Labels.

How: napari reads the zarr through dask, computing a slice when it is shown, and keeps a cache
of recent slices. On each event that cache is emptied (resized to nothing and back) and the
layers refreshed, so each computes its slice again, and the server answers with the current
pipeline (its addresses carry no digest, so they always mean the pipeline as it is now).
Between edits napari caches as it always does.
"""

import json
import sys
import urllib.request
from collections.abc import Iterator
from urllib.parse import parse_qs, urlsplit


def changes(server: str, token: str | None = None) -> Iterator[dict]:
    """The server's ``change`` events, each the JSON it carries (version, datasets with their
    digests and sources), after the first, which is the state as it is when the stream opens."""
    url = f"{server}/api/events" + (f"?token={token}" if token else "")
    request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(request) as stream:  # noqa: S310  (the server the user named)
        first = True
        for raw in stream:
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line.startswith("data:"):
                continue  # event names and keepalives
            if first:
                first = False
                continue
            yield json.loads(line[len("data:"):])


def split_link(link: str) -> tuple[str, str | None]:
    """The server and its token from the control UI link the server printed."""
    url = urlsplit(link.strip())
    if not url.scheme or not url.netloc:
        raise SystemExit(f"not a link to a chunkmirage server: {link!r}")
    return f"{url.scheme}://{url.netloc}", parse_qs(url.query).get("token", [None])[0]


def main(link: str, *names: str) -> None:
    import napari
    from napari.qt.threading import thread_worker
    from napari.utils import resize_dask_cache
    from napari_ome_zarr import napari_get_reader

    server, token = split_link(link)
    names = names or ("raw", "processed")
    viewer = napari.Viewer(title=f"chunkmirage: {' '.join(names)}")
    for name in names:
        # the reader itself, not viewer.open: napari offers a reader only to paths matching
        # its patterns (*.zarr), and the server's addresses end in /zarr
        url = f"{server}/{name}/zarr"
        reader = napari_get_reader(url)
        if reader is None:
            raise SystemExit(f"{url}: nothing readable as OME-Zarr there (is {name!r} a dataset of this server?)")
        for data, kwargs, kind in reader(url):
            getattr(viewer, f"add_{kind}")(data, **{**kwargs, "name": name})

    def refresh(change: dict) -> None:
        resize_dask_cache(0)  # the slices napari has: else one shown once stays as it was
        resize_dask_cache()  # and caching again, at its usual size, until the next edit
        for layer in viewer.layers:
            layer.refresh()
        print(f"chunkmirage: version {change.get('version')}, layers refreshed")

    @thread_worker(connect={"yielded": refresh})  # refresh runs in napari's thread
    def follow():
        yield from changes(server, token)

    follow()
    print(f"chunkmirage: following {server}; edit a pipeline and napari refreshes")
    napari.run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
