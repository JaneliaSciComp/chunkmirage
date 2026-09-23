# CLI

```
chunkmirage serve SOURCE [options]
chunkmirage ops
chunkmirage inspect SOURCE
```

## `serve`

| option                | default                                   | meaning |
| --------------------- | ----------------------------------------- | ------- |
| `--name`              | `data`                                    | dataset name in URLs |
| `--op`, `-o`          |                                           | op spec, repeatable; `name:k=v,k=v` or JSON |
| `--raw` / `--no-raw`  | on                                        | also serve the unprocessed source as `raw`; shares the cache, appears as a second layer |
| `--chunk`             | source chunks                             | output chunk shape, e.g. `64,64,64` |
| `--host` / `--port`   | `0.0.0.0` / `8000`                        | bind address |
| `--https`             | off                                       | serve https; a self-signed certificate is generated in `~/.cache/chunkmirage/` on first use (needs the `https` extra or the `openssl` CLI) |
| `--cert` / `--key`    | auto-generated                            | use your own certificate and key with `--https` |
| `--public-url`        | `http(s)://<lan-ip>:PORT`                 | address clients use in every printed link and layer URL; defaults to this machine's network address when binding `0.0.0.0`, `localhost` when binding `127.0.0.1`; set explicitly behind a tunnel or proxy |
| `--cache-gb`          | `2.0`                                     | in-process chunk cache |
| `--source-cache-gb`   | `0.5`                                     | tensorstore raw-byte cache |
| `--viewer`            | `https://neuroglancer-demo.appspot.com`   | viewer for the printed link |
| `--format`            | `zarr3`                                   | format used in the printed link |
| `--workers`           | `1`                                       | uvicorn worker processes (caches are per process) |
| `--python-viewer`     | off                                       | also start a python-neuroglancer viewer whose layers follow live edits (needs the `viewer` extra) |
| `--ng-client`         | `bundled`                                 | client build for the python viewer: `bundled`, `appspot`, or a URL |
| `--viewer-host`       | same as `--host`                          | bind address of the python viewer; `0.0.0.0` lets other machines open it |
| `--viewer-port`       | random                                    | fixed port for the python viewer (handy for port forwarding) |

`serve` prints the source URL, an appspot link, the control page (`/ui`), the control API
URL and, with `--python-viewer`, the viewer URL.

## `ops`

Lists registered ops with halo, cache flag and parameters.

## `inspect`

Prints shape, chunks, dtype, voxel size, units and axes per scale level as chunkmirage sees
them, which is useful for checking metadata detection before serving.
