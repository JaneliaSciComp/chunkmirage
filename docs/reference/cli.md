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
| `--chunk`             | source chunks                             | output chunk shape, e.g. `64,64,64` |
| `--host` / `--port`   | `0.0.0.0` / `8000`                        | bind address |
| `--public-url`        | `http://localhost:PORT`                   | address clients use (tunnel / proxy) |
| `--cache-gb`          | `2.0`                                     | in-process chunk cache |
| `--source-cache-gb`   | `0.5`                                     | tensorstore raw-byte cache |
| `--viewer`            | `https://neuroglancer-demo.appspot.com`   | viewer for the printed link |
| `--format`            | `zarr3`                                   | format used in the printed link |
| `--workers`           | `1`                                       | uvicorn worker processes (caches are per process) |

## `ops`

Lists registered ops with halo, cache flag and parameters.

## `inspect`

Prints shape, chunks, dtype, voxel size, units and axes per scale level as chunkmirage sees
them, which is useful for checking metadata detection before serving.
