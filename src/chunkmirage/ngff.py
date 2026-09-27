"""OME-Zarr 0.6 (RFC-5) coordinate systems, transformations and scenes.

``Scene(url)`` reads a scene group (or a single multiscale image) into
``chunkmirage.transforms`` objects and a graph whose nodes are coordinate systems, so the
transform between any two coordinate systems can be looked up with ``Scene.transform``.
Array-backed parameters (displacement and coordinate fields, matrices stored under
``path``) are opened with tensorstore; fields are read lazily, window by window.

Also accepted, because real files use them: fields stored as a bare array with the vector
axis last (the 0.6 drafts, BigWarp's export), and string ``input``/``output`` references.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from chunkmirage.sources.tensorstore_source import (
    _detect_driver,
    _node_attrs,
    _open_kvstore,
    open_tensorstore,
)
from chunkmirage.transforms import (
    Affine,
    Bijection,
    ByDimension,
    Coordinates,
    Displacements,
    Sequence,
    Transform,
    VectorField,
    simplify,
)

# A coordinate system is identified by (image path relative to the scene root, name).
# The path is None for coordinate systems defined in the scene metadata itself and "" for
# a multiscale image at the root of the hierarchy.
Node = tuple[str | None, str]


@dataclass(frozen=True)
class FieldOptions:
    """How displacement and coordinate fields are read.

    Fields are sampled again for every output chunk, so their decoded chunks are cached
    (``cache_bytes`` per field). A store decodes whole chunks: a field saved in huge chunks
    would cost one full decode per output chunk, several at once under a viewer's parallel
    requests, which can exhaust the machine's memory. Such fields are refused unless
    ``allow_large_chunks``, and then decoded one chunk at a time.
    """

    cache_bytes: int = 64 << 20
    max_chunk_bytes: int = 256 << 20
    allow_large_chunks: bool = False


DEFAULT_FIELDS = FieldOptions()


class LargeFieldChunks(ValueError):
    pass


class UnsupportedTransform(ValueError):
    pass


@dataclass
class CoordinateSystem:
    name: str
    axes: list[dict]

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def names(self) -> list[str]:
        return [str(a.get("name", f"d{i}")) for i, a in enumerate(self.axes)]

    @property
    def types(self) -> list[str]:
        return [str(a.get("type") or "") for a in self.axes]

    @property
    def units(self) -> list[str]:
        return [str(a.get("unit") or "") for a in self.axes]

    @property
    def spatial(self) -> list[int]:
        return [i for i, t in enumerate(self.types) if t == "space"]


def _join(base: str, path: str | None) -> str:
    if not path:
        return base.rstrip("/")
    return base.rstrip("/") + "/" + str(path).strip("/")


def _ref(ref) -> tuple[str | None, str | None]:
    """``{"name", "path"}`` (0.6) or a bare name (0.6 drafts) -> (path, name)."""
    if isinstance(ref, str):
        return None, ref
    if isinstance(ref, dict):
        return (str(ref.get("path") or "").strip("/") or None), ref.get("name")
    return None, None


def _multiscales(attrs: dict) -> list[dict]:
    ms = attrs.get("multiscales")
    if isinstance(ms, dict):  # 0.6 drafts
        return [ms]
    return [m for m in ms if isinstance(m, dict)] if isinstance(ms, list) else []


def _axes(raw) -> list[dict]:
    """Axes as dicts; 0.3-style axis names become ``{"name": n, "type": ...}``."""
    kinds = {"x": "space", "y": "space", "z": "space", "t": "time", "c": "channel"}
    out = []
    for a in raw or []:
        out.append(a if isinstance(a, dict) else {"name": str(a), "type": kinds.get(str(a), "")})
    return out


def _read_array(url: str) -> np.ndarray:
    return np.asarray(open_tensorstore(url).read().result(), dtype=float)


def parse_transform(
    obj: dict,
    base: str,
    ndim_in: int | None = None,
    ndim_out: int | None = None,
    *,
    fields: FieldOptions = DEFAULT_FIELDS,
) -> Transform:
    """One transformation object -> ``Transform``. ``base`` is the URL of the group whose
    metadata holds ``obj``; ``path`` parameters resolve against it. ``ndim_in``/``ndim_out``
    are the dimensionalities of the referenced coordinate systems, where known.
    ``fields`` says how the displacement/coordinate fields it opens are read."""
    kw = {"fields": fields}
    if not isinstance(obj, dict):
        raise ValueError(f"not a transformation: {obj!r}")
    typ = obj.get("type")

    def params(key: str) -> np.ndarray:
        if key in obj:
            return np.asarray(obj[key], dtype=float)
        if obj.get("path"):
            return _read_array(_join(base, obj["path"]))
        raise ValueError(f"{typ} transformation needs {key!r} or 'path'")

    if typ == "identity":
        n = ndim_in if ndim_in is not None else ndim_out
        if n is None:
            raise ValueError("identity transformation of unknown dimensionality")
        return Affine.identity(n)
    if typ == "scale":
        return Affine.scale_translation(params("scale").ravel())
    if typ == "translation":
        t = params("translation").ravel()
        return Affine.from_linear(np.eye(len(t)), t)
    if typ == "affine":
        m = params("affine")
        if m.ndim == 1 and ndim_in is not None:
            m = m.reshape(-1, ndim_in + 1)
        return Affine(m)
    if typ == "rotation":
        return Affine.from_linear(params("rotation"))
    if typ == "mapAxis":
        perm = [int(i) for i in obj["mapAxis"]]
        m = np.zeros((len(perm), len(perm) + 1))
        for i, p in enumerate(perm):
            m[i, p] = 1.0  # output axis i is input axis perm[i]
        return Affine(m)
    if typ == "projectAxis":
        dropped = {int(i) for i in obj.get("droppedInputs", [])}
        created = {int(i) for i in obj.get("createdOutputs", [])}
        n_in = ndim_in if ndim_in is not None else (ndim_out - len(created) + len(dropped))
        kept = [i for i in range(n_in) if i not in dropped]
        n_out = len(kept) + len(created)
        m = np.zeros((n_out, n_in + 1))
        for j, i in zip([j for j in range(n_out) if j not in created], kept):
            m[j, i] = 1.0  # created outputs stay 0
        return Affine(m)
    if typ == "sequence":
        parts, d = [], ndim_in
        items = obj.get("transformations") or []
        for k, child in enumerate(items):
            t = parse_transform(child, base, d, ndim_out if k == len(items) - 1 else None, **kw)
            parts.append(t)
            d = t.ndim_out
        return Sequence(parts)
    if typ in ("displacements", "coordinates"):
        if not obj.get("path"):
            raise ValueError(f"{typ} transformation needs 'path'")
        order = 0 if obj.get("interpolation") == "nearest" else 1  # cubic is read as linear
        f = open_field(_join(base, obj["path"]), order=order, fields=fields)
        return Displacements(f) if typ == "displacements" else Coordinates(f)
    if typ == "bijection":
        fwd = parse_transform(obj["forward"], base, ndim_in, ndim_out, **kw)
        inv = parse_transform(obj["inverse"], base, ndim_out, ndim_in, **kw)
        return Bijection(fwd, inv)
    if typ == "byDimension":
        parts = []
        for item in obj.get("transformations") or []:
            # camelCase per the 0.6 spec; snake_case in earlier drafts
            ia = list(item["inputAxes"] if "inputAxes" in item else item["input_axes"])
            oa = list(item["outputAxes"] if "outputAxes" in item else item["output_axes"])
            parts.append(
                (parse_transform(item["transformation"], base, len(ia), len(oa), **kw), ia, oa)
            )
        n_in = ndim_in if ndim_in is not None else 1 + max(a for _, ia, _ in parts for a in ia)
        n_out = ndim_out if ndim_out is not None else 1 + max(a for _, _, oa in parts for a in oa)
        return ByDimension(parts, n_in, n_out)
    raise UnsupportedTransform(f"unsupported coordinate transformation type {typ!r}")


def _to_affine(cts, ndim: int, base: str) -> Affine:
    """A dataset's ``coordinateTransformations`` list (index -> intrinsic) as one affine."""
    t: Transform = Affine.identity(ndim)
    for ct in cts if isinstance(cts, list) else []:
        t = Sequence([t, parse_transform(ct, base, ndim, ndim)])
    t = simplify(t)
    if not isinstance(t, Affine):
        raise ValueError(f"{base}: dataset transformations must be scale/translation")
    return t


def open_field(url: str, order: int = 1, fields: FieldOptions = DEFAULT_FIELDS) -> VectorField:
    """A displacement or coordinate field: an OME-Zarr multiscale image (0.6), or a bare
    array whose own ``ome`` attributes carry the axes and grid transform (0.6 drafts)."""
    kv = _open_kvstore(url)
    driver = _detect_driver(kv)
    attrs = _node_attrs(kv)
    if driver is not None and driver.endswith("-group"):
        ms = (_multiscales(attrs) or [None])[0]
        if ms is None or not ms.get("datasets"):
            raise ValueError(f"{url}: field group has no multiscales datasets")
        ds = ms["datasets"][0]
        array_url = _join(url, ds["path"])
        systems = ms.get("coordinateSystems") or [{"axes": ms.get("axes", [])}]
        axes, cts = _axes(systems[0].get("axes")), ds.get("coordinateTransformations", [])
    else:
        array_url = url
        systems = attrs.get("coordinateSystems") or [{"axes": attrs.get("axes", [])}]
        axes, cts = _axes(systems[0].get("axes")), attrs.get("coordinateTransformations", [])
    store = open_tensorstore(array_url, cache_bytes=fields.cache_bytes)
    shape = tuple(int(s) for s in store.shape)
    read_chunk = tuple(int(c) for c in store.chunk_layout.read_chunk.shape)
    chunk_bytes = math.prod(read_chunk) * store.dtype.numpy_dtype.itemsize
    large = chunk_bytes > fields.max_chunk_bytes
    if large and not fields.allow_large_chunks:
        gib = chunk_bytes / 2**30
        raise LargeFieldChunks(
            f"{array_url}: field chunks are {read_chunk} ({gib:.1f} GiB each). Every output "
            "chunk would decode a whole one, several at once under a viewer, which can exhaust "
            "memory. Rechunk the field to about 64^3 (or shard it with small inner chunks), or "
            f"add large_field_chunks=1 to decode them one at a time (then field_cache_gb above "
            f"{gib:.1f} decodes each only once; memory is then about that plus {2 * gib:.0f} GiB)."
        )
    types = [a.get("type") for a in axes] if len(axes) == len(shape) else []
    vec = next((i for i, t in enumerate(types) if t in ("displacement", "coordinate")), None)
    if vec is None:
        vec = len(shape) - 1  # drafts, BigWarp: vectors last
    # The grid transform may cover all array axes (vector axis included) or only the grid.
    try:
        grid = _to_affine(cts, len(shape), url)
        keep = [i for i in range(len(shape)) if i != vec]
        grid = Affine(np.hstack([grid.linear[np.ix_(keep, keep)], grid.offset[keep, None]]))
    except (ValueError, IndexError):
        grid = _to_affine(cts, len(shape) - 1, url)
    lock = threading.Lock() if large else None

    def read(sl):
        if lock is None:
            return store[sl].read().result()
        with lock:  # allowed large chunks: one decode at a time bounds transient memory
            return store[sl].read().result()

    return VectorField(read, shape, vec, grid, key=array_url, order=order)


@dataclass
class Image:
    """One multiscale image of a scene: its coordinate systems, levels, and own transforms."""

    path: str
    url: str
    systems: dict[str, CoordinateSystem]
    intrinsic: str
    datasets: list[str]
    level_affines: list[Affine]  # array index -> intrinsic, per level
    edges: list[tuple[Node, Node, Transform]] = field(default_factory=list)
    is_label: bool = False

    @property
    def cs(self) -> CoordinateSystem:
        return self.systems[self.intrinsic]


def read_image(root: str, path: str, fields: FieldOptions = DEFAULT_FIELDS) -> Image | None:
    """The multiscale image at ``root/path`` (``path=""``: the root itself), or None."""
    url = _join(root, path)
    attrs = _node_attrs(_open_kvstore(url))
    ms_list = _multiscales(attrs)
    if not ms_list:
        return None
    ms = ms_list[0]
    datasets = [d for d in ms.get("datasets", []) if isinstance(d, dict) and "path" in d]
    if not datasets:
        raise ValueError(f"{url}: multiscales has no datasets")
    systems = {
        cs["name"]: CoordinateSystem(cs["name"], _axes(cs.get("axes")))
        for cs in ms.get("coordinateSystems", [])
        if isinstance(cs, dict) and cs.get("name")
    }
    v06 = bool(systems)
    if not v06:  # 0.4/0.5: one unnamed system; multiscale-level transforms apply to all levels
        systems = {"physical": CoordinateSystem("physical", _axes(ms.get("axes")))}
    first = (datasets[0].get("coordinateTransformations") or [{}])[0]
    intrinsic = _ref(first.get("output"))[1] if v06 else "physical"
    if intrinsic not in systems:
        intrinsic = next(iter(systems))
    ndim = systems[intrinsic].ndim
    affines = [_to_affine(d.get("coordinateTransformations", []), ndim, url) for d in datasets]
    img = Image(
        path=path,
        url=url,
        systems=systems,
        intrinsic=intrinsic,
        datasets=[str(d["path"]).strip("/") for d in datasets],
        level_affines=affines,
        is_label="image-label" in attrs or "/labels/" in f"/{path}",
    )
    top = ms.get("coordinateTransformations") or []
    if not v06:
        shared = _to_affine(top, ndim, url)
        img.level_affines = [a.then(shared) for a in affines]
        return img
    for ct in top:
        (ip, iname), (op, oname) = _ref(ct.get("input")), _ref(ct.get("output"))
        src = (_join(path, ip).lstrip("/") if ip else path, iname or intrinsic)
        dst = (_join(path, op).lstrip("/") if op else path, oname or intrinsic)
        n_in = systems[src[1]].ndim if src[0] == path and src[1] in systems else None
        n_out = systems[dst[1]].ndim if dst[0] == path and dst[1] in systems else None
        img.edges.append((src, dst, parse_transform(ct, url, n_in, n_out, fields=fields)))
    return img


class Scene:
    """The transformation graph of an OME-Zarr 0.6 scene or single multiscale image."""

    def __init__(self, url: str, *, fields: FieldOptions = DEFAULT_FIELDS):
        self.url = url.rstrip("/")
        self.fields = fields
        attrs = _node_attrs(_open_kvstore(self.url))
        scene = attrs.get("scene")
        if not isinstance(scene, dict) and "coordinateTransformations" in attrs:
            scene = attrs  # 0.6 drafts kept the scene graph at the top level
        scene = scene if isinstance(scene, dict) else {}
        self.systems = {
            cs["name"]: CoordinateSystem(cs["name"], _axes(cs.get("axes")))
            for cs in scene.get("coordinateSystems", [])
            if isinstance(cs, dict) and cs.get("name")
        }
        self.images: dict[str, Image] = {}
        self.edges: list[tuple[Node, Node, Transform]] = []  # images add theirs on load
        if _multiscales(attrs):
            self._load("")
        for ct in scene.get("coordinateTransformations", []):
            src, dst = _ref(ct.get("input")), _ref(ct.get("output"))
            if not src[1] or not dst[1]:
                raise ValueError(f"{self.url}: scene transformation needs input and output names")
            self.edges.append(
                (
                    src,
                    dst,
                    parse_transform(
                        ct,
                        self.url,
                        self._ndim(src),
                        self._ndim(dst),
                        fields=fields,
                    ),
                )
            )
        if not self.images and not self.systems and not self.edges:
            raise ValueError(f"{self.url}: no OME-Zarr scene or multiscales metadata")

    def _load(self, path: str) -> Image:
        if path not in self.images:
            img = read_image(self.url, path, self.fields)
            if img is None:
                raise ValueError(f"{_join(self.url, path)}: not an OME-Zarr multiscale image")
            self.images[path] = img
            self.edges.extend(img.edges)
        return self.images[path]

    def image(self, path: str) -> Image:
        return self._load(path.strip("/"))

    def system(self, node: Node) -> CoordinateSystem:
        path, name = node
        systems = self.systems if path is None else self.image(path).systems
        if name not in systems:
            where = "the scene" if path is None else f"image {path!r}"
            raise KeyError(f"no coordinate system {name!r} in {where}; have {sorted(systems)}")
        return systems[name]

    def _ndim(self, node: Node) -> int | None:
        try:
            return self.system(node).ndim
        except (KeyError, ValueError):
            return None

    def transform(self, src: Node, dst: Node, *, approximate: bool = False) -> Transform:
        """The transform taking points in ``src`` to points in ``dst``, composed along the
        shortest path of the graph. Edges are walked backwards only when the transform has
        a closed-form (or, with ``approximate``, an estimated) inverse."""
        self.system(src), self.system(dst)  # validate both ends
        if src == dst:
            return Affine.identity(self.system(src).ndim)
        for allow in (False, True) if approximate else (False,):
            adj: dict[Node, list[tuple[Node, Transform]]] = defaultdict(list)
            for a, b, t in self.edges:
                adj[a].append((b, t))
                inv = t.approximate_inverse() if allow else t.inverse()
                if inv is not None:
                    adj[b].append((a, inv))
            prev: dict[Node, tuple[Node, Transform] | None] = {src: None}
            queue = deque([src])
            while queue:
                n = queue.popleft()
                if n == dst:
                    chain = []
                    while prev[n] is not None:
                        n, t = prev[n]
                        chain.append(t)
                    return Sequence(chain[::-1])
                for m, t in adj[n]:
                    if m not in prev:
                        prev[m] = (n, t)
                        queue.append(m)
        hint = "" if approximate else " (inverse=approx allows estimated inverses)"
        raise ValueError(f"no transformation path from {src} to {dst}{hint}")
