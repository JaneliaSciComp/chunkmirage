"""Chunk-size sources: a sharded volume's finest level as stored, and coarser levels read
from its shard indexes alone, nothing decoded.

URL form (accepted by ``open_source``)::

    chunksizes://precomputed://gs://bucket/volume?range=0.17,0.4

The volume is a Neuroglancer precomputed one whose finest scale is sharded
(``neuroglancer_uint64_sharded_v1``, identity hash). Its levels:

0. the finest scale, as stored;
1. one voxel per chunk of it: the chunk's compressed size, from its minishard's index;
2. one voxel per minishard (a block of chunks, 8³ for hemibrain): the bytes between its
   index and the one before, from the shard's own index. Offered only where that is exact:
   a minishard holds one block of chunks (``preshift_bits`` > 0, and the shard bits reach
   the grid's last), and its chunks lie just before its index, as tensorstore and
   Neuroglancer's writers lay a shard out (checked on one shard when the volume is opened).

A voxel of levels 1 and 2 is the compressed size as a fraction of the raw size of the
chunks it covers, ``range`` (default ``0,1``) mapped onto the volume's dtype, black to
white (0 to 255 for uint8). Compressed size follows content: empty resin and padding
compress to almost nothing, busy tissue far less well. So the coarse levels are a
thumbnail of the data from a few small reads: a slice through all of hemibrain's 8 nm EM
is 334,000 chunks at full resolution, and needs about 5,700 index reads at one voxel per
chunk and 340 at one per minishard (read in cubes of 16³ voxels, two minishards deep, about
12,000 in all). ``range`` picks the fractions shown as black and white, so the
thumbnail's greys can match the data's in one layer (hemibrain: ``0.17,0.4``).
"""

from __future__ import annotations

import gzip
import math
import threading
from collections import OrderedDict
from urllib.parse import parse_qs

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source

INDEXES_KEPT = 65536  # shard indexes kept (hemibrain's s0 has 7140 shards of 1 KB indexes)


class ShardIndexes:
    """A sharded precomputed scale's indexes, read by byte range through a tensorstore
    kvstore: each shard's index (16 bytes a minishard) kept, minishard indexes read when
    asked. Chunk ids are compressed Morton codes of chunk grid positions (x, y, z)."""

    def __init__(self, kv, scale: dict):
        sh = scale["sharding"]
        if sh.get("@type") != "neuroglancer_uint64_sharded_v1" or sh.get("hash") != "identity":
            raise ValueError(f"{scale['key']}: sharding {sh} is not read here (identity hash)")
        self.kv, self.key = kv, scale["key"].strip("/")
        self.pre, self.mbits, self.sbits = sh["preshift_bits"], sh["minishard_bits"], sh["shard_bits"]
        self.gzip = sh.get("minishard_index_encoding", "raw") == "gzip"
        self.grid = [-(-s // c) for s, c in zip(scale["size"], scale["chunk_size"])]  # x, y, z
        bits = [max(0, math.ceil(math.log2(g))) for g in self.grid]
        self.bit_axis = [a for i in range(max(bits)) for a in range(3) if i < bits[a]]
        rank = [0, 0, 0]
        self.bit_rank = []  # each code bit's place among its axis's bits
        for a in self.bit_axis:
            self.bit_rank.append(rank[a])
            rank[a] += 1
        # the chunks sharing a minishard: a block, the lowest `pre` code bits' extent per axis
        self.block = [1 << sum(1 for a in self.bit_axis[: self.pre] if a == ax) for ax in range(3)]
        self._kept: OrderedDict[int, np.ndarray | None] = OrderedDict()
        self._lock = threading.Lock()

    def codes(self, pos: np.ndarray) -> np.ndarray:
        """Chunk ids (uint64) of grid positions ``pos`` (n, 3: x, y, z)."""
        pos = np.asarray(pos, np.uint64).reshape(-1, 3)
        out = np.zeros(len(pos), np.uint64)
        for j, (a, r) in enumerate(zip(self.bit_axis, self.bit_rank)):
            out |= ((pos[:, a] >> np.uint64(r)) & np.uint64(1)) << np.uint64(j)
        return out

    @property
    def exact_blocks(self) -> bool:
        """Whether each minishard holds exactly one block of chunks: its low ``preshift_bits``
        vary within it (so more than one chunk), and no code bits lie past the shard bits (else
        blocks far apart share a minishard)."""
        return self.pre > 0 and len(self.bit_axis) <= self.pre + self.mbits + self.sbits

    def positions(self, ids: np.ndarray) -> np.ndarray:
        """Grid positions (n, 3: x, y, z) of chunk ids."""
        out = np.zeros((len(ids), 3), np.int64)
        for j, (a, r) in enumerate(zip(self.bit_axis, self.bit_rank)):
            out[:, a] |= ((ids >> np.uint64(j)) & np.uint64(1)).astype(np.int64) << r
        return out

    def where(self, block_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shards and minishards of the blocks of chunks at ``block_pos`` (n, 3: x, y, z, in
        blocks)."""
        h = self.codes(np.asarray(block_pos).reshape(-1, 3) * self.block) >> np.uint64(self.pre)
        mini = (h & np.uint64((1 << self.mbits) - 1)).astype(np.int64)
        return ((h >> np.uint64(self.mbits)) & np.uint64((1 << self.sbits) - 1)).astype(np.int64), mini

    @staticmethod
    def data_before_indexes(index: np.ndarray) -> bool | None:
        """Whether each stored minishard of a shard lies just before its index (a gap of data
        before every index), as minishard_bytes counts on; None if the shard cannot tell
        (fewer than two minishards stored)."""
        stored = [m for m in np.lexsort((index[:, 1], index[:, 0])) if index[m, 1] > index[m, 0]]
        if len(stored) < 2:
            return None
        end = 0
        for m in stored:
            if index[m, 0] <= end:
                return False  # indexes back to back: the data lies elsewhere
            end = index[m, 1]
        return True

    def _file(self, shard: int) -> str:
        return f"{self.key}/{shard:0{-(-self.sbits // 4)}x}.shard"

    def shard_indexes(self, shards) -> dict[int, np.ndarray | None]:
        """Each shard's index, (minishards, 2) byte ranges of their indexes after it; None for
        a shard not stored. Those not kept are read at once."""
        with self._lock:
            out = {s: self._kept[s] for s in shards if s in self._kept}
        todo = [s for s in dict.fromkeys(shards) if s not in out]
        n = 16 << self.mbits
        futures = [self.kv.read(self._file(s), byte_range=slice(0, n)) for s in todo]
        for s, f in zip(todo, futures):
            r = f.result()
            out[s] = None if r.state == "missing" else np.frombuffer(r.value, "<u8").reshape(-1, 2).astype(np.int64)
        with self._lock:
            for s in todo:
                self._kept[s] = out[s]
            while len(self._kept) > INDEXES_KEPT:
                self._kept.popitem(last=False)
        return out

    def minishard_bytes(self, index: np.ndarray) -> np.ndarray:
        """Bytes of each minishard's chunks, from its shard's index alone: the gap before its
        index, in file order (the chunks of a minishard lie just before its index)."""
        out, end = np.zeros(len(index), np.int64), 0
        for m in np.lexsort((index[:, 1], index[:, 0])):
            start, stop = index[m]
            if stop > start:
                out[m], end = max(0, start - end), stop
        return out

    def minishards(self, wanted: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
        """Chunk ids and compressed sizes of each (shard, minishard) in ``wanted``, read at once."""
        wanted = list(dict.fromkeys(wanted))
        idx = self.shard_indexes([s for s, _ in wanted])
        base = 16 << self.mbits
        todo = [(s, m) for s, m in wanted if idx[s] is not None and idx[s][m, 1] > idx[s][m, 0]]
        futures = [
            self.kv.read(self._file(s), byte_range=slice(base + int(idx[s][m, 0]), base + int(idx[s][m, 1])))
            for s, m in todo
        ]
        out = {w: (np.zeros(0, np.uint64), np.zeros(0, np.uint64)) for w in wanted}
        for w, f in zip(todo, futures):
            raw = bytes(f.result().value)
            v = np.frombuffer(gzip.decompress(raw) if self.gzip else raw, "<u8")
            n = len(v) // 3
            out[w] = (np.cumsum(v[:n], dtype=np.uint64), v[2 * n :])
        return out


class SizeLevel(Source):
    """Compressed size as a fraction of raw, one voxel per ``per`` (z, y, x) chunks of the
    finest scale: per chunk (``per`` 1) from minishard indexes, or per minishard block
    (``per`` the block) from shard indexes; mapped through ``lo``..``hi`` to ``dtype``."""

    def __init__(self, index: ShardIndexes, s0: ArrayInfo, per: tuple[int, int, int],
                 lo: float, hi: float, name: str):
        self.index, self.s0, self.per, self.lo, self.hi = index, s0, per, lo, hi
        c = s0.chunk_shape
        grid = tuple(index.grid[::-1])  # z, y, x
        shape = tuple(-(-g // p) for g, p in zip(grid, per))
        size = tuple(v * ch * p for v, ch, p in zip(s0.voxel_size, c, per))
        # cubes of two index reads across (per chunk: two minishards, 16³ for hemibrain; per
        # minishard: 16³ minishards), for slices any way and 3-D alike
        e = index.block[::-1]
        chunk = tuple(2 * b for b in e) if per == (1, 1, 1) else (16, 16, 16)
        self._info = ArrayInfo(
            shape=shape,
            dtype=s0.dtype,
            chunk_shape=tuple(min(s, k) for s, k in zip(shape, chunk)),
            voxel_size=size,
            units=s0.units,
            axes=s0.axes,
            # voxel 0 spans chunks 0 .. per - 1: its corner is s0's voxel 0's corner
            translation=tuple(t - v / 2 + s / 2 for t, v, s in zip(s0.translation, s0.voxel_size, size)),
            kind="image",
        )
        self._key = f"chunksizes:{name}:{per}:{lo},{hi}"

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def _raw_bytes(self, lo, hi) -> np.ndarray:
        """Raw bytes of the chunks under each voxel of [lo, hi) (z, y, x): edge chunks hold
        fewer voxels."""
        parts = []
        for a in range(3):
            c, n, p = self.s0.chunk_shape[a], self.s0.shape[a], self.per[a]
            first = np.arange(lo[a], hi[a]) * p * c  # first voxel of each output voxel's span
            parts.append(np.clip(n - first, 0, p * c).astype(np.float64))
        return np.einsum("i,j,k->ijk", *parts) * self.s0.dtype.itemsize

    def _scaled(self, nbytes: np.ndarray, lo, hi) -> np.ndarray:
        f = (nbytes / np.maximum(self._raw_bytes(lo, hi), 1) - self.lo) / (self.hi - self.lo)
        dt = self._info.dtype
        top = np.iinfo(dt).max if dt.kind in "ui" else 1.0
        return np.clip(np.rint(f * top) if dt.kind in "ui" else f, 0, top).astype(dt)

    def read(self, box: Box) -> np.ndarray:
        lo, hi = box.start, box.stop
        nbytes = np.zeros(box.shape, np.float64)
        ix = self.index
        if self.per == (1, 1, 1):
            e = ix.block[::-1]  # z, y, x chunks a minishard
            zyx = np.stack(np.meshgrid(*(np.arange(a // b, (z - 1) // b + 1) for a, z, b in zip(lo, hi, e)), indexing="ij"), -1)
            shards, minis = ix.where(zyx.reshape(-1, 3)[:, ::-1])
            for ids, sizes in ix.minishards(list(zip(shards.tolist(), minis.tolist()))).values():
                if not len(ids):
                    continue
                p = ix.positions(ids)[:, ::-1]  # z, y, x
                inside = np.all((p >= lo) & (p < hi), axis=1)
                q = p[inside] - lo
                nbytes[q[:, 0], q[:, 1], q[:, 2]] = sizes[inside]
        else:  # one voxel per block: its minishard's bytes, from its shard's index
            zyx = np.stack(np.meshgrid(*(np.arange(a, z) for a, z in zip(lo, hi)), indexing="ij"), -1)
            shards, minis = ix.where(zyx.reshape(-1, 3)[:, ::-1])
            idx = ix.shard_indexes(shards.tolist())
            per_shard = {s: ix.minishard_bytes(v) for s, v in idx.items() if v is not None}
            flat = nbytes.reshape(-1)
            for i, (s, m) in enumerate(zip(shards.tolist(), minis.tolist())):
                if s in per_shard:
                    flat[i] = per_shard[s][m]
        return self._scaled(nbytes, lo, hi)


def open_chunksizes(url: str, *, cache_bytes: int = 0, cache=None) -> MultiscaleSource:
    import tensorstore as ts

    from chunkmirage.sources.registry import open_source
    from chunkmirage.sources.tensorstore_source import shared_context

    rest = url[len("chunksizes://") :]
    location, _, query = rest.rpartition("?") if "?" in rest else (rest, "", "")
    q = {k: v[-1] for k, v in parse_qs(query).items()}
    if extra := set(q) - {"range"}:
        raise ValueError(f"chunksizes:// takes only range=lo,hi, not {sorted(extra)}")
    lo, hi = (float(v) for v in q.get("range", "0,1").split(","))
    if not hi > lo:
        raise ValueError(f"chunksizes:// range={q['range']}: the second must be larger")
    data = open_source(location, cache_bytes=cache_bytes, cache=cache)
    s0 = data.levels[0]
    spec = s0.store.spec().to_json() if hasattr(s0, "store") else {}
    scale = spec.get("scale_metadata", {})
    if spec.get("driver") != "neuroglancer_precomputed" or "sharding" not in scale:
        raise ValueError(f"chunksizes:// reads a sharded precomputed volume's indexes; {location} is not one")
    if s0.info.ndim != 3:
        raise ValueError(f"chunksizes:// {location}: one channel, z, y, x, not {s0.info.axes}")
    if s0.info.dtype.kind in "iu" and s0.info.dtype.itemsize > 4:
        raise ValueError(f"chunksizes:// {location}: {s0.info.dtype} values (a fraction of their range is not exact)")
    kv = ts.KvStore.open(spec["kvstore"], context=shared_context(cache_bytes)).result()
    index = ShardIndexes(kv, scale)
    levels = [s0, SizeLevel(index, s0.info, (1, 1, 1), lo, hi, location)]
    if index.exact_blocks:  # the minishard level, if a shard here is laid out as it counts on
        middle = np.array([g // 2 // b for g, b in zip(index.grid, index.block)])
        shard = int(index.where(middle)[0][0])
        sample = index.shard_indexes([shard])[shard]
        if sample is not None and index.data_before_indexes(sample):
            levels.append(SizeLevel(index, s0.info, tuple(index.block[::-1]), lo, hi, location))
    return MultiscaleSource(levels, name=url)
