"""Passes of a computation chained through dataset:// and warp://, the acceptance scenario
registration-station asked for: a toy block matcher, an op whose output is one vector per
lattice node (a coarser grid than its input) and whose blocks overlap by a node, blended
where they meet, undoes a known rotation in two passes, the second fitted on the image the
first corrected. Editing the second pass's parameter rebuilds only it and what follows, the
first pass's chunks staying cached; the result does not depend on the order chunks are
asked for; and the coverage layer marks exactly the blocks fitted."""

from typing import ClassVar

import numpy as np
import pytest
from pydantic import Field, PrivateAttr

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops import base
from chunkmirage.ops.base import Op
from chunkmirage.server import DatasetRegistry


class BlockMatch(Op):
    """One translation per block of ``stride`` voxels: the whole-voxel shift within
    ``radius`` at which the moving channel best matches the fixed one (least squared
    difference), as a displacement in the data's units, the three components first."""

    name: ClassVar[str] = "block_match_test"
    overlap: ClassVar[int] = 1  # a node beyond each chunk, so neighbouring chunks' fits blend
    stride: int = Field(8, description="voxels per lattice node along each axis")
    radius: int = Field(4, description="largest shift tried, in voxels")
    calls: ClassVar[int] = 0
    _voxel: tuple[float, ...] = PrivateAttr((1.0, 1.0, 1.0))

    @property
    def halo(self):  # the moving block moved by up to radius
        return (self.radius,) * 3

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        out = info.rescaled([v * self.stride for v in info.voxel_size[1:]])
        return out.with_(shape=(3, *out.shape[1:]), chunk_shape=(3, *out.chunk_shape[1:]), dtype=np.dtype("float32"), kind="image")

    def for_level(self, info: ArrayInfo) -> Op:
        op = self.model_copy()
        op._voxel = tuple(float(v) / self.stride for v in info.voxel_size[-3:])  # the input's voxels
        op._cache = self._cache
        return op

    def apply(self, block: np.ndarray) -> np.ndarray:
        BlockMatch.calls += 1
        r, s = self.radius, self.stride
        fixed, moving = block[0].astype(np.float32), block[1].astype(np.float32)
        core = fixed[r:-r, r:-r, r:-r]
        n = tuple(c // s for c in core.shape)  # the core is whole blocks: the chunk times the stride
        best = np.full(n, np.inf, np.float32)
        shift = np.zeros((3, *n), np.float32)
        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    m = moving[r + dz : r + dz + core.shape[0], r + dy : r + dy + core.shape[1], r + dx : r + dx + core.shape[2]]
                    d = ((m - core) ** 2).reshape(n[0], s, n[1], s, n[2], s).sum((1, 3, 5))
                    better = d < best
                    best = np.where(better, d, best)
                    for a, v in enumerate((dz, dy, dx)):
                        shift[a] = np.where(better, v, shift[a])
        return shift * np.asarray(self._voxel, np.float32)[:, None, None, None]


# textured enough to match on (blobs alone are sparse this small); 8 nm voxels, so displacements are in nm
@pytest.fixture(autouse=True)
def registered(monkeypatch):
    """The toy op known by name for this test only, so the ops reference and the schema stay those of the shipped ops."""
    monkeypatch.setitem(base._REGISTRY, BlockMatch.name, BlockMatch)


FIXED = "synthetic://blobs+noise?shape=64,64,64&chunk=16,16,16&levels=1&seed=5"
MOVING = f"warp://{FIXED}?field=swirl&angle=5&radius=1000"  # turned 5 degrees about z: up to 4 voxels at the corners
WHOLE = Box((0, 0, 0), (64, 64, 64))


def build(reg: DatasetRegistry, radius2: int = 2) -> None:
    reg.add("fixed", {"source": FIXED})
    reg.add("moving", {"source": MOVING})
    reg.add("pass1", {"source": "stack://dataset://fixed|dataset://moving", "ops": [{"op": "block_match_test", "stride": 8, "radius": 4, "cache": True}]})
    reg.add("corrected1", {"source": "warp://dataset://moving?field=dataset://pass1&target=dataset://fixed"})
    reg.add("pass2", {"source": "stack://dataset://fixed|dataset://corrected1", "ops": [{"op": "block_match_test", "stride": 8, "radius": radius2, "cache": True}]})
    # the original through both passes' fields at once, the latest first: one resample
    reg.add("final", {"source": "warp://dataset://moving?field=dataset://pass2|dataset://pass1&target=dataset://fixed"})


def read(reg: DatasetRegistry, name: str) -> np.ndarray:
    return reg.get(name).read(0, WHOLE).astype(float)


def error(a: np.ndarray, fixed: np.ndarray) -> float:
    """Mean absolute difference from the fixed image, away from the edges (where a turned
    image reads outside itself)."""
    inner = (slice(8, -8),) * 3
    return float(np.abs(a[inner] - fixed[inner]).mean())


def test_two_passes_undo_a_rotation_and_only_the_edited_pass_recomputes():
    reg = DatasetRegistry()
    build(reg)
    pass1 = reg.get("pass1")
    assert pass1.info(0).shape == (3, 8, 8, 8) and pass1.info(0).voxel_size == (1.0, 64.0, 64.0, 64.0)  # a node every 8 voxels of 8 nm
    assert pass1.info(0).chunk_shape == (3, 2, 2, 2), "a lattice's chunks follow its grid: the input's 16 voxels are 2 nodes"
    assert pass1.levels[0].blocks.chunk_box((0, 1, 1, 1)) == Box((0, 1, 1, 1), (3, 5, 5, 5)), "a block is its chunk and a node around it"
    assert reg.upstreams("final") == {"fixed", "moving", "pass1", "corrected1", "pass2"}

    fixed = read(reg, "fixed")
    e0, e1, e2 = (error(read(reg, n), fixed) for n in ("moving", "corrected1", "final"))
    assert e1 < 0.5 * e0, f"the first pass should undo most of the rotation: {e0:.3f} -> {e1:.3f}"
    # the second pass finds whole-voxel shifts in a residual under a voxel: it may add a little
    # noise of its own, but a wrong sign in the chain would bring the rotation back
    assert e2 < 0.5 * e0 and e2 <= e1 * 1.25, f"the second pass must keep what the first undid: {e0:.3f}, {e1:.3f} -> {e2:.3f}"

    # the second pass's parameter changes: pass1 and the image it corrected stand, as do
    # pass1's cached chunks; only pass2 and the final image are built and computed again
    corrected1, final = reg.get("corrected1"), reg.get("final")
    key1 = pass1.levels[0].cache_key()
    kept = {k for k in reg.cache._data if k[0] == key1}
    assert len(kept) == np.prod(pass1.levels[0].info.chunk_grid)
    calls = BlockMatch.calls
    reg.add("pass2", {"source": "stack://dataset://fixed|dataset://corrected1", "ops": [{"op": "block_match_test", "stride": 8, "radius": 3, "cache": True}]})
    assert reg.get("pass1") is pass1 and reg.get("corrected1") is corrected1
    assert reg.get("final") is not final and reg.get("final").digest() != final.digest()
    read(reg, "final")
    assert BlockMatch.calls - calls == np.prod(reg.get("pass2").levels[0].info.chunk_grid), "only the second pass ran, each block once"
    assert {k for k in reg.cache._data if k[0] == key1} == kept


def test_the_coverage_layer_marks_exactly_the_blocks_fitted():
    reg = DatasetRegistry()
    build(reg)
    cov = reg.add("where", {"source": "coverage://pass1"})
    whole = Box((0, 0, 0, 0), cov.info(0).shape)
    assert cov.info(0).shape == (1, 4, 4, 4) and cov.info(0).voxel_size[1:] == (128.0,) * 3  # a voxel per chunk of 2 nodes of 64 nm
    assert not cov.read(0, whole).any()
    calls = BlockMatch.calls
    reg.get("pass1").chunk(0, (0, 0, 0, 0))  # one chunk of the lattice
    fitted = np.zeros((4, 4, 4), bool)
    fitted[:2, :2, :2] = True  # its own block and the neighbours whose overlap reaches it
    marked = cov.read(0, whole)[0]
    np.testing.assert_array_equal(marked == 3, fitted)
    assert set(np.unique(marked)) <= {0, 3} and BlockMatch.calls - calls == fitted.sum()
    pass1 = reg.get("pass1")
    pass1.read(0, Box((0, 0, 0, 0), pass1.info(0).shape))
    assert (cov.read(0, whole) == 3).all()


def test_the_result_does_not_depend_on_the_order_chunks_are_asked_in():
    a, b = DatasetRegistry(), DatasetRegistry()
    build(a)
    build(b)
    fa, fb = a.get("final"), b.get("final")
    idx = list(np.ndindex(*fa.levels[0].info.chunk_grid))
    forward = [fa.chunk(0, i) for i in idx]
    backward = [fb.chunk(0, i) for i in reversed(idx)][::-1]
    for x, y in zip(forward, backward):
        np.testing.assert_array_equal(x, y)
