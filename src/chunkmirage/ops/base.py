from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from chunkmirage.core import ArrayInfo


class Op(BaseModel):
    """A block-wise operation.

    Subclass, set ``name``, declare parameters as pydantic fields, and implement ``apply``.

    * ``halo``: voxels of upstream context needed on every side (int or per-axis tuple).
      The framework reads the padded region, calls ``apply`` on it, and crops the result.
    * ``cache``: whether this stage's output chunks should be memoized. Turn on for
      expensive stages (inference) so cheap downstream tweaks (threshold) are free. A
      pipeline can override it per op: ``{"op": "gaussian", "sigma": 4, "cache": true}``.
    * ``output_dtype`` / ``output_info``: describe the result; default is unchanged.
    * ``output_kind``: what the result's values are (``ArrayInfo.kind``): ``image``,
      ``label`` or ``mask``; ``None`` (the default) if the op does not say.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ClassVar[str] = ""
    halo: ClassVar[int | tuple[int, ...]] = 0
    cache: ClassVar[bool] = False
    # packages its apply imports beyond numpy (the browser engine loads them up front)
    packages: ClassVar[tuple[str, ...]] = ()
    output_kind: ClassVar[str | None] = None
    _cache: bool | None = PrivateAttr(None)  # this op's own setting, over the class's

    @property
    def cached(self) -> bool:
        """Whether this op's output is memoized: its own setting, else its class's."""
        return self.cache if self._cache is None else self._cache

    def halo_for(self, ndim: int) -> tuple[int, ...]:
        h = self.halo
        return tuple(h) if isinstance(h, (tuple, list)) else (int(h),) * ndim

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype(in_dtype)

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        return info.with_(dtype=self.output_dtype(info.dtype), kind=self.output_kind)

    def apply(self, block: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def input_voxel_size(self) -> tuple[float, ...] | None:
        """The voxel size this op must read, along the data's last axes and in the source's
        units: a model trained at one resolution. The pipeline then runs it on one level (the
        source's at that size, else one resampled to it), caches what it makes, and makes the
        coarser levels by downsampling that. ``None`` (the default): it runs on every level."""
        return None

    def for_level(self, info: ArrayInfo) -> Op:
        """This op as it runs on a scale level described by ``info`` (the stage's output, on
        the input's grid). Ops that work in physical units override it to take the level's
        voxel size: a slope in degrees needs the pixel spacing, which doubles from level to
        level. The default is the op itself."""
        return self

    def apply_at(self, block: np.ndarray, box) -> np.ndarray:
        """Like ``apply`` but told where ``block`` sits (``box`` = its halo-padded extent in
        voxels of this scale level). Override when the result must depend on position, e.g. to
        make per-chunk labels globally unique. Default delegates to ``apply``."""
        return self.apply(block)

    # --- identity / serialization -------------------------------------------------
    def spec(self) -> dict[str, Any]:
        spec = {"op": self.name, **self.model_dump(mode="json")}
        if self._cache is not None:
            spec["cache"] = self._cache
        return spec

    def digest(self) -> str:
        """Identity of what the op computes: whether it is cached does not change that."""
        payload = json.dumps(
            {"op": self.name, **self.model_dump(mode="json")}, sort_keys=True, default=str
        ).encode()
        return hashlib.sha1(payload).hexdigest()[:12]


_REGISTRY: dict[str, type[Op]] = {}
_ENTRYPOINTS_LOADED = False


def register(cls: type[Op]) -> type[Op]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a class-level `name`")
    _REGISTRY[cls.name] = cls
    return cls


def _load_entrypoints() -> None:
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True
    for ep in entry_points(group="chunkmirage.ops"):
        try:
            cls = ep.load()
            if isinstance(cls, type) and issubclass(cls, Op):
                _REGISTRY.setdefault(cls.name or ep.name, cls)
        except Exception:  # noqa: BLE001 - a broken plugin must not take the server down
            continue


def get_op(name: str) -> type[Op]:
    _load_entrypoints()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown op {name!r}; known: {sorted(_REGISTRY)}") from None


def list_ops() -> dict[str, dict]:
    _load_entrypoints()
    return {
        name: {
            "halo": "dynamic" if isinstance(cls.halo, property) else cls.halo,
            "cache": cls.cache,
            "doc": (cls.__doc__ or "").strip(),
            "schema": cls.model_json_schema(),
        }
        for name, cls in sorted(_REGISTRY.items())
    }


def op_from_spec(spec: dict[str, Any] | Op) -> Op:
    if isinstance(spec, Op):
        return spec
    spec = dict(spec)
    name = spec.pop("op")
    cache = spec.pop("cache", None)
    op = get_op(name)(**spec)
    if cache is not None:
        op._cache = bool(cache)
    return op


def ops_from_specs(specs: Sequence[dict[str, Any] | Op]) -> list[Op]:
    return [op_from_spec(s) for s in specs]
