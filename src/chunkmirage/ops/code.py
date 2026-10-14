"""``python``: an op whose code is part of the pipeline spec, edited live as a Neuroglancer
shader is. Lines starting ``#halo``, ``#output`` and ``#uicontrol`` say what it needs and
makes; they are read as text, so a pipeline is planned without running the code."""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register

KINDS = ("image", "label", "mask")

TEMPLATE = '''\
# Python run on each block of this view as it is fetched; edit it and the view updates.
# Lines starting with these are read before it runs:
#   #halo 2                 voxels of context apply needs on each side
#   #output uint32 label    the dtype it returns, then image, label or mask (default: as given)
#   #uicontrol float low slider(min=0, max=255, default=128)
#                           a slider, as in Neuroglancer's shaders; apply gets its value
# numpy is np; scipy and scikit-image can be imported.
def apply(block):
    return block
'''


class Control(BaseModel):
    """A ``#uicontrol`` line: a slider (``float``, ``int``) or a checkbox (``bool``)."""

    name: str
    type: Literal["float", "int", "bool"]
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: float | int | bool

    def cast(self, value: Any) -> float | int | bool:
        return {"float": float, "int": int, "bool": bool}[self.type](value)


class Directives(BaseModel):
    halo: tuple[int, ...] = (0,)
    dtype: str | None = None
    kind: str | None = None
    controls: list[Control] = []


_CONTROL = re.compile(r"(\w+)\s+([A-Za-z_]\w*)\s+(\w+)\s*\((.*)\)\s*$")


def _number(text: str, line: int) -> float | bool:
    if text in ("true", "false"):
        return text == "true"
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"line {line}: {text!r} is not a number") from None


def _control(rest: str, line: int) -> Control:
    m = _CONTROL.match(rest)
    if not m:
        raise ValueError(f"line {line}: #uicontrol takes `type name slider(min=..., max=..., default=...)`")
    kind, name, widget, args = m.groups()
    values = {}
    for part in filter(None, (p.strip() for p in args.split(","))):
        key, eq, value = part.partition("=")
        if not eq:
            raise ValueError(f"line {line}: #uicontrol arguments are key=value, not {part!r}")
        values[key.strip()] = _number(value.strip(), line)
    if name in ("block", "box"):
        raise ValueError(f"line {line}: #uicontrol {name!r}: apply gets that name itself")
    if name in _reserved():
        raise ValueError(f"line {line}: #uicontrol {name!r}: the op has that name itself; pick another")
    if kind in ("float", "int") and widget == "slider":
        if "min" not in values or "max" not in values:
            raise ValueError(f"line {line}: a slider needs min and max")
        values.setdefault("default", values["min"])
        if kind == "int":
            values.setdefault("step", 1)
        if values["min"] > values["max"]:
            raise ValueError(f"line {line}: a slider's min is above its max")
        if not values["min"] <= values["default"] <= values["max"]:
            raise ValueError(f"line {line}: a slider's default is outside min..max")
        if "step" in values and values["step"] <= 0:
            raise ValueError(f"line {line}: a slider's step must be above 0")
    elif kind == "bool" and widget == "checkbox":
        values.setdefault("default", False)
    else:
        raise ValueError(f"line {line}: #uicontrol {kind} {widget}: use float or int slider, or bool checkbox")
    unknown = set(values) - {"min", "max", "step", "default"}
    if unknown:
        raise ValueError(f"line {line}: #uicontrol takes min, max, step and default, not {sorted(unknown)}")
    c = Control(name=name, type=kind, **values)
    return c.model_copy(update={"default": c.cast(c.default)})


@lru_cache(maxsize=1)
def _reserved() -> frozenset[str]:
    """Names a ``#uicontrol`` may not take: the op's own attributes and fields, which a
    plugin made of the code (``python_op``) would otherwise shadow."""
    return frozenset(dir(_Coded)) | frozenset(PythonOp.model_fields)


MAX_HALO = 1024


@lru_cache(maxsize=256)
def directives(code: str) -> Directives:
    """What the ``#`` lines of ``code`` say (lines starting at the first column, so an
    indented comment in ``apply`` is not one); ``ValueError`` naming the line if one is
    wrong."""
    found: dict[str, Any] = {"controls": []}
    for n, raw in enumerate(code.splitlines(), 1):
        line = raw.strip()
        word = line.split(maxsplit=1)[0] if raw.startswith("#") else ""
        rest = line[len(word):].strip()
        if word in ("#halo", "#output") and word[1:] in found:
            raise ValueError(f"line {n}: a second {word}")
        if word == "#halo":
            try:
                found["halo"] = tuple(int(v) for v in rest.split())
            except ValueError:
                raise ValueError(f"line {n}: #halo takes whole voxels, one or one per axis") from None
            if not found["halo"] or min(found["halo"]) < 0:
                raise ValueError(f"line {n}: #halo takes whole voxels, one or one per axis")
            if max(found["halo"]) > MAX_HALO:
                raise ValueError(f"line {n}: #halo above {MAX_HALO} voxels: every chunk would read that much more")
        elif word == "#output":
            found["output"] = True
            for token in rest.split():
                if token in KINDS:
                    found["kind"] = token
                    continue
                try:
                    dt = np.dtype(token)
                except TypeError:
                    raise ValueError(f"line {n}: #output {token!r}: a numpy dtype, or one of {KINDS}") from None
                if dt.kind not in "uif":
                    raise ValueError(f"line {n}: #output {token!r}: an integer or float dtype")
                found["dtype"] = dt.name
        elif word == "#uicontrol":
            c = _control(rest, n)
            if any(o.name == c.name for o in found["controls"]):
                raise ValueError(f"line {n}: a second #uicontrol {c.name!r}")
            found["controls"].append(c)
    found.pop("output", None)
    return Directives(**found)


def check(code: str) -> None:
    """``code`` parses and defines ``apply(block, ...)`` taking its controls; checked without
    running it."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"line {e.lineno}: {e.msg}") from None
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply"]
    if not defs:
        raise ValueError("the code defines no `def apply(block, ...)`")
    fn = defs[-1]  # the last def is what runs
    for later in tree.body[tree.body.index(fn) + 1:]:  # ... unless the name is rebound after it
        targets = getattr(later, "targets", [getattr(later, "target", None)])
        if isinstance(later, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)) and any(
            isinstance(t, ast.Name) and t.id == "apply" for t in targets
        ):
            raise ValueError(f"line {later.lineno}: apply is no longer the function after this")
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if not names and a.vararg is None:
        raise ValueError(f"line {fn.lineno}: apply takes the block first")
    controls = {c.name for c in directives(code).controls}
    for c in controls:
        if c not in names and a.kwarg is None:
            raise ValueError(f"line {fn.lineno}: apply has no parameter {c!r} for its #uicontrol")
        if names and c == names[0]:
            raise ValueError(f"line {fn.lineno}: {c!r} is apply's first parameter, the block, and a #uicontrol")
    # every other parameter is a slider's, box, or has a default: nothing else is passed
    positional = a.posonlyargs + a.args
    defaulted = {x.arg for x in positional[len(positional) - len(a.defaults):]} | {
        x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d is not None
    }
    for x in names[1:]:
        if x not in controls and x != "box" and x not in defaulted:
            raise ValueError(f"line {fn.lineno}: apply's parameter {x!r} has no #uicontrol and no default")


@lru_cache(maxsize=4)  # a few versions, while an edit is in flight; each holds its module's namespace
def _compiled(code: str):
    """``apply`` from ``code``, run once per process, and whether it takes ``box``."""
    ns: dict[str, Any] = {"np": np, "__name__": "chunkmirage_python_op"}
    exec(compile(code, "<python op>", "exec"), ns)  # noqa: S102 - the point of this op; gated by the server
    fn = ns["apply"]
    params = fn.__code__.co_varnames[: fn.__code__.co_argcount + fn.__code__.co_kwonlyargcount]
    return fn, "box" in params


class _Coded(Op):
    """What an op made from code has: halo, output and ``apply`` from its code's text."""

    def _code(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _values(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def directives(self) -> Directives:
        return directives(self._code())

    @property
    def halo(self):  # type: ignore[override]
        return self.directives.halo

    def halo_for(self, ndim: int) -> tuple[int, ...]:
        h = self.directives.halo
        if len(h) == 1:
            return h * ndim
        if len(h) > ndim:
            raise ValueError(f"#halo gives {len(h)} axes; the data has {ndim}")
        return (0,) * (ndim - len(h)) + h  # the last axes: z y x of c z y x

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        d = self.directives.dtype
        return np.dtype(d) if d else np.dtype(in_dtype)

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        return info.with_(dtype=self.output_dtype(info.dtype), kind=self.directives.kind or info.kind)

    def apply(self, block: np.ndarray) -> np.ndarray:
        return self.apply_at(block, None)

    def apply_at(self, block: np.ndarray, box) -> np.ndarray:
        fn, wants_box = _compiled(self._code())
        out = fn(block, **self._values(), **({"box": box} if wants_box else {}))
        return np.asarray(out).astype(self.output_dtype(block.dtype), copy=False)


@register
class PythonOp(_Coded):
    """Python of your own, carried in the spec: ``apply(block, ...)`` returns the block
    computed (or its interior, shaved by the halo). ``#halo``, ``#output`` and
    ``#uicontrol`` lines say what it reads and makes; a server runs it only when started
    with ``--allow-code``."""

    name = "python"
    code: str = Field(
        TEMPLATE,
        description="Python defining apply(block, ...), with #halo, #output and #uicontrol lines.",
        json_schema_extra={"format": "python"},
    )
    params: dict[str, float | int | bool] = Field(
        default_factory=dict, description="Values of the code's #uicontrol sliders, over their defaults."
    )

    @field_validator("code")
    @classmethod
    def _valid(cls, v: str) -> str:
        directives(v)
        check(v)
        return v

    @model_validator(mode="before")
    @classmethod
    def _current(cls, data: Any) -> Any:
        """Values of sliders the code no longer has dropped: edited live, the code renames
        and removes them, and the value of one gone is no part of what the op computes."""
        code = data.get("code", TEMPLATE) if isinstance(data, dict) else None
        if isinstance(code, str) and isinstance(data.get("params"), dict):
            try:
                names = {c.name for c in directives(code).controls}
            except ValueError:
                return data  # the code's own check says what is wrong
            data = {**data, "params": {k: v for k, v in data["params"].items() if k in names}}
        return data

    def _code(self) -> str:
        return self.code

    def _values(self) -> dict[str, Any]:
        return {c.name: c.cast(self.params.get(c.name, c.default)) for c in self.directives.controls}


def python_op(name: str, code: str, doc: str | None = None) -> type[Op]:
    """An op class from ``code`` written for the ``python`` op: its ``#uicontrol`` lines
    become its parameters, so it is a plugin like any other (register it through the
    ``chunkmirage.ops`` entry point, or ``register`` it)."""
    check(code)
    controls = directives(code).controls
    body: dict[str, Any] = {
        "__doc__": doc or f"``{name}``, from code written for the ``python`` op.",
        "__module__": __name__,
        "__annotations__": {c.name: {"float": float, "int": int, "bool": bool}[c.type] for c in controls},
        "name": name,
        "_code": lambda self: code,
        "_values": lambda self: {c.name: getattr(self, c.name) for c in controls},
    }
    for c in controls:
        body[c.name] = Field(c.default, **({} if c.type == "bool" else {"ge": c.min, "le": c.max}))
    return type("".join(w.title() for w in name.split("_")), (_Coded,), body)
