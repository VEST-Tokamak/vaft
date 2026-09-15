"""One canonical spelling for the IMAS Data Dictionary paths a plot reads.

The two data models spell the same DD path differently::

    canonical   magnetics/ip(:)/data          this module; the DD's own full_path
    recipe      magnetics.ip.{i}.data         OMAS dotted, {i} the enumerated index
    OMAS        magnetics.ip.0.data           OMAS dotted, a concrete index
    OMAS info   magnetics.ip.:.data           what omas.omas_info_node accepts
    IMAS        ip/data                       relative to the IDS, no index

A :class:`DDPath` is parsed once from a recipe template
(:func:`from_template`) or from the canonical spelling (:func:`parse`) and
translated to whichever the caller needs (:func:`to_omas`,
:func:`to_omas_template`, :func:`to_info`, :func:`to_imas`).
:func:`resolve` answers what the Data Dictionary itself says about the leaf --
units, coordinates, documentation -- without any data loaded, and
:func:`dd_paths` derives the paths a canonical plot declares from its recipe,
so ``vaft.omas.dd_<stem>()`` and ``vaft.imas.dd_<stem>()`` share one answer
(umbrella #434).

Grammar: segments joined by ``/``; an array-of-structures segment carries its
index in parentheses -- ``(:)`` for the enumerated index a recipe expands, an
integer for a fixed one (``profiles_2d(0)``: a fixed index is a declaration,
so it is kept).  The IDS is the first segment and carries no index.

Nothing here imports ``omas`` or ``imas`` at module level: the grammar is
pure string work, and only :func:`resolve` reaches the Data Dictionary.

Computed views (``CallableRecipe``) declare only what their registry spec
lists under ``required_paths``/``optional_paths``; the full input declaration
of the 47 computed views is sub-issue #439's work, and their ``DDPath`` tuples
carry ``attrs["declared_by"] == "spec"`` to say so.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "DEFAULT_DD_VERSION",
    "DDInfo",
    "DDPath",
    "ROLES",
    "dd_paths",
    "from_template",
    "normalise_units",
    "parse",
    "resolve",
    "to_imas",
    "to_info",
    "to_omas",
    "to_omas_template",
]

#: The Data Dictionary version the packaged samples are written against and
#: :func:`resolve` reads by default; ``omas`` ships this one and older.
DEFAULT_DD_VERSION = "3.41.0"

#: What a path is to the plot that declares it, in the order
#: :func:`dd_paths` lists them.
ROLES = (
    "data",          # the quantity drawn
    "fallback",      # an alternative spelling of the quantity, tried when the first holds nothing
    "coordinate",    # the abscissa or grid the quantity is read against
    "abscissa",      # a sibling quantity the plot may be drawn against (issue #481)
    "label",         # where a channel's name is read
    "weight",        # a per-channel multiplier (turns)
    "divisor",       # a scalar the quantity is divided by
    "container",     # the array of structures a channel or slice is enumerated in
    "geometry",      # an outline or point set of a geometry layer
    "boundary",      # the boundary outline drawn over a field
    "required",      # declared by the registry spec, without a finer role
    "optional",      # declared optional by the registry spec
)

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_PLACEHOLDER = re.compile(r"^\{([a-z])\}$")
_CANONICAL_SEGMENT = re.compile(r"^([a-z][a-z0-9_]*)(?:\((:|\d+)\))?$")
_PLACEHOLDER_LETTERS = "ijklmn"

# One segment is ``(name, index)``: index ``None`` for a structure, ``":"``
# for the enumerated index, an ``int`` for a fixed one.
_Segments = tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class DDPath:
    """One Data Dictionary path a plot declares, in the canonical spelling.

    ``template`` keeps the recipe's own spelling verbatim (``{i}`` and all) so
    :func:`to_omas_template` reproduces it exactly; ``role`` is one of
    :data:`ROLES`; ``coordinate`` and ``fallback_coordinate`` are the
    canonical spellings of the abscissae a ``data`` path is read against, as
    the recipe declares them; ``units`` is what the recipe declares for the
    quantity (empty when it declares nothing -- ask :func:`resolve` for the
    DD's own); ``attrs`` carries the small facts a role needs (the layer index
    of a geometry path, the member a composite took a path from).
    """

    ids: str
    canonical: str
    template: str = field(default="", compare=False)
    role: str = "data"
    coordinate: str = ""
    fallback_coordinate: tuple[str, ...] = ()
    units: str = ""
    attrs: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown DDPath role {self.role!r}; expected one of {ROLES}")
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))
        object.__setattr__(self, "fallback_coordinate", tuple(self.fallback_coordinate))
        if not self.template:
            object.__setattr__(self, "template", to_omas_template(self))

    def __str__(self) -> str:
        return self.canonical

    def __hash__(self) -> int:
        return hash((self.canonical, self.role))


@dataclass(frozen=True)
class DDInfo:
    """What the Data Dictionary says about one leaf, without any data loaded.

    ``coordinates`` are canonical spellings (``magnetics/ip(:)/time``) or the
    DD's own ``1...N`` for an index without a coordinate leaf; ``source`` is
    ``"omas"`` when only ``omas.omas_info_node`` answered and
    ``"omas+imas"`` when imas-python's ``IDSMetadata`` confirmed it.
    """

    units: str
    coordinates: tuple[str, ...]
    documentation: str
    lifecycle_status: str
    data_type: str
    dd_version: str
    source: str = "omas"


# ---------------------------------------------------------------------------
# the grammar
# ---------------------------------------------------------------------------


def _segments(path: DDPath | str) -> _Segments:
    canonical = path.canonical if isinstance(path, DDPath) else path
    if not canonical:
        raise ValueError("a DD path cannot be empty")
    out: list[tuple[str, Any]] = []
    for position, segment in enumerate(canonical.split("/")):
        match = _CANONICAL_SEGMENT.match(segment)
        if match is None:
            raise ValueError(
                f"malformed DD path {canonical!r}: segment {segment!r} is not "
                "name, name(:) or name(<int>)"
            )
        name, index = match.groups()
        if index is not None and position == 0:
            raise ValueError(f"malformed DD path {canonical!r}: the IDS segment takes no index")
        out.append((name, None if index is None else (":" if index == ":" else int(index))))
    return tuple(out)


def _canonical(segments: _Segments) -> str:
    return "/".join(name if index is None else f"{name}({index})" for name, index in segments)


def parse(canonical: str, *, role: str = "data", **fields: Any) -> DDPath:
    """A :class:`DDPath` from its canonical spelling (``magnetics/ip(:)/data``)."""
    segments = _segments(canonical)
    return DDPath(ids=segments[0][0], canonical=_canonical(segments), role=role, **fields)


def from_template(template: str, *, role: str = "data", **fields: Any) -> DDPath:
    """A :class:`DDPath` from a recipe template (``magnetics.ip.{i}.data``).

    ``{i}``, ``{j}``, ... and a bare ``:`` become the enumerated index
    ``(:)``; a digit segment becomes a fixed index; the template itself is
    kept on the path so it can be reproduced verbatim.
    """
    if not template:
        raise ValueError("a DD path template cannot be empty")
    segments: list[tuple[str, Any]] = []
    for piece in template.split("."):
        if _PLACEHOLDER.match(piece) or piece == ":":
            index: Any = ":"
        elif piece.isdigit():
            index = int(piece)
        elif _NAME.match(piece):
            segments.append((piece, None))
            continue
        else:
            raise ValueError(f"malformed DD path template {template!r}: segment {piece!r}")
        if not segments or segments[-1][1] is not None:
            raise ValueError(
                f"malformed DD path template {template!r}: index {piece!r} follows no structure"
            )
        segments[-1] = (segments[-1][0], index)
    if segments[0][1] is not None:
        raise ValueError(f"malformed DD path template {template!r}: the IDS segment takes no index")
    return DDPath(
        ids=segments[0][0], canonical=_canonical(tuple(segments)), template=template,
        role=role, **fields,
    )


def to_omas(path: DDPath | str, index: int | Mapping[str, int] = 0) -> str:
    """The OMAS dotted spelling with concrete indices (``magnetics.ip.0.data``).

    ``index`` fills every enumerated index: one integer for all of them, or a
    mapping by placeholder letter (``{"i": 1, "j": 2}``); a letter the mapping
    does not name is ``0``, as the recipes probe deeper indices.
    """
    pieces: list[str] = []
    enumerated = 0
    for name, idx in _segments(path):
        pieces.append(name)
        if idx is None:
            continue
        if idx == ":":
            letter = _PLACEHOLDER_LETTERS[enumerated] if enumerated < len(_PLACEHOLDER_LETTERS) else str(enumerated)
            value = index if isinstance(index, int) else int(index.get(letter, 0))
            enumerated += 1
            pieces.append(str(value))
        else:
            pieces.append(str(idx))
    return ".".join(pieces)


def to_omas_template(path: DDPath | str) -> str:
    """The recipe spelling (``magnetics.ip.{i}.data``).

    A :class:`DDPath` built by :func:`from_template` gives its template back
    verbatim; otherwise the enumerated indices are lettered ``{i}``, ``{j}``,
    ``{k}`` in order of appearance.
    """
    if isinstance(path, DDPath) and path.template:
        return path.template
    pieces: list[str] = []
    enumerated = 0
    for name, idx in _segments(path):
        pieces.append(name)
        if idx == ":":
            pieces.append("{" + _PLACEHOLDER_LETTERS[enumerated] + "}")
            enumerated += 1
        elif idx is not None:
            pieces.append(str(idx))
    return ".".join(pieces)


def to_info(path: DDPath | str) -> str:
    """The wildcard spelling ``omas.omas_info_node`` accepts (``magnetics.ip.:.data``).

    Every index -- enumerated *and* fixed -- becomes ``:``: the DD describes
    the structure, not one element of it.
    """
    pieces: list[str] = []
    for name, idx in _segments(path):
        pieces.append(name)
        if idx is not None:
            pieces.append(":")
    return ".".join(pieces)


def to_imas(path: DDPath | str) -> str:
    """The imas-python spelling relative to the IDS, no indices (``ip/data``)."""
    return "/".join(name for name, _ in _segments(path)[1:])


def _from_omas_bracketed(spelling: str) -> str:
    """``magnetics.ip[:].time`` (omas_info_node's coordinates) to canonical."""
    if not re.match(r"^[a-z]", spelling):
        return spelling  # ``1...N``
    pieces = []
    for piece in spelling.split("."):
        match = re.match(r"^([a-z][a-z0-9_]*)(?:\[(:|\d+)\])?$", piece)
        if match is None:
            return spelling
        name, index = match.groups()
        pieces.append(name if index is None else f"{name}({index})")
    return "/".join(pieces)


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

_UNIT_TOKEN = re.compile(r"^([A-Za-z%°µ]+)(?:\^(-?\d+))?$")


def normalise_units(text: str) -> str:
    """One spelling for a unit whichever way it was written.

    ``A.m^-2``, ``A/m^2`` and ``A m^-2`` are the same unit: the DD joins
    factors with ``.`` and writes negative exponents, VAFT's recipes write
    ``/`` and spaces.  Factors are sorted, so the result compares by
    equality; ``''``, ``'-'`` and ``'1'`` are all dimensionless.  A token
    that is not a plain symbol with an exponent (``A-turns``) is kept whole,
    so it compares unequal to anything else on purpose.
    """
    text = (text or "").strip()
    if text in ("", "-", "1"):
        return ""
    numerator, _, denominator = text.partition("/")
    exponents: dict[str, int] = {}

    def take(part: str, sign: int) -> None:
        for token in re.split(r"[.\s]+", part.strip()):
            if not token:
                continue
            match = _UNIT_TOKEN.match(token)
            if match is None:
                symbol, power = token, 1
            else:
                symbol, exponent = match.groups()
                power = int(exponent) if exponent else 1
            exponents[symbol] = exponents.get(symbol, 0) + sign * power

    take(numerator, +1)
    take(denominator, -1)
    factors = [
        symbol if power == 1 else f"{symbol}^{power}"
        for symbol, power in sorted(exponents.items())
        if power != 0
    ]
    return " ".join(factors)


# ---------------------------------------------------------------------------
# the Data Dictionary
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _omas_info(info_path: str, dd_version: str) -> Mapping[str, Any]:
    from omas import omas_info_node

    return MappingProxyType(dict(omas_info_node(info_path, imas_version=dd_version)))


@functools.lru_cache(maxsize=None)
def _imas_factory(dd_version: str) -> Any:
    import imas

    return imas.IDSFactory(dd_version)


def _imas_metadata(path: DDPath, dd_version: str) -> Any:
    """imas-python's ``IDSMetadata`` for ``path``, or ``None`` when it has none."""
    factory = _imas_factory(dd_version)
    try:
        node = getattr(factory, path.ids)().metadata
    except AttributeError:
        return None
    for name in to_imas(path).split("/"):
        try:
            node = node[name]
        except KeyError:
            return None
    return node


def resolve(
    path: DDPath | str,
    *,
    dd_version: str = DEFAULT_DD_VERSION,
    cross_check: bool | str = "auto",
) -> DDInfo:
    """What the Data Dictionary says about ``path``, with no data loaded.

    Reads ``omas.omas_info_node`` and, when imas-python is importable and
    ``cross_check`` is not ``False``, confirms the leaf and its units against
    ``imas.IDSFactory(dd_version)``'s metadata; the two disagreeing is a
    ``ValueError`` naming both.  A path the DD does not define is a
    ``KeyError`` naming the version, whichever side found it missing.
    """
    if isinstance(path, str):
        path = parse(path) if "/" in path else from_template(path)
    info = _omas_info(to_info(path), dd_version)
    if not info or info.get("full_path") is None:
        raise KeyError(f"{path.canonical} is not in Data Dictionary {dd_version}")
    units = str(info.get("units") or "")
    result = DDInfo(
        units=units,
        coordinates=tuple(_from_omas_bracketed(str(c)) for c in info.get("coordinates") or ()),
        documentation=str(info.get("documentation") or ""),
        lifecycle_status=str(info.get("lifecycle_status") or ""),
        data_type=str(info.get("data_type") or ""),
        dd_version=dd_version,
    )
    if cross_check is False:
        return result
    try:
        import imas  # noqa: F401
    except ImportError:
        if cross_check is True:
            raise
        return result
    metadata = _imas_metadata(path, dd_version)
    if metadata is None:
        raise KeyError(
            f"{path.canonical} is in omas' Data Dictionary {dd_version} but imas-python "
            f"({to_imas(path)!r} under {path.ids}) does not define it"
        )
    imas_units = str(getattr(metadata, "units", "") or "")
    if normalise_units(imas_units) != normalise_units(units):
        raise ValueError(
            f"{path.canonical}: omas says units {units!r}, imas-python says "
            f"{imas_units!r} (Data Dictionary {dd_version})"
        )
    return DDInfo(**{**result.__dict__, "source": "omas+imas"})


# ---------------------------------------------------------------------------
# what a plot declares
# ---------------------------------------------------------------------------

_ROLE_ORDER = {role: position for position, role in enumerate(ROLES)}


def dd_paths(name: str) -> tuple[DDPath, ...]:
    """The Data Dictionary paths canonical plot ``name`` reads, derived from its recipe.

    Data paths first, then coordinates, then the rest, in declaration order;
    one entry per canonical spelling, keeping the most specific role.  A
    composite lists its members' paths (``attrs["member"]`` names the member)
    and a computed view what its registry spec declares
    (``attrs["declared_by"] == "spec"``).  Touches no data.
    """
    from vaft.plot.registry import get_spec

    from . import recipes as R

    spec = get_spec(name)
    recipe = R.RECIPES.get(name)
    found: list[DDPath] = []

    def add(template: str, role: str, **extra: Any) -> None:
        if template:
            found.append(from_template(template, role=role, **extra))

    def canon(template: str) -> str:
        return from_template(template).canonical if template else ""

    if isinstance(recipe, R.LineRecipe):
        coordinates = tuple(canon(p) for p in recipe.x_paths)
        for position, template in enumerate((recipe.y_path, *recipe.fallback_y_paths)):
            add(
                template, "data" if position == 0 else "fallback",
                coordinate=coordinates[0] if coordinates else "",
                fallback_coordinate=coordinates[1:], units=recipe.y_unit,
            )
        for template in recipe.x_paths:
            add(template, "coordinate", units=recipe.x_unit)
        for abscissa in recipe.abscissae:
            for template in abscissa.paths:
                add(template, "abscissa", units=abscissa.unit, attrs={"abscissa": abscissa.name})
        add(recipe.label_path, "label")
        add(recipe.weight_path, "weight")
        add(recipe.divide_by_path, "divisor")
    elif isinstance(recipe, R.ProfileRecipe):
        coordinates = list(recipe.coordinate_paths.items())
        if not coordinates and recipe.y_path.startswith("equilibrium."):
            # An equilibrium profile offers every PROFILE_COORDINATES: the two
            # stored leaves, sqrt(phi_N) from phi, and the radial pair from the
            # midplane crossings (assembled in the builder from two leaves).
            coordinates = list(R._EQUILIBRIUM_COORDINATES.items()) + [
                ("sqrt_phi_norm", "equilibrium.time_slice.{i}.profiles_1d.phi"),
                ("r_major|r_minor", "equilibrium.time_slice.{i}.profiles_1d.r_inboard"),
                ("r_major|r_minor", "equilibrium.time_slice.{i}.profiles_1d.r_outboard"),
            ]
        default = canon(dict(coordinates).get(recipe.default_coordinate, ""))
        others = tuple(canon(p) for k, p in coordinates if k != recipe.default_coordinate)
        for position, template in enumerate((recipe.y_path, *recipe.fallback_y_paths)):
            add(
                template, "data" if position == 0 else "fallback",
                coordinate=default, fallback_coordinate=others, units=recipe.y_unit,
            )
        for key, template in coordinates:
            add(template, "coordinate", attrs={"coordinate": key})
        add(recipe.slice_container, "container")
        add(recipe.label_path, "label")
    elif isinstance(recipe, R.ChannelProfileRecipe):
        coordinates = tuple(canon(p) for p in recipe.time_paths)
        for position, template in enumerate((recipe.y_path, *recipe.fallback_y_paths)):
            add(
                template, "data" if position == 0 else "fallback",
                coordinate=coordinates[0] if coordinates else "",
                fallback_coordinate=coordinates[1:], units=recipe.y_unit,
            )
        for template in recipe.time_paths:
            add(template, "coordinate")
        add(recipe.label_path, "label")
    elif isinstance(recipe, R.GeometryRecipe):
        for layer, (kind, r_template, z_template, container, label_template, _style) in enumerate(recipe.layers):
            add(r_template, "geometry", attrs={"layer": layer, "kind": kind, "axis": "r"})
            add(z_template, "geometry", attrs={"layer": layer, "kind": kind, "axis": "z"})
            add(container, "container", attrs={"layer": layer})
            # A layer's label is a path template or a literal caption.
            try:
                add(label_template, "label", attrs={"layer": layer})
            except ValueError:
                pass
    elif isinstance(recipe, R.FieldRecipe):
        add(
            recipe.value_path, "data", coordinate=canon(recipe.r_path),
            units=_bracket_unit(recipe.value_label),
            attrs={"coordinates": (canon(recipe.r_path), canon(recipe.z_path))},
        )
        add(recipe.r_path, "coordinate", attrs={"axis": "r"})
        add(recipe.z_path, "coordinate", attrs={"axis": "z"})
        for template in recipe.boundary_paths:
            add(template, "boundary")
        # The other quantities the map can draw (field=, issue #483): a stored
        # or derived 2-D leaf beside the declared one, or a 1-D profile mapped
        # onto the grid where it is drawn.
        slice_prefix, _, _ = recipe.value_path.partition(".profiles_2d")
        grid_prefix = recipe.value_path.rsplit(".", 1)[0]
        for field_name in recipe.fields:
            field = R.EQUILIBRIUM_FIELDS[field_name]
            template = (
                f"{slice_prefix}.{field.mapped_from}" if field.mapped_from else f"{grid_prefix}.{field.leaf}"
            )
            if template != recipe.value_path:
                add(template, "optional", units=field.unit, attrs={"field": field_name})
    elif isinstance(recipe, (R.SpectrogramRecipe, R.PowerSpectrumRecipe)):
        coordinates = tuple(canon(p) for p in recipe.time_paths)
        for position, template in enumerate((recipe.signal_path, *recipe.fallback_signal_paths)):
            add(
                template, "data" if position == 0 else "fallback",
                coordinate=coordinates[0] if coordinates else "",
                fallback_coordinate=coordinates[1:],
            )
        for template in recipe.time_paths:
            add(template, "coordinate")
        add(recipe.container, "container")
        add(recipe.label_path, "label")
    elif isinstance(recipe, R.PanelRecipe):
        for member in recipe.members:
            for path in dd_paths(member):
                found.append(DDPath(**{**path.__dict__, "attrs": {**path.attrs, "member": member}}))
    # Every plot also declares paths on its registry spec; a computed view
    # declares nothing else (sub-issue #439 owns its full input declaration).
    declared_by = {"declared_by": "spec"} if isinstance(recipe, R.CallableRecipe) else {}
    for template in spec.required_paths:
        add(template, "required", attrs=declared_by)
    for template in spec.optional_paths:
        add(template, "optional", attrs=declared_by)

    unique: dict[str, DDPath] = {}
    for path in found:
        kept = unique.get(path.canonical)
        if kept is None:
            unique[path.canonical] = path
        elif _ROLE_ORDER[path.role] < _ROLE_ORDER[kept.role]:
            unique[path.canonical] = _merged(path, kept)
        else:
            unique[path.canonical] = _merged(kept, path)
    ordered = sorted(
        unique.values(),
        key=lambda p: _ROLE_ORDER[p.role] if p.role in ("data", "fallback", "coordinate", "abscissa") else 4,
    )
    return tuple(ordered)


def _merged(kept: DDPath, other: DDPath) -> DDPath:
    """``kept`` with the composite members ``other`` was also declared by."""
    members = tuple(dict.fromkeys(
        m for path in (kept, other) for m in (path.attrs.get("members") or (path.attrs.get("member"),)) if m
    ))
    if len(members) <= 1:
        return kept
    return DDPath(**{**kept.__dict__, "attrs": {**kept.attrs, "members": members}})


def _bracket_unit(label: str) -> str:
    """The ``[unit]`` suffix of an axis label, or ``""``."""
    match = re.search(r"\[([^\]]*)\]\s*$", label or "")
    return match.group(1) if match else ""
