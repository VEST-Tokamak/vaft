"""GPEC-native coil input generation for any machine's 3-D coil sets.

Coil geometry is a :class:`~vaft.machine_mapping.coils_non_axisymmetric_geometry.CoilSet3D`
mapping supplied by the caller (VEST's packaged configuration is the default
when ``machine == "vest"``).  This module owns the GPEC-specific
serialization: staging ``.dat`` files into a run's coil data directory as
``<machine>_<set>.dat`` and writing a ``coil.in`` whose ``coil_name``/
``coil_cur`` block expresses a run-specific excitation.  The machine-mapping
layer never emits GPEC files; only this adapter does.

``ip_direction``/``bt_direction`` (GPEC: "positive for CCW or negative for CW
from a top down view") are machine facts.  VEST's pair is
:data:`vaft.machine_mapping.conventions.VEST_GPEC_COIL_DIRECTIONS`; any other
machine must pass its own explicitly -- the adapter never inherits a
template's or another machine's words.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from vaft.machine_mapping.coils_non_axisymmetric_geometry import (
    CoilExcitation,
    CoilSet3D,
    Vest3DCoilConfig,
    load_vest_3d_coil_config,
)
from vaft.machine_mapping.conventions import VEST_GPEC_COIL_DIRECTIONS

from . import _runtime as rt

__all__ = [
    "CoilInputSpec",
    "resolve_coil_inputs",
    "stage_coil_data",
    "emit_coil_dat",
    "read_coil_in",
    "write_coil_in",
]

DEFAULT_MACHINE = "vest"
_DIRECTION_WORDS = ("positive", "negative")


_MACHINE_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def resolve_coil_inputs(
    machine: str,
    coil_config: Mapping[str, CoilSet3D] | Vest3DCoilConfig | None,
    ip_direction: str | None,
    bt_direction: str | None,
    names: Sequence[str],
) -> tuple[Mapping[str, CoilSet3D], str, str]:
    """Resolve geometry and direction words for ``machine``; refuse to guess.

    For ``"vest"`` the packaged configuration and
    :data:`VEST_GPEC_COIL_DIRECTIONS` fill whatever is not given.  For any
    other machine ``coil_config``, ``ip_direction`` and ``bt_direction`` are
    all required (``ValueError`` otherwise).  ``names`` are the sets to
    activate: non-empty, unique, each present in ``coil_config`` under a key
    equal to the set's own ``name`` (the staged file and ``coil.in`` both use
    it).  Every resolved set must be non-empty with closed filaments of one
    point count, so a hand-built :class:`CoilSet3D` gets the same checks as
    one from :func:`coil_set_from_xyz_loops`.
    """
    if not isinstance(machine, str) or not _MACHINE_WORD.fullmatch(machine):
        raise ValueError(
            f"machine must be a plain word (letters, digits, '_', '-') used as the .dat file "
            f"prefix and the coil.in machine word, got {machine!r}"
        )
    names = list(names)
    if not names:
        raise ValueError("at least one coil set name is required")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"coil set(s) {duplicates} listed more than once; GPEC would apply them twice")
    if isinstance(coil_config, Vest3DCoilConfig):
        coil_config = coil_config.coil_sets
    if coil_config is None:
        if machine != DEFAULT_MACHINE:
            raise ValueError(
                f"machine {machine!r}: coil_config (name -> CoilSet3D) is required; only "
                f"{DEFAULT_MACHINE!r} has packaged coil geometry"
            )
        coil_config = load_vest_3d_coil_config(coil_sets=names).coil_sets
    missing = [name for name in names if name not in coil_config]
    if missing:
        raise ValueError(f"machine {machine!r}: coil set(s) {missing} not in coil_config {sorted(coil_config)}")
    for name in names:
        coil_set = coil_config[name]
        if coil_set.name != name:
            raise ValueError(
                f"coil_config key {name!r} holds a set named {coil_set.name!r}; the key must equal "
                "the set name because both the staged .dat file and coil.in use it"
            )
        if not coil_set.filaments:
            raise ValueError(f"coil set {name!r} has no filaments")
        npts = {f.points_xyz.shape[0] for f in coil_set.filaments}
        if len(npts) != 1:
            raise ValueError(f"coil set {name!r}: filaments have differing point counts {sorted(npts)}")
        open_loops = [k for k, f in enumerate(coil_set.filaments) if not f.is_closed]
        if open_loops:
            raise ValueError(f"coil set {name!r}: filament(s) {open_loops} are not closed")
    if ip_direction is None or bt_direction is None:
        if machine != DEFAULT_MACHINE:
            raise ValueError(
                f"machine {machine!r}: ip_direction and bt_direction must be given explicitly "
                f"({_DIRECTION_WORDS}); they are machine facts and are never inherited"
            )
        if ip_direction is None:
            ip_direction = VEST_GPEC_COIL_DIRECTIONS["ip_direction"]
        if bt_direction is None:
            bt_direction = VEST_GPEC_COIL_DIRECTIONS["bt_direction"]
    for label, value in (("ip_direction", ip_direction), ("bt_direction", bt_direction)):
        if value not in _DIRECTION_WORDS:
            raise ValueError(f"{label} must be one of {_DIRECTION_WORDS}, got {value!r}")
    return coil_config, ip_direction, bt_direction


@dataclass(frozen=True)
class CoilInputSpec:
    """One activated coil set for a GPEC run: canonical name plus sector currents (A)."""

    name: str
    currents_a: tuple[float, ...]

    @classmethod
    def from_excitation(cls, excitation: CoilExcitation) -> "CoilInputSpec":
        return cls(name=excitation.coil_set, currents_a=tuple(excitation.currents_a))


def stage_coil_data(
    coil_sets: Sequence[CoilSet3D], data_dir: Path, *, machine: str
) -> tuple[Path, ...]:
    """Stage each set as ``<data_dir>/<machine>_<coil_name>.dat``.

    A set backed by a file (``dat_path``) is copied byte-identical (a missing
    file is an error, never silently re-emitted); a set built in memory
    (:func:`coil_set_from_xyz_loops`, ``dat_path is None``) is written with
    :func:`emit_coil_dat`.  GPEC resolves the file from the ``machine`` word
    written to ``coil.in``, so ``machine`` is required here and must be the
    word passed to :func:`write_coil_in`.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for coil_set in coil_sets:
        target = data_dir / f"{machine}_{coil_set.name}.dat"
        if coil_set.dat_path is not None:
            shutil.copy2(coil_set.dat_path, target)
        else:
            emit_coil_dat(coil_set, target)
        staged.append(target)
    return tuple(staged)


def emit_coil_dat(coil_set: CoilSet3D, path: Path) -> Path:
    """Write a GPEC ``.dat`` (header :data:`GPEC_COIL_DAT_HEADER`) from the arrays.

    This is the production writer for every set built in memory (any
    non-VEST machine, or VEST sets constructed from loops); file-backed sets
    are copied by :func:`stage_coil_data` instead so their bytes are kept.
    Coordinates are written with seven significant digits; header fields are
    space-separated so no count can run into its neighbour.
    """
    npts = coil_set.filaments[0].points_xyz.shape[0]
    lines = [f"{len(coil_set.filaments)} 1 {npts} {coil_set.turns:.6g}"]
    for filament in coil_set.filaments:
        lines.extend(f"{x:15.6e}{y:15.6e}{z:15.6e}" for x, y, z in filament.points_xyz)
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_coil_in(path: str | Path) -> tuple[CoilInputSpec, ...]:
    """Read the activated coil sets and sector currents from a GPEC ``coil.in``.

    Tolerant of both the generated layout and hand-written references:
    ``coil_name(i)`` / ``coil_cur(i,j)`` entries are collected per set index
    (multi-value ``coil_cur(i,1..k)=a,b,c`` rows included), honoring
    ``coil_num`` when present.
    """
    text = Path(path).read_text(encoding="utf-8")
    body = text.split("&COIL_CONTROL", 1)[-1]
    names: dict[int, str] = {}
    currents: dict[int, list[float]] = {}
    coil_num: int | None = None
    for line in body.splitlines():
        line = line.split("!", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        name_match = re.fullmatch(r"coil_name\((\d+)\)", key)
        cur_match = re.fullmatch(r"coil_cur\((\d+)(?:,\d+(?:\.\.\d+)?)?\)", key)
        if name_match:
            names[int(name_match.group(1))] = value.strip("\"'")
        elif cur_match:
            set_index = int(cur_match.group(1))
            currents.setdefault(set_index, []).extend(
                float(token) for token in value.replace(",", " ").split()
            )
        elif key == "coil_num":
            coil_num = int(float(value))
    active = sorted(names)
    if coil_num is not None:
        active = active[:coil_num]
    return tuple(
        CoilInputSpec(name=names[index], currents_a=tuple(currents.get(index, ())))
        for index in active
    )


def _render_coil_block(specs: Sequence[CoilInputSpec]) -> str:
    lines = []
    for set_index, spec in enumerate(specs, start=1):
        lines.append(f'    coil_name({set_index})="{spec.name}"')
        for sector_index, current in enumerate(spec.currents_a, start=1):
            lines.append(
                f"    coil_cur({set_index},{sector_index})={current:g}"
            )
        lines.append("")
    return "\n".join(lines)


def write_coil_in(
    template_path: Path,
    out_path: Path,
    *,
    data_dir: Path,
    specs: Sequence[CoilInputSpec],
    machine: str,
    coil_config: Mapping[str, CoilSet3D] | None = None,
    ip_direction: str | None = None,
    bt_direction: str | None = None,
) -> Path:
    """Write a GPEC ``coil.in`` activating exactly ``specs`` for ``machine``.

    ``machine`` has no default because it is the one word that names the
    machine; GPEC uses it only as the ``<machine>_<set>.dat`` file prefix
    (the enumeration in the template comment is upstream documentation, and
    the packaged VEST runs use a word outside it).  Everything not listed
    above (vacuum-grid resolution, ``&COIL_OUTPUT``) is inherited from
    ``template_path``; a machine needing different values passes its own
    template.

    Scalar keys (``data_dir``, ``machine``, ``coil_num``, ``ip_direction``,
    ``bt_direction``) are patched through the template; the template's fixed
    ``coil_name``/``coil_cur`` block is replaced wholesale because it cannot
    express a variable number of activated sets.  ``coil_num`` is always the
    number of ``specs``.  See :func:`resolve_coil_inputs` for what may be
    omitted for VEST and what is mandatory for every other machine.
    """
    out_path = Path(out_path)
    template_path = Path(template_path)
    if not specs:
        raise ValueError("write_coil_in requires at least one CoilInputSpec")
    known, ip_direction, bt_direction = resolve_coil_inputs(
        machine, coil_config, ip_direction, bt_direction, [spec.name for spec in specs]
    )
    for spec in specs:
        expected = len(known[spec.name].filaments)
        if len(spec.currents_a) != expected:
            raise ValueError(
                f"Coil set {spec.name!r} has {expected} sectors but the spec "
                f"carries {len(spec.currents_a)} currents"
            )

    rt.write_template(
        template_path,
        out_path,
        {
            "data_dir": str(Path(data_dir)),
            "machine": machine,
            "coil_num": len(specs),
            "ip_direction": ip_direction,
            "bt_direction": bt_direction,
        },
    )
    text = out_path.read_text(encoding="utf-8")
    stripped = re.sub(
        r"^\s*coil_(?:name|cur)\([^)]*\)\s*=.*\n", "", text, flags=re.MULTILINE
    )
    # Insert the generated activation block just before COIL_CONTROL's
    # closing "/" (the first namelist terminator in the file).
    terminator = re.search(r"^/\s*$", stripped, flags=re.MULTILINE)
    if terminator is None:
        raise ValueError(f"no namelist terminator found in template {template_path}")
    position = terminator.start()
    rendered = (
        stripped[:position].rstrip("\n")
        + "\n\n"
        + _render_coil_block(specs)
        + stripped[position:]
    )
    out_path.write_text(rendered, encoding="utf-8")
    return out_path
