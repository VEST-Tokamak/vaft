"""GPEC-native coil input generation from the canonical VEST configuration.

The canonical VEST 3D coil geometry lives in
:mod:`vaft.machine_mapping.coil_geometry_3d` (three packaged GPEC-format
``.dat`` files plus set metadata).  This module owns the GPEC-specific
serialization: staging the ``.dat`` files into a run's coil data directory
and writing a ``coil.in`` whose ``coil_name``/``coil_cur`` block expresses a
run-specific excitation.  The machine-mapping layer never emits GPEC files;
only this adapter does.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from vaft.machine_mapping.coil_geometry_3d import (
    CoilExcitation,
    CoilSet3D,
    load_vest_3d_coil_config,
)

from . import _runtime as rt

__all__ = [
    "CoilInputSpec",
    "stage_coil_data",
    "emit_coil_dat",
    "read_coil_in",
    "write_coil_in",
]

_GPEC_MACHINE = "vest"


@dataclass(frozen=True)
class CoilInputSpec:
    """One activated coil set for a GPEC run: canonical name plus sector currents (A)."""

    name: str
    currents_a: tuple[float, ...]

    @classmethod
    def from_excitation(cls, excitation: CoilExcitation) -> "CoilInputSpec":
        return cls(name=excitation.coil_set, currents_a=tuple(excitation.currents_a))


def stage_coil_data(coil_sets: Sequence[CoilSet3D], data_dir: Path) -> tuple[Path, ...]:
    """Copy each set's packaged ``.dat`` byte-identical into ``data_dir``.

    GPEC resolves ``<data_dir>/<machine>_<coil_name>.dat``, so the staged name
    is derived from the canonical set name.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for coil_set in coil_sets:
        target = data_dir / f"{_GPEC_MACHINE}_{coil_set.name}.dat"
        shutil.copy2(coil_set.dat_path, target)
        staged.append(target)
    return tuple(staged)


def emit_coil_dat(coil_set: CoilSet3D, path: Path) -> Path:
    """Regenerate a GPEC ``.dat`` from the parsed arrays.

    Exists for reconstruction/validation: the staged copies from
    :func:`stage_coil_data` remain the authoritative byte-identical inputs.
    """
    points = np.concatenate([f.points_xyz for f in coil_set.filaments])
    npts = coil_set.filaments[0].points_xyz.shape[0]
    lines = [
        f"{len(coil_set.filaments):5d}{1:5d}{npts:5d}{coil_set.turns:9.2f}"
    ]
    lines.extend(
        f"{x:15.6e}{y:15.6e}{z:15.6e}" for x, y, z in points
    )
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
    ip_direction: str = "positive",
    bt_direction: str = "negative",
) -> Path:
    """Write a GPEC ``coil.in`` activating exactly ``specs``.

    Scalar keys are patched through the packaged template; the template's
    fixed ``coil_name``/``coil_cur`` block is replaced wholesale because it
    cannot express a variable number of activated sets.
    """
    if not specs:
        raise ValueError("write_coil_in requires at least one CoilInputSpec")
    known = load_vest_3d_coil_config(coil_sets=[spec.name for spec in specs])
    for spec in specs:
        expected = len(known[spec.name].filaments)
        if len(spec.currents_a) != expected:
            raise ValueError(
                f"Coil set {spec.name!r} has {expected} sectors but the spec "
                f"carries {len(spec.currents_a)} currents"
            )

    rt.write_template(
        Path(template_path),
        Path(out_path),
        {
            "data_dir": str(Path(data_dir)),
            "machine": _GPEC_MACHINE,
            "coil_num": len(specs),
            "ip_direction": ip_direction,
            "bt_direction": bt_direction,
        },
    )
    text = Path(out_path).read_text(encoding="utf-8")
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
    Path(out_path).write_text(rendered, encoding="utf-8")
    return Path(out_path)
