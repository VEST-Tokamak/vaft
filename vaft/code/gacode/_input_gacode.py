"""Read and write ``input.gacode``, in pure Python.

GACODE ships its own reader as ``pygacode``, but that is an f2py extension
built from ``f2py/expro/expro.f90``, so using it would make a Fortran toolchain
a requirement for reading a text file.  The format is a tagged flat file and is
reproduced here directly against ``expro_write`` and its helpers in
``f2py/expro/expro_util.f90``.

Layout::

    #  *original : ...        six free-text header lines, fixed order
    ...
    #
    # nexp                    integer, format i0
    51
    # torfluxa | Wb/radian    scalar, format 1pe14.7
     6.1675847E-01
    # rho | -                 profile, format (i3,1x,1pe14.7)
      1  0.0000000E+00
    # ni | 10^19/m^3          per-ion profile, format (i3,1x,10(1pe14.7,1x))
      1  5.3635000E+00  1.7599000E-01

Two behaviours of ``expro`` the writer here reproduces deliberately:

* **All-zero objects are omitted.**  ``expro_writev`` skips a vector whose
  absolute sum is below 1e-16.  So a tag's *absence* from a file carries no
  information beyond "not set or identically zero", and the reader must not
  invent a zero array for a missing tag.
* **The unit strings in an existing file may be stale.**  ``qpar_beam`` is
  labelled ``MW/m^3`` in files written by older versions and ``1/m^3/s`` today.
  The reader therefore keys on the tag and ignores the unit; the writer emits
  the current one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ._profiles import SHAPE_COS_FIELDS, SHAPE_SIN_FIELDS, SOURCE_FIELDS, GACODEProfile

#: The six free-text header lines, in the order ``expro_write`` emits them.
HEADER_KEYS = ("original", "statefile", "gfile", "cerfile", "vgen", "tgyro")

#: Integer tags, written by ``expro_writei`` and omitted when not positive.
INTEGER_TAGS = ("nexp", "nion", "shot", "time")

#: Whitespace-separated string lists, one entry per ion.
STRING_TAGS = ("name", "type")

#: Bare floats with no unit and no index: one value, or one per ion.
BARE_FLOAT_TAGS = ("masse", "mass", "ze", "z")

#: Unit-carrying scalars, written by ``expro_writes``.
SCALAR_TAGS: tuple[tuple[str, str], ...] = (
    ("torfluxa", "Wb/radian"),
    ("rcentr", "m"),
    ("bcentr", "T"),
    ("current", "MA"),
)

#: Unit-carrying profiles, in ``expro_write`` order.  ``per_ion`` marks the
#: tags written by ``expro_writea`` with one column per ion species.
PROFILE_TAGS: tuple[tuple[str, str, bool], ...] = (
    ("rho", "-", False),
    ("rmin", "m", False),
    ("polflux", "Wb/radian", False),
    ("q", "-", False),
    ("w0", "rad/s", False),
    ("rmaj", "m", False),
    ("zmag", "m", False),
    ("kappa", "-", False),
    ("delta", "-", False),
    ("zeta", "-", False),
    *((name, "-", False) for name in SHAPE_COS_FIELDS),
    *((name, "-", False) for name in SHAPE_SIN_FIELDS),
    ("ne", "10^19/m^3", False),
    ("ni", "10^19/m^3", True),
    ("te", "keV", False),
    ("ti", "keV", True),
    ("ptot", "Pa", False),
    ("fpol", "T-m", False),
    ("johm", "MA/m^2", False),
    ("jbs", "MA/m^2", False),
    ("jrf", "MA/m^2", False),
    ("jnb", "MA/m^2", False),
    ("jbstor", "MA/m^2", False),
    ("sigmapar", "MSiemens/m", False),
    ("z_eff", "-", False),
    ("vpol", "m/s", True),
    ("vtor", "m/s", True),
    ("qohme", "MW/m^3", False),
    ("qbeame", "MW/m^3", False),
    ("qbeami", "MW/m^3", False),
    ("qrfe", "MW/m^3", False),
    ("qrfi", "MW/m^3", False),
    ("qfuse", "MW/m^3", False),
    ("qfusi", "MW/m^3", False),
    ("qbrem", "MW/m^3", False),
    ("qsync", "MW/m^3", False),
    ("qline", "MW/m^3", False),
    ("qei", "MW/m^3", False),
    ("qione", "MW/m^3", False),
    ("qioni", "MW/m^3", False),
    ("qcxi", "MW/m^3", False),
    ("qpar_beam", "1/m^3/s", False),
    ("qpar_wall", "1/m^3/s", False),
    ("qmom", "N/m^2", False),
)

_PROFILE_UNITS = {name: unit for name, unit, _ in PROFILE_TAGS}
_PER_ION = {name for name, _, per_ion in PROFILE_TAGS if per_ion}

#: expro_writev/writes skip anything whose absolute sum is below this.
ZERO_TOLERANCE = 1e-16


def _fortran_float(value: float) -> str:
    """Format one value as Fortran ``1pe14.7``: a signed 14-character field."""
    return f"{float(value): .7E}"


def _split_sections(lines: Iterable[str]) -> tuple[dict[str, str], list[tuple[str, list[str]]]]:
    """Split the file into its header block and its ``# tag`` sections."""
    header: dict[str, str] = {}
    sections: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    in_header = True

    for raw in lines:
        line = raw.rstrip("\n\r")
        if in_header:
            stripped = line.strip()
            if stripped == "#":
                in_header = False
                continue
            if stripped.startswith("#") and ":" in stripped:
                key, _, value = stripped[1:].partition(":")
                header[key.strip().lstrip("*")] = value.strip()
                continue
            # A file with no header block at all: fall through and treat this
            # line as the start of the data.
            in_header = False

        if line.startswith("#"):
            tag = line[1:].split("|")[0].strip()
            current = (tag, [])
            sections.append(current)
        elif current is not None and line.strip():
            current[1].append(line)
    return header, sections


def _profile_columns(rows: list[str], tag: str) -> np.ndarray:
    """Parse ``index value...`` rows into ``(columns, rows)``, dropping the index."""
    parsed = []
    for row in rows:
        fields = row.split()
        if len(fields) < 2:
            raise ValueError(f"input.gacode: malformed row in section {tag!r}: {row!r}")
        parsed.append([float(value) for value in fields[1:]])
    widths = {len(row) for row in parsed}
    if len(widths) != 1:
        raise ValueError(
            f"input.gacode: section {tag!r} has rows of differing width {sorted(widths)}"
        )
    array = np.asarray(parsed, dtype=float)
    return array[:, 0] if array.shape[1] == 1 else array.T


def read_input_gacode(path: str | Path) -> GACODEProfile:
    """Read an ``input.gacode`` file into a :class:`GACODEProfile`.

    Every tag is preserved: modelled ones land on their field, the volumetric
    source terms in ``sources``, shape harmonics in ``shape``, and anything
    unrecognised in ``extra`` so that a round trip is lossless.

    Raises
    ------
    ValueError
        The file has no ``rho`` section, ragged rows, or a per-ion section whose
        width disagrees with the declared ion count.
    """
    source = Path(path)
    header, sections = _split_sections(
        source.read_text(encoding="utf-8", errors="replace").splitlines()
    )

    values: dict[str, Any] = {}
    unknown: dict[str, Any] = {}
    for tag, rows in sections:
        if not rows:
            continue
        if tag in INTEGER_TAGS:
            values[tag] = int(float(rows[0].split()[0]))
        elif tag in STRING_TAGS:
            values[tag] = tuple(rows[0].split())
        elif tag in BARE_FLOAT_TAGS:
            numbers = [float(value) for value in " ".join(rows).split()]
            values[tag] = numbers[0] if len(numbers) == 1 else np.asarray(numbers)
        elif tag in dict(SCALAR_TAGS):
            values[tag] = float(rows[0].split()[0])
        elif tag in _PROFILE_UNITS:
            values[tag] = _profile_columns(rows, tag)
        else:
            unknown[tag] = _profile_columns(rows, tag)

    if "rho" not in values:
        raise ValueError(f"input.gacode: {source} has no 'rho' section")

    charge = values.get("z")
    if charge is None:
        raise ValueError(f"input.gacode: {source} has no 'z' section")
    charge = np.atleast_1d(np.asarray(charge, dtype=float))

    declared_ions = values.get("nion")
    if declared_ions is not None and int(declared_ions) != charge.size:
        raise ValueError(
            f"input.gacode: nion is {declared_ions} but 'z' lists {charge.size} species"
        )
    declared_points = values.get("nexp")
    rho = np.asarray(values["rho"], dtype=float)
    if declared_points is not None and int(declared_points) != rho.size:
        raise ValueError(
            f"input.gacode: nexp is {declared_points} but 'rho' has {rho.size} points"
        )

    for tag in _PER_ION:
        array = values.get(tag)
        if array is None:
            continue
        array = np.atleast_2d(array)
        if array.shape[0] != charge.size:
            raise ValueError(
                f"input.gacode: section {tag!r} has {array.shape[0]} columns "
                f"but there are {charge.size} ion species"
            )
        values[tag] = array

    shape = {
        name: values.pop(name)
        for name in (*SHAPE_COS_FIELDS, *SHAPE_SIN_FIELDS)
        if name in values
    }
    sources = {name: values.pop(name) for name in SOURCE_FIELDS if name in values}
    mass = values.get("mass")

    profile = GACODEProfile(
        rho=rho,
        z=charge,
        shape=shape,
        sources=sources,
        extra=unknown,
        header=header,
        mass=None if mass is None else np.atleast_1d(np.asarray(mass, dtype=float)),
        masse=float(values.get("masse", 5.4488741e-04)),
        ze=float(values.get("ze", -1.0)),
        name=values.get("name", ()),
        type=values.get("type", ()),
        shot=values.get("shot"),
        time=values.get("time"),
        **{
            key: values[key]
            for key in (
                "rmin", "polflux", "q", "w0", "rmaj", "zmag", "kappa", "delta",
                "zeta", "ne", "ni", "te", "ti", "ptot", "z_eff", "vpol", "vtor",
                "fpol", "johm", "jbs", "jrf", "jnb", "jbstor", "sigmapar",
                "torfluxa", "rcentr", "bcentr", "current",
            )
            if key in values
        },
    )
    profile.provenance = {
        name: {"kind": "caller_supplied", "source": str(source)}
        for name in (*values, *shape, *sources, *unknown)
        if name not in {"nexp", "nion", "rho", "z"}
    }
    return profile


def _section(tag: str, unit: str | None) -> str:
    return f"# {tag}\n" if unit is None else f"# {tag} | {unit}\n"


def _write_profile(tag: str, unit: str, values: np.ndarray) -> str:
    """Render one profile section, or nothing when it is identically zero."""
    array = np.asarray(values, dtype=float)
    if array.size == 0 or float(np.sum(np.abs(array))) <= ZERO_TOLERANCE:
        return ""
    text = [_section(tag, unit)]
    if array.ndim == 1:
        for index, value in enumerate(array, start=1):
            text.append(f"{index:3d} {_fortran_float(value)}\n")
    else:
        for index in range(array.shape[1]):
            columns = " ".join(_fortran_float(value) for value in array[:, index])
            text.append(f"{index + 1:3d} {columns}\n")
    return "".join(text)


def write_input_gacode(profile: GACODEProfile, path: str | Path) -> Path:
    """Write a :class:`GACODEProfile` as ``input.gacode`` and return the path.

    The output reproduces ``expro_write``: the same tag order, the same
    ``1pe14.7`` formatting, and the same omission of identically-zero objects,
    so that GACODE reads back exactly what it would have written.
    """
    target = Path(path)
    header = dict(profile.header)
    # expro declares these as character(len=70) with the starred tag
    # right-aligned to column 12, and Fortran writes the whole fixed-width
    # field, so the trailing padding is part of the format.
    text = [
        f"# {'*' + key:>10s} : {header.get(key, 'null')}".ljust(70) + "\n"
        for key in HEADER_KEYS
    ]
    text.append("#\n")

    integers = {
        "nexp": profile.n_exp,
        "nion": profile.n_ion,
        "shot": profile.shot,
        "time": profile.time,
    }
    for tag in INTEGER_TAGS:
        value = integers[tag]
        if value is not None and int(value) > 0:
            text.append(_section(tag, None))
            text.append(f"{int(value)}\n")

    names = tuple(profile.name) or tuple(f"i{i + 1}" for i in range(profile.n_ion))
    kinds = tuple(profile.type) or ("[therm]",) * profile.n_ion
    # expro's "(20(a,1x))" would leave a trailing blank, and its
    # "(10(1pe14.7,1x))" one per row. The files GACODE ships carry none, and
    # its parser splits on whitespace either way, so the reference artifact is
    # what is reproduced here.
    text.append(_section("name", None))
    text.append(" ".join(names) + "\n")
    text.append(_section("type", None))
    text.append(" ".join(kinds) + "\n")
    text.append(_section("masse", None))
    text.append(_fortran_float(profile.masse) + "\n")
    if profile.mass is not None:
        text.append(_section("mass", None))
        text.append("".join(_fortran_float(v) for v in profile.mass) + "\n")
    text.append(_section("ze", None))
    text.append(_fortran_float(profile.ze) + "\n")
    text.append(_section("z", None))
    text.append("".join(_fortran_float(v) for v in profile.z) + "\n")

    for tag, unit in SCALAR_TAGS:
        value = getattr(profile, tag, None)
        if value is not None and abs(float(value)) > ZERO_TOLERANCE:
            text.append(_section(tag, unit))
            text.append(_fortran_float(value) + "\n")

    for tag, unit, _per_ion in PROFILE_TAGS:
        if tag in profile.shape:
            values = profile.shape[tag]
        elif tag in profile.sources:
            values = profile.sources[tag]
        else:
            values = getattr(profile, tag, None)
        if values is None:
            continue
        text.append(_write_profile(tag, unit, values))

    for tag, values in profile.extra.items():
        text.append(_write_profile(tag, "-", np.asarray(values)))

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(text), encoding="utf-8")
    return target
