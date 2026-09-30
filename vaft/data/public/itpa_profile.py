"""ITPA International Multi-Tokamak Confinement Profile Database (PR08) reader and ODS mapping.

Source: https://tokamak-profiledb.ccfe.ac.uk/PR08/itpa_pr08/<machine>/<shot>/,
one directory per discharge with four files: ``pr08_<m>_<shot>_0d.dat`` (global
quantities), ``_1d.dat`` (time traces), ``_2d.dat`` (radial profiles against
time) and ``_com.dat`` (free-text comments: analysis codes, assumptions,
contacts).  Public, no login.  Terms of use, from the release page: anyone may
analyse, present and publish from the public release provided they cite
D. Boucher et al., Nucl. Fusion 40 (2000) 1955 and the ITER Physics Basis,
Nucl. Fusion 39 (1999) 2175 (or the reference in the discharge's comment
file).  Nothing is said about redistribution, so VAFT fetches on demand with a
pinned checksum and ships no PR08 file.

Format facts this reader relies on (PR08 manual, ``DOCS/PR08MAN/pdbman.html``)
-------------------------------------------------------------------------------
* 1D and 2D files are concatenated ASCII UFILEs (PPPL, D. McCune): a header
  whose lines end in ``;-LABEL-`` markers, then the independent variables and
  the data with X varying fastest, then ``;----END-OF-DATA`` and a comment
  block.
* The 2D radial coordinate is ``rho = sqrt(Phi / Phi_a)``, the normalised
  toroidal flux, 0 on axis and 1 at the separatrix -- IMAS ``rho_tor_norm``
  without any conversion.  The file header leaves the X label blank.
* Signals of one discharge sit on **different radial grids** (TRANSP zone
  centres, zone boundaries, the measured-profile grids).  Nothing is
  interpolated: a target block takes the grid of its reference signal and only
  signals on that same grid (and the same times); the rest are listed as
  unmapped with the reason in :func:`pr08_mapping_coverage`.
* Missing values are ``-9.999E-09`` (real), ``-9999999`` (integer) and
  ``????????`` (string).
* Variables ending in ``XP`` are measured profiles, ``EB`` their error bars,
  the plain names fitted/interpretive (e.g. TRANSP) profiles.
* 0D files come either as CSV (header row, one row per time) or as
  fixed-width 11-character fields wrapped seven to a line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import os
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from ._fetch import FetchError, cache_dir, fetch, sha256_of

__all__ = [
    "PR08_BASE_URL",
    "PR08_PINNED",
    "PR08_REFERENCE",
    "Pr08Discharge",
    "UFileSignal",
    "fetch_pr08",
    "pr08_mapping_coverage",
    "pr08_to_omas",
    "read_pr08",
    "read_pr08_0d",
    "read_ufiles",
]

PR08_BASE_URL = "https://tokamak-profiledb.ccfe.ac.uk/PR08/itpa_pr08"
PR08_REFERENCE = (
    "D. Boucher et al., 'The International Multi-Tokamak Profile Database', "
    "Nucl. Fusion 40 (2000) 1955, doi:10.1088/0029-5515/40/12/302; ITPA PR08 "
    "public release, tokamak-profiledb.ccfe.ac.uk"
)
PR08_KINDS = ("0d", "1d", "2d", "com")

#: SHA-256 of the files of the discharges VAFT has checked, keyed (machine, shot).
PR08_PINNED: dict[tuple[str, str], dict[str, str]] = {
    ("mast", "8302"): {
        "0d": "b3783c0936c76b402fad9655668c581c0be481a42872a2b05de3293cc966c63e",
        "1d": "897ae40a9f91a5888fd0fb0f70b8307758542d3719d315382e33d5a07a24e388",
        "2d": "9806d5de04f95936f6d258e363cac18dcf55fe2249f28922571260592b213a62",
        "com": "5c713817220ae550d1991e4f39b43f867a88cf4bf2659e26c8d4b2657d3048c8",
    },
    ("d3d", "81507"): {
        "0d": "c02217ad9450fb857b102de7aa65ee591b7c17de4ea2f306cd1b1edbd3f97352",
        "1d": "df5189960f1e1977b5eec2fb3e892818eeb83a5f9bacdba5beb2c33d064b72fc",
        "2d": "9fd3e67dc84a5cbda02592bc8642d11c362f6feb69bd1396668576908b25d52c",
        "com": "75a11da44c802486bc6f65640aa48a730f0f684c5c07a42ba3cd1735eb3fb070",
    },
    ("jet", "19649"): {
        "0d": "63e92d9f91fc9ccb4e4909473645e2723d23331218986caab055a8e58a5be7e5",
        "1d": "916a7456804532ebf7f8cf845d51514a406859a8fecd43323c0ab246cc457a32",
        "2d": "c395c50f3e704b2502c83b997d06bea6c28e8fb8df3a3dd4b5a3ba7985f045dd",
        "com": "dab082254a1cf40df2ca50df23f3d89f01418e869da05578ecf7454cbf200ded",
    },
}

_REAL_MISSING = -9.999e-09
_INT_MISSING = -9999999
_STRING_MISSING = "????????"
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")


# ---------------------------------------------------------------- data model


@dataclass(frozen=True)
class UFileSignal:
    """One UFILE block, in the file's own names and units.

    Attributes
    ----------
    name : str
        Dependent-variable label, e.g. ``"TE"``.
    unit : str
        Unit as written in the file, e.g. ``"eV"``.
    time : numpy.ndarray
        Time base [s] (the Y variable of a 2D file, the X of a 1D file).
    values : numpy.ndarray
        ``(n_time,)`` for 1D, ``(n_time, n_rho)`` for 2D; missing is NaN.
    rho : numpy.ndarray or None
        Radial grid of a 2D signal, ``sqrt(Phi/Phi_a)``; ``None`` for 1D.
    comment : str
        Comment block after the data.
    """

    name: str
    unit: str
    time: np.ndarray
    values: np.ndarray
    rho: np.ndarray | None = None
    comment: str = ""


@dataclass
class Pr08Discharge:
    """One PR08 discharge, as released.

    Attributes
    ----------
    machine : str
        Machine directory name, lower case (e.g. ``"mast"``).
    shot : str
        Shot directory name (PR08 shots may carry letter suffixes).
    zero_d : pandas.DataFrame
        One row per 0D time, source names, missing as NaN / None.
    one_d : dict of str to UFileSignal
        Time traces.
    two_d : dict of str to UFileSignal
        Radial profiles against time.
    comments : str
        Content of the comment file.
    files : dict of str to str
        SHA-256 of each file read, keyed by kind (``"0d"`` ...).
    verification : dict of str to str
        Per kind, ``"pinned"`` (checked against a known hash) or
        ``"unverified"`` (downloaded without one); empty for a local read.
    """

    machine: str
    shot: str
    zero_d: pd.DataFrame
    one_d: dict[str, UFileSignal]
    two_d: dict[str, UFileSignal]
    comments: str = ""
    files: dict[str, str] = field(default_factory=dict)
    verification: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------- UFILE reader


def _numbers(lines: list[str]) -> np.ndarray:
    # Fixed-format numbers may abut ("1.2E-02-3.4E-02"); a regex splits them.
    return np.array(
        [float(token.replace("D", "E").replace("d", "e")) for line in lines for token in _NUMBER.findall(line)],
        dtype=float,
    )


def _labelled(lines: list[str], marker: str) -> int:
    for index, line in enumerate(lines):
        if marker in line.upper():
            return index
    raise ValueError(f"UFILE block has no line marked {marker!r}")


def _parse_block(block: str) -> UFileSignal | None:
    head, _, comment = block.partition(";----END-OF-DATA")
    lines = [line for line in head.splitlines() if line.strip() and not set(line.strip()) <= {"*"}]
    if not any("-SHOT #-" in line.upper() for line in lines):
        return None
    label_line = lines[_labelled(lines, "-DEPENDENT VARIABLE LABEL-")]
    fields = label_line.split(";")[0].split()
    name = fields[0] if fields else ""
    unit = " ".join(fields[1:])
    two_d = any("# OF X PTS" in line.upper() for line in lines)
    if two_d:
        nx_line = _labelled(lines, "# OF X PTS")
        ny_line = _labelled(lines, "# OF Y PTS")
        nx = int(lines[nx_line].split(";")[0].split()[0])
        ny = int(lines[ny_line].split(";")[0].split()[0])
        data = _numbers(lines[ny_line + 1:])
        if data.size != nx + ny + nx * ny:
            raise ValueError(f"UFILE {name}: expected {nx + ny + nx * ny} numbers, found {data.size}")
        rho, time = data[:nx], data[nx:nx + ny]
        values = data[nx + ny:].reshape(ny, nx)  # X varies fastest
    else:
        n_line = _labelled(lines, "# OF PTS")
        n = int(lines[n_line].split(";")[0].split()[0])
        data = _numbers(lines[n_line + 1:])
        if data.size != 2 * n:
            raise ValueError(f"UFILE {name}: expected {2 * n} numbers, found {data.size}")
        rho, time, values = None, data[:n], data[n:]
    values = np.where(np.isclose(values, _REAL_MISSING, rtol=0.0, atol=1e-12), np.nan, values)
    text = "\n".join(
        line for line in comment.splitlines()[1:] if line.strip() and not set(line.strip()) <= {"*"}
    )
    return UFileSignal(name=name, unit=unit, time=time, values=values, rho=rho, comment=text)


def read_ufiles(path: str | os.PathLike[str]) -> dict[str, UFileSignal]:
    """Read a file of concatenated ASCII UFILEs (PR08 ``_1d`` / ``_2d`` files).

    Parameters
    ----------
    path : path-like
        UFILE file [path].

    Returns
    -------
    dict of str to UFileSignal
        Signals keyed by their label, in file order [signal].

    Raises
    ------
    ValueError
        A block's point counts do not match its data, or a label repeats.
    """
    text = Path(path).read_text(errors="replace")
    signals: dict[str, UFileSignal] = {}
    # Each block ends with END-OF-DATA plus a comment; the next one starts at
    # the following SHOT line, so split just before every SHOT line.
    starts = [m.start() for m in re.finditer(r"(?im)^.*;-SHOT #-", text)]
    for begin, end in zip(starts, starts[1:] + [len(text)]):
        signal = _parse_block(text[begin:end])
        if signal is None:
            continue
        if signal.name in signals:
            raise ValueError(f"{path}: signal {signal.name!r} appears twice")
        signals[signal.name] = signal
    return signals


def _typed(value: str):
    value = value.strip()
    if value == "" or value == _STRING_MISSING:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if np.isclose(number, _REAL_MISSING, rtol=0.0, atol=1e-12) or number == _INT_MISSING:
        return np.nan
    return number


def read_pr08_0d(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a PR08 ``_0d.dat`` file, CSV or fixed-width.

    Parameters
    ----------
    path : path-like
        0D file [path].

    Returns
    -------
    pandas.DataFrame
        One row per record (time), source names; numbers as float with missing
        NaN, strings with missing ``None`` [table].
    """
    text = Path(path).read_text(errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and "," in lines[0]:
        rows = list(csv.reader(lines))
        header = [name.strip() for name in rows[0]]
        records = [dict(zip(header, (_typed(v) for v in row))) for row in rows[1:]]
        return pd.DataFrame(records, columns=header)
    # Fixed width: 11-character fields, names first, then values.  A names line
    # holds only identifiers; the first line with a number or a missing-string
    # marker starts the values (a value such as 'D3D' alone looks like a name).
    def fields(line: str) -> list[str]:
        # A blank cell inside a line is a value (missing), not a gap to skip;
        # only the padding after the last field is dropped.
        cells = [line[i:i + 11].strip() for i in range(0, len(line.rstrip()), 11)]
        return cells

    def is_value_line(line: str) -> bool:
        return any(
            token == _STRING_MISSING or _NUMBER.fullmatch(token) for token in fields(line)
        )

    split = next((i for i, line in enumerate(lines) if is_value_line(line)), len(lines))
    names = [token for line in lines[:split] for token in fields(line) if token]
    values = [token for line in lines[split:] for token in fields(line)]
    if not names or len(values) % len(names):
        raise ValueError(f"{path}: {len(names)} names but {len(values)} values")
    records = [
        dict(zip(names, (_typed(v) for v in values[i:i + len(names)])))
        for i in range(0, len(values), len(names))
    ]
    return pd.DataFrame(records, columns=names)


def _file(directory: Path, machine: str, shot: str, kind: str) -> Path:
    return directory / f"pr08_{machine}_{shot}_{kind}.dat"


def read_pr08(directory: str | os.PathLike[str], machine: str, shot) -> Pr08Discharge:
    """Read one PR08 discharge from a local directory.

    Parameters
    ----------
    directory : path-like
        Directory holding ``pr08_<machine>_<shot>_{0d,1d,2d,com}.dat`` [path].
    machine : str
        PR08 machine name, e.g. ``"mast"``, ``"d3d"``, ``"jet"`` [str].
    shot : int or str
        Shot, as in the file names [str].

    Returns
    -------
    Pr08Discharge
        The discharge, source names and units [record].

    Raises
    ------
    FileNotFoundError
        The 2D file is absent (0D, 1D and comments are optional).
    """
    directory = Path(directory)
    machine, shot = str(machine).lower(), str(shot)
    two_d_path = _file(directory, machine, shot, "2d")
    if not two_d_path.exists():
        raise FileNotFoundError(two_d_path)
    files = {kind: sha256_of(p) for kind in PR08_KINDS if (p := _file(directory, machine, shot, kind)).exists()}
    zero_path = _file(directory, machine, shot, "0d")
    one_path = _file(directory, machine, shot, "1d")
    com_path = _file(directory, machine, shot, "com")
    return Pr08Discharge(
        machine=machine,
        shot=shot,
        zero_d=read_pr08_0d(zero_path) if zero_path.exists() else pd.DataFrame(),
        one_d=read_ufiles(one_path) if one_path.exists() else {},
        two_d=read_ufiles(two_d_path),
        comments=com_path.read_text(errors="replace").strip() if com_path.exists() else "",
        files=files,
    )


def fetch_pr08(
    machine: str,
    shot,
    *,
    cache: str | os.PathLike[str] | None = None,
    sha256: dict[str, str] | None = None,
    allow_unpinned: bool = False,
    timeout: float = 120.0,
) -> Pr08Discharge:
    """Download one PR08 discharge (checksum-verified) and read it.

    Parameters
    ----------
    machine : str
        PR08 machine directory, e.g. ``"mast"`` [str].
    shot : int or str
        Shot directory [str].
    cache : path-like or None, optional
        Cache root, default ``None`` for the per-user cache [path].
    sha256 : dict or None, optional
        Hashes by kind (``"0d"``, ``"1d"``, ``"2d"``, ``"com"``); default
        ``None`` uses :data:`PR08_PINNED` [str].
    allow_unpinned : bool, optional
        Accept files with no known hash -- all of them for an unpinned
        discharge, or the kinds a partial ``sha256`` leaves out -- recording
        (not checking) their hashes and marking them ``"unverified"`` in the
        provenance; default ``False`` [bool].
    timeout : float, optional
        Network timeout per file, default 120 [s].

    Returns
    -------
    Pr08Discharge
        The discharge [record].

    Raises
    ------
    KeyError
        A file has no known hash and ``allow_unpinned`` is false.
    vaft.data.public.FetchError
        The upstream was unreachable.
    vaft.data.public.ChecksumError
        A file does not match its pinned hash.
    """
    from ._fetch import _download

    machine, shot = str(machine).lower(), str(shot)
    pins = dict(sha256 if sha256 is not None else PR08_PINNED.get((machine, shot), {}))
    unpinned = [kind for kind in PR08_KINDS if kind not in pins]
    if unpinned and not allow_unpinned:
        raise KeyError(
            f"No hash for PR08 {machine} {shot} files {unpinned}; pass sha256= or allow_unpinned=True"
        )
    directory = cache_dir(cache) / "pr08" / machine / shot
    verification: dict[str, str] = {}
    for kind in PR08_KINDS:
        name = f"pr08_{machine}_{shot}_{kind}.dat"
        url = f"{PR08_BASE_URL}/{machine}/{shot}/{name}"
        if kind in pins:
            fetch(url, sha256=pins[kind], filename=name, cache=directory, timeout=timeout)
            verification[kind] = "pinned"
            continue
        target = directory / name
        if target.exists():
            warnings.warn(f"PR08 {name}: using a cached file that no hash verifies", stacklevel=2)
            verification[kind] = "unverified"
            continue
        try:
            _download(url, target, timeout, sha256=None)
        except FetchError as exc:
            if kind == "2d":
                raise
            warnings.warn(f"PR08 {name} could not be downloaded ({exc}); continuing without it", stacklevel=2)
            continue
        warnings.warn(f"PR08 {name} was downloaded without a known hash", stacklevel=2)
        verification[kind] = "unverified"
    discharge = read_pr08(directory, machine, shot)
    discharge.verification = verification
    return discharge


# ---------------------------------------------------------------- ODS mapping

#: (PR08 2D name, relative path, factor, note).  Each block's grid and time
#: base are its reference signal's; see :func:`_fill_all`.
_CORE_PROFILES = (
    ("TE", "electrons.temperature", 1.0, ""),
    ("NE", "electrons.density", 1.0, ""),
    ("TI", "ion.0.temperature", 1.0, "main ion (ion.0), species from 0D PGASA/PGASZ"),
    ("NM1", "ion.0.density", 1.0, "main ion density"),
    ("ZEFFR", "zeff", 1.0, ""),
    ("CURTOT", "j_tor", 1.0, "toroidal current density, +ve anti-clockwise from above = IMAS phi"),
)
#: Q is handled separately: PR08 gives |q|, IMAS COCOS 11 wants sign(Ip B0) |q|.
_EQUILIBRIUM = (
    ("VOLUME", "profiles_1d.volume", 1.0, ""),
    ("SURF", "profiles_1d.surface", 1.0, ""),
    ("KAPPAR", "profiles_1d.elongation", 1.0, ""),
    ("DELTARU", "profiles_1d.triangularity_upper", 1.0, ""),
    ("DELTARL", "profiles_1d.triangularity_lower", 1.0, ""),
    ("PRES", "profiles_1d.pressure", 1.0, "scalar MHD pressure used in the Grad-Shafranov solution"),
)
#: core_sources: identifier (index, name) -> ((PR08, path, factor, note), ...)
_SOURCES = {
    (2, "nbi"): (
        ("QNBIE", "electrons.energy", 1.0, ""),
        ("QNBII", "total_ion_energy", 1.0, "includes fast-ion thermalisation"),
        ("SNBIE", "electrons.particles", 1.0, ""),
    ),
    (3, "ec"): (("QECHE", "electrons.energy", 1.0, ""), ("QECHI", "total_ion_energy", 1.0, "")),
    (4, "lh"): (("QLHE", "electrons.energy", 1.0, ""), ("QLHI", "total_ion_energy", 1.0, "")),
    (5, "ic"): (("QICRHE", "electrons.energy", 1.0, ""), ("QICRHI", "total_ion_energy", 1.0, "")),
    (6, "fusion"): (("QFUSE", "electrons.energy", 1.0, ""), ("QFUSI", "total_ion_energy", 1.0, "")),
    (7, "ohmic"): (("QOHM", "electrons.energy", 1.0, ""),),
    (11, "collisional_equipartition"): (
        ("QEI", "electrons.energy", -1.0, "PR08 QEI is electrons -> ions; electrons lose it"),
        ("QEI", "total_ion_energy", 1.0, "ions gain QEI"),
    ),
    (200, "radiation"): (
        ("QRAD", "electrons.energy", -1.0, "PR08 QRAD is radiated power (a magnitude); IMAS radiation is a negative source"),
    ),
}
_TRANSPORT = (
    ("CHIE", "electrons.energy.d", 1.0, "power-balance diffusivity; excludes <|grad rho|^2>"),
    ("CHII", "total_ion_energy.d", 1.0, "power-balance diffusivity; excludes <|grad rho|^2>"),
)
_ION_SPECIES = {(1, 1): "H", (2, 1): "D", (3, 1): "T", (3, 2): "He3", (4, 2): "He"}

_UNMAPPED_REASONS = {
    "VROT": "rotation of a species named only in the comment file",
    "VROTXP": "measured rotation on its own grid",
    "BPOL": "no flux-surface-averaged B_pol slot",
    "RMAJOR": "definition of the radial position not in the manual text",
    "RMINOR": "definition of the radial position not in the manual text",
    "GRHO1": "<|grad rho|> with dimensionless rho; IMAS gm7 needs rho_tor in m",
    "GRHO2": "<|grad rho|^2> with dimensionless rho; IMAS gm3 needs rho_tor in m",
    "DELTAR": "mean triangularity; IMAS carries upper and lower only",
    "DNER": "time derivative, no core_profiles slot",
    "DWER": "time derivative, no core_profiles slot",
    "DWIR": "time derivative, no core_profiles slot",
    "QWALLE": "wall-neutral loss: no agreed core_sources identifier",
    "QWALLI": "wall-neutral loss: no agreed core_sources identifier",
    "SWALL": "wall-neutral particle source: no agreed core_sources identifier",
    "SNBII": "ion particle source needs a species; not assumed",
    "CURNBI": "beam-driven toroidal current density; core_sources carries j_parallel only",
    "NFAST1": "fast-ion density: species in 0D NFAST1A/Z, not mapped yet",
    "NFAST2": "fast-ion density: species in 0D NFAST2A/Z, not mapped yet",
    "NM2": "second ion density: species in 0D, not mapped yet",
    "NHYA": "hydrogen isotope ratio profile, no IMAS slot",
    "TORQ": "torque density: source split not stated",
}


@dataclass
class _Outcome:
    """Where one PR08 variable went: slices written out of slices targeted."""

    target: str
    written: int
    total: int
    reason: str = ""


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.allclose(a, b, rtol=0.0, atol=1e-6)


def _provenance(ods, ids: str, entries: list[tuple[str, str]], discharge: Pr08Discharge, extra: str = "") -> None:
    ods[f"{ids}.ids_properties.homogeneous_time"] = 1
    ods[f"{ids}.ids_properties.comment"] = (
        f"Mapped by vaft.data.public.itpa_profile from ITPA PR08 {discharge.machine} "
        f"{discharge.shot}; {PR08_REFERENCE}.  No interpolation: only signals on the "
        "reference radial grid and time base of each block are mapped." + extra
    )
    for i, (path, source) in enumerate(entries):
        ods[f"{ids}.ids_properties.provenance.node.{i}.path"] = path
        ods[f"{ids}.ids_properties.provenance.node.{i}.sources"] = [source]


def _source_label(discharge: Pr08Discharge, signal: UFileSignal) -> str:
    unit = signal.unit or "unit not given in the file"
    return f"PR08 {discharge.machine}/{discharge.shot} {signal.name} [{unit}]"


def _zero_d_value(discharge: Pr08Discharge, name: str):
    if discharge.zero_d.empty or name not in discharge.zero_d.columns:
        return None
    values = discharge.zero_d[name].dropna()
    unique = values.unique()
    return unique[0] if len(unique) == 1 else None


def _at_times(signal: UFileSignal | None, times: np.ndarray) -> np.ndarray:
    """1D values at exactly matching times; NaN where there is no sample."""
    out = np.full(len(times), np.nan)
    if signal is None:
        return out
    for index, time in enumerate(times):
        match = np.flatnonzero(np.isclose(signal.time, time, rtol=0.0, atol=1e-6))
        if match.size == 1:
            out[index] = signal.values[match[0]]
    return out


def _sign_check(discharge: Pr08Discharge, name: str, values: np.ndarray) -> str:
    """Empty if every finite 1D sample and the 0D value(s) share one sign."""
    signs = set(np.sign(values[np.isfinite(values) & (values != 0.0)]).tolist())
    if not discharge.zero_d.empty and name in discharge.zero_d.columns:
        zero = pd.to_numeric(discharge.zero_d[name], errors="coerce").to_numpy(float)
        zero_signs = set(np.sign(zero[np.isfinite(zero) & (zero != 0.0)]).tolist())
        if signs and zero_signs and signs != zero_signs:
            return f"{name} sign differs between the 0D ({sorted(zero_signs)}) and 1D ({sorted(signs)}) files"
        signs |= zero_signs
    if len(signs) > 1:
        return f"{name} changes sign within the release"
    return ""


def _map(discharge: Pr08Discharge, ods) -> dict[str, _Outcome]:
    """Fill ``ods``; return what happened to each PR08 variable that was targeted."""
    outcomes: dict[str, _Outcome] = {}
    _fill_all(discharge, ods, discharge.two_d, outcomes)
    return outcomes


def pr08_to_omas(discharge: Pr08Discharge, ods=None):
    """Map a PR08 discharge into an ODS, without interpolation or invented data.

    Parameters
    ----------
    discharge : Pr08Discharge
        Output of :func:`read_pr08` or :func:`fetch_pr08` [record].
    ods : omas.ODS or None, optional
        ODS to fill, default ``None`` creates one [ODS].

    Returns
    -------
    omas.ODS
        ``dataset_description``, ``core_profiles``, ``equilibrium`` (profiles
        subset), ``core_sources`` and ``core_transport`` as far as the discharge
        carries them.  Every IDS records its source variables and units under
        ``ids_properties.provenance`` [ODS].

    Notes
    -----
    * Times are the 2D file's; each block keeps only its reference signal's
      grid and times, and 1D scalars are attached only at exactly matching
      times.
    * ``rho_tor_norm`` is PR08's ``sqrt(Phi/Phi_a)`` unchanged.
    * Error bars (``<NAME>EB`` on the same grid) go to ``*_error_upper``.
    * Signs: ``QRAD`` enters the ``radiation`` source negative; ``QEI``
      leaves the electrons and enters the ions; ``CURTOT`` is the toroidal
      current density (``j_tor``), anti-clockwise positive as IMAS phi.
    * PR08 ``Q`` is ``|q|``.  It is written as ``sign(Ip) sign(B0) |q|``
      (COCOS 11) only where ``Ip`` and ``B0`` are known at that time and their
      signs agree across the 0D and 1D files; otherwise q, ``ip`` and ``b0``
      are left out and the coverage table says why.
    * The equilibrium carries profiles only -- no psi, no boundary, no 2D
      field; it is not a reconstruction.
    """
    import omas

    if ods is None:
        ods = omas.ODS()
    _map(discharge, ods)
    return ods


def _fill_all(discharge: Pr08Discharge, ods, two_d, outcomes: dict[str, _Outcome]) -> None:
    ods["dataset_description.data_entry.machine"] = discharge.machine.upper()
    try:
        ods["dataset_description.data_entry.pulse"] = int(discharge.shot)
    except ValueError:
        pass
    ods["dataset_description.ids_properties.homogeneous_time"] = 2
    ods["dataset_description.ids_properties.comment"] = (
        f"ITPA PR08 public release, {discharge.machine} {discharge.shot}. {PR08_REFERENCE}. "
        "Profiles are the contributors' fitted/interpretive (e.g. TRANSP) analysis, "
        "not raw measurements.  Comment file follows.\n" + discharge.comments
    )

    def fill(block, prefix_of, reference, entries, ids):
        """Write entries on the reference signal's grid and times; return provenance."""
        provenance = []
        n = len(reference.time)
        for t_index, time in enumerate(reference.time):
            prefix = prefix_of(t_index)
            ods[f"{prefix}.{block}"] = reference.rho
            ods[f"{prefix}.time"] = float(time)
        for name, path, factor, _note in entries:
            signal = two_d.get(name)
            if signal is None:
                continue
            if not (_same(signal.rho, reference.rho) and _same(signal.time, reference.time)):
                outcomes.setdefault(name, _Outcome(
                    f"{ids}:{path}", 0, n,
                    f"on a different radial grid or time base than {reference.name}; not interpolated",
                ))
                continue
            for t_index in range(n):
                ods[f"{prefix_of(t_index)}.{path}"] = factor * signal.values[t_index]
            outcomes[name] = _Outcome(f"{ids}:{path}", n, n)
            provenance.append((f"{prefix_of(':')}.{path}", _source_label(discharge, signal)))
            error = two_d.get(f"{name}EB")
            if error is not None:
                if _same(error.rho, reference.rho) and _same(error.time, reference.time):
                    for t_index in range(n):
                        ods[f"{prefix_of(t_index)}.{path}_error_upper"] = np.abs(error.values[t_index])
                    outcomes[error.name] = _Outcome(f"{ids}:{path}_error_upper", n, n)
                    provenance.append((f"{prefix_of(':')}.{path}_error_upper", _source_label(discharge, error)))
                else:
                    outcomes[error.name] = _Outcome(
                        f"{ids}:{path}_error_upper", 0, n, "error bar on a different grid than its value; not interpolated"
                    )
        return provenance

    # core_profiles
    reference = two_d.get("TE") or two_d.get("NE")
    if reference is not None:
        ods["core_profiles.time"] = reference.time
        provenance = fill(
            "grid.rho_tor_norm", lambda t: f"core_profiles.profiles_1d.{t}", reference, _CORE_PROFILES, "core_profiles"
        )
        extra = ""
        mass, charge = _zero_d_value(discharge, "PGASA"), _zero_d_value(discharge, "PGASZ")
        if (outcomes.get("TI") and outcomes["TI"].written) or (outcomes.get("NM1") and outcomes["NM1"].written):
            known = mass is not None and charge is not None and np.isfinite(mass) and np.isfinite(charge)
            integer = known and abs(mass - round(mass)) < 1e-6
            label = _ION_SPECIES.get((int(round(mass)), int(round(charge)))) if integer else None
            for t_index in range(len(reference.time)):
                prefix = f"core_profiles.profiles_1d.{t_index}.ion.0"
                if known:
                    ods[f"{prefix}.element.0.a"] = float(mass)
                    ods[f"{prefix}.element.0.z_n"] = float(charge)
                    ods[f"{prefix}.z_ion"] = float(charge)
                if label:
                    ods[f"{prefix}.label"] = label
            if known and not integer:
                extra = (
                    f"  ion.0 is a lumped main-ion pseudo-element: A = PGASA = {mass:g} is an "
                    "effective mass of a fuel mixture, so no species label is given."
                )
        _provenance(ods, "core_profiles", provenance, discharge, extra)

    # equilibrium profiles subset, on Q's grid
    if "Q" in two_d:
        q = two_d["Q"]
        n = len(q.time)
        ods["equilibrium.time"] = q.time
        provenance = fill(
            "profiles_1d.rho_tor_norm", lambda t: f"equilibrium.time_slice.{t}", q, _EQUILIBRIUM, "equilibrium"
        )
        ip = _at_times(discharge.one_d.get("IP"), q.time)
        bt = _at_times(discharge.one_d.get("BT"), q.time)
        problems = [p for p in (_sign_check(discharge, "IP", ip), _sign_check(discharge, "BT", bt)) if p]
        if problems:
            reason = "; ".join(problems) + "; current direction ambiguous, so sign-bearing quantities are left out"
            outcomes["Q"] = _Outcome("equilibrium:profiles_1d.q", 0, n, reason)
            outcomes["IP"] = _Outcome("equilibrium:global_quantities.ip", 0, n, reason)
            outcomes["BT"] = _Outcome("equilibrium:vacuum_toroidal_field.b0", 0, n, reason)
        else:
            written = 0
            for t_index in range(n):
                if np.isfinite(ip[t_index]):
                    ods[f"equilibrium.time_slice.{t_index}.global_quantities.ip"] = float(ip[t_index])
                if np.isfinite(ip[t_index]) and np.isfinite(bt[t_index]):
                    magnitude = np.abs(q.values[t_index]) if np.nanmin(q.values) >= 0.0 else q.values[t_index]
                    ods[f"equilibrium.time_slice.{t_index}.profiles_1d.q"] = (
                        np.sign(ip[t_index]) * np.sign(bt[t_index]) * magnitude
                    )
                    written += 1
            missing = "no IP/BT sample at some equilibrium times"
            outcomes["Q"] = _Outcome("equilibrium:profiles_1d.q", written, n, "" if written == n else missing)
            outcomes["IP"] = _Outcome(
                "equilibrium:global_quantities.ip", int(np.isfinite(ip).sum()), n,
                "" if np.isfinite(ip).all() else missing,
            )
            if written:
                provenance.append((
                    "equilibrium.time_slice.:.profiles_1d.q",
                    _source_label(discharge, q) + " as |q|, signed sign(IP) sign(BT) (COCOS 11)",
                ))
            if np.isfinite(ip).any():
                provenance.append(("equilibrium.time_slice.:.global_quantities.ip",
                                   _source_label(discharge, discharge.one_d["IP"])))
            rgeo = _zero_d_value(discharge, "RGEO")
            if np.isfinite(bt).all() and rgeo is not None and np.isfinite(rgeo):
                # PR08 0D follows the H-mode database conventions: BT at RGEO.
                ods["equilibrium.vacuum_toroidal_field.b0"] = bt
                ods["equilibrium.vacuum_toroidal_field.r0"] = float(rgeo)
                outcomes["BT"] = _Outcome("equilibrium:vacuum_toroidal_field.b0", n, n)
                provenance.append(("equilibrium.vacuum_toroidal_field.b0",
                                   _source_label(discharge, discharge.one_d["BT"]) + " at 0D RGEO"))
            elif "BT" in discharge.one_d:
                outcomes["BT"] = _Outcome(
                    "equilibrium:vacuum_toroidal_field.b0", 0, n,
                    "b0 needs a BT sample at every equilibrium time and a single 0D RGEO",
                )
        _provenance(ods, "equilibrium", provenance, discharge)

    # core_sources: one time base for the IDS
    source_index = 0
    source_provenance = []
    source_time = None
    for (index, label), entries in _SOURCES.items():
        present = [entry for entry in entries if entry[0] in two_d]
        if not present:
            continue
        reference = two_d[present[0][0]]
        if source_time is None:
            source_time = reference.time
        elif not _same(reference.time, source_time):
            for name, path, _factor, _note in present:
                outcomes[name] = _Outcome(
                    f"core_sources:{path}", 0, len(reference.time),
                    "source on a different time base than core_sources.time; not interpolated",
                )
            continue
        root = f"core_sources.source.{source_index}"
        ods[f"{root}.identifier.index"] = index
        ods[f"{root}.identifier.name"] = label
        source_provenance += fill(
            "grid.rho_tor_norm", lambda t, r=root: f"{r}.profiles_1d.{t}", reference, present, "core_sources"
        )
        source_index += 1
    if source_index:
        ods["core_sources.time"] = source_time
        _provenance(ods, "core_sources", source_provenance, discharge)

    # core_transport
    if "CHIE" in two_d or "CHII" in two_d:
        reference = two_d["CHIE"] if "CHIE" in two_d else two_d["CHII"]
        root = "core_transport.model.0"
        ods[f"{root}.identifier.index"] = 2
        ods[f"{root}.identifier.name"] = "transport_solver"
        ods[f"{root}.identifier.description"] = (
            "Interpretive power-balance diffusivities from the contributor's analysis "
            "(e.g. TRANSP), as released in ITPA PR08"
        )
        ods["core_transport.time"] = reference.time
        provenance = fill(
            "grid_d.rho_tor_norm", lambda t: f"{root}.profiles_1d.{t}", reference, _TRANSPORT, "core_transport"
        )
        _provenance(ods, "core_transport", provenance, discharge)

    ods["dataset_description.ids_properties.provenance.node.0.path"] = "dataset_description"
    ods["dataset_description.ids_properties.provenance.node.0.sources"] = [
        f"{PR08_BASE_URL}/{discharge.machine}/{discharge.shot}/ sha256 "
        + ", ".join(
            f"{kind}={digest[:12]} ({discharge.verification.get(kind, 'read locally')})"
            for kind, digest in sorted(discharge.files.items())
        )
    ]


def pr08_mapping_coverage(discharge: Pr08Discharge) -> pd.DataFrame:
    """Which PR08 variables reached the ODS, where, and why the others did not.

    Parameters
    ----------
    discharge : Pr08Discharge
        The discharge [record].

    Returns
    -------
    pandas.DataFrame
        One row per 2D and 1D variable: ``kind``, ``variable``, ``unit``,
        ``n_rho``, ``status`` (``mapped``, ``partial`` or ``unmapped``),
        ``target``, ``slices`` (``"written/targeted"``) and ``reason`` [table].
    """
    import omas

    outcomes = _map(discharge, omas.ODS())
    rows = []
    for kind, signals in (("2d", discharge.two_d), ("1d", discharge.one_d)):
        for name, signal in signals.items():
            outcome = outcomes.get(name)
            if outcome is not None:
                if outcome.written == outcome.total and outcome.total:
                    status = "mapped"
                elif outcome.written:
                    status = "partial"
                else:
                    status = "unmapped"
                target, slices, reason = outcome.target, f"{outcome.written}/{outcome.total}", outcome.reason
            else:
                target, slices, status = "", "", "unmapped"
                if name.endswith("XP") or name.endswith("XPEB"):
                    reason = "measured profile on its own grid; not interpolated"
                elif name.endswith("EB"):
                    reason = "error bar of an unmapped variable"
                else:
                    reason = _UNMAPPED_REASONS.get(
                        name, "no mapping defined" if kind == "2d" else "global trace; see the 0D file"
                    )
            rows.append({
                "kind": kind,
                "variable": name,
                "unit": signal.unit,
                "n_rho": None if signal.rho is None else len(signal.rho),
                "status": status,
                "target": target,
                "slices": slices,
                "reason": reason,
            })
    return pd.DataFrame(rows, columns=["kind", "variable", "unit", "n_rho", "status", "target", "slices", "reason"])
