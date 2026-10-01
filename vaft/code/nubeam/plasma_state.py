"""Build NUBEAM's input Plasma State with VAFT's own generator.

NUBEAM reads its plasma -- equilibrium, kinetic profiles, species, beams --
from a Plasma State netCDF file. The public NTCC distribution ships the
Plasma State library but no program that creates a state from scratch, so
VAFT carries one: ``install/nubeam/plasma_state/vaft_plasma_state.f90``,
compiled by the installers against the user's own NTCC build and installed as
``$NUBEAMHOME/bin/vaft_plasma_state``.

A pure-Python writer is not an option. On read, NUBEAM rebuilds the
equilibrium in xplasma from the state's Fourier moments, surface geometry and
metric averages, and checks them against each other; only the NTCC library
computes those consistently. The division of labour is therefore:

* this module turns VAFT data -- a G-EQDSK or an equilibrium IDS, profiles
  from arrays or ``core_profiles`` -- into one ``&vaft_plasma_state``
  namelist, validating everything it can in Python;
* the Fortran program loads the G-EQDSK through ``ps_update_equilibrium``,
  maps the profiles onto the state grid and writes the file.

Units in this module are SI except temperatures and beam energy (keV), which
is what the Plasma State itself stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Optional, Sequence

import numpy as np

from .inputs import NUBEAMInputError

#: Name of the namelist file the generator reads from its working directory.
PLASMA_STATE_NAMELIST = "vaft_plasma_state.nml"

#: Coordinates the generator accepts for the profile abscissa.
#:
#: ``rho_tor`` is sqrt(normalised toroidal flux), the Plasma State's own
#: coordinate. ``sqrt_psi_n`` is sqrt(normalised poloidal flux); the generator
#: maps it onto rho_tor through the equilibrium it has just loaded, so profiles
#: and geometry share one flux-surface labelling. Prefer it for IDS data:
#: packaged VAFT equilibria store a sqrt(psi_N) proxy under ``rho_tor_norm``.
PROFILE_COORDINATES = ("rho_tor", "sqrt_psi_n")

#: Fixed array bounds compiled into the generator.
GENERATOR_MAX_POINTS = 2001
GENERATOR_MAX_IONS = 20
GENERATOR_MAX_BEAMS = 32
GENERATOR_MAX_GAS_SOURCES = 16

#: Longest path the generator's ``character(len=512)`` fields hold.
GENERATOR_PATH_CHARS = 512


class PlasmaStateInputError(NUBEAMInputError):
    """Raised when a Plasma State cannot be specified from the given data."""


def _array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise PlasmaStateInputError(f"{name} is not finite everywhere")
    return array


@dataclass(frozen=True)
class PlasmaStateProfiles:
    """Kinetic profiles tabulated on one radial abscissa.

    ``ion_densities`` holds one profile per thermal ion species, in the order
    the sconfig namelist declares them (``iZatom_S`` / ``iAMU_S``). Electrons
    are ``ne``. One ion temperature applies to every ion species, as in the
    Plasma State.
    """

    x: np.ndarray
    ne: np.ndarray
    te_kev: np.ndarray
    ti_kev: np.ndarray
    zeff: np.ndarray
    ion_densities: tuple[np.ndarray, ...]
    coordinate: str = "rho_tor"
    omegat: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.coordinate not in PROFILE_COORDINATES:
            raise PlasmaStateInputError(
                f"coordinate must be one of {PROFILE_COORDINATES}, got {self.coordinate!r}"
            )
        x = _array("x", self.x)
        object.__setattr__(self, "x", x)
        if not 2 <= x.size <= GENERATOR_MAX_POINTS:
            raise PlasmaStateInputError(
                f"profiles need 2..{GENERATOR_MAX_POINTS} points, got {x.size}"
            )
        if abs(x[0]) > 1e-9 or abs(x[-1] - 1.0) > 1e-9:
            raise PlasmaStateInputError(
                f"x must run from 0 to 1 (got {x[0]:g} .. {x[-1]:g}); "
                "VAFT does not extrapolate profiles to the axis or the edge"
            )
        if np.any(np.diff(x) <= 0.0):
            raise PlasmaStateInputError("x must increase strictly")

        def profile(name: str, values: Any) -> np.ndarray:
            array = _array(name, values)
            if array.shape != x.shape:
                raise PlasmaStateInputError(
                    f"{name} has {array.size} points, x has {x.size}"
                )
            return array

        for name in ("ne", "te_kev", "ti_kev", "zeff"):
            object.__setattr__(self, name, profile(name, getattr(self, name)))
        for name in ("ne", "te_kev", "ti_kev"):
            if np.any(getattr(self, name) <= 0.0):
                raise PlasmaStateInputError(f"{name} must be positive everywhere")
        if np.any(self.zeff < 1.0):
            raise PlasmaStateInputError(f"zeff falls to {self.zeff.min():g} (< 1)")

        ions = tuple(
            profile(f"ion_densities[{k}]", values)
            for k, values in enumerate(self.ion_densities)
        )
        if not 1 <= len(ions) <= GENERATOR_MAX_IONS:
            raise PlasmaStateInputError(
                f"need 1..{GENERATOR_MAX_IONS} ion density profiles, got {len(ions)}"
            )
        if any(np.any(ion < 0.0) for ion in ions):
            raise PlasmaStateInputError("ion densities must be >= 0")
        object.__setattr__(self, "ion_densities", ions)
        if self.omegat is not None:
            object.__setattr__(self, "omegat", profile("omegat", self.omegat))


@dataclass(frozen=True)
class PlasmaStateSpec:
    """Everything the generator needs for one Plasma State.

    ``mdescr`` and ``sconfig`` are NTCC's own machine-description and
    shot-configuration namelists. They fix the beam geometry, the species
    lists and the neutral gas sources; the per-beam operating point and the
    neutral source values here must match their counts.
    """

    mdescr: Path
    sconfig: Path
    gfile: Path
    profiles: PlasmaStateProfiles
    beam_power_w: tuple[float, ...]
    beam_energy_kev: tuple[float, ...]
    t0: float = 0.0
    t1: float = 0.0
    runid: str = "NUBEAM"
    output_name: str = "NUBEAM.cdf"
    #: Hydrogenic neutral density beyond the plasma boundary [m^-3].
    edge_neutral_density: float = 0.0
    #: Mean energy of each neutral gas source [keV], one per sconfig source.
    neutral_energy_kev: tuple[float, ...] = ()
    #: Whether each neutral gas source is recycling (1) or not (0).
    recycling: tuple[int, ...] = ()
    #: State radial grid size (zone boundaries) and poloidal angle points.
    nrho: int = 101
    nth_eq: int = 101
    #: ``ps_update_equilibrium`` boundary-curvature limit; 0.08 is NTCC's
    #: documented default. It picks a boundary a safe step inside an X-point.
    bdy_crat: float = 0.08
    #: Fourier moments kept for the flux-surface shapes (1..64).
    nmom: int = 64
    #: Take the limiter from the G-EQDSK rather than the mdescr namelist.
    limiter_from_geqdsk: bool = True
    label: str = "vaft_plasma_state"

    def __post_init__(self) -> None:
        if len(self.beam_power_w) != len(self.beam_energy_kev):
            raise PlasmaStateInputError(
                "beam_power_w and beam_energy_kev must have one entry per beam"
            )
        if len(self.beam_power_w) > GENERATOR_MAX_BEAMS:
            raise PlasmaStateInputError(f"at most {GENERATOR_MAX_BEAMS} beams")
        if any(p < 0.0 for p in self.beam_power_w) or any(
            e < 0.0 for e in self.beam_energy_kev
        ):
            raise PlasmaStateInputError("beam power and energy must be >= 0")
        if self.recycling and len(self.recycling) != len(self.neutral_energy_kev):
            raise PlasmaStateInputError(
                "recycling needs one flag per neutral gas source"
            )
        if len(self.neutral_energy_kev) > GENERATOR_MAX_GAS_SOURCES:
            raise PlasmaStateInputError(
                f"at most {GENERATOR_MAX_GAS_SOURCES} neutral gas sources"
            )
        if self.t1 < self.t0:
            raise PlasmaStateInputError("t1 must not precede t0")
        if not 3 <= self.nrho <= GENERATOR_MAX_POINTS:
            raise PlasmaStateInputError(f"nrho must be in 3..{GENERATOR_MAX_POINTS}")
        if not 1 <= self.nmom <= 64:
            raise PlasmaStateInputError("nmom must be in 1..64")
        if not 0.01 <= self.bdy_crat <= 0.15:
            # ps_update_equilibrium silently clamps to this range.
            raise PlasmaStateInputError("bdy_crat must be in 0.01..0.15")


# --------------------------------------------------------------------------- #
# the namelist
# --------------------------------------------------------------------------- #


def _fortran_string(value: str, name: str) -> str:
    if "'" in value or "\n" in value:
        raise PlasmaStateInputError(f"{name} cannot contain quotes or newlines: {value!r}")
    return f"'{value}'"


def _fortran_path(path: Path, workdir: Path, name: str) -> str:
    path = Path(path)
    try:
        text = str(path.resolve().relative_to(workdir.resolve()))
    except ValueError:
        text = str(path.resolve())
    if len(text) > GENERATOR_PATH_CHARS:
        raise PlasmaStateInputError(
            f"{name} path is {len(text)} characters; the generator holds {GENERATOR_PATH_CHARS}"
        )
    return _fortran_string(text, name)


def _reals(values: Sequence[float]) -> str:
    return ", ".join(repr(float(v)) for v in values)


def render_plasma_state_namelist(spec: PlasmaStateSpec, workdir: str | Path) -> str:
    """The ``&vaft_plasma_state`` namelist for *spec*, run from *workdir*."""
    workdir = Path(workdir)
    profiles = spec.profiles
    lines = [
        "&vaft_plasma_state",
        f" mdescr_file = {_fortran_path(spec.mdescr, workdir, 'mdescr')}",
        f" sconfig_file = {_fortran_path(spec.sconfig, workdir, 'sconfig')}",
        f" geqdsk_file = {_fortran_path(spec.gfile, workdir, 'gfile')}",
        f" output_file = {_fortran_string(spec.output_name, 'output_name')}",
        f" runid = {_fortran_string(spec.runid, 'runid')}",
        f" global_label = {_fortran_string(spec.label, 'label')}",
        f" t0 = {float(spec.t0)!r}, t1 = {float(spec.t1)!r}",
        f" bdy_crat = {float(spec.bdy_crat)!r}, nmom = {int(spec.nmom)}",
        f" limiter_from_geqdsk = {'.true.' if spec.limiter_from_geqdsk else '.false.'}",
        f" nrho = {int(spec.nrho)}, nth_eq = {int(spec.nth_eq)}",
        f" x_coordinate = {_fortran_string(profiles.coordinate, 'coordinate')}",
        f" nx = {profiles.x.size}",
        f" x = {_reals(profiles.x)}",
        f" ne = {_reals(profiles.ne)}",
        f" te = {_reals(profiles.te_kev)}",
        f" ti = {_reals(profiles.ti_kev)}",
        f" zeff = {_reals(profiles.zeff)}",
        f" nion = {len(profiles.ion_densities)}",
    ]
    for k, density in enumerate(profiles.ion_densities, start=1):
        lines.append(f" ni(1:{density.size},{k}) = {_reals(density)}")
    if profiles.omegat is not None:
        lines.append(f" omegat = {_reals(profiles.omegat)}")
    lines.append(f" nbeam = {len(spec.beam_power_w)}")
    if spec.beam_power_w:
        lines.append(f" power_nbi = {_reals(spec.beam_power_w)}")
        lines.append(f" kvolt_nbi = {_reals(spec.beam_energy_kev)}")
    lines.append(f" dn0out = {float(spec.edge_neutral_density)!r}")
    lines.append(f" ngas = {len(spec.neutral_energy_kev)}")
    if spec.neutral_energy_kev:
        lines.append(f" e0_av = {_reals(spec.neutral_energy_kev)}")
        flags = spec.recycling or (1,) * len(spec.neutral_energy_kev)
        lines.append(" is_recycling = " + ", ".join(str(int(f)) for f in flags))
    lines.append("/")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# NTCC namelist facts the spec has to agree with
# --------------------------------------------------------------------------- #


_ASSIGNMENT = re.compile(r"\b(?P<key>[A-Za-z_]\w*)\s*(?:\((?P<index>\d+)\))?\s*=")
_VALUE_TOKEN = re.compile(r"'[^']*'|\"[^\"]*\"|[^,\s]+")
_REPEAT = re.compile(r"^(?P<count>\d+)\*(?P<value>.*)$")


def _namelist_assignments(text: str, key: str) -> list[tuple[int, str]]:
    """``(line number, value)`` for every value assigned to ``key`` / ``key(i)``.

    Reads the forms a Fortran namelist accepts on one record: values
    separated by commas or blanks, ``r*value`` repeat counts and several
    ``name = ...`` groups per line.  The list is in array order; an index
    never assigned reads as ``""``.
    """
    values: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("!", 1)[0]
        assignments = list(_ASSIGNMENT.finditer(line))
        for n, match in enumerate(assignments):
            if match.group("key").lower() != key.lower():
                continue
            stop = assignments[n + 1].start() if n + 1 < len(assignments) else len(line)
            items: list[str] = []
            for token in _VALUE_TOKEN.findall(line[match.end():stop]):
                repeat = _REPEAT.match(token)
                if repeat and token[0] not in "'\"":
                    items.extend([repeat.group("value")] * int(repeat.group("count")))
                else:
                    items.append(token)
            start = int(match.group("index") or 1)
            while len(values) < start - 1 + len(items):
                values.append((number, ""))
            for offset, item in enumerate(items):
                values[start - 1 + offset] = (number, item)
    return values


def _namelist_entries(text: str, key: str) -> list[str]:
    """Values assigned to ``key`` / ``key(i)`` in an NTCC namelist, in order."""
    return [value for _, value in _namelist_assignments(text, key)]


def _namelist_integers(path: Path, text: str, key: str) -> list[int]:
    """The integer values of ``key``; an unreadable token names where it is."""
    integers: list[int] = []
    for line, value in _namelist_assignments(text, key):
        if not value:
            continue
        try:
            integers.append(int(float(value)))
        except ValueError:
            raise PlasmaStateInputError(
                f"{path}:{line}: cannot read {key} value {value!r} as an integer"
            ) from None
    return integers


@dataclass(frozen=True)
class ShotConfiguration:
    """The parts of an sconfig namelist the profiles must line up with."""

    ion_charge_numbers: tuple[int, ...]
    ion_mass_numbers: tuple[int, ...]
    gas_sources: tuple[str, ...]


def read_shot_configuration(path: str | Path) -> ShotConfiguration:
    """Thermal ion species and neutral gas sources declared by an sconfig file."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    atoms = _namelist_integers(path, text, "iZatom_S")
    masses = _namelist_integers(path, text, "iAMU_S")
    if not atoms or len(atoms) != len(masses):
        raise PlasmaStateInputError(
            f"{path}: iZatom_S and iAMU_S must declare the same, non-empty species list"
        )
    gases = tuple(v.strip("'\" ") for v in _namelist_entries(text, "gs_name") if v)
    return ShotConfiguration(tuple(atoms), tuple(masses), gases)


def count_beams(path: str | Path) -> int:
    """Number of beam sources an mdescr namelist declares (``nbi_src_name``)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return len([v for v in _namelist_entries(text, "nbi_src_name") if v])


def ion_densities_from_zeff(
    ne: Any, zeff: Any, charges: Sequence[int]
) -> tuple[np.ndarray, ...]:
    """Main-ion and impurity densities from ne and Zeff, for two species.

    Solves quasi-neutrality and the Zeff definition for a main ion of charge
    Z1 and one impurity of charge Z2, both fully stripped::

        n1 Z1 + n2 Z2 = ne,    n1 Z1^2 + n2 Z2^2 = Zeff ne

    Refuses a Zeff the pair cannot produce rather than clipping a density.
    """
    if len(charges) != 2:
        raise PlasmaStateInputError(
            f"ion densities can be derived from Zeff for exactly two ion species; "
            f"sconfig declares {len(charges)}. Pass ion_densities explicitly"
        )
    z1, z2 = (float(z) for z in charges)
    if z1 == z2:
        raise PlasmaStateInputError("the two ion species have the same charge")
    ne = np.asarray(ne, dtype=float)
    zeff = np.asarray(zeff, dtype=float)
    n2 = ne * (zeff - z1) / (z2 * (z2 - z1))
    n1 = (ne - z2 * n2) / z1
    if np.any(n1 < 0.0) or np.any(n2 < 0.0):
        raise PlasmaStateInputError(
            f"Zeff in [{zeff.min():g}, {zeff.max():g}] is outside what charges "
            f"{int(z1)} and {int(z2)} can produce"
        )
    return (n1, n2)


def check_spec_against_namelists(spec: PlasmaStateSpec) -> None:
    """Refuse a spec whose counts disagree with its mdescr / sconfig files.

    The generator checks the same and stops, but only after loading the
    equilibrium; doing it here gives the message before anything runs.
    """
    config = read_shot_configuration(spec.sconfig)
    ions = len(spec.profiles.ion_densities)
    if ions != len(config.ion_charge_numbers):
        raise PlasmaStateInputError(
            f"{spec.sconfig.name} declares {len(config.ion_charge_numbers)} thermal ion "
            f"species (Z = {list(config.ion_charge_numbers)}); {ions} ion density "
            "profiles were given"
        )
    beams = count_beams(spec.mdescr)
    if beams != len(spec.beam_power_w):
        raise PlasmaStateInputError(
            f"{spec.mdescr.name} declares {beams} beam source(s); "
            f"{len(spec.beam_power_w)} beam power(s) were given"
        )
    if len(config.gas_sources) != len(spec.neutral_energy_kev):
        raise PlasmaStateInputError(
            f"{spec.sconfig.name} declares {len(config.gas_sources)} neutral gas "
            f"source(s); {len(spec.neutral_energy_kev)} neutral energies were given"
        )


# --------------------------------------------------------------------------- #
# legacy case directories
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LegacyProfiles:
    """Contents of a legacy ``profiles`` file, as far as VAFT reads it."""

    profiles: PlasmaStateProfiles
    beam_power_w: tuple[float, ...]
    beam_energy_kev: tuple[float, ...]
    edge_neutral_density: float
    neutral_energy_kev: tuple[float, ...]


def read_legacy_profiles(path: str | Path) -> LegacyProfiles:
    """Read the ``profiles`` file of a legacy NUBEAM case directory.

    The layout is the one the VEST case in ``vaft/data/nubeam/vest_case``
    uses. It was established from that file and the Plasma State produced from
    it, not from a specification, so every block VAFT does not map must be
    zero -- a non-zero value there is refused rather than silently dropped::

        n                      grid points, rho_tor = 0 .. 1 uniform
        3 lines                anomalous-transport header (not mapped)
        n lines                (must be zero)
        nion                   number of thermal ion species
        (nion+1) x n lines     "density temperature" per species; electrons
                               first, then ions in sconfig order
        3 x n lines            (must be zero)
        n lines                Zeff
        nbeam, then nbeam lines "power[W] energy[keV]"
        dn0out                 edge neutral density
        ngas, then ngas lines  neutral source energy [keV]
    """
    path = Path(path)
    tokens = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
    tokens = [t for t in tokens if t]
    cursor = 0

    def take(width: int = 1) -> list[float]:
        nonlocal cursor
        if cursor >= len(tokens):
            raise PlasmaStateInputError(f"{path}: ends early (line {cursor + 1})")
        row = tokens[cursor]
        cursor += 1
        if len(row) < width:
            raise PlasmaStateInputError(
                f"{path}: line {cursor} has {len(row)} values, expected {width}"
            )
        return [float(value) for value in row[:width]]

    def block(n: int, width: int = 1) -> np.ndarray:
        return np.array([take(width) for _ in range(n)])

    def zeros(n: int, what: str) -> None:
        values = block(n)
        if np.any(values != 0.0):
            raise PlasmaStateInputError(
                f"{path}: {what} is non-zero; VAFT does not map that block, so it "
                "refuses the file rather than drop the data"
            )

    n = int(take()[0])
    take()  # anomalous-transport header: count,
    take()  #   locations,
    take()  #   and flags
    zeros(n, "the first profile block")
    nion = int(take()[0])
    species = [block(n, 2) for _ in range(nion + 1)]
    for k in range(3):
        zeros(n, f"profile block {k + 2} after the species")
    zeff = block(n)[:, 0]
    nbeam = int(take()[0])
    beams = [take(2) for _ in range(nbeam)]
    edge = take()[0]
    ngas = int(take()[0])
    gases = [take()[0] for _ in range(ngas)]

    electrons, ions = species[0], species[1:]
    ion_temperatures = {tuple(ion[:, 1]) for ion in ions}
    if len(ion_temperatures) > 1:
        raise PlasmaStateInputError(
            f"{path}: ion species carry different temperatures; the Plasma State "
            "has one Ti"
        )
    profiles = PlasmaStateProfiles(
        x=np.linspace(0.0, 1.0, n),
        ne=electrons[:, 0],
        te_kev=electrons[:, 1],
        ti_kev=ions[0][:, 1],
        zeff=zeff,
        ion_densities=tuple(ion[:, 0] for ion in ions),
        coordinate="rho_tor",
    )
    return LegacyProfiles(
        profiles=profiles,
        beam_power_w=tuple(b[0] for b in beams),
        beam_energy_kev=tuple(b[1] for b in beams),
        edge_neutral_density=float(edge),
        neutral_energy_kev=tuple(gases),
    )


def _inputf_value(lines: list[str], number: int, path: Path) -> str:
    if len(lines) < number:
        raise PlasmaStateInputError(f"{path}: expected at least {number} lines")
    value = lines[number - 1].split("!", 1)[0].strip()
    if not value:
        raise PlasmaStateInputError(f"{path}: line {number} is blank")
    return value


def legacy_case_spec(
    workdir: str | Path, *, gfile: str | Path, inputf: str = "inputf"
) -> PlasmaStateSpec:
    """The spec for a staged legacy case directory (``inputf`` + ``profiles``).

    ``inputf`` gives the time window, the mdescr / sconfig names, the state to
    write, the run id and the profiles file; *gfile* replaces its equilibrium
    line.
    """
    workdir = Path(workdir)
    path = workdir / inputf
    lines = path.read_text(encoding="utf-8").splitlines()
    times = _inputf_value(lines, 1, path).replace(",", " ").split()
    if len(times) < 2:
        raise PlasmaStateInputError(f"{path}: line 1 must give two times")
    if _inputf_value(lines, 7, path).split()[0] != "1":
        raise PlasmaStateInputError(
            f"{path}: line 7 selects analytic profiles; VAFT reads only a profiles file (1)"
        )
    word = lambda number: _inputf_value(lines, number, path).split()[0]  # noqa: E731
    legacy = read_legacy_profiles(workdir / word(8))
    return PlasmaStateSpec(
        mdescr=workdir / word(3),
        sconfig=workdir / word(4),
        gfile=Path(gfile),
        profiles=legacy.profiles,
        beam_power_w=legacy.beam_power_w,
        beam_energy_kev=legacy.beam_energy_kev,
        t0=float(times[0]),
        t1=float(times[1]),
        output_name=word(5),
        runid=word(6),
        edge_neutral_density=legacy.edge_neutral_density,
        neutral_energy_kev=legacy.neutral_energy_kev,
    )


# --------------------------------------------------------------------------- #
# IMAS
# --------------------------------------------------------------------------- #


def _get(ods: Any, path: str) -> Any:
    """``ods[path]`` or ``None``, without materialising a missing path."""
    node = ods
    for part in path.split("."):
        try:
            key = int(part) if part.isdigit() else part
            if key not in node:
                return None
            node = node[key]
        except (TypeError, KeyError, IndexError):
            return None
    return node


def _nearest(times: Any, time: float, tolerance: float, what: str) -> int:
    stamps = np.asarray(times, dtype=float).reshape(-1)
    if stamps.size == 0 or not np.any(np.isfinite(stamps)):
        raise PlasmaStateInputError(f"{what} has no time base")
    distance = np.where(np.isfinite(stamps), np.abs(stamps - time), np.inf)
    index = int(np.argmin(distance))
    if distance[index] > tolerance:
        raise PlasmaStateInputError(
            f"{what} has no sample within {tolerance:g} s of t = {time:g} s "
            f"(nearest is {stamps[index]:g} s)"
        )
    return index


def _equilibrium_index(ods: Any, time: float, tolerance: float) -> int:
    times = _get(ods, "equilibrium.time")
    if times is None:
        slices = _get(ods, "equilibrium.time_slice")
        if not slices:
            raise PlasmaStateInputError("the ODS has no equilibrium time slices")
        times = [float(_get(ods, f"equilibrium.time_slice.{i}.time")) for i in range(len(slices))]
    return _nearest(times, time, tolerance, "equilibrium")


def profiles_from_core_profiles(
    ods: Any,
    *,
    time: float,
    equilibrium_index: int,
    ion_charges: Sequence[int],
    ion_masses: Sequence[int],
    time_tolerance: float = 1e-3,
    zeff: Optional[float] = None,
    n_points: int = 101,
    minimum_density: Optional[float] = None,
    minimum_temperature_ev: Optional[float] = None,
) -> tuple[PlasmaStateProfiles, dict[str, Any]]:
    """Kinetic profiles from ``core_profiles`` on sqrt(psi_N), plus provenance.

    The abscissa is built from ``grid.psi`` against the equilibrium's own axis
    and boundary flux, so the ODS psi convention cancels; ``rho_tor_norm`` is
    not used (packaged VAFT equilibria store a sqrt(psi_N) proxy under it).

    Ion densities come from ``ion[*]`` matched to the sconfig species by
    charge and mass number. When ``core_profiles`` carries no ion densities
    and sconfig declares two ion species, they are derived from ne and Zeff
    (:func:`ion_densities_from_zeff`). Ti falls back to the first ion that has
    a temperature; there is no fallback to Te.

    The Plasma State needs strictly positive densities and temperatures. Fits
    that reach zero at the separatrix are refused unless a floor is given:
    *minimum_density* [m^-3] and *minimum_temperature_ev* clamp ne, the ion
    densities and both temperatures from below, and the number of points
    raised is recorded in the provenance.
    """
    index = _nearest(
        _get(ods, "core_profiles.time") if _get(ods, "core_profiles.time") is not None else [],
        time,
        time_tolerance,
        "core_profiles",
    )
    base = f"core_profiles.profiles_1d.{index}"
    psi = _get(ods, f"{base}.grid.psi")
    if psi is None:
        raise PlasmaStateInputError(
            f"{base}.grid.psi is missing; it is required (rho_tor_norm is not trusted)"
        )
    psi = np.asarray(psi, dtype=float)
    glob = f"equilibrium.time_slice.{equilibrium_index}.global_quantities"
    psi_axis, psi_boundary = (_get(ods, f"{glob}.{name}") for name in ("psi_axis", "psi_boundary"))
    if psi_axis is None or psi_boundary is None or float(psi_axis) == float(psi_boundary):
        raise PlasmaStateInputError("equilibrium psi_axis / psi_boundary are missing or equal")
    psi_n = (psi - float(psi_axis)) / (float(psi_boundary) - float(psi_axis))
    usable = np.isfinite(psi_n) & (psi_n >= -1e-9) & (psi_n <= 1.0 + 1e-9)
    source = np.sqrt(np.clip(psi_n, 0.0, None))
    order = np.argsort(source[usable])
    rho = source[usable][order]
    if rho.size < 3:
        raise PlasmaStateInputError("core_profiles has fewer than three points inside the plasma")
    if rho[0] > 0.02 or rho[-1] < 0.98:
        raise PlasmaStateInputError(
            f"core_profiles spans sqrt(psi_N) = [{rho[0]:.3f}, {rho[-1]:.3f}]; "
            "NUBEAM needs the whole plasma, and VAFT does not extrapolate"
        )
    grid = np.linspace(0.0, 1.0, int(n_points))
    provenance: dict[str, Any] = {
        "core_profiles_time_s": float(np.asarray(_get(ods, "core_profiles.time"), dtype=float).reshape(-1)[index]),
        "coordinate": "sqrt_psi_n",
    }

    def on_grid(path: str, what: str, *, required: bool = True) -> Optional[np.ndarray]:
        values = _get(ods, path)
        if values is None:
            if required:
                raise PlasmaStateInputError(f"{what}: {path} is missing")
            return None
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape != psi.shape:
            raise PlasmaStateInputError(f"{path} has {values.size} points, grid.psi has {psi.size}")
        values = values[usable][order]
        if not np.all(np.isfinite(values)):
            raise PlasmaStateInputError(f"{path} is not finite inside the plasma")
        provenance[what] = path
        return np.interp(grid, rho, values)

    def floor(name: str, values: np.ndarray, minimum: Optional[float], unit: str) -> np.ndarray:
        if minimum is None:
            if np.any(values <= 0.0):
                option = "minimum_density" if unit == "m^-3" else "minimum_temperature_ev"
                raise PlasmaStateInputError(
                    f"core_profiles: {name} reaches {values.min():g} {unit} at sqrt(psi_N) = "
                    f"{grid[np.argmin(values)]:.3f}; the Plasma State needs it positive. "
                    f"Pass {option} to floor it explicitly"
                )
            return values
        raised = int(np.count_nonzero(values < minimum))
        if raised:
            provenance.setdefault("floored_points", {})[name] = raised
        return np.maximum(values, float(minimum))

    ne = on_grid(f"{base}.electrons.density_thermal", "ne", required=False)
    if ne is None:
        ne = on_grid(f"{base}.electrons.density", "ne")
    ne = floor("ne", ne, minimum_density, "m^-3")
    te = floor("te", on_grid(f"{base}.electrons.temperature", "te"), minimum_temperature_ev, "eV")

    if zeff is not None:
        zeff_profile = np.full(grid.shape, float(zeff))
        provenance["zeff"] = "argument (uniform)"
    else:
        zeff_profile = on_grid(f"{base}.zeff", "zeff")

    ions = _get(ods, f"{base}.ion") or []
    by_species: dict[tuple[int, int], int] = {}
    for k in range(len(ions)):
        z = _get(ods, f"{base}.ion.{k}.element.0.z_n")
        a = _get(ods, f"{base}.ion.{k}.element.0.a")
        if z is not None and a is not None:
            by_species[(int(round(float(z))), int(round(float(a))))] = k

    ti = None
    for k in range(len(ions)):
        ti = on_grid(f"{base}.ion.{k}.temperature", "ti", required=False)
        if ti is not None:
            break
    if ti is None:
        raise PlasmaStateInputError(f"{base}.ion[*].temperature is missing; NUBEAM needs Ti")
    ti = floor("ti", ti, minimum_temperature_ev, "eV")

    densities: list[np.ndarray] = []
    for z, a in zip(ion_charges, ion_masses):
        k = by_species.get((int(z), int(a)))
        density = None
        if k is not None:
            density = on_grid(f"{base}.ion.{k}.density_thermal", f"n(Z={z},A={a})", required=False)
            if density is None:
                density = on_grid(f"{base}.ion.{k}.density", f"n(Z={z},A={a})", required=False)
        if density is None:
            densities = []
            break
        densities.append(density)
    if not densities:
        # A partial match is not used, so it must not be recorded as a source.
        for name in [key for key in provenance if key.startswith("n(Z=")]:
            del provenance[name]
        # From the floored ne, so the derived pair stays non-negative.
        densities = list(ion_densities_from_zeff(ne, zeff_profile, ion_charges))
        provenance["ion_densities"] = "derived from ne and Zeff"

    profiles = PlasmaStateProfiles(
        x=grid,
        ne=ne,
        te_kev=te / 1000.0,
        ti_kev=ti / 1000.0,
        zeff=zeff_profile,
        ion_densities=tuple(densities),
        coordinate="sqrt_psi_n",
    )
    return profiles, provenance


def spec_from_ods(
    ods: Any,
    *,
    time: float,
    workdir: str | Path,
    mdescr: str | Path,
    sconfig: str | Path,
    beam_power_w: Sequence[float],
    beam_energy_kev: Sequence[float],
    time_tolerance: float = 1e-3,
    duration: float = 0.0,
    zeff: Optional[float] = None,
    minimum_density: Optional[float] = None,
    minimum_temperature_ev: Optional[float] = None,
    edge_neutral_density: float = 0.0,
    neutral_energy_kev: Optional[Sequence[float]] = None,
    runid: str = "NUBEAM",
    output_name: str = "NUBEAM.cdf",
    **options: Any,
) -> tuple[PlasmaStateSpec, dict[str, Any]]:
    """Spec for one time from ``equilibrium`` and ``core_profiles`` in *ods*.

    The equilibrium is written as a G-EQDSK into *workdir* with
    :func:`vaft.data.eqdsk.from_omas`; the profiles are read by
    :func:`profiles_from_core_profiles`. Beam power and energy are run
    choices, not machine description, so they are arguments. *options* are
    passed through to :class:`PlasmaStateSpec` (``nrho``, ``bdy_crat``, ...).
    """
    from vaft.data.eqdsk import from_omas, write_geqdsk

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    index = _equilibrium_index(ods, float(time), time_tolerance)
    gfile = write_geqdsk(from_omas(ods, index), workdir / "equilibrium.gfile")
    config = read_shot_configuration(sconfig)
    profiles, provenance = profiles_from_core_profiles(
        ods,
        time=float(time),
        equilibrium_index=index,
        ion_charges=config.ion_charge_numbers,
        ion_masses=config.ion_mass_numbers,
        time_tolerance=time_tolerance,
        zeff=zeff,
        n_points=int(options.get("nrho", 101)),
        minimum_density=minimum_density,
        minimum_temperature_ev=minimum_temperature_ev,
    )
    if neutral_energy_kev is None:
        # e0_av: NTCC notes some models impose a 0.005 keV minimum.
        neutral_energy_kev = (0.005,) * len(config.gas_sources)
    provenance["equilibrium_index"] = index
    spec = PlasmaStateSpec(
        mdescr=Path(mdescr),
        sconfig=Path(sconfig),
        gfile=Path(gfile),
        profiles=profiles,
        beam_power_w=tuple(float(p) for p in beam_power_w),
        beam_energy_kev=tuple(float(e) for e in beam_energy_kev),
        t0=float(time),
        t1=float(time) + float(duration),
        runid=runid,
        output_name=output_name,
        edge_neutral_density=float(edge_neutral_density),
        neutral_energy_kev=tuple(float(e) for e in neutral_energy_kev),
        **options,
    )
    return spec, provenance


__all__ = [
    "GENERATOR_MAX_BEAMS",
    "GENERATOR_MAX_GAS_SOURCES",
    "GENERATOR_MAX_IONS",
    "GENERATOR_MAX_POINTS",
    "LegacyProfiles",
    "PLASMA_STATE_NAMELIST",
    "PROFILE_COORDINATES",
    "PlasmaStateInputError",
    "PlasmaStateProfiles",
    "PlasmaStateSpec",
    "ShotConfiguration",
    "check_spec_against_namelists",
    "count_beams",
    "ion_densities_from_zeff",
    "legacy_case_spec",
    "profiles_from_core_profiles",
    "read_legacy_profiles",
    "read_shot_configuration",
    "render_plasma_state_namelist",
    "spec_from_ods",
]
