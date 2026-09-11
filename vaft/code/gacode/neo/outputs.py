"""Parse a NEO run directory into a complete, solver-native result.

Nothing here writes an IDS.  Following the split `vaft/code/nubeam/outputs.py`
documents, the IDS-populating layer reads this container rather than re-parsing
solver output itself, and the container stays a faithful transcript of what NEO
produced -- in NEO's units, on NEO's grid, with NEO's sign conventions.  The
audit that decides which of these quantities has a defensible IMAS home is
deliberately not done here (issue #550, phase 5).

**Column layouts are taken from NEO's writers, not from `pygacode`.**
`pygacode/neo/data.py`'s `read_theory` is stale with respect to
`neo/src/neo_theory.f90`: it reads the per-species block as three-wide
(`HSGamma`, `HSQ`, `KjparB`), while the writer emits two values per species and
then two trailing scalars.  For the three-species reg18 case that is 24 columns
against the 23 the file actually has.  The layout used here is the writer's, and
it is checked against stored runs with two *and* three species.

**Exit status is not success.**  NEO reports an input or physics error by
writing it to ``out.neo.run`` and then exiting **zero** (``neo_error`` sets a
flag and ``neo_do`` jumps to cleanup), and the launcher creates ``out.neo.run``
and ``out.neo.version`` before NEO starts.  So the presence of output files says
nothing; :attr:`NeoOutputs.errors` and :attr:`NeoOutputs.solved` are what do.

**Absent is not zero.**  NEO writes `out.neo.expnorm` and `out.neo.exprhon` only
when `PROFILE_MODEL >= 2`, and several files only under a rotation model.  A
missing file leaves its field ``None``; it is never filled with zeros, because a
zero flux is a physical result and must stay distinguishable from an
unevaluated one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

#: Bumped when the stored shape changes incompatibly; `from_dict` refuses a
#: payload written by a newer version rather than silently misreading it.
SCHEMA = "vaft.code.gacode.neo.NeoOutputs"
SCHEMA_VERSION = 1

#: The analytic models NEO evaluates alongside its own solve, in the order
#: `neo/src/neo_theory.f90` writes them. Every one is a comparison point, and
#: two of them -- `sauter_bootstrap_current` and `redl_bootstrap_current` -- are
#: what `vaft.formula.neoclassical` is verified against.
THEORY_SCALARS = (
    "hinton_hazeltine_particle_flux",
    "hinton_hazeltine_ion_energy_flux",
    "hinton_hazeltine_electron_energy_flux",
    "hinton_hazeltine_bootstrap_current",
    "hinton_hazeltine_k",
    "hinton_hazeltine_uparB",
    "hinton_hazeltine_poloidal_velocity",
    "chang_hinton_ion_energy_flux",
    "taguchi_ion_energy_flux",
    "sauter_bootstrap_current",
    "sauter_k",
    "sauter_uparB",
    "sauter_poloidal_velocity",
    "hinton_rosenbluth_potential_squared",
)

#: What each transported quantity means, and in what normalisation. From the
#: declarations in `neo/src/neo_transport.f90`.
QUANTITY_DESCRIPTIONS: Mapping[str, str] = {
    "particle_flux": "Gamma / (n_0 v_t0), per species",
    "energy_flux": "Q / (n_0 v_t0 T_0), per species",
    "momentum_flux": "Pi / (n_0 a T_0), per species",
    "uparB": "<u_par B> / (v_t0 B_0), per species",
    "k": "poloidal-flow coefficient, per species",
    "K": "<u_par B> n / (v_t0 B_0 n_0), per species",
    "poloidal_velocity": "v_theta / v_t0, per species",
    "toroidal_velocity": "v_phi / v_t0, per species",
    "bootstrap_current": "<sum_s Z_s u_par,s B n_s>, normalised",
    "potential_squared": "<delta phi^2>, normalised",
}

_TRANSPORT_PER_SPECIES = (
    "particle_flux", "energy_flux", "momentum_flux", "uparB", "k", "K",
    "poloidal_velocity", "toroidal_velocity",
)


def _load(path: Path) -> Optional[np.ndarray]:
    """Read a whitespace table, returning None when the file is absent or empty."""
    if not path.is_file():
        return None
    try:
        array = np.loadtxt(path)
    except (ValueError, OSError):
        return None
    return array if array.size else None


def _rows(array: np.ndarray) -> np.ndarray:
    """Present a table as two-dimensional, even for a single radial point."""
    return array[None, :] if array.ndim == 1 else array


def _check_columns(table: np.ndarray, expected: int, name: str, n_species: int) -> None:
    """Refuse a table whose width disagrees with the species count.

    Every per-species block here is read by striding, and a stride over a table
    of the wrong width does not fail -- it silently assigns one species' flux to
    another. Checking the width is what makes that impossible.
    """
    if table.shape[1] != expected:
        raise ValueError(
            f"{name} has {table.shape[1]} columns but {n_species} species imply "
            f"{expected}; refusing to stride over it rather than mis-assign species"
        )


@dataclass
class NeoGrid:
    """The discretisation NEO actually used."""

    n_species: int
    n_energy: int
    n_xi: int
    n_theta: int
    theta: np.ndarray
    n_radial: int
    r_over_a: np.ndarray


@dataclass
class NeoNormalisation:
    """`out.neo.expnorm`: everything needed to put NEO's output into SI.

    Written only for ``PROFILE_MODEL >= 2``, so it is absent from a purely
    local run -- and without it the normalised outputs cannot be dimensionalised
    at all, which is why its absence is recorded rather than worked around.
    """

    r_over_a: np.ndarray
    a_meters: np.ndarray
    mass_deuterium: np.ndarray
    density_norm: np.ndarray
    temperature_norm: np.ndarray
    velocity_norm_times_a: np.ndarray
    b_unit: np.ndarray


@dataclass
class NeoOutputs:
    """One NEO run, in NEO's own terms.

    Attributes
    ----------
    theory
        The analytic models NEO evaluates for comparison, keyed by
        :data:`THEORY_SCALARS` plus ``nclass_bootstrap_current`` and
        ``redl_bootstrap_current``, each shaped ``(n_radial,)``.
    coordinates
        ``out.neo.exprhon``: the bridge from NEO's ``r/a`` back to
        ``rho_tor_norm`` and ``psi_norm``, and so back into IMAS.
    """

    directory: str
    grid: Optional[NeoGrid] = None
    species_mass: Optional[np.ndarray] = None
    species_charge: Optional[np.ndarray] = None
    normalisation: Optional[NeoNormalisation] = None
    coordinates: Optional[Mapping[str, np.ndarray]] = None
    equilibrium: Optional[Mapping[str, np.ndarray]] = None
    transport: Optional[Mapping[str, np.ndarray]] = None
    transport_gyroviscous: Optional[Mapping[str, np.ndarray]] = None
    transport_experimental: Optional[Mapping[str, np.ndarray]] = None
    transport_gyrobohm: Optional[Mapping[str, np.ndarray]] = None
    theory: Optional[Mapping[str, np.ndarray]] = None
    rotation: Optional[Mapping[str, np.ndarray]] = None
    geometry: Optional[Mapping[str, float]] = None
    precision: Optional[float] = None
    version: Optional[Mapping[str, str]] = None
    errors: tuple[str, ...] = ()
    files: tuple[str, ...] = ()

    @property
    def n_species(self) -> Optional[int]:
        if self.grid is not None:
            return self.grid.n_species
        if self.species_charge is not None:
            return int(np.size(self.species_charge))
        return None

    @property
    def bootstrap_current(self) -> Optional[np.ndarray]:
        """NEO's own drift-kinetic ``<j_par B>``, normalised."""
        if self.transport is None:
            return None
        return self.transport.get("bootstrap_current")

    @property
    def trapped_fraction(self) -> Optional[float]:
        """The trapped fraction NEO computed from the surface geometry.

        Integrated over the field-strength distribution, so it is the value to
        prefer over
        :func:`vaft.formula.neoclassical.trapped_particle_fraction`'s circular
        approximation whenever a run is available.
        """
        return None if self.geometry is None else self.geometry.get("f_trap")

    @property
    def solved(self) -> bool:
        """Whether NEO completed a solve, as opposed to merely leaving files.

        True only when NEO logged no error, wrote its transport product, and the
        drift-kinetic current it wrote is finite. The last clause is not
        pedantry: a degenerate geometry -- ``kappa`` absent from input.gacode,
        which expro reads as zero -- produces NaN with no error logged.
        """
        if self.errors or self.transport is None:
            return False
        current = self.bootstrap_current
        return current is not None and bool(np.all(np.isfinite(current)))

    def describe(self, name: str) -> str:
        """What a transported quantity means and how it is normalised."""
        try:
            return QUANTITY_DESCRIPTIONS[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a NEO transport quantity; known names are "
                f"{', '.join(sorted(QUANTITY_DESCRIPTIONS))}"
            ) from None

    def missing(self) -> tuple[str, ...]:
        """Products this run did not write, so absent never reads as zero."""
        return tuple(
            f.name
            for f in fields(self)
            if f.name not in {"directory", "files", "errors"} and getattr(self, f.name) is None
        )

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready payload that ``from_dict`` reads back exactly."""

        def encode(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return {"__array__": value.tolist()}
            if isinstance(value, (NeoGrid, NeoNormalisation)):
                return {
                    "__record__": type(value).__name__,
                    "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)},
                }
            if isinstance(value, Mapping):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [encode(item) for item in value]
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            return value

        payload = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION}
        payload.update({f.name: encode(getattr(self, f.name)) for f in fields(self)})
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NeoOutputs":
        version = int(payload.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"this payload is schema version {version} but this VAFT reads at most "
                f"{SCHEMA_VERSION}; upgrade rather than reading it partially"
            )

        records = {"NeoGrid": NeoGrid, "NeoNormalisation": NeoNormalisation}

        def decode(value: Any) -> Any:
            if isinstance(value, Mapping):
                if "__array__" in value:
                    return np.asarray(value["__array__"])
                if "__record__" in value:
                    record = records[value["__record__"]]
                    return record(**{k: decode(v) for k, v in value["fields"].items()})
                return {key: decode(item) for key, item in value.items()}
            return value

        known = {f.name for f in fields(cls)}
        arguments = {
            key: decode(value) for key, value in payload.items() if key in known
        }
        if isinstance(arguments.get("files"), list):
            arguments["files"] = tuple(arguments["files"])
        return cls(**arguments)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return target

    @classmethod
    def read_json(cls, path: str | Path) -> "NeoOutputs":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _parse_grid(directory: Path) -> Optional[NeoGrid]:
    values = _load(directory / "out.neo.grid")
    if values is None:
        return None
    flat = np.atleast_1d(values).ravel()
    n_species, n_energy, n_xi, n_theta = (int(flat[i]) for i in range(4))
    theta = flat[4 : 4 + n_theta]
    n_radial = int(flat[4 + n_theta])
    return NeoGrid(
        n_species=n_species,
        n_energy=n_energy,
        n_xi=n_xi,
        n_theta=n_theta,
        theta=theta,
        n_radial=n_radial,
        r_over_a=flat[5 + n_theta : 5 + n_theta + n_radial],
    )


def _parse_equilibrium(directory: Path, n_species: int) -> Optional[Mapping[str, np.ndarray]]:
    values = _load(directory / "out.neo.equil")
    if values is None:
        return None
    table = _rows(values)
    _check_columns(table, 7 + 5 * n_species, "out.neo.equil", n_species)
    return {
        "r_over_a": table[:, 0],
        "dphi0dr": table[:, 1],
        "q": table[:, 2],
        "rho_star": table[:, 3],
        "rmaj_over_a": table[:, 4],
        "omega0": table[:, 5],
        "domega0dr": table[:, 6],
        "density": table[:, 7 + 0 :: 5].T,
        "temperature": table[:, 7 + 1 :: 5].T,
        "dlnndr": table[:, 7 + 2 :: 5].T,
        "dlntdr": table[:, 7 + 3 :: 5].T,
        "collision_rate": table[:, 7 + 4 :: 5].T,
    }


def _parse_transport(path: Path, n_species: int) -> Optional[Mapping[str, np.ndarray]]:
    """`out.neo.transport` and `out.neo.transport_exp` share a layout."""
    values = _load(path)
    if values is None:
        return None
    table = _rows(values)
    _check_columns(
        table, 5 + len(_TRANSPORT_PER_SPECIES) * n_species, path.name, n_species
    )
    parsed = {
        "r_over_a": table[:, 0],
        "potential_squared": table[:, 1],
        "bootstrap_current": table[:, 2],
        "poloidal_velocity_zeroth": table[:, 3],
        "uparB_zeroth": table[:, 4],
    }
    for offset, name in enumerate(_TRANSPORT_PER_SPECIES):
        parsed[name] = table[:, 5 + offset :: 8].T
    return parsed


def _parse_transport_gv(path: Path, n_species: int) -> Optional[Mapping[str, np.ndarray]]:
    values = _load(path)
    if values is None:
        return None
    table = _rows(values)
    _check_columns(table, 1 + 3 * n_species, path.name, n_species)
    return {
        "r_over_a": table[:, 0],
        "particle_flux": table[:, 1 + 0 :: 3].T,
        "energy_flux": table[:, 1 + 1 :: 3].T,
        "momentum_flux": table[:, 1 + 2 :: 3].T,
    }


def _parse_transport_flux(path: Path, n_species: int) -> Optional[Mapping[str, np.ndarray]]:
    """`out.neo.transport_flux`: three blocks of n_species rows per radius."""
    values = _load(path)
    if values is None or n_species < 1:
        return None
    table = _rows(values)
    stride = 3 * n_species
    if table.shape[0] % stride:
        return None
    blocks = ("drift_kinetic", "gyroviscous", "total")
    parsed: dict[str, np.ndarray] = {}
    for block_index, block in enumerate(blocks):
        for column, name in enumerate(("particle_flux", "energy_flux", "momentum_flux"), start=1):
            parsed[f"{block}_{name}"] = np.stack(
                [table[block_index * n_species + s :: stride, column] for s in range(n_species)]
            )
    return parsed


def _parse_theory(directory: Path, n_species: int) -> Optional[Mapping[str, np.ndarray]]:
    values = _load(directory / "out.neo.theory")
    if values is None:
        return None
    table = _rows(values)
    expected = len(THEORY_SCALARS) + 1 + 2 * n_species + 2
    if table.shape[1] != expected:
        raise ValueError(
            f"out.neo.theory has {table.shape[1]} columns but {n_species} species imply "
            f"{expected}. The layout is neo_theory.f90's THEORY_do, not pygacode's."
        )
    parsed: dict[str, np.ndarray] = {"r_over_a": table[:, 0]}
    for offset, name in enumerate(THEORY_SCALARS, start=1):
        parsed[name] = table[:, offset]
    base = 1 + len(THEORY_SCALARS)
    parsed["hirshman_sigmar_particle_flux"] = table[:, base + 0 : base + 2 * n_species : 2].T
    parsed["hirshman_sigmar_energy_flux"] = table[:, base + 1 : base + 2 * n_species : 2].T
    parsed["nclass_bootstrap_current"] = table[:, base + 2 * n_species]
    parsed["redl_bootstrap_current"] = table[:, base + 2 * n_species + 1]
    return parsed


def _parse_geometry(directory: Path) -> Optional[Mapping[str, float]]:
    path = directory / "out.neo.diagnostic_geo"
    if not path.is_file():
        return None
    parsed: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("#") or "=" not in line:
            continue
        name, _, number = line[1:].partition("=")
        try:
            parsed[name.strip()] = float(number)
        except ValueError:
            continue
    return parsed or None


def _parse_version(directory: Path) -> Optional[Mapping[str, str]]:
    path = directory / "out.neo.version"
    if not path.is_file():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    keys = ("revision", "platform", "date")
    return {key: value for key, value in zip(keys, lines)}


def _parse_errors(directory: Path) -> tuple[str, ...]:
    """The error lines NEO wrote to out.neo.run, in order.

    `neo_error` writes each message verbatim, and every one NEO raises begins
    with ``ERROR:``.
    """
    path = directory / "out.neo.run"
    if not path.is_file():
        return ()
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "ERROR" in line
    )


def collect_neo_outputs(workdir: str | Path) -> Optional[NeoOutputs]:
    """Read a NEO run directory without re-running it.

    Returns ``None`` when the directory holds no NEO output at all, and a
    partially populated container when a run wrote only some of its products.
    Every field that has no file stays ``None``; see :meth:`NeoOutputs.missing`.
    """
    directory = Path(workdir)
    if not directory.is_dir():
        return None
    produced = sorted(path.name for path in directory.glob("out.neo.*"))
    if not produced:
        return None

    grid = _parse_grid(directory)
    species = _load(directory / "out.neo.species")
    species_mass = species_charge = None
    if species is not None:
        flat = np.atleast_1d(species).ravel()
        species_mass, species_charge = flat[0::2], flat[1::2]

    n_species = 0
    if grid is not None:
        n_species = grid.n_species
    elif species_charge is not None:
        n_species = int(species_charge.size)

    normalisation = None
    expnorm = _load(directory / "out.neo.expnorm")
    if expnorm is not None:
        table = _rows(expnorm)
        normalisation = NeoNormalisation(
            r_over_a=table[:, 0],
            a_meters=table[:, 1],
            mass_deuterium=table[:, 2],
            density_norm=table[:, 3],
            temperature_norm=table[:, 4],
            velocity_norm_times_a=table[:, 5],
            b_unit=table[:, 6],
        )

    coordinates = None
    exprhon = _load(directory / "out.neo.exprhon")
    if exprhon is not None:
        table = _rows(exprhon)
        coordinates = {
            "r_over_a": table[:, 0],
            "rho_tor_norm": table[:, 1],
            "psi_norm": table[:, 2],
        }

    rotation = None
    rotation_table = _load(directory / "out.neo.rotation")
    if rotation_table is not None:
        table = _rows(rotation_table)
        rotation = {"r_over_a": table[:, 0], "raw": table}

    precision = _load(directory / "out.neo.prec")
    return NeoOutputs(
        directory=str(directory),
        grid=grid,
        species_mass=species_mass,
        species_charge=species_charge,
        normalisation=normalisation,
        coordinates=coordinates,
        equilibrium=_parse_equilibrium(directory, n_species) if n_species else None,
        transport=_parse_transport(directory / "out.neo.transport", n_species),
        transport_gyroviscous=_parse_transport_gv(
            directory / "out.neo.transport_gv", n_species
        ),
        transport_experimental=_parse_transport(
            directory / "out.neo.transport_exp", n_species
        ),
        transport_gyrobohm=_parse_transport_flux(
            directory / "out.neo.transport_flux", n_species
        ),
        theory=_parse_theory(directory, n_species) if n_species else None,
        rotation=rotation,
        geometry=_parse_geometry(directory),
        precision=None if precision is None else float(np.atleast_1d(precision).ravel()[0]),
        version=_parse_version(directory),
        errors=_parse_errors(directory),
        files=tuple(produced),
    )
