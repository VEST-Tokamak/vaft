"""Collect one TGLF run into a solver-native container.

The same principle as :mod:`vaft.code.gacode.neo.outputs`: what TGLF produced, in
TGLF's own units and on TGLF's own grid, stays here. Nothing is projected into IMAS at
this layer, and a file TGLF did not write leaves its field ``None`` rather than a zero --
a zero flux is a physical result and must stay distinguishable from an unevaluated one.

**The launcher writes ``out.tglf.run`` before the solve starts**, exactly as NEO's does,
so the presence of output proves nothing about success. :attr:`TglfOutputs.solved` is
what answers that, and it requires the gyro-Bohm fluxes to exist and be finite.

Column layouts are read from TGLF's own Fortran writers rather than from ``pygacode``:
``neo/outputs.py`` documents ``pygacode``'s reader being stale against the writer it
claims to read, and nothing protects this suite member from the same drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

#: Bumped when the stored shape changes incompatibly; `from_dict` refuses a payload
#: written by a newer version rather than silently misreading it.
SCHEMA = "vaft.code.gacode.tglf.TglfOutputs"
SCHEMA_VERSION = 1

#: `out.tglf.gbflux` is one flat row, quantity-major: every species' particle flux,
#: then every species' energy flux, and so on. Verified against the per-species table
#: TGLF prints at the end of `out.tglf.run`.
GBFLUX_QUANTITIES = ("particle", "energy", "momentum", "exchange")

__all__ = [
    "GBFLUX_QUANTITIES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "TglfOutputs",
    "collect_tglf_outputs",
]


@dataclass
class TglfOutputs:
    """One TGLF run, in TGLF's own terms.

    Attributes
    ----------
    gbflux
        Gyro-Bohm normalised fluxes keyed by :data:`GBFLUX_QUANTITIES`, each an array
        over species in TGLF's order -- **electrons first**. Normalised, not SI: the
        conversion needs a gyro-Bohm unit this layer does not compute.
    errors
        Lines TGLF logged as errors. Non-empty means the run is not solved, whatever
        the exit status said.
    """

    directory: str
    gbflux: Optional[Mapping[str, np.ndarray]] = None
    ky_spectrum: Optional[np.ndarray] = None
    eigenvalue_spectrum: Optional[np.ndarray] = None
    grid: Optional[Mapping[str, int]] = None
    precision: Optional[float] = None
    version: Optional[Mapping[str, str]] = None
    errors: tuple[str, ...] = ()
    files: tuple[str, ...] = ()

    @property
    def n_species(self) -> Optional[int]:
        if self.grid is not None and "n_species" in self.grid:
            return int(self.grid["n_species"])
        if self.gbflux is not None and GBFLUX_QUANTITIES[0] in self.gbflux:
            return int(np.size(self.gbflux[GBFLUX_QUANTITIES[0]]))
        return None

    @property
    def energy_flux(self) -> Optional[np.ndarray]:
        """Gyro-Bohm normalised energy flux per species, electrons first."""
        return None if self.gbflux is None else self.gbflux.get("energy")

    @property
    def particle_flux(self) -> Optional[np.ndarray]:
        return None if self.gbflux is None else self.gbflux.get("particle")

    @property
    def solved(self) -> bool:
        """Whether TGLF completed the problem it was asked to solve.

        True only when TGLF logged no error and wrote finite gyro-Bohm fluxes. The
        launcher creates ``out.tglf.run`` before the solve, so neither its presence nor
        a zero exit status is evidence.
        """
        if self.errors:
            return False
        if self.gbflux is None:
            return False
        values = [np.asarray(v, dtype=float) for v in self.gbflux.values()]
        if not values or any(v.size == 0 for v in values):
            return False
        return bool(all(np.all(np.isfinite(v)) for v in values))

    def missing(self) -> tuple[str, ...]:
        """Fields this run did not produce, in declaration order."""
        skip = {"directory", "files", "errors"}
        return tuple(
            f.name for f in fields(self)
            if f.name not in skip and getattr(self, f.name) is None
        )

    def to_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return {"__array__": value.tolist()}
            if isinstance(value, Mapping):
                return {str(k): encode(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            return value

        payload: dict[str, Any] = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION}
        for entry in fields(self):
            payload[entry.name] = encode(getattr(self, entry.name))
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TglfOutputs":
        version = int(payload.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"this payload was written by schema version {version}; this build "
                f"reads {SCHEMA_VERSION}. Upgrade rather than reading it partially."
            )

        def decode(value: Any) -> Any:
            if isinstance(value, Mapping):
                if "__array__" in value:
                    return np.asarray(value["__array__"], dtype=float)
                return {str(k): decode(v) for k, v in value.items()}
            return value

        known = {entry.name for entry in fields(cls)}
        data = {key: decode(value) for key, value in payload.items() if key in known}
        for key in ("errors", "files"):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        return cls(**data)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return target

    @classmethod
    def read_json(cls, path: str | Path) -> "TglfOutputs":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _load(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        values = np.loadtxt(path, dtype=float, ndmin=1)
    except (ValueError, OSError):
        return None
    return values if values.size else None


def _parse_gbflux(directory: Path, n_species: Optional[int]) -> Optional[dict]:
    """Split the flat gyro-Bohm row into its quantities.

    The row is quantity-major, so the split needs the species count. It comes from
    ``out.tglf.grid`` when that was written, and otherwise from the row itself, which
    must divide by the four quantities.
    """
    values = _load(directory / "out.tglf.gbflux")
    if values is None:
        return None
    flat = np.ravel(values)
    count = n_species
    if count is None:
        if flat.size % len(GBFLUX_QUANTITIES):
            return None
        count = flat.size // len(GBFLUX_QUANTITIES)
    if count <= 0 or flat.size != count * len(GBFLUX_QUANTITIES):
        return None
    return {
        name: flat[index * count:(index + 1) * count]
        for index, name in enumerate(GBFLUX_QUANTITIES)
    }


def _parse_grid(directory: Path) -> Optional[dict]:
    values = _load(directory / "out.tglf.grid")
    if values is None or values.size < 2:
        return None
    return {"n_species": int(values[0]), "n_xgrid": int(values[1])}


def _parse_version(directory: Path) -> Optional[dict]:
    path = directory / "out.tglf.version"
    if not path.is_file():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None
    keys = ("revision", "platform", "date")
    return {key: value for key, value in zip(keys, lines)}


def _parse_precision(directory: Path) -> Optional[float]:
    values = _load(directory / "out.tglf.prec")
    if values is None:
        return None
    return float(np.ravel(values)[0])


def _parse_errors(directory: Path) -> tuple[str, ...]:
    """Error lines TGLF logged, from the run log it writes before solving."""
    path = directory / "out.tglf.run"
    if not path.is_file():
        return ()
    found = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("ERROR") or "ERROR:" in stripped.upper():
            found.append(stripped)
    return tuple(found)


def collect_tglf_outputs(workdir: str | Path) -> Optional[TglfOutputs]:
    """Read a finished TGLF directory, or ``None`` when it holds no TGLF output.

    A partially populated container is a valid answer: a run that failed still wrote
    files, and what it wrote is what says why.
    """
    directory = Path(workdir)
    products = sorted(directory.glob("out.tglf.*"))
    if not products:
        return None
    grid = _parse_grid(directory)
    return TglfOutputs(
        directory=str(directory),
        gbflux=_parse_gbflux(directory, None if grid is None else grid["n_species"]),
        ky_spectrum=_load(directory / "out.tglf.ky_spectrum"),
        eigenvalue_spectrum=_load(directory / "out.tglf.eigenvalue_spectrum"),
        grid=grid,
        precision=_parse_precision(directory),
        version=_parse_version(directory),
        errors=_parse_errors(directory),
        files=tuple(path.name for path in products),
    )
