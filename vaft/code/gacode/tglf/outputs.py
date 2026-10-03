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
SCHEMA_VERSION = 3

#: `out.tglf.gbflux` is one flat row, quantity-major: every species' particle flux,
#: then every species' energy flux, and so on. Verified against the per-species table
#: TGLF prints at the end of `out.tglf.run`.
GBFLUX_QUANTITIES = ("particle", "energy", "momentum", "exchange")

#: The five columns of each ky row in `out.tglf.sum_flux_spectrum`, in the order
#: `write_tglf_sum_flux_spectrum` (tglf_inout.f90) writes them. Each value is the
#: ky-integration weight times the mode-summed flux, so a sum over ky is the total.
FLUX_SPECTRUM_QUANTITIES = ("particle", "energy", "toroidal_stress", "parallel_stress", "exchange")

#: Field index in `out.tglf.sum_flux_spectrum`: 1 electrostatic potential, 2 the
#: perpendicular magnetic perturbation (USE_BPER), 3 the parallel one (USE_BPAR).
FLUX_SPECTRUM_FIELDS = ("phi", "a_par", "b_par")

#: The four columns of `out.tglf.field_spectrum` per (ky, mode), as
#: `write_tglf_field_spectrum` writes them: the "vector" total, then the
#: electrostatic potential, A_parallel and B_parallel intensities. A field the run did
#: not include (`a_par_no` / `b_par_no` in the header) is stored as NaN, not as the
#: zero TGLF writes, so an absent field cannot be read as a measured null.
FIELD_SPECTRUM_COLUMNS = ("vector", "phi", "a_par", "b_par")

#: The four moments of `out.tglf.intensity_spectrum` per (species, ky, mode),
#: `write_tglf_intensity_spectrum`: density, temperature, parallel velocity, parallel
#: energy -- gyro-Bohm normalised fluctuation *intensities* (squared amplitudes).
INTENSITY_MOMENTS = ("density", "temperature", "parallel_velocity", "parallel_energy")

__all__ = [
    "FIELD_SPECTRUM_COLUMNS",
    "FLUX_SPECTRUM_FIELDS",
    "FLUX_SPECTRUM_QUANTITIES",
    "GBFLUX_QUANTITIES",
    "INTENSITY_MOMENTS",
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
    ky_spectrum
        The ky grid (``ky rho_s``), from ``out.tglf.ky_spectrum``.
    eigenvalue_spectrum
        ``out.tglf.eigenvalue_spectrum`` as written: one row per ky, columns
        ``gamma_1, omega_1, gamma_2, omega_2, ...`` over the NMODES modes, in
        gyro-Bohm units (``c_s/a``). :attr:`growth_rate` and :attr:`frequency` split it.
    sum_flux_spectrum
        ``out.tglf.sum_flux_spectrum`` shaped ``(species, field, ky, quantity)`` with
        quantities :data:`FLUX_SPECTRUM_QUANTITIES` and fields
        :data:`FLUX_SPECTRUM_FIELDS` (only those the run included). Electrons first.
    errors
        Lines TGLF logged as errors. Non-empty means the run is not solved, whatever
        the exit status said.

    The spectral diagnostics below are written by TGLF's transport-model path only
    (``USE_TRANSPORT_MODEL=T``); each is ``None`` when its file is absent or does not
    match the layout its writer in ``tglf/src/tglf_inout.f90`` produces (GACODE
    b493397). All are gyro-Bohm normalised and on the ky grid of :attr:`ky_spectrum`.

    field_spectrum
        ``(ky, mode, 4)``, columns :data:`FIELD_SPECTRUM_COLUMNS`
        (``write_tglf_field_spectrum``). Saturated field intensities per mode.
    intensity_spectrum
        ``(species, ky, mode, 4)``, moments :data:`INTENSITY_MOMENTS`
        (``write_tglf_intensity_spectrum``); species are :attr:`spectral_species`.
    density_spectrum, temperature_spectrum
        ``(ky, species)``: ``sqrt(sum over modes of the density/temperature
        intensity)`` -- fluctuation *amplitudes* (``write_tglf_density_spectrum``,
        ``..._temperature_spectrum``), species ``1..ns`` (electrons first).
    nete_crossphase_spectrum
        ``(ky, mode)`` electron density-temperature cross phase [rad].
    nsts_crossphase_spectrum
        ``(species, ky, mode)`` density-temperature cross phase per species [rad].
    ql_flux_spectrum
        ``(species, field, mode, ky, 5)`` quasilinear *weights* per mode, quantities
        :data:`FLUX_SPECTRUM_QUANTITIES` (``write_tglf_QL_flux_spectrum``). These are
        not fluxes: the saturated flux is weight x intensity, which
        :attr:`sum_flux_spectrum` already carries.
    width_spectrum, spectral_shift_spectrum, ave_p0_spectrum
        ``(ky,)`` model internals: the Gaussian width of the Hermite basis, the
        ``kx/ky`` spectral shift (its model depends on SAT_RULE and UNITS), and the SAT0
        normalisation.
    saturation_parameters
        ``out.tglf.scalar_saturation_parameters`` as ``{KEY: value}``: SAT_RULE, UNITS,
        XNU_MODEL, the SAT geometry factors, ``grad_r0_out``, ``kymax_out``,
        ``vzf_out`` .... Note that TGLF's presets (``tglf_startup.f90``) set
        ``XNU_MODEL=3`` and ``WDIA_TRAPPED=1`` for SAT2/3, so the *linear* eigenvalues of
        a SAT2/3 run differ from a SAT0/1 run on the same input; ``UNITS=CGYRO`` only
        moves the ky grid points (``ky_factor = grad_r0``), not the units ky is reported
        in.
    spectral_species
        1-based TGLF species indices the per-species spectra cover: ``1..ns``, or
        ``2..ns`` when the electrons are adiabatic (``ns0 = 2``).
    """

    directory: str
    gbflux: Optional[Mapping[str, np.ndarray]] = None
    ky_spectrum: Optional[np.ndarray] = None
    eigenvalue_spectrum: Optional[np.ndarray] = None
    sum_flux_spectrum: Optional[np.ndarray] = None
    field_spectrum: Optional[np.ndarray] = None
    intensity_spectrum: Optional[np.ndarray] = None
    density_spectrum: Optional[np.ndarray] = None
    temperature_spectrum: Optional[np.ndarray] = None
    nete_crossphase_spectrum: Optional[np.ndarray] = None
    nsts_crossphase_spectrum: Optional[np.ndarray] = None
    ql_flux_spectrum: Optional[np.ndarray] = None
    width_spectrum: Optional[np.ndarray] = None
    spectral_shift_spectrum: Optional[np.ndarray] = None
    ave_p0_spectrum: Optional[np.ndarray] = None
    saturation_parameters: Optional[Mapping[str, Any]] = None
    spectral_species: Optional[tuple[int, ...]] = None
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

    def _eigen_columns(self, offset: int) -> Optional[np.ndarray]:
        values = self.eigenvalue_spectrum
        if values is None:
            return None
        table = np.atleast_2d(np.asarray(values, dtype=float))
        if table.shape[1] % 2:
            return None
        return table[:, offset::2]

    @property
    def growth_rate(self) -> Optional[np.ndarray]:
        """Linear growth rate ``(ky, mode)`` in ``c_s/a``, from the eigenvalue spectrum."""
        return self._eigen_columns(0)

    @property
    def frequency(self) -> Optional[np.ndarray]:
        """Real frequency ``(ky, mode)`` in ``c_s/a``; TGLF's sign convention, unlabelled."""
        return self._eigen_columns(1)

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
        for key in ("errors", "files", "spectral_species"):
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


def _numeric_rows(path: Path) -> Optional[list[list[float]]]:
    """Rows of numbers in a TGLF list-directed file, text header lines dropped.

    TGLF's spectrum writers open with prose lines (``index limits: nky``), which
    ``np.loadtxt`` rejects outright; a row that does not parse as numbers is a header.
    """
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        try:
            rows.append([float(part) for part in parts])
        except ValueError:
            continue
    return rows or None


def _parse_ky_spectrum(directory: Path) -> Optional[np.ndarray]:
    """``out.tglf.ky_spectrum``: a prose line, ``nky``, then one ky per line."""
    rows = _numeric_rows(directory / "out.tglf.ky_spectrum")
    if not rows or len(rows[0]) != 1:
        return None
    count = int(rows[0][0])
    values = [row[0] for row in rows[1:] if len(row) == 1]
    if count <= 0 or len(values) != count:
        return None
    return np.asarray(values, dtype=float)


def _parse_eigenvalue_spectrum(directory: Path, nky: Optional[int]) -> Optional[np.ndarray]:
    """``out.tglf.eigenvalue_spectrum``: two prose lines, then ``nky`` rows of
    ``(gamma_n, freq_n), n = 1..NMODES``."""
    rows = _numeric_rows(directory / "out.tglf.eigenvalue_spectrum")
    if not rows:
        return None
    widths = {len(row) for row in rows}
    if len(widths) != 1 or next(iter(widths)) % 2:
        return None
    if nky is not None and len(rows) != nky:
        return None
    return np.asarray(rows, dtype=float)


def _parse_sum_flux_spectrum(
    directory: Path, n_species: Optional[int], nky: Optional[int]
) -> Optional[np.ndarray]:
    """``out.tglf.sum_flux_spectrum``: per (species, field) a two-line header
    (``species = i field = j`` and a column legend), then ``nky`` rows of the five
    :data:`FLUX_SPECTRUM_QUANTITIES`."""
    path = directory / "out.tglf.sum_flux_spectrum"
    if not path.is_file() or n_species is None or nky is None:
        return None
    blocks: dict[tuple[int, int], list[list[float]]] = {}
    current: Optional[tuple[int, int]] = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("species"):
            numbers = [part for part in text.replace("=", " ").split() if part.lstrip("-").isdigit()]
            if len(numbers) != 2:
                return None
            current = (int(numbers[0]), int(numbers[1]))
            blocks[current] = []
            continue
        try:
            row = [float(part) for part in text.split()]
        except ValueError:
            continue
        if current is None or len(row) != len(FLUX_SPECTRUM_QUANTITIES):
            return None
        blocks[current].append(row)
    if not blocks:
        return None
    species = sorted({key[0] for key in blocks})
    field_ids = sorted({key[1] for key in blocks})
    if len(species) != n_species or field_ids != list(range(1, len(field_ids) + 1)):
        return None
    out = np.full((len(species), len(field_ids), nky, len(FLUX_SPECTRUM_QUANTITIES)), np.nan)
    for (spec, fld), rows in blocks.items():
        if len(rows) != nky:
            return None
        out[species.index(spec), fld - 1] = np.asarray(rows, dtype=float)
    return out


def _header_numbers(path: Path, label: str) -> Optional[list[int]]:
    """The integer line that follows ``index limits: ...`` (or the line holding
    ``label``) in a TGLF spectrum file."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if label in line and index + 1 < len(lines):
            try:
                return [int(part) for part in lines[index + 1].split()]
            except ValueError:
                return None
    return None


def _parse_field_spectrum(directory: Path, nky: Optional[int]) -> Optional[np.ndarray]:
    """``out.tglf.field_spectrum``: 3 prose lines, ``nky nmodes``, ``a_par_yes|no``,
    ``b_par_yes|no``, then rows ``v phi a_par b_par`` looped ky (outer), mode."""
    path = directory / "out.tglf.field_spectrum"
    if not path.is_file() or nky is None:
        return None
    limits = _header_numbers(path, "index limits")
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = [r for r in _numeric_rows(path) or [] if len(r) == 4]
    if not limits or len(limits) != 2 or limits[0] != nky or len(rows) != nky * limits[1]:
        return None
    out = np.asarray(rows, dtype=float).reshape(nky, limits[1], 4)
    if "a_par_no" in text:
        out[..., 2] = np.nan
    if "b_par_no" in text:
        out[..., 3] = np.nan
    return out


def _species_rows(count: int, per_species: int, n_species: Optional[int]) -> Optional[tuple[int, ...]]:
    """Which TGLF species a species-major block covers: ``1..ns`` or ``2..ns``."""
    if n_species is None or per_species <= 0 or count % per_species:
        return None
    present = count // per_species
    if present == n_species:
        return tuple(range(1, n_species + 1))
    if present == n_species - 1:
        return tuple(range(2, n_species + 1))
    return None


def _parse_intensity_spectrum(
    directory: Path, n_species: Optional[int], nky: Optional[int]
) -> tuple[Optional[np.ndarray], Optional[tuple[int, ...]]]:
    """``out.tglf.intensity_spectrum``: 3 prose lines, ``ns nky nmodes``, then rows of
    the four :data:`INTENSITY_MOMENTS` looped species (``ns0..ns``), ky, mode."""
    path = directory / "out.tglf.intensity_spectrum"
    if not path.is_file() or nky is None:
        return None, None
    limits = _header_numbers(path, "index limits")
    rows = [r for r in _numeric_rows(path) or [] if len(r) == 4]
    if not limits or len(limits) != 3 or limits[1] != nky:
        return None, None
    nmodes = limits[2]
    species = _species_rows(len(rows), nky * nmodes, n_species if n_species else limits[0])
    if species is None:
        return None, None
    return np.asarray(rows, dtype=float).reshape(len(species), nky, nmodes, 4), species


def _parse_amplitude_spectrum(directory: Path, name: str, nky: Optional[int]) -> Optional[np.ndarray]:
    """``out.tglf.{density,temperature}_spectrum``: 2 prose lines, ``nky`` rows of ``ns``."""
    rows = _numeric_rows(directory / f"out.tglf.{name}")
    if not rows or nky is None or len(rows) != nky or len({len(r) for r in rows}) != 1:
        return None
    return np.asarray(rows, dtype=float)


def _parse_nete_crossphase(directory: Path, nky: Optional[int]) -> Optional[np.ndarray]:
    """``out.tglf.nete_crossphase_spectrum``: 2 prose lines, ``nky`` rows of NMODES phases."""
    return _parse_amplitude_spectrum(directory, "nete_crossphase_spectrum", nky)


def _parse_nsts_crossphase(directory: Path, nky: Optional[int]) -> Optional[np.ndarray]:
    """``out.tglf.nsts_crossphase_spectrum``: per species a ``species index = is`` line,
    a legend, then ``nky`` rows of NMODES phases."""
    path = directory / "out.tglf.nsts_crossphase_spectrum"
    if not path.is_file() or nky is None:
        return None
    blocks: list[list[list[float]]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if text.startswith("species index"):
            blocks.append([])
            continue
        try:
            row = [float(part) for part in text.split()]
        except ValueError:
            continue
        if row and blocks:
            blocks[-1].append(row)
    if not blocks or any(len(b) != nky for b in blocks) or len({len(r) for b in blocks for r in b}) != 1:
        return None
    return np.asarray(blocks, dtype=float)


def _parse_ql_flux_spectrum(
    directory: Path, nky: Optional[int]
) -> tuple[Optional[np.ndarray], Optional[tuple[int, ...]]]:
    """``out.tglf.QL_flux_spectrum``: 3 prose lines, ``5 ns nfield nky nmodes``, then per
    (species, field) a ``species = is field = j`` line and per mode a ``mode = m`` line
    followed by ``nky`` rows of the five quantities."""
    path = directory / "out.tglf.QL_flux_spectrum"
    if not path.is_file() or nky is None:
        return None, None
    limits = _header_numbers(path, "index limits")
    if not limits or len(limits) != 5 or limits[0] != 5 or limits[3] != nky:
        return None, None
    _, _, nfield, _, nmodes = limits
    blocks: dict[tuple[int, int, int], list[list[float]]] = {}
    species_field: Optional[tuple[int, int]] = None
    current: Optional[tuple[int, int, int]] = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        numbers = [part for part in text.replace("=", " ").split() if part.lstrip("-").isdigit()]
        if text.startswith("species") and len(numbers) == 2:
            species_field = (int(numbers[0]), int(numbers[1]))
            continue
        if text.startswith("mode") and len(numbers) == 1 and species_field is not None:
            current = (*species_field, int(numbers[0]))
            blocks[current] = []
            continue
        try:
            row = [float(part) for part in text.split()]
        except ValueError:
            continue
        if current is not None and len(row) == 5:
            blocks[current].append(row)
    if not blocks:
        return None, None
    species = tuple(sorted({key[0] for key in blocks}))
    if any(len(rows) != nky for rows in blocks.values()) or len(blocks) != len(species) * nfield * nmodes:
        return None, None
    out = np.full((len(species), nfield, nmodes, nky, 5), np.nan)
    for (spec, fld, mode), rows in blocks.items():
        out[species.index(spec), fld - 1, mode - 1] = np.asarray(rows, dtype=float)
    return out, species


def _parse_ky_scalar_spectrum(directory: Path, name: str, nky: Optional[int]) -> Optional[np.ndarray]:
    """``width``/``spectral_shift``/``ave_p0`` spectra: prose lines, ``nky``, ``nky`` values."""
    rows = _numeric_rows(directory / f"out.tglf.{name}")
    if not rows or nky is None:
        return None
    values = [row[0] for row in rows if len(row) == 1]
    if len(values) == nky + 1 and int(values[0]) == nky:
        values = values[1:]
    if len(values) != nky:
        return None
    return np.asarray(values, dtype=float)


def _parse_saturation_parameters(directory: Path) -> Optional[dict]:
    """``out.tglf.scalar_saturation_parameters``: ``KEY = value`` lines, ``!`` comments."""
    path = directory / "out.tglf.scalar_saturation_parameters"
    if not path.is_file():
        return None
    out: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.split("!")[0].strip()
        if "=" not in text:
            continue
        key, _, raw = text.partition("=")
        raw = raw.strip()
        try:
            value: Any = int(raw) if raw.lstrip("-").isdigit() else float(raw)
        except ValueError:
            value = raw
        out[key.strip()] = value
    return out or None


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
    n_species = None if grid is None else grid["n_species"]
    ky = _parse_ky_spectrum(directory)
    nky = None if ky is None else int(ky.size)
    intensity, intensity_species = _parse_intensity_spectrum(directory, n_species, nky)
    ql_flux, ql_species = _parse_ql_flux_spectrum(directory, nky)
    return TglfOutputs(
        directory=str(directory),
        gbflux=_parse_gbflux(directory, n_species),
        ky_spectrum=ky,
        eigenvalue_spectrum=_parse_eigenvalue_spectrum(directory, nky),
        sum_flux_spectrum=_parse_sum_flux_spectrum(directory, n_species, nky),
        field_spectrum=_parse_field_spectrum(directory, nky),
        intensity_spectrum=intensity,
        density_spectrum=_parse_amplitude_spectrum(directory, "density_spectrum", nky),
        temperature_spectrum=_parse_amplitude_spectrum(directory, "temperature_spectrum", nky),
        nete_crossphase_spectrum=_parse_nete_crossphase(directory, nky),
        nsts_crossphase_spectrum=_parse_nsts_crossphase(directory, nky),
        ql_flux_spectrum=ql_flux,
        width_spectrum=_parse_ky_scalar_spectrum(directory, "width_spectrum", nky),
        spectral_shift_spectrum=_parse_ky_scalar_spectrum(directory, "spectral_shift_spectrum", nky),
        ave_p0_spectrum=_parse_ky_scalar_spectrum(directory, "ave_p0_spectrum", nky),
        saturation_parameters=_parse_saturation_parameters(directory),
        spectral_species=intensity_species or ql_species,
        grid=grid,
        precision=_parse_precision(directory),
        version=_parse_version(directory),
        errors=_parse_errors(directory),
        files=tuple(path.name for path in products),
    )
