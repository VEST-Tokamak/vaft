"""Collect one CGYRO run into a solver-native container.

Same principle as the NEO and TGLF containers: what CGYRO produced, in CGYRO's units
(``a``, ``c_s = sqrt(T_e/m_D)``, ``rho_s``, gyro-Bohm fluxes) and on CGYRO's grids,
stays here. Nothing is projected into IMAS at this layer, and a file CGYRO did not write
leaves its field ``None``.

Success is judged from ``out.cgyro.info``, never from the exit status: CGYRO writes
``ERROR: (CGYRO) ...`` there and exits 0 when it rejects an input, and appends
``EXIT: (CGYRO) <reason>`` only when the kernel finishes (``cgyro_final_kernel.F90``).
The reason distinguishes ``Linear converged`` from ``Linear terminated at max time``,
which is the line between a run that executed and one that is qualified (#1354).

File layouts follow CGYRO's own reader, ``f2py/pygacode/cgyro/data.py`` at the pinned
GACODE revision, and are checked against a real run in the tests rather than trusted:

* ``out.cgyro.grids`` -- packed ASCII: 11 integers/floats of sizes, then ``p``, theta,
  energy, xi, the ballooning angle, ``ky``, and dissipation arrays;
* ``out.cgyro.equilibrium`` -- packed ASCII local parameters, normalisations, species;
* ``out.cgyro.time`` -- four columns per print step, time first;
* ``bin.cgyro.freq`` -- ``(2, n_n, n_time)`` Fortran order, ``[omega, gamma]``;
* ``bin.cgyro.{phib,aparb,bparb}`` -- complex ``(n_theta*n_radial, n_time)``, the
  ballooning-space field;
* ``bin.cgyro.ky_flux`` -- ``(n_species, 3, n_field, n_n, n_time)``, moments
  ``[particle, energy, momentum]``.

Binary files are single precision unless ``HIPREC_FLAG=1``, which CGYRO records as the
last entry of ``out.cgyro.equilibrium``. A run cut off by a wall-clock limit writes a
partial final record, so time lengths are taken from the file sizes, not from the
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional

import numpy as np

SCHEMA = "vaft.code.gacode.cgyro.CgyroOutputs"
SCHEMA_VERSION = 1

#: The three moments of `bin.cgyro.ky_flux`, in file order.
FLUX_MOMENTS = ("particle", "energy", "momentum")
#: The fields, in file order; a run with N_FIELD=n carries the first n.
FIELD_NAMES = ("phi", "a_parallel", "b_parallel")
_BALLOONING_FILES = {"phi": "phib", "a_parallel": "aparb", "b_parallel": "bparb"}

#: The convention :attr:`CgyroOutputs.frequency_ion_negative` is expressed in -- the
#: one TGLF writes, so the two spectra overlay without a per-run sign flip.
FREQUENCY_SIGN_CONVENTION = "ion_diamagnetic_negative"

__all__ = [
    "FIELD_NAMES",
    "FLUX_MOMENTS",
    "FREQUENCY_SIGN_CONVENTION",
    "SCHEMA",
    "SCHEMA_VERSION",
    "CgyroOutputs",
    "collect_cgyro_outputs",
    "parse_equilibrium",
    "parse_grids",
]


@dataclass
class CgyroOutputs:
    """One CGYRO run, in CGYRO's own terms.

    Attributes
    ----------
    grid
        Sizes and grids from ``out.cgyro.grids``: ``n_n``, ``n_species``, ``n_field``,
        ``n_radial``, ``n_theta``, ``n_energy``, ``n_xi``, ``ky`` (``k_y rho_s`` per
        toroidal mode), ``theta``, ``thetab`` (the extended ballooning angle).
    equilibrium
        Local parameters and normalisations CGYRO actually used, from
        ``out.cgyro.equilibrium``; ``species`` is a sub-dict of arrays.
    time
        ``t`` in ``a/c_s`` per print step.
    frequency, growth_rate
        ``(n_n, n_time)`` in ``c_s/a``, **as CGYRO wrote them**: the frequency sign is
        the native ``exp(-i omega t)`` one, which depends on the field orientation.
    ion_direction
        ``+1`` when CGYRO reported "Ion direction: omega > 0" for this run, ``-1`` for
        "omega < 0", ``None`` when it did not say.
    ballooning
        Final-time ballooning-space fields, ``{"phi": complex (n_theta*n_radial,)}``
        etc., on :attr:`grid` ``["thetab"]``, exactly as CGYRO wrote them (no
        renormalisation; a consumer that wants ``max|phi| = 1`` divides).
    flux
        ``(n_species, 3, n_field, n_n, n_time)`` gyro-Bohm fluxes from
        ``bin.cgyro.ky_flux``; species in CGYRO's order (electrons last here).
    info
        Every line of ``out.cgyro.info``.
    errors
        ``ERROR:`` lines. Non-empty means not executed, whatever the exit status.
    exit_message
        The text after ``EXIT: (CGYRO)``, or ``None`` when the kernel never finished.
    """

    directory: str
    grid: Optional[dict[str, Any]] = None
    equilibrium: Optional[dict[str, Any]] = None
    time: Optional[np.ndarray] = None
    time_error: Optional[np.ndarray] = None
    frequency: Optional[np.ndarray] = None
    growth_rate: Optional[np.ndarray] = None
    ion_direction: Optional[int] = None
    ballooning: dict[str, np.ndarray] = field(default_factory=dict)
    flux: Optional[np.ndarray] = None
    info: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    exit_message: Optional[str] = None
    version: Optional[dict[str, Any]] = None
    files: tuple[str, ...] = ()

    # -- derived ---------------------------------------------------------------

    @property
    def ky(self) -> Optional[np.ndarray]:
        """``|k_y rho_s|`` per toroidal mode.

        CGYRO writes ``ky`` signed (``rho`` carries ``-BTCCW``, ``cgyro_make_profiles``),
        so a run with ``BTCCW = +1`` reports ``-0.3`` for ``KY=0.3``. The magnitude is the
        wavenumber; the orientation is already in the ion-direction statement.
        """
        if self.grid is None:
            return None
        return np.abs(np.asarray(self.grid.get("ky"), dtype=float))

    @property
    def nonlinear(self) -> bool:
        return bool(self.grid is not None and int(self.grid.get("n_n", 1)) > 1
                    and self.flux is not None)

    @property
    def final_frequency(self) -> Optional[np.ndarray]:
        """Native ``omega`` at the last print step, per toroidal mode."""
        return None if self.frequency is None else self.frequency[:, -1]

    @property
    def final_growth_rate(self) -> Optional[np.ndarray]:
        return None if self.growth_rate is None else self.growth_rate[:, -1]

    @property
    def frequency_ion_negative(self) -> Optional[np.ndarray]:
        """Final ``omega`` with the ion diamagnetic direction negative, as TGLF writes it.

        Derived from CGYRO's own statement of the ion direction for this run, not from a
        sign assumed from the inputs: ``None`` when CGYRO did not state it.
        """
        final = self.final_frequency
        if final is None or self.ion_direction is None:
            return None
        return -float(self.ion_direction) * final

    @property
    def converged(self) -> bool:
        """CGYRO's linear frequency met ``FREQ_TOL`` before ``MAX_TIME``."""
        return self.exit_message == "Linear converged"

    @property
    def decayed(self) -> bool:
        """The field decayed below CGYRO's 1e-12 floor and the run stopped.

        ``cgyro_freq.F90`` raises "Underflow in calculation of frequency error" when
        ``|omega|`` underflows -- in practice a strongly damped (stable) mode whose
        amplitude fell out of single precision. It is still an error and not
        :attr:`solved` (no eigenvalue was measured), but it is evidence of stability,
        not of a broken run, and callers classify it separately.
        """
        return any("Underflow in calculation of frequency error" in line for line in self.errors)

    @property
    def solved(self) -> bool:
        """No error, a finished kernel, and finite physics at the end of the run."""
        if self.errors or self.exit_message is None:
            return False
        if self.exit_message.startswith("Linear"):
            final = self.final_growth_rate
            omega = self.final_frequency
            return (
                final is not None and omega is not None
                and bool(np.all(np.isfinite(final))) and bool(np.all(np.isfinite(omega)))
            )
        return self.flux is not None and bool(np.all(np.isfinite(self.flux)))

    def flux_time_trace(self) -> Optional[np.ndarray]:
        """``(n_species, 3, n_time)``: fluxes summed over fields and toroidal modes."""
        if self.flux is None:
            return None
        return np.sum(self.flux, axis=(2, 3))

    def time_average_flux(self, window: tuple[float, float]) -> Optional[np.ndarray]:
        """``(n_species, 3)`` time average over ``[t0, t1]`` in ``a/c_s``."""
        trace = self.flux_time_trace()
        if trace is None or self.time is None:
            return None
        t = np.asarray(self.time, dtype=float)[: trace.shape[-1]]
        mask = (t >= window[0]) & (t <= window[1])
        if np.count_nonzero(mask) < 2:
            return None
        integrate = getattr(np, "trapezoid", None) or np.trapz
        return integrate(trace[..., mask], t[mask], axis=-1) / (t[mask][-1] - t[mask][0])

    def missing(self) -> tuple[str, ...]:
        return tuple(
            name for name in ("grid", "equilibrium", "time", "frequency", "growth_rate")
            if getattr(self, name) is None
        )

    # -- serialisation ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                if np.iscomplexobj(value):
                    return {"__complex__": True, "real": value.real.tolist(),
                            "imag": value.imag.tolist()}
                return value.tolist()
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            return value

        payload = {f.name: encode(getattr(self, f.name)) for f in fields(self)}
        payload["schema"] = SCHEMA
        payload["schema_version"] = SCHEMA_VERSION
        payload["frequency_sign_convention_derived"] = FREQUENCY_SIGN_CONVENTION
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CgyroOutputs":
        version = int(payload.get("schema_version", 0))
        if payload.get("schema") != SCHEMA or version > SCHEMA_VERSION:
            raise ValueError(
                f"not a {SCHEMA} payload this version can read "
                f"(schema={payload.get('schema')!r}, version={version})"
            )

        def decode(value: Any) -> Any:
            if isinstance(value, dict) and value.get("__complex__"):
                return np.asarray(value["real"]) + 1j * np.asarray(value["imag"])
            return value

        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in payload:
                continue
            value = payload[f.name]
            if f.name in ("time", "time_error", "frequency", "growth_rate", "flux"):
                value = None if value is None else np.asarray(value, dtype=float)
            elif f.name == "ballooning":
                value = {key: decode(item) for key, item in (value or {}).items()}
            elif f.name in ("info", "errors", "files"):
                value = tuple(value or ())
            elif f.name in ("grid", "equilibrium") and value is not None:
                value = _arrays(value)
            kwargs[f.name] = value
        return cls(**kwargs)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return target

    @classmethod
    def read_json(cls, path: str | Path) -> "CgyroOutputs":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _arrays(value: Mapping[str, Any]) -> dict[str, Any]:
    """Lists back to arrays, one level down (grid and equilibrium are shallow)."""
    out: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, list):
            out[key] = np.asarray(item, dtype=float)
        elif isinstance(item, dict):
            out[key] = _arrays(item)
        else:
            out[key] = item
    return out


# -- parsers -----------------------------------------------------------------


def _ascii(path: Path) -> Optional[np.ndarray]:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return np.asarray(path.read_text(encoding="utf-8").split(), dtype=float)
    except ValueError:
        return None


def parse_grids(path: str | Path) -> Optional[dict[str, Any]]:
    """``out.cgyro.grids``, unpacked in ``pygacode``'s order."""
    data = _ascii(Path(path))
    if data is None or data.size < 11:
        return None
    grid: dict[str, Any] = {
        "n_n": int(data[0]), "n_species": int(data[1]), "n_field": int(data[2]),
        "n_radial": int(data[3]), "n_theta": int(data[4]), "n_energy": int(data[5]),
        "n_xi": int(data[6]), "m_box": int(data[7]), "length": float(data[8]),
        "n_global": int(data[9]), "theta_plot": int(data[10]),
    }
    mark = 11

    def take(count: int) -> np.ndarray:
        nonlocal mark
        chunk = data[mark:mark + count]
        mark += count
        return chunk

    grid["p"] = take(grid["n_radial"]).astype(int)
    grid["theta"] = take(grid["n_theta"])
    grid["energy"] = take(grid["n_energy"])
    grid["xi"] = take(grid["n_xi"])
    grid["thetab"] = take(grid["n_theta"] * (grid["n_radial"] // max(grid["m_box"], 1)))
    grid["ky"] = take(grid["n_n"])
    return grid


_EQUILIBRIUM_SCALARS = (
    "rmin", "rmaj", "q", "shear", "shift", "kappa", "s_kappa", "delta", "s_delta",
    "zeta", "s_zeta", "zmag", "dzmag",
)
_EQUILIBRIUM_NORMS = (
    "rho", "ky0", "betae_unit", "beta_star", "lambda_star", "gamma_e", "gamma_p",
    "mach", "a_meters", "b_unit", "b_gs2", "dens_norm", "temp_norm", "vth_norm",
    "mass_norm", "rho_star_norm", "gamma_gb_norm", "q_gb_norm", "pi_gb_norm",
)
_SPECIES_FIELDS = ("z", "mass", "dens", "temp", "dlnndr", "dlntdr", "nu")


def parse_equilibrium(path: str | Path, n_species: int) -> Optional[dict[str, Any]]:
    """``out.cgyro.equilibrium``, unpacked in ``pygacode``'s order.

    Shape coefficients (4 sin + 7 cos pairs) are skipped by count, not kept: VAFT's
    Miller projection does not use them and they are all zero in its inputs.
    """
    data = _ascii(Path(path))
    if data is None:
        return None
    values = list(data)
    position = 0

    def take() -> Optional[float]:
        nonlocal position
        if position >= len(values):
            return None
        position += 1
        return float(values[position - 1])

    equilibrium: dict[str, Any] = {name: take() for name in _EQUILIBRIUM_SCALARS}
    for _ in range(2 * (4 + 7)):
        take()
    equilibrium.update({name: take() for name in _EQUILIBRIUM_NORMS})
    species = {name: [] for name in _SPECIES_FIELDS}
    for _ in range(int(n_species)):
        for name in _SPECIES_FIELDS:
            species[name].append(take())
    equilibrium["species"] = {
        name: np.asarray([np.nan if v is None else v for v in column], dtype=float)
        for name, column in species.items()
    }
    for _ in range(2 * int(n_species)):  # sdlnndr, sdlntdr
        take()
    equilibrium["sbeta"] = take()
    equilibrium["z_eff"] = take()
    flag = take()
    equilibrium["hiprec_flag"] = None if flag is None else int(flag)
    return equilibrium


def _info(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], Optional[str], Optional[int]]:
    if not path.is_file():
        return (), (), None, None
    lines = tuple(path.read_text(encoding="utf-8", errors="replace").splitlines())
    errors = tuple(line.strip() for line in lines if line.lstrip().startswith("ERROR"))
    exit_message = None
    direction = None
    for line in lines:
        stripped = line.strip()
        match = re.match(r"EXIT:\s*\(CGYRO\)\s*(.*)$", stripped)
        if match:
            exit_message = match.group(1).strip()
        match = re.search(r"Ion direction:\s*omega\s*([<>])\s*0", stripped)
        if match:
            direction = 1 if match.group(1) == ">" else -1
    return lines, errors, exit_message, direction


def _version(path: Path) -> Optional[dict[str, Any]]:
    """Last line of ``out.cgyro.version``: ``date [version][platform][simtime]``.

    The version tag itself carries brackets (``[b493397 [2026-08-20]]``), so the fields
    are split on ``][`` after the first ``[`` rather than matched bracket by bracket.
    """
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
             if line.strip()]
    if not lines:
        return None
    last = lines[-1]
    start = last.find("[")
    tags = [] if start < 0 else last[start + 1:].rstrip().rstrip("]").split("][")
    revision = tags[0].strip() if tags else None
    return {
        "line": last,
        "revision": revision,
        "commit": None if not revision else revision.split()[0],
        "platform": tags[1] if len(tags) > 1 else None,
        "start_time": tags[2] if len(tags) > 2 else None,
        "starts": len(lines),
    }


def _binary(directory: Path, name: str, dtype: str) -> Optional[np.ndarray]:
    path = directory / f"bin.cgyro.{name}"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return np.fromfile(path, dtype=dtype)


def collect_cgyro_outputs(workdir: str | Path) -> Optional[CgyroOutputs]:
    """Parse everything CGYRO wrote in *workdir*; ``None`` when it wrote nothing."""
    directory = Path(workdir)
    produced = sorted(
        p.name for p in directory.glob("*.cgyro.*")
        if p.is_file() and (p.name.startswith("out.") or p.name.startswith("bin."))
    )
    if not produced:
        return None

    info, errors, exit_message, direction = _info(directory / "out.cgyro.info")
    grid = parse_grids(directory / "out.cgyro.grids")
    equilibrium = None
    if grid is not None:
        equilibrium = parse_equilibrium(directory / "out.cgyro.equilibrium", grid["n_species"])
    hiprec = bool(equilibrium and equilibrium.get("hiprec_flag"))
    real, complex_ = ("float64", "complex128") if hiprec else ("float32", "complex64")

    outputs = CgyroOutputs(
        directory=str(directory),
        grid=grid,
        equilibrium=equilibrium,
        info=info,
        errors=errors,
        exit_message=exit_message,
        ion_direction=direction,
        version=_version(directory / "out.cgyro.version"),
        files=tuple(produced),
    )

    time_data = _ascii(directory / "out.cgyro.time")
    if time_data is not None and time_data.size >= 4:
        count = time_data.size // 4
        table = time_data[: 4 * count].reshape(count, 4)
        outputs.time = table[:, 0]
        outputs.time_error = table[:, 1:3]

    if grid is None:
        return outputs
    n_n = grid["n_n"]

    freq = _binary(directory, "freq", real)
    if freq is None:
        freq = _ascii(directory / "out.cgyro.freq")
    if freq is not None and freq.size >= 2 * n_n:
        steps = freq.size // (2 * n_n)
        cube = freq[: 2 * n_n * steps].reshape((2, n_n, steps), order="F")
        outputs.frequency = cube[0].astype(float)
        outputs.growth_rate = cube[1].astype(float)

    spatial = grid["n_theta"] * grid["n_radial"]
    for name, stem in _BALLOONING_FILES.items():
        data = _binary(directory, stem, complex_)
        if data is not None and data.size >= spatial:
            steps = data.size // spatial
            outputs.ballooning[name] = (
                data[: spatial * steps].reshape((spatial, steps), order="F")[:, -1]
                .astype(complex)
            )

    flux = _binary(directory, "ky_flux", real)
    if flux is not None:
        outputs.flux = _ky_flux(flux, grid, None if outputs.time is None else outputs.time.size)
    return outputs


def _ky_flux(data: np.ndarray, grid: Mapping[str, Any], n_time: Optional[int]) -> Optional[np.ndarray]:
    """Reshape ``bin.cgyro.ky_flux``; the moment count is taken from the file, as
    pygacode does (``m = size // (ns*nf*nn*nt)``), not assumed.

    With the time count known, a build that writes a fourth moment (exchange) is read
    correctly and the first three kept; without it, three moments are assumed and the
    step count follows from the size. A size that fits neither is refused (``None``)
    rather than reshaped into scrambled species/moment/time.
    """
    base = grid["n_species"] * grid["n_field"] * grid["n_n"]
    moments = len(FLUX_MOMENTS)
    if n_time:
        if data.size % (base * n_time) == 0 and data.size // (base * n_time) >= moments:
            m, steps = data.size // (base * n_time), n_time
        elif data.size >= base * moments and (data.size // (base * moments)) <= n_time:
            # a run cut short mid-record: whole steps only, three moments
            m, steps = moments, data.size // (base * moments)
        else:
            return None
    else:
        if data.size < base * moments:
            return None
        m, steps = moments, data.size // (base * moments)
    cube = data[: base * m * steps].reshape(
        (grid["n_species"], m, grid["n_field"], grid["n_n"], steps), order="F")
    return cube[:, :moments].astype(float)
