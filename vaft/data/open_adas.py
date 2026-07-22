"""Small, dependency-free OPEN-ADAS ADF11 client used by VAFT.

The ADF11 parsing and default-file selection logic in this module is adapted
from MIT-licensed work, Copyright (c) 2021 Francesco Sciortino. See the
third-party notices at the end of the project README.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import tempfile
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


OPEN_ADAS_DOWNLOAD_URL = "https://open.adas.ac.uk/download/adf11"


class ADASDataError(RuntimeError):
    """Base exception for OPEN-ADAS lookup and ADF11 data failures."""


class ADASDownloadError(ADASDataError):
    """Raised when an ADF11 file cannot be downloaded."""


class ADF11FormatError(ADASDataError, ValueError):
    """Raised when an ADF11 file is malformed or inconsistent."""


@dataclass(frozen=True)
class ADF11Data:
    """Parsed unresolved ADF11 coefficient table in native log10 units."""

    path: Path
    file_type: str
    log_density_cm3: np.ndarray
    log_temperature_eV: np.ndarray
    log_coefficients: np.ndarray
    charge: np.ndarray
    metastables: np.ndarray

    @property
    def n_charge_states(self) -> int:
        return int(self.log_coefficients.shape[0])


_DEFAULT_FILES: Mapping[str, Mapping[str, str]] = {
    "H": {"acd": "acd12_h.dat", "scd": "scd12_h.dat", "plt": "plt12_h.dat"},
    "D": {"acd": "acd12_h.dat", "scd": "scd12_h.dat", "plt": "plt12_h.dat"},
    "He": {"acd": "acd96_he.dat", "scd": "scd96_he.dat", "plt": "plt96_he.dat"},
    "Li": {"acd": "acd96_li.dat", "scd": "scd96_li.dat", "plt": "plt96_li.dat"},
    "Be": {"acd": "acd96_be.dat", "scd": "scd96_be.dat", "plt": "plt96_be.dat"},
    "B": {"acd": "acd89_b.dat", "scd": "scd89_b.dat", "plt": "plt89_b.dat"},
    "C": {"acd": "acd96_c.dat", "scd": "scd96_c.dat", "plt": "plt96_c.dat"},
    "N": {"acd": "acd96_n.dat", "scd": "scd96_n.dat", "plt": "plt96_n.dat"},
    "O": {"acd": "acd96_o.dat", "scd": "scd96_o.dat", "plt": "plt96_o.dat"},
    "F": {"acd": "acd89_f.dat", "scd": "scd89_f.dat", "plt": "plt89_f.dat"},
    "Ne": {"acd": "acd96_ne.dat", "scd": "scd96_ne.dat", "plt": "plt96_ne.dat"},
    "Al": {"acd": "acd89_al.dat", "scd": "scd89_al.dat", "plt": "plt89_al.dat"},
    "Si": {"acd": "acd96_si.dat", "scd": "scd96_si.dat", "plt": "plt96_si.dat"},
    "S": {"acd": "acd89_s.dat", "scd": "scd89_s.dat", "plt": "plt89_s.dat"},
    "Cl": {"acd": "acd89_cl.dat", "scd": "scd89_cl.dat", "plt": "plt89_cl.dat"},
    "Ar": {"acd": "acd89_ar.dat", "scd": "scd89_ar.dat", "plt": "plt41_ar.dat"},
    "Ca": {"acd": "acd85_ca.dat", "scd": "scd85_ca.dat", "plt": "plt85_ca.dat"},
    "Ti": {"acd": "acd00_ti.dat", "scd": "scd00_ti.dat", "plt": "plt00_ti.dat"},
    "Fe": {"acd": "acd89_fe.dat", "scd": "scd89_fe.dat", "plt": "plt41_fe.dat"},
    "Ni": {"acd": "acd85_ni.dat", "scd": "scd85_ni.dat", "plt": "plt89_ni.dat"},
    "Kr": {"acd": "acd89_kr.dat", "scd": "scd89_kr.dat", "plt": "plt41_kr.dat"},
    "Mo": {"acd": "acd89_mo.dat", "scd": "scd89_mo.dat", "plt": "plt89_mo.dat"},
    "Xe": {"acd": "acd89_xe.dat", "scd": "scd89_xe.dat", "plt": "plt41_xe.dat"},
    "W": {"acd": "acd89_w.dat", "scd": "scd89_w.dat", "plt": "plt41_w.dat"},
}


def _normalize_species(species: str) -> str:
    text = str(species).strip()
    if not text:
        raise ValueError("Atomic species must not be empty")
    return text[0].upper() + text[1:].lower()


def default_adf11_files(species: str) -> dict[str, str]:
    """Return VAFT's default ``acd``, ``scd`` and ``plt`` files for *species*."""

    symbol = _normalize_species(species)
    try:
        return dict(_DEFAULT_FILES[symbol])
    except KeyError as exc:
        raise KeyError(f"No default ADF11 files are configured for species {symbol!r}") from exc


def _user_cache_dir() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "vaft" / "open_adas" / "adf11"
    if sys_platform() == "darwin":
        return Path.home() / "Library" / "Caches" / "vaft" / "open_adas" / "adf11"
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "vaft" / "open_adas" / "adf11"


def sys_platform() -> str:
    """Isolated for deterministic platform-path tests."""

    import sys

    return sys.platform


def _resolve_cache_dir(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    configured = os.environ.get("VAFT_ADAS_DIR")
    if configured:
        root = Path(configured).expanduser()
        adf11 = root / "adf11"
        return adf11 if adf11.is_dir() or not root.name.lower().startswith("adf11") else root
    return _user_cache_dir()


def _validate_filename(filename: str) -> str:
    name = str(filename).strip()
    if not name or Path(name).name != name or not name.lower().endswith(".dat"):
        raise ValueError(f"Invalid ADF11 filename: {filename!r}")
    prefix = name.split("_", 1)[0]
    if len(prefix) < 4 or prefix[:3].lower() not in {"acd", "scd", "plt"}:
        raise ValueError(f"Unsupported ADF11 filename: {filename!r}")
    return name


def _download_adf11(filename: str, destination: Path, timeout: float) -> None:
    dataset = filename.split("_", 1)[0]
    url = f"{OPEN_ADAS_DOWNLOAD_URL}/{dataset}/{filename}"
    request = Request(url, headers={"User-Agent": "vaft-open-adas/1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            payload = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ADASDownloadError(f"Could not download {filename!r} from OPEN-ADAS: {exc}") from exc
    if status != 200:
        raise ADASDownloadError(f"OPEN-ADAS returned HTTP {status} for {filename!r}")
    if len(payload) < 1000:
        raise ADASDownloadError(f"OPEN-ADAS returned an invalid short response for {filename!r}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{filename}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def get_adf11_path(
    filename: str,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    timeout: float = 30.0,
) -> Path:
    """Locate *filename* in the VAFT cache, downloading it from OPEN-ADAS if absent."""

    name = _validate_filename(filename)
    destination = _resolve_cache_dir(cache_dir) / name
    if destination.is_file():
        return destination
    _download_adf11(name, destination, float(timeout))
    return destination


def _numbers(lines: list[str], cursor: int, count: int, label: str) -> tuple[list[float], int]:
    values: list[float] = []
    while len(values) < count:
        if cursor >= len(lines):
            raise ADF11FormatError(f"Unexpected end of file while reading {label}")
        try:
            values.extend(float(value) for value in lines[cursor].split())
        except ValueError as exc:
            raise ADF11FormatError(f"Non-numeric value in {label} at line {cursor + 1}") from exc
        cursor += 1
    if len(values) != count:
        raise ADF11FormatError(f"Expected {count} values for {label}, found {len(values)}")
    return values, cursor


@lru_cache(maxsize=128)
def read_adf11(path: str | os.PathLike[str]) -> ADF11Data:
    """Parse an unresolved ADF11 file and return its native log10 coefficient table."""

    source = Path(path).expanduser().resolve()
    try:
        lines = source.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ADF11FormatError(f"Could not read ADF11 file {source}: {exc}") from exc
    if len(lines) < 5:
        raise ADF11FormatError(f"ADF11 file {source} is too short")

    try:
        n_blocks, n_density, n_temperature = (int(value) for value in lines[0].split()[:3])
    except (ValueError, TypeError) as exc:
        raise ADF11FormatError(f"Invalid ADF11 header in {source}") from exc
    if min(n_blocks, n_density, n_temperature) <= 0:
        raise ADF11FormatError(f"ADF11 dimensions must be positive in {source}")

    cursor = 2
    first = lines[cursor].split()
    if first and all(value.isdigit() for value in first):
        metastable_values, cursor = _numbers(lines, cursor, n_blocks + 1, "metastable counts")
        metastables = np.asarray(metastable_values, dtype=int)
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
    else:
        metastables = np.ones(n_blocks + 1, dtype=int)

    density, cursor = _numbers(lines, cursor, n_density, "density grid")
    temperature, cursor = _numbers(lines, cursor, n_temperature, "temperature grid")

    blocks: list[np.ndarray] = []
    charges: list[int] = []
    for block in range(n_blocks):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            raise ADF11FormatError(f"Missing subheader for coefficient block {block + 1}")
        subheader = lines[cursor]
        cursor += 1
        charge = block + 1
        for part in subheader.replace("-", " ").split("/"):
            if "Z=" in part.upper():
                try:
                    charge = int(part.split("=", 1)[1].split()[0])
                except (ValueError, IndexError):
                    pass
        raw, cursor = _numbers(lines, cursor, n_density * n_temperature, f"coefficient block {block + 1}")
        blocks.append(np.asarray(raw, dtype=float).reshape(n_temperature, n_density))
        charges.append(charge)

    density_array = np.asarray(density, dtype=float)
    temperature_array = np.asarray(temperature, dtype=float)
    coefficients = np.stack(blocks)
    if not (
        np.all(np.isfinite(density_array))
        and np.all(np.isfinite(temperature_array))
        and np.all(np.isfinite(coefficients))
        and np.all(np.diff(density_array) > 0)
        and np.all(np.diff(temperature_array) > 0)
    ):
        raise ADF11FormatError(f"ADF11 grids and coefficients must be finite with increasing grids: {source}")

    return ADF11Data(
        path=source,
        file_type=source.name[:3].lower(),
        log_density_cm3=density_array,
        log_temperature_eV=temperature_array,
        log_coefficients=coefficients,
        charge=np.asarray(charges, dtype=int),
        metastables=metastables,
    )


__all__ = [
    "ADF11Data",
    "ADF11FormatError",
    "ADASDataError",
    "ADASDownloadError",
    "OPEN_ADAS_DOWNLOAD_URL",
    "default_adf11_files",
    "get_adf11_path",
    "read_adf11",
]
