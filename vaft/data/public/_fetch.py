"""Checksum-pinned download and cache for public multi-machine databases.

A published database is fetched once into a per-user cache and verified
against the SHA-256 recorded in :data:`SOURCES`.  A mismatch is an error, not a
silent re-download: a changed upstream file is a new release, and a new release
needs a new registry entry.

:class:`FetchError` means the upstream could not be reached; it is kept apart
from :class:`ChecksumError` (the bytes arrived but are not the pinned release)
and from parse errors raised by the dataset readers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

__all__ = [
    "ChecksumError",
    "FetchError",
    "PublicSource",
    "SOURCES",
    "cache_dir",
    "fetch",
    "fetch_source",
    "sha256_of",
]


class FetchError(RuntimeError):
    """The upstream host could not deliver the file (network or HTTP failure)."""


class ChecksumError(RuntimeError):
    """The file does not match the SHA-256 pinned for its release."""


@dataclass(frozen=True)
class PublicSource:
    """One pinned file of a public database release.

    Attributes
    ----------
    key : str
        Registry key, e.g. ``"itpa_db5.2.3"``.
    database : str
        Database name as published, e.g. ``"ITPA global H-mode confinement database"``.
    release : str
        Release label, e.g. ``"DB5.2.3"``.
    url : str
        Direct download URL.
    filename : str
        Cache file name.
    sha256 : str
        Hex SHA-256 of the pinned file.
    size_bytes : int
        Size of the pinned file.
    licence : str
        Licence of the file as stated by the distributor.
    reference : str
        Publication that must be cited when the data are used.
    doi : str
        DOI of the reference.
    landing_page : str
        Human-readable page of the distribution.
    """

    key: str
    database: str
    release: str
    url: str
    filename: str
    sha256: str
    size_bytes: int
    licence: str
    reference: str
    doi: str
    landing_page: str


SOURCES: dict[str, PublicSource] = {
    "itpa_db5.2.3": PublicSource(
        key="itpa_db5.2.3",
        database="ITPA global H-mode confinement database",
        release="DB5.2.3",
        url="https://osf.io/download/zhwa3/",
        filename="DB5.2.3.csv",
        sha256="7a48e34379663e3e298924990f05cd8f16b8581516bcfa7bb8f438f83ae80ab6",
        size_bytes=11_685_289,
        licence="CC BY 4.0",
        reference=(
            "G. Verdoolaege et al., 'The updated ITPA global H-mode confinement "
            "database: description and analysis', Nucl. Fusion 61 (2021) 076006"
        ),
        doi="10.1088/1741-4326/abdb91",
        landing_page="https://osf.io/drwcq/",
    ),
}


def cache_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Cache directory for public database files.

    Parameters
    ----------
    override : path-like or None, optional
        Explicit directory; default ``None`` uses ``$VAFT_PUBLIC_DATA_DIR`` when
        set, otherwise the per-user cache (``~/Library/Caches/vaft/public`` on
        macOS, ``%LOCALAPPDATA%/vaft/public`` on Windows,
        ``$XDG_CACHE_HOME/vaft/public`` elsewhere) [path].

    Returns
    -------
    pathlib.Path
        The directory; it is not created here [path].
    """
    if override is not None:
        return Path(override).expanduser()
    env = os.environ.get("VAFT_PUBLIC_DATA_DIR")
    if env:
        return Path(env).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "vaft" / "public"


def sha256_of(path: str | os.PathLike[str]) -> str:
    """Hex SHA-256 of a file.

    Parameters
    ----------
    path : path-like
        File to hash [path].

    Returns
    -------
    str
        Lower-case hex digest [str].
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, timeout: float) -> None:
    request = Request(url, headers={"User-Agent": "vaft-public-data/1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            payload = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise FetchError(f"Could not download {url}: {exc}") from exc
    if status != 200:
        raise FetchError(f"{url} returned HTTP {status}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch(
    url: str,
    *,
    sha256: str,
    filename: str,
    cache: str | os.PathLike[str] | None = None,
    timeout: float = 120.0,
) -> Path:
    """Download ``url`` once into the cache and verify its SHA-256.

    Parameters
    ----------
    url : str
        Direct download URL [str].
    sha256 : str
        Expected hex SHA-256 of the file [str].
    filename : str
        Cache file name; a bare name, no directories [str].
    cache : path-like or None, optional
        Cache directory, default ``None`` for :func:`cache_dir` [path].
    timeout : float, optional
        Network timeout, default 120 [s].

    Returns
    -------
    pathlib.Path
        Path of the verified cached file [path].

    Raises
    ------
    FetchError
        The download failed.
    ChecksumError
        The cached or downloaded file does not match ``sha256``.  A corrupt
        download is removed; a mismatching file already in the cache is left
        in place for inspection.
    """
    if Path(filename).name != filename or not filename:
        raise ValueError(f"filename must be a bare file name, got {filename!r}")
    destination = cache_dir(cache) / filename
    downloaded = False
    if not destination.exists():
        _download(url, destination, timeout)
        downloaded = True
    actual = sha256_of(destination)
    if actual != sha256.lower():
        if downloaded:
            destination.unlink(missing_ok=True)
        raise ChecksumError(
            f"{destination} has SHA-256 {actual}, expected {sha256}. "
            "The upstream file changed or the cached copy is corrupt; a new "
            "release needs a new registry entry."
        )
    return destination


def fetch_source(
    key: str,
    *,
    cache: str | os.PathLike[str] | None = None,
    timeout: float = 120.0,
) -> Path:
    """Fetch a registered public database file by its :data:`SOURCES` key.

    Parameters
    ----------
    key : str
        Registry key, e.g. ``"itpa_db5.2.3"`` [str].
    cache : path-like or None, optional
        Cache directory, default ``None`` for :func:`cache_dir` [path].
    timeout : float, optional
        Network timeout, default 120 [s].

    Returns
    -------
    pathlib.Path
        Path of the verified cached file [path].
    """
    if key not in SOURCES:
        raise KeyError(f"Unknown public source {key!r}. Registered: {sorted(SOURCES)}")
    source = SOURCES[key]
    return fetch(
        source.url, sha256=source.sha256, filename=source.filename,
        cache=cache, timeout=timeout,
    )
