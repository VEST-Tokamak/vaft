"""Canonical thomson_scattering builders integrated under machine_mapping."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.io import loadmat

from .utils import resolve_data_root, set_path

try:
    import uncertainties.unumpy as unumpy
except ImportError:
    unumpy = None


_CHANNEL_META = (
    (0, 0.475, "Polychrometer 1R1", "poly1R1"),
    (1, 0.425, "Polychrometer 2R2", "poly2R2"),
    (2, 0.370, "Polychrometer 3R3", "poly3R3"),
    (3, 0.310, "Polychrometer 4R4", "poly4R4"),
    (4, 0.255, "Polychrometer 5R5", "poly5R5"),
)


def _uarray_or_values(values: Any, errors: Any) -> Any:
    if unumpy is None:
        return values
    return unumpy.uarray(values, errors)


def _as_real_array(values: Any) -> np.ndarray:
    return np.real(np.asarray(values)).astype(float)


def _sanitize_sigma(values: Any) -> np.ndarray:
    """Return sigma as float array with invalid entries masked to NaN.

    MATLAB fit failures show up as complex sigmas (nonzero imaginary part) or
    zero/negative sigmas; storing those as-is gives the point effectively
    infinite weight in downstream weighted fits.
    """
    arr = np.asarray(values)
    real = np.real(arr).astype(float).copy()
    invalid = ~np.isfinite(real) | (real <= 0)
    if np.iscomplexobj(arr):
        invalid |= np.imag(arr) != 0
    real[invalid] = np.nan
    return real


def _normalize_thomson_time_to_seconds(time_values: np.ndarray) -> np.ndarray:
    """Return Thomson time in seconds (MAT time is treated as milliseconds)."""
    time = np.asarray(time_values, dtype=float).reshape(-1)
    return time / 1e3


#: Directories searched for a shot's Thomson MAT, nearest first. Only the
#: directory is ranked here -- which *file* wins inside them is decided by
#: :func:`thomson_source_rank`, so a revision is preferred wherever it sits.
_THOMSON_SEARCH_SUBDIRS = ("thomson_scattering", "legacy", "")

#: Filename layouts, most-preferred family last. The trailing number is the
#: analysis version and ``_rev`` marks a revision of that version, so the rank
#: is ordered rather than a lookup: a later version beats an earlier one, and a
#: revision beats the version it revises.
_THOMSON_NAME_PATTERNS = (
    # (regex, family rank). The regex must anchor the shot so 40330 never
    # matches a file belonging to 4033.
    (re.compile(r"^(?P<shot>\d+)_NeTe\.mat$", re.IGNORECASE), 0),
    (re.compile(r"^NeTe_Shot(?P<shot>\d+)\.mat$", re.IGNORECASE), 1),
    (re.compile(r"^NeTe[_-](?P<shot>\d+)\.mat$", re.IGNORECASE), 2),
    (
        re.compile(
            r"^NeTe_Shot(?P<shot>\d+)_v(?P<version>\d+)(?P<rev>_rev)?\.mat$",
            re.IGNORECASE,
        ),
        3,
    ),
    # The same two layouts without the `NeTe_` prefix. `Shot40330_v10.mat` is
    # the first example in the corrective updater's own filename docstring, and
    # that updater parses a shot number out of it -- so a rank that did not
    # recognise it made the updater drop the file instead of ingesting it.
    (re.compile(r"^Shot(?P<shot>\d+)\.mat$", re.IGNORECASE), 1),
    (
        re.compile(
            r"^Shot(?P<shot>\d+)_v(?P<version>\d+)(?P<rev>_rev)?\.mat$",
            re.IGNORECASE,
        ),
        3,
    ),
)


def thomson_source_rank(filename: str, shotnumber: int) -> tuple[int, int, int] | None:
    """Rank one Thomson MAT filename for ``shotnumber``, or ``None`` if unrelated.

    Higher sorts better. This is the single place the preference between
    competing files is expressed, so the library resolver and the corrective
    updater cannot disagree about which file is authoritative -- previously the
    resolver read a fixed list that happened to put ``_v9`` before
    ``_v9_rev``, while the updater passed whichever path ``os.listdir`` yielded
    last, and the two could pick different files for the same shot.

    The order is ``(family, version, revision)``:

    ``family``
        ``{shot}_NeTe`` < ``NeTe_Shot{shot}`` < ``NeTe_{shot}`` < the versioned
        ``NeTe_Shot{shot}_v{n}`` family. No VEST shot carries files from both
        the ``NeTe_{shot}`` campaign and the versioned family, so this ordering
        never has to arbitrate between two live analyses.
    ``version``
        The ``v{n}`` number, so a future ``_v10`` outranks ``_v9`` without this
        table being edited.
    ``revision``
        ``_rev`` outranks the version it revises. Shots 40323-40331 each carry
        both, and their ``T_e`` differs by up to 58%, so reading the wrong one
        is a scientific error rather than a cosmetic one.
    """
    name = Path(filename).name
    for pattern, family in _THOMSON_NAME_PATTERNS:
        match = pattern.match(name)
        if match is None:
            continue
        if int(match.group("shot")) != int(shotnumber):
            return None
        groups = match.groupdict()
        version = int(groups["version"]) if groups.get("version") else 0
        revision = 1 if groups.get("rev") else 0
        return (family, version, revision)
    return None


def _discover_thomson_sources(shotnumber: int, search_root: Path) -> list[Path]:
    """Return every Thomson MAT for ``shotnumber`` under ``search_root``.

    Ordered best-first by :func:`thomson_source_rank`, then by directory
    proximity, then by name. Directory iteration order never reaches the
    result: the listing is sorted before it is ranked, so two machines with
    different filesystem orders resolve the same file.
    """
    found: list[tuple[tuple[int, int, int], int, str, Path]] = []
    for depth, subdir in enumerate(_THOMSON_SEARCH_SUBDIRS):
        directory = search_root / subdir if subdir else search_root
        try:
            names = sorted(entry.name for entry in directory.iterdir() if entry.is_file())
        except OSError:
            continue
        for name in names:
            rank = thomson_source_rank(name, shotnumber)
            if rank is not None:
                found.append((rank, -depth, name, directory / name))
    # Two passes so the name tiebreak stays ascending: a single reverse=True
    # would flip it along with the rank, contradicting "then by name".
    found.sort(key=lambda item: item[2])
    found.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [path for _, _, _, path in found]


def _candidate_thomson_paths(shotnumber: int, data_root: Path) -> list[Path]:
    """Thomson MAT candidates for ``shotnumber``, best first.

    Discovery is by listing rather than by probing a fixed set of names, so a
    version this module has never heard of is still found and ranked.
    """
    return _discover_thomson_sources(shotnumber, Path(data_root))


def _resolve_thomson_mat_file(
    shotnumber: int,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
) -> Path:
    """Resolve a Thomson MAT path.

    ``data_root`` may be a directory (default search root) or a legacy positional
    path to a specific ``*.mat`` file (as used by ``thomson_scattering(ods, shot, filepath)``).
    A ``*.mat`` path must not be used as a directory when probing ``NeTe_Shot{shot}_v9`` etc.;
    in that case the parent directory (or package ``vaft/data``) is used for those patterns.
    """
    pkg_data = resolve_data_root(None)
    candidates: list[Path] = []

    if mat_file is not None:
        explicit = Path(mat_file)
        if explicit.is_absolute():
            candidates.append(explicit)
        else:
            if data_root is None:
                base = pkg_data
            else:
                dr = Path(data_root)
                if dr.suffix.lower() == ".mat" or (dr.exists() and dr.is_file()):
                    base = dr.parent if dr.parent.is_dir() else pkg_data
                else:
                    base = dr
            candidates.append(base / explicit)
            candidates.append(explicit)
    else:
        if data_root is None:
            search_root = pkg_data
        else:
            dr = Path(data_root)
            if dr.suffix.lower() == ".mat" or dr.is_file():
                # Legacy: third positional argument is often a full path to a MAT file.
                if dr.is_absolute():
                    candidates.append(dr)
                else:
                    candidates.append(dr)
                    candidates.append(pkg_data / dr)
                search_root = dr.parent if dr.parent.is_dir() else pkg_data
            else:
                search_root = dr

        candidates.extend(_candidate_thomson_paths(shotnumber, search_root))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if candidates:
        searched = ", ".join(str(path) for path in candidates)
    else:
        # Discovery found nothing, so there are no probed paths to list. Name
        # the directories and the layouts instead: an operator chasing a
        # missing shot needs to know where we looked, and the old fixed-list
        # error told them that.
        roots = ", ".join(
            str(search_root / sub) if sub else str(search_root)
            for sub in _THOMSON_SEARCH_SUBDIRS
        )
        searched = (
            f"{roots} (for names NeTe_Shot{shotnumber}_v<n>[_rev].mat, "
            f"NeTe_Shot{shotnumber}.mat, NeTe_{shotnumber}.mat, "
            f"{shotnumber}_NeTe.mat, Shot{shotnumber}[_v<n>[_rev]].mat)"
        )
    raise FileNotFoundError(
        f"Cannot find Thomson MAT file for shot {shotnumber}; searched: {searched}"
    )


def _has_v9_channel_keys(mat_data: dict[str, Any]) -> bool:
    """Whether every polychromator key the v9 reader indexes is present."""
    return all(
        f"{tag}_{quantity}" in mat_data
        for _, _, _, tag in _CHANNEL_META
        for quantity in ("Te", "sigmaTe", "Ne", "sigmaNe")
    )


def _set_dynamic_from_v9(mat_data: dict[str, Any], ods: Any) -> None:
    time = _normalize_thomson_time_to_seconds(_as_real_array(mat_data["time_TS"]))
    set_path(ods, "thomson_scattering.time", time)
    for channel, _, _, tag in _CHANNEL_META:
        set_path(
            ods,
            f"thomson_scattering.channel.{channel}.t_e.data",
            _uarray_or_values(
                _as_real_array(mat_data[f"{tag}_Te"]).reshape(-1),
                _sanitize_sigma(mat_data[f"{tag}_sigmaTe"]).reshape(-1),
            ),
        )
        set_path(
            ods,
            f"thomson_scattering.channel.{channel}.n_e.data",
            _uarray_or_values(
                _as_real_array(mat_data[f"{tag}_Ne"]).reshape(-1),
                _sanitize_sigma(mat_data[f"{tag}_sigmaNe"]).reshape(-1),
            ),
        )


def _suffixed_key_root(mat_data: dict[str, Any], shotnumber: int) -> str | None:
    """Return the shot suffix (e.g. '_48224') of a suffixed NeTe MAT, if present."""
    preferred = f"_{int(shotnumber)}"
    if f"tsTime{preferred}" in mat_data:
        return preferred
    for key in mat_data:
        if key.startswith("tsTime_"):
            return key[len("tsTime"):]
    return None


def _set_dynamic_from_suffixed(mat_data: dict[str, Any], ods: Any, suffix: str) -> None:
    """Load the shot-suffixed schema (tsTime_<shot>, Te_<shot>, ..., Rposition_<shot>).

    Channel count and R positions come from the in-file Rposition vector (mm),
    so 7-channel files (including the outboard R=0.650 m point) are fully ingested.
    """
    time = _normalize_thomson_time_to_seconds(_as_real_array(mat_data[f"tsTime{suffix}"]))
    r_positions_m = _as_real_array(mat_data[f"Rposition{suffix}"]).reshape(-1) / 1e3
    te = _as_real_array(mat_data[f"Te{suffix}"])
    ne = _as_real_array(mat_data[f"Ne{suffix}"])
    sigma_te = _sanitize_sigma(mat_data[f"sigmaTe{suffix}"])
    sigma_ne = _sanitize_sigma(mat_data[f"sigmaNe{suffix}"])

    # Dead-channel sentinels: the MATLAB pipeline writes Te=0.1 eV / Ne=1e17 m^-3
    # (with tiny sigmas) for failed measurements — mask them to NaN so they
    # cannot dominate weighted profile fits downstream.
    te_sentinel = np.isclose(te, 0.1, rtol=1e-6, atol=0.0)
    ne_sentinel = np.isclose(ne, 1e17, rtol=1e-6, atol=0.0)
    te[te_sentinel] = np.nan
    sigma_te[te_sentinel] = np.nan
    ne[ne_sentinel] = np.nan
    sigma_ne[ne_sentinel] = np.nan

    set_path(ods, "thomson_scattering.time", time)
    for channel, r_pos in enumerate(r_positions_m):
        prefix = f"thomson_scattering.channel.{channel}"
        set_path(ods, f"{prefix}.position.r", float(r_pos))
        set_path(ods, f"{prefix}.position.z", 0)
        set_path(ods, f"{prefix}.name", f"Polychrometer {channel + 1}")
        set_path(
            ods,
            f"{prefix}.t_e.data",
            _uarray_or_values(
                _extract_channel_series(te, channel, time.size),
                _extract_channel_series(sigma_te, channel, time.size),
            ),
        )
        set_path(
            ods,
            f"{prefix}.n_e.data",
            _uarray_or_values(
                _extract_channel_series(ne, channel, time.size),
                _extract_channel_series(sigma_ne, channel, time.size),
            ),
        )


def _extract_channel_series(matrix: np.ndarray, channel: int, time_len: int) -> np.ndarray:
    if matrix.ndim == 1:
        return matrix.reshape(-1)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {matrix.shape}")
    if matrix.shape[0] == time_len and matrix.shape[1] > channel:
        return matrix[:, channel]
    if matrix.shape[1] == time_len and matrix.shape[0] > channel:
        return matrix[channel, :]
    raise ValueError(
        f"Cannot infer time/channel axes for shape {matrix.shape} and time_len={time_len}"
    )


def _recover_simple_time(
    mat_data: dict[str, Any],
    shotnumber: int,
    source_file: Path,
) -> np.ndarray:
    """Return the time axis for a simple-schema MAT, from a sibling if needed.

    ``39915_NeTe.mat`` carries ``Ne``/``Te`` but no ``time``, while its
    ``39916`` counterpart has one -- the old export was not consistent. Rather
    than synthesising a timebase, the other ranked sources for the same shot
    are consulted and one is adopted only when its length matches the data's
    time axis unambiguously. A fabricated timebase would silently misplace
    every measurement, which is worse than refusing to map the shot.
    """
    if "time" in mat_data:
        return _as_real_array(mat_data["time"])

    samples = _as_real_array(mat_data["Te"]).shape[0]
    matches: list[tuple[Path, np.ndarray]] = []
    for sibling in _discover_thomson_sources(shotnumber, source_file.parent):
        if sibling == source_file:
            continue
        try:
            other = loadmat(str(sibling))
        except (OSError, ValueError):
            continue
        # Every timebase this module knows how to read, including the suffixed
        # campaign's `tsTime_<shot>`. Checked to exhaustion rather than
        # stopping at the first key that happens to be present: a file holding
        # a wrong-length `time` beside a correct-length `time_TS` still has a
        # usable timebase.
        keys = ("time", "time_TS", f"tsTime_{int(shotnumber)}")
        for key in keys:
            if key not in other:
                continue
            candidate = _as_real_array(other[key]).reshape(-1)
            if candidate.size == samples:
                matches.append((sibling, candidate))
                break

    if not matches:
        raise KeyError(
            f"{source_file.name} has no 'time' field and no sibling Thomson file "
            f"for shot {shotnumber} carries a time axis of {samples} samples to "
            "match it against. Refusing to invent timestamps."
        )

    distinct = {tuple(np.round(values, 12)) for _, values in matches}
    if len(distinct) > 1:
        names = ", ".join(sorted(path.name for path, _ in matches))
        raise KeyError(
            f"{source_file.name} has no 'time' field, and the sibling files "
            f"({names}) disagree about the {samples}-sample timebase for shot "
            f"{shotnumber}. Refusing to choose one arbitrarily."
        )
    return matches[0][1]


def _set_dynamic_from_simple(
    mat_data: dict[str, Any],
    ods: Any,
    *,
    shotnumber: int | None = None,
    source_file: Path | None = None,
) -> None:
    if "time" in mat_data or shotnumber is None or source_file is None:
        raw_time = _as_real_array(mat_data["time"])
    else:
        raw_time = _recover_simple_time(mat_data, shotnumber, source_file)
    time = _normalize_thomson_time_to_seconds(raw_time)
    te = _as_real_array(mat_data["Te"])
    ne = _as_real_array(mat_data["Ne"])
    sigma_te = _sanitize_sigma(mat_data["sigmaTe"])
    sigma_ne = _sanitize_sigma(mat_data["sigmaNe"])

    set_path(ods, "thomson_scattering.time", time)
    for channel, _, _, _ in _CHANNEL_META:
        te_values = _extract_channel_series(te, channel, time.size)
        ne_values = _extract_channel_series(ne, channel, time.size)
        te_err = _extract_channel_series(sigma_te, channel, time.size)
        ne_err = _extract_channel_series(sigma_ne, channel, time.size)
        set_path(
            ods,
            f"thomson_scattering.channel.{channel}.t_e.data",
            _uarray_or_values(te_values, te_err),
        )
        set_path(
            ods,
            f"thomson_scattering.channel.{channel}.n_e.data",
            _uarray_or_values(ne_values, ne_err),
        )


def vfit_thomson_scattering_static(ods: Any) -> None:
    set_path(ods, "thomson_scattering.ids_properties.homogeneous_time", 1)
    for channel, r_pos, name, _ in _CHANNEL_META:
        prefix = f"thomson_scattering.channel.{channel}"
        set_path(ods, f"{prefix}.position.r", r_pos)
        set_path(ods, f"{prefix}.position.z", 0)
        set_path(ods, f"{prefix}.name", name)


def vfit_thomson_scattering_dynamic(
    ods: Any,
    shotnumber: int,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
) -> None:
    source_file = _resolve_thomson_mat_file(
        shotnumber=shotnumber,
        data_root=data_root,
        mat_file=mat_file,
    )
    mat_data = loadmat(str(source_file))

    # `time_TS` alone does not make a file readable by the v9 reader: shot
    # 22027 carries `time_TS` next to `poly1R1_REM_Te` / `poly2R3_...`, a
    # polychromator naming this module does not map. Check the keys the reader
    # will actually index so such a file reports the schema it has, rather than
    # raising a bare KeyError on the first tag it happens to miss.
    if "time_TS" in mat_data and _has_v9_channel_keys(mat_data):
        _set_dynamic_from_v9(mat_data, ods)
        return

    suffix = _suffixed_key_root(mat_data, shotnumber)
    if suffix is not None:
        _set_dynamic_from_suffixed(mat_data, ods, suffix)
        return

    # 'time' is deliberately not required here: some old exports omit it, and
    # _set_dynamic_from_simple recovers it from a sibling of the same shot or
    # raises. Requiring it made those files fall through to the "unsupported
    # schema" error, which named the wrong problem.
    simple_keys = {"Te", "sigmaTe", "Ne", "sigmaNe"}
    if simple_keys.issubset(mat_data):
        _set_dynamic_from_simple(
            mat_data, ods, shotnumber=shotnumber, source_file=source_file
        )
        return

    available = sorted(key for key in mat_data.keys() if not key.startswith("__"))
    raise KeyError(
        f"Unsupported Thomson MAT schema in {source_file}. "
        f"Expected v9 keys (time_TS/poly*), suffixed keys (tsTime_<shot>/Te_<shot>/...), "
        f"or simple keys {sorted(simple_keys)}; "
        f"available keys: {available}"
    )


def thomson_scattering(
    ods: Any,
    shotnumber: int,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
) -> None:
    vfit_thomson_scattering_static(ods)
    vfit_thomson_scattering_dynamic(
        ods,
        shotnumber,
        data_root=data_root,
        mat_file=mat_file,
    )


__all__ = [
    "thomson_scattering",
    "thomson_source_rank",
    "vfit_thomson_scattering_dynamic",
    "vfit_thomson_scattering_static",
]
