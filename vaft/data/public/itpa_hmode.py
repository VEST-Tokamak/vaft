"""ITPA global H-mode confinement database (DB5.2.3) reader and normaliser.

Source: G. Verdoolaege et al., Nucl. Fusion 61 (2021) 076006, distributed on
OSF (https://osf.io/drwcq/) under CC BY 4.0.  The file is fetched on demand and
verified against a pinned SHA-256 (:data:`vaft.data.public.SOURCES`); it is not
shipped with VAFT.

The full ``DB5.2.3.csv`` is SI throughout (A, T, m^-3, W, J, s, m, amu), with
signed ``IP`` and ``BT``.  Its first line holds column numbers and the header
is line two; it starts with a byte-order mark.  The ``STD5*.csv`` convenience
files use engineering units and ``DELTA1 = 1 + delta`` and are not read here --
the ``SELDB5`` flag of the full file selects the DB5 standard set instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ._fetch import SOURCES, fetch_source
from .schema import validate_confinement_table

__all__ = [
    "DB5_COLUMN_MAP",
    "DB5_DEFINITIONS",
    "normalize_db5",
    "read_db5",
]

_SOURCE = SOURCES["itpa_db5.2.3"]

#: Canonical column <- DB5.2.3 variable.  ``epsilon`` is derived (DB5 has no
#: EPS column in the full file) and ``i_p_A`` / ``b_t_T`` take magnitudes.
DB5_COLUMN_MAP: dict[str, str] = {
    "machine": "TOK",
    "shot": "SHOT",
    "time_s": "TIME",
    "regime": "PHASE",
    "i_p_A": "IP",
    "b_t_T": "BT",
    "n_e_line_avg_m3": "NEL",
    "p_loss_W": "PLTH",
    "w_th_J": "WTH",
    "tau_e_th_s": "TAUTH",
    "w_global_J": "WTOT",
    "tau_e_global_s": "TAUTOT",
    "r_geo_m": "RGEO",
    "a_m": "AMIN",
    "kappa": "KAPPA",
    "kappa_area": "KAREA",
    "delta": "DELTA",
    "m_eff_amu": "MEFF",
    "selected": "SELDB5",
}

#: Definitions carried on every DB5 row, from the DB5.2.3 variable description.
DB5_DEFINITIONS: dict[str, str] = {
    "p_loss_definition": (
        "DB5 PLTH = PL - PFLOSS, PL = POHM + auxiliary heating - dW/dt "
        "(machine-specific terms); beam charge-exchange and unconfined-orbit "
        "losses removed; radiation NOT subtracted"
    ),
    "w_th_definition": (
        "DB5 WTH: WMHD, WDIA or WKIN minus the fast-ion content, "
        "machine-specific recipe"
    ),
    "tau_e_definition": "DB5 TAUTH = WTH / PLTH",
    "w_global_definition": "DB5 WTOT: total plasma stored energy, fast ions included, machine-specific recipe",
    "tau_e_global_definition": (
        "DB5 TAUTOT = WTOT / PL, PL = POHM + auxiliary heating - dW/dt; unlike TAUTH, "
        "fast-ion losses (PFLOSS) are NOT removed from the power"
    ),
    "b_t_definition": "DB5 BT: vacuum toroidal field at RGEO",
    "n_e_definition": (
        "DB5 NEL: central line-averaged density from interferometry "
        "(approximated where NELFORM says so)"
    ),
    "m_eff_source": "source",
}

#: Source columns read when present: the global stored energy and confinement time
#: (#1713). A file without them still reads; the canonical columns stay NaN.
_OPTIONAL_SOURCE_COLUMNS = ("WTOT", "TAUTOT")

_NUMERIC_SOURCE_COLUMNS = (
    "SHOT", "TIME", "TIME_ID", "IP", "BT", "NEL", "PLTH", "WTH", "TAUTH",
    "RGEO", "AMIN", "KAPPA", "KAREA", "DELTA", "MEFF", "SELDB5",
)


def read_db5(
    path: str | os.PathLike[str] | None = None,
    *,
    cache: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Read the raw DB5.2.3 CSV with its original column names.

    Parameters
    ----------
    path : path-like or None, optional
        Local copy of ``DB5.2.3.csv``; default ``None`` fetches the pinned OSF
        release into the cache (network on first use) [path].
    cache : path-like or None, optional
        Cache directory for the fetch, default ``None`` for the per-user cache
        [path].

    Returns
    -------
    pandas.DataFrame
        One row per DB5 record, source column names and source (SI) units,
        blanks as ``NaN`` [table].

    Raises
    ------
    vaft.data.public.FetchError
        The file had to be fetched and the upstream was unreachable.
    ValueError
        The file lacks the DB5 columns this module relies on.
    """
    if path is None:
        path = fetch_source(_SOURCE.key, cache=cache)
    raw = pd.read_csv(
        Path(path), skiprows=1, encoding="utf-8-sig", low_memory=False,
    )
    # The column-number line leaves an unnamed row-index column in front.
    raw = raw.drop(columns=[c for c in raw.columns if str(c).startswith("Unnamed")])
    missing = [c for c in ("TOK", "PHASE", *_NUMERIC_SOURCE_COLUMNS) if c not in raw.columns]
    if missing:
        raise ValueError(f"{path} is not a DB5.2.3 table; missing columns {missing}")
    for column in (*_NUMERIC_SOURCE_COLUMNS, *(c for c in _OPTIONAL_SOURCE_COLUMNS if c in raw.columns)):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return raw


def normalize_db5(raw: pd.DataFrame, *, release: str = _SOURCE.release) -> pd.DataFrame:
    """Map a raw DB5.2.3 table into the canonical confinement table.

    Parameters
    ----------
    raw : pandas.DataFrame
        Output of :func:`read_db5` (source column names, SI units) [table].
    release : str, optional
        Release label recorded in ``source_release``; default ``"DB5.2.3"``
        [str].

    Returns
    -------
    pandas.DataFrame
        Canonical confinement table (:data:`vaft.data.public.CONFINEMENT_COLUMNS`),
        one row per source row, in source order [table].

    Notes
    -----
    * Units are unchanged (DB5 is SI); ``IP`` and ``BT`` become magnitudes.
    * ``epsilon = AMIN / RGEO`` is the only derived column.
    * ``record_id`` is ``"<TOK>:<SHOT>:<TIME_ID>"`` using the source's own
      ``TIME_ID``.  It is not always milliseconds: it is off by one from
      ``round(1000 TIME)`` on ~2,800 JET rows (and some COMPASS / JT60U rows)
      and in units of 1e-5 s for TUMAN3M.
    * Blank source cells stay ``NaN``; nothing is filled.
    * DB5's own ``HIPB98Y2`` multiplies ``TAUTH`` by the ``TAUC92`` correction
      (not 1 only for a few old machines).  ``tau_e_th_s`` is the uncorrected
      ``TAUTH``, so an H-factor computed from this table can differ from
      ``HIPB98Y2`` on those rows.
    """
    table = pd.DataFrame(index=raw.index)
    for canonical, source in DB5_COLUMN_MAP.items():
        table[canonical] = raw[source] if source in raw.columns else np.nan
    table["machine"] = raw["TOK"].astype(str).str.strip().str.upper()
    table["regime"] = raw["PHASE"].astype(str).str.strip()
    table["shot"] = raw["SHOT"].astype("Int64")
    table["i_p_A"] = raw["IP"].abs()
    table["b_t_T"] = raw["BT"].abs()
    table["epsilon"] = raw["AMIN"] / raw["RGEO"]
    table["selected"] = raw["SELDB5"].fillna(0).astype(int).eq(1)
    table["record_id"] = (
        table["machine"] + ":" + raw["SHOT"].astype("Int64").astype(str)
        + ":" + raw["TIME_ID"].astype("Int64").astype(str)
    )
    for column, text in DB5_DEFINITIONS.items():
        table[column] = text
    table["source_database"] = _SOURCE.database
    table["source_release"] = release
    table["source_reference"] = f"{_SOURCE.reference}, doi:{_SOURCE.doi}"
    # Numeric columns as float so NaN is the only missing marker.
    for column in (
        "time_s", "i_p_A", "b_t_T", "n_e_line_avg_m3", "p_loss_W", "w_th_J",
        "tau_e_th_s", "w_global_J", "tau_e_global_s", "r_geo_m", "a_m", "epsilon", "kappa", "kappa_area",
        "delta", "m_eff_amu",
    ):
        table[column] = table[column].astype(float)
    # DB5 marks a missing WTOT / TAUTOT on a few AUG rows with -1e-8 instead of a
    # blank; read it as missing, not as a (negative) stored energy.
    for column in ("w_global_J", "tau_e_global_s"):
        table.loc[table[column] < 0.0, column] = np.nan
    return validate_confinement_table(table)
