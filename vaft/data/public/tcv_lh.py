"""TCV L-H transition database (public limited dataset) reader and normaliser.

Source: B. Labit et al., Plasma Phys. Control. Fusion 67 (2025) 055010, data on
Zenodo (doi:10.5281/zenodo.14996664) under CC BY 4.0.  ``lhdatabase.h5`` is a
MATLAB v7.3 file: one ``(1, N)`` dataset per variable, each with
``Description`` and ``Units`` attributes.  It is fetched on demand and verified
against a pinned SHA-256; it is not shipped with VAFT.

What the file says, and what it does not
-----------------------------------------
* ``ILH`` flags whether the L-H transition was observed.  ``ILH = 0`` rows are
  L-mode records at which the transition was sought and not seen; they become
  rows with ``transition_observed = False``, not a different event.
* ``TIME`` is the L-H transition time.  ``THL`` is labelled "time of H-L
  transition" but is populated on ``ILH = 0`` rows too and takes a handful of
  window-end values (2.50 s on 53 of 92 rows), so it is **not** turned into an
  ``H_to_L`` event.
* ``PLH`` is the Martin 2008 scaling computed by the source (it matches
  ``0.0488 n20^0.717 B^0.803 S^0.941`` to 0.4 %, without isotope correction).
  VAFT has no L-H threshold formula yet (#670, #1066), so the source value is
  carried with that provenance rather than recomputed.
* ``nRyter`` is the Ryter (2014) density of minimum threshold power, also
  computed by the source; the density branch is assigned against it.
* The file gives no magnetic configuration, so ``divertor_configuration`` is
  missing; ``BAFFLES`` is carried as ``divertor_closure``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ._fetch import SOURCES, fetch_source
from .schema import make_record_id, validate_transition_table

__all__ = [
    "TCV_LH_DEFINITIONS",
    "normalize_tcv_lh",
    "read_tcv_lh",
]

_SOURCE = SOURCES["tcv_lh_2025"]

#: Main ion from (mass number A, charge number Z).
_MAIN_ION = {(1, 1): "H", (2, 1): "D", (3, 1): "T", (3, 2): "He3", (4, 2): "He"}

TCV_LH_DEFINITIONS: dict[str, str] = {
    "p_loss_definition": (
        "TCV PLMW, 'experimental loss power' (source definition; on this file it "
        "agrees on average with coupled power PTOTMW_A - dW_MHD/dt); radiation "
        "NOT subtracted"
    ),
    "b_t_definition": "TCV BT, 'toroidal magnetic field' (reference radius not stated in the file)",
    "n_e_definition": "TCV NEL, line-averaged density (interferometer)",
    "isotope_definition": (
        "main ion from the source's A and Z; hydrogen fraction cH from CNPA; "
        "helium fraction cHe as given by the source"
    ),
    "density_branch_definition": (
        "n_e_min_m3 = source nRyter (Ryter 2014 minimum-threshold density, "
        "source-computed); 'low' if n_e_line_avg_m3 < n_e_min_m3 else 'high'"
    ),
    "p_lh_scaling_definition": (
        "source PLH: Martin 2008 ITPA scaling 0.0488 n20^0.717 B^0.803 S^0.941 "
        "[MW], source-computed, no isotope correction"
    ),
}

_REQUIRED = (
    "SHOT", "TIME", "ILH", "PLMW", "PRAD", "IP", "BT", "NEL", "SPLASMA", "RGEO",
    "AMIN", "KAPPA", "DELTA", "Q95", "ZEFF", "A", "Z", "cH", "cHe", "BAFFLES",
    "nRyter", "PLH",
)


def read_tcv_lh(
    path: str | os.PathLike[str] | None = None,
    *,
    cache: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Read the TCV ``lhdatabase.h5`` with its original variable names.

    Parameters
    ----------
    path : path-like or None, optional
        Local copy of ``lhdatabase.h5``; default ``None`` fetches the pinned
        Zenodo release into the cache (network on first use) [path].
    cache : path-like or None, optional
        Cache directory for the fetch, default ``None`` for the per-user cache
        [path].

    Returns
    -------
    pandas.DataFrame
        One row per record, source variable names and source units; the
        per-variable units and descriptions are in ``attrs["units"]`` and
        ``attrs["descriptions"]`` [table].

    Raises
    ------
    vaft.data.public.FetchError
        The file had to be fetched and the upstream was unreachable.
    ValueError
        The file lacks the variables this module relies on, or they do not
        share one length.
    """
    import h5py

    if path is None:
        path = fetch_source(_SOURCE.key, cache=cache)
    columns, units, descriptions = {}, {}, {}
    with h5py.File(Path(path), "r") as handle:
        for name, item in handle.items():
            if not isinstance(item, h5py.Dataset):
                continue
            columns[name] = np.ravel(item[()])
            units[name] = _attribute_text(item.attrs.get("Units"))
            descriptions[name] = _attribute_text(item.attrs.get("Description"))
    missing = [name for name in _REQUIRED if name not in columns]
    if missing:
        raise ValueError(f"{path} is not the TCV L-H database; missing variables {missing}")
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError(f"{path}: variables have different lengths {sorted(lengths)}")
    raw = pd.DataFrame(columns)
    raw.attrs["units"] = units
    raw.attrs["descriptions"] = descriptions
    return raw


def _attribute_text(value) -> str:
    """MATLAB string attribute as text; absent or empty (``h5py.Empty``) is ``""``."""
    if isinstance(value, bytes):  # includes numpy.bytes_
        return value.decode("utf-8", "replace").strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def normalize_tcv_lh(raw: pd.DataFrame, *, release: str = _SOURCE.release) -> pd.DataFrame:
    """Map a raw TCV L-H table into the canonical transition table.

    Parameters
    ----------
    raw : pandas.DataFrame
        Output of :func:`read_tcv_lh` (source names and units) [table].
    release : str, optional
        Release label recorded in ``source_release``; default the pinned
        Zenodo record [str].

    Returns
    -------
    pandas.DataFrame
        Canonical transition table, one ``L_to_H`` row per source record, in
        source order [table].

    Notes
    -----
    Unit conversions: ``IP`` MA -> A, ``PLMW`` and ``PLH`` MW -> W, ``nRyter``
    1e20 m^-3 -> m^-3; ``PRAD`` W, ``NEL`` m^-3, ``SPLASMA`` m^2, ``RGEO`` and
    ``AMIN`` m are already SI.  ``IP`` and ``BT`` become magnitudes.  Missing
    source values stay ``NaN``.
    """
    def col(name: str) -> np.ndarray:
        return pd.to_numeric(raw[name], errors="coerce").to_numpy(float)

    shots = col("SHOT")
    times = col("TIME")
    ilh = col("ILH")
    mass = col("A")
    charge = col("Z")
    n_e = col("NEL")
    n_min = col("nRyter") * 1e20

    table = pd.DataFrame(index=raw.index)
    table["machine"] = "TCV"
    table["shot"] = pd.array(np.where(np.isfinite(shots), shots, np.nan), dtype="Float64").astype("Int64")
    table["time_s"] = times
    table["record_id"] = [make_record_id("TCV", s, t) for s, t in zip(shots, times)]
    table["transition"] = "L_to_H"
    table["source_regime"] = "L_mode"
    table["target_regime"] = "H_mode"
    if not np.all(np.isin(ilh[np.isfinite(ilh)], (0.0, 1.0))):
        raise ValueError("ILH must be 0 or 1")
    table["transition_observed"] = ilh == 1.0

    table["p_loss_W"] = col("PLMW") * 1e6
    table["p_rad_W"] = col("PRAD")
    table["i_p_A"] = np.abs(col("IP")) * 1e6
    table["b_t_T"] = np.abs(col("BT"))
    table["n_e_line_avg_m3"] = n_e
    table["surface_area_m2"] = col("SPLASMA")
    table["r_geo_m"] = col("RGEO")
    table["a_m"] = col("AMIN")
    table["kappa"] = col("KAPPA")
    table["delta"] = col("DELTA")
    table["q95"] = col("Q95")
    table["z_eff"] = col("ZEFF")
    table["main_ion"] = [
        _MAIN_ION.get((int(a), int(z))) if np.isfinite(a) and np.isfinite(z) else None
        for a, z in zip(mass, charge)
    ]
    table["main_ion_mass_amu"] = mass
    table["hydrogen_fraction"] = col("cH")
    table["helium_fraction"] = col("cHe")
    table["divertor_configuration"] = None
    baffles = col("BAFFLES")
    table["divertor_closure"] = [
        f"TCV BAFFLES={value:g}" if np.isfinite(value) else None for value in baffles
    ]
    table["n_e_min_m3"] = n_min
    branch_known = np.isfinite(n_e) & np.isfinite(n_min)
    table["density_branch"] = np.where(
        branch_known, np.where(n_e < n_min, "low", "high"), None
    )
    table["p_lh_scaling_W"] = col("PLH") * 1e6
    for column, text in TCV_LH_DEFINITIONS.items():
        table[column] = text
    table["source_database"] = _SOURCE.database
    table["source_release"] = release
    table["source_reference"] = f"{_SOURCE.reference}, doi:{_SOURCE.doi}"
    return validate_transition_table(table)
