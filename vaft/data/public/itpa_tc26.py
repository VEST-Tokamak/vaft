"""ITPA TC-26 metal-wall L-H threshold database reader and normaliser.

Source: E. Delabie et al., 'Empirical scaling of the L-H threshold power for
metal wall tokamaks using a multi-device database', Nucl. Fusion 66 (2026)
036016 (open access, CC BY 4.0).  The database is the paper's auxiliary file
``nfae39f2supp2.csv`` with its definitions in ``nfae39f2supp1.pdf``
(dated 2025-12-09).

**Local path only.**  The IOP supplementary-data page is behind a bot
challenge, and whether the article licence covers the auxiliary file is not
stated in either file, so VAFT neither downloads nor ships it.  Download it
from the article page in a browser and pass the path.  :data:`TC26_SHA256` is
the hash of the release this reader was written against; a different file is
read, but reported.

What the file says, and what it does not
-----------------------------------------
* Every record is an L-H transition point: ``PHASE = 'LH'`` is L-mode just
  before the transition, taken at ``TIME``; ``LHTIME`` is the transition time.
  Other labels (``LHblip``, ``LHL``, ``LHLH``, ``LHST``, ``LHLHST``; AUG only)
  are carried verbatim in ``source_phase``.  On 14 rows ``LHTIME`` precedes
  ``TIME`` (by up to 0.877 s), 9 of them labelled plain ``'LH'``; there the
  conditions were taken after the recorded transition.  This is kept as given
  and visible as ``transition_time_s < time_s``.
* Missing values are written four ways: blank, ``-`` (``ZEFF``, ``PRADCORE``),
  ``-1e-08`` (``LHTIME``, ``FRACNMIN``) and ``0`` in quantities that cannot be
  zero -- ``ZEFF`` (all AUG rows, 25 C-Mod rows; Z_eff >= 1), ``PRADCORE``
  (29 AUG rows; the smallest AUG value otherwise is 38.8 kW) and ``FRACNMIN``
  (all JET rows).  All become missing.  ``PFLOSS = 0`` is a real zero (no
  beam) and is kept.
* ``PLTH`` is the loss power the TC-26 scalings use.  The definition file says
  ``PLTH = PL - PFLOSS``; that holds (to 0.1 %) on every JET and C-Mod row but
  not on 192 AUG rows (67 of them with ``PFLOSS = 0``), where ``PLTH`` differs
  by up to 49 % of itself -- 98 % of ``PL`` -- and exceeds ``PL`` on 74 rows,
  which a loss-corrected power cannot.  ``PLTH`` is carried as given, being
  what the paper's scalings were fitted on.
* ``PRADCORE`` is core radiation (to r/a = 0.95 for JET/AUG, 1 for C-Mod), not
  total radiation.
* ``PGASA`` is the effective fuel mass (1-3, fractional for mixtures, incl.
  JET T and DT).  ``main_ion`` is 'H', 'D' or 'T' within 0.05 of 1, 2, 3 and
  'mixed' otherwise; ``hydrogenic_mix`` then says from the mass alone which
  range it lies in ('M_eff 1-2' or 'M_eff 2-3') -- it cannot tell H/D from H/T.
* ``FRACNMIN`` is not defined in the definition file; it is not mapped.  The
  density branch is 'high' where ``SELEC2024 = 1`` ("selected for the TC-26
  high density branch scalings") and missing otherwise -- a row outside the
  selection is not thereby low-density.
* The file has no triangularity (``delta`` missing) and no threshold
  prediction; VAFT has no L-H scaling yet (#670, #1066), so
  ``p_lh_scaling_W`` is missing and so is the margin.
* All records are single null with the X-point at the bottom and the ion
  grad-B drift toward it (``CONFIG``, ``IGRADB = 1``).
* The release holds one exact duplicate row (JET 98969 at 50.135 s); exact
  duplicates are dropped, any other repeated key is an error.
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from ._fetch import sha256_of
from .schema import make_record_id, validate_transition_table

__all__ = [
    "TC26_DEFINITIONS",
    "TC26_REFERENCE",
    "TC26_SHA256",
    "normalize_tc26",
    "read_tc26",
]

#: SHA-256 of ``nfae39f2supp2.csv`` as downloaded 2026-09-29 (689 rows).
TC26_SHA256 = "b900e13fcb7973ae9b85f4df5264e65cc19ed3bef026076739331e3868878582"

TC26_REFERENCE = (
    "E. Delabie et al., 'Empirical scaling of the L-H threshold power for metal "
    "wall tokamaks using a multi-device database', Nucl. Fusion 66 (2026) 036016, "
    "doi:10.1088/1741-4326/ae39f2 (auxiliary file nfae39f2supp2.csv)"
)

_MISSING_TEXT = {"", "-"}
_SENTINEL = -1.0e-08
#: Quantities that are strictly positive physically; a non-positive value is
#: the release's "not measured".
_POSITIVE_ONLY = ("ZEFF", "PRADCORE", "FRACNMIN")
_STRING_COLUMNS = (
    "TOK", "PHASE", "CONFIG", "WALMAT", "DIVMAT", "LIMMAT", "EVAP", "DIVNAME",
    "DIVCON", "AUXHEAT",
)
_NUMERIC_COLUMNS = (
    "SHOT", "TIME", "PGASA", "RGEO", "AMIN", "KAPPA", "SPLASMA", "IGRADB", "BT",
    "IP", "Q95", "NEL", "ZEFF", "PL", "PLTH", "PFLOSS", "PRADCORE", "FRACNMIN",
    "LHTIME", "SELEC2024",
)

TC26_DEFINITIONS: dict[str, str] = {
    "p_loss_definition": (
        "TC-26 PLTH, loss power corrected for charge-exchange and unconfined-orbit "
        "losses, as given (stated PLTH = PL - PFLOSS fails on 192 AUG rows, PLTH > PL "
        "on 74); radiation NOT subtracted"
    ),
    "p_rad_definition": "TC-26 PRADCORE: CORE radiation from bolometry (r/a < 0.95 JET/AUG, < 1 C-Mod), not total",
    "b_t_definition": "TC-26 BT: vacuum toroidal field at RGEO",
    "n_e_definition": "TC-26 NEL: line-averaged density from a core interferometer chord",
    "isotope_definition": (
        "main_ion from TC-26 PGASA (effective fuel mass): 'H'/'D'/'T' within 0.05 "
        "of 1/2/3, else 'mixed' with hydrogenic_mix 'M_eff 1-2' or 'M_eff 2-3' "
        "(mass only); main_ion_mass_amu = PGASA; no concentrations given"
    ),
    "density_branch_definition": (
        "'high' where TC-26 SELEC2024 = 1 (selected for the high-density-branch "
        "scalings), missing otherwise; no n_min given (FRACNMIN is undefined)"
    ),
    "p_lh_scaling_definition": "not given by TC-26; VAFT has no L-H scaling yet (#670, #1066)",
}


def read_tc26(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read the TC-26 CSV with its original column names, missing markers resolved.

    Parameters
    ----------
    path : path-like
        Local copy of ``nfae39f2supp2.csv`` downloaded from the article page;
        there is no automatic download [path].

    Returns
    -------
    pandas.DataFrame
        One row per source row, source names and units (SI), strings stripped,
        blank / ``-`` / ``-1e-08`` and non-positive ``ZEFF``, ``PRADCORE``,
        ``FRACNMIN`` as missing.  ``attrs["sha256"]`` holds the file hash
        [table].

    Raises
    ------
    ValueError
        The file lacks the TC-26 columns.

    Warns
    -----
    UserWarning
        The file's hash differs from :data:`TC26_SHA256` (another release).
    """
    path = Path(path)
    digest = sha256_of(path)
    if digest != TC26_SHA256:
        warnings.warn(
            f"{path.name} has SHA-256 {digest}, not the release this reader was "
            f"written against ({TC26_SHA256}); check the definitions still hold.",
            stacklevel=2,
        )
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in (*_STRING_COLUMNS, *_NUMERIC_COLUMNS) if c not in raw.columns]
    if missing:
        raise ValueError(f"{path} is not the TC-26 database; missing columns {missing}")
    for column in raw.columns:
        raw[column] = raw[column].str.strip()
    for column in _STRING_COLUMNS:
        raw[column] = raw[column].where(~raw[column].isin(_MISSING_TEXT), None)
    for column in _NUMERIC_COLUMNS:
        text = raw[column].where(~raw[column].isin(_MISSING_TEXT))
        values = pd.to_numeric(text, errors="raise").astype(float)
        values = values.where(values != _SENTINEL)
        if column in _POSITIVE_ONLY:
            values = values.where(~(values <= 0.0))
        raw[column] = values
    raw.attrs["sha256"] = digest
    return raw


def _main_ion(mass: float):
    if not np.isfinite(mass):
        return None
    for label, number in (("H", 1.0), ("D", 2.0), ("T", 3.0)):
        # Rounded first so 1.05, 1.95 and 2.95 are all inside the band.
        if round(abs(mass - number), 6) <= 0.05:
            return label
    return "mixed"


def _mass_range(mass: float, label):
    if label != "mixed":
        return None
    return "M_eff 1-2" if mass < 2.0 else "M_eff 2-3"


def normalize_tc26(raw: pd.DataFrame, *, release: str | None = None) -> pd.DataFrame:
    """Map a raw TC-26 table into the canonical transition table.

    Parameters
    ----------
    raw : pandas.DataFrame
        Output of :func:`read_tc26` [table].
    release : str or None, optional
        Release label for ``source_release``; default ``None`` records the
        file hash [str].

    Returns
    -------
    pandas.DataFrame
        Canonical transition table, one ``L_to_H`` row per distinct source
        record, in source order [table].

    Raises
    ------
    ValueError
        Two different records share machine, shot and time.

    Notes
    -----
    The file is SI already (A, T, m^-3, W, m, m^2, s); ``IP`` and ``BT``
    become magnitudes.  ``time_s = TIME`` (conditions, L-mode just before the
    transition) and ``transition_time_s = LHTIME``.
    """
    digest = raw.attrs.get("sha256", "unknown")
    raw = raw.drop_duplicates(ignore_index=True)

    def col(name: str) -> np.ndarray:
        return raw[name].to_numpy(dtype=float)

    machines = raw["TOK"].str.upper().to_numpy()
    shots, times = col("SHOT"), col("TIME")
    table = pd.DataFrame(index=raw.index)
    table["machine"] = machines
    table["shot"] = pd.array(shots.astype(np.int64), dtype="Int64")
    table["time_s"] = times
    table["transition_time_s"] = col("LHTIME")
    table["record_id"] = [make_record_id(m, s, t) for m, s, t in zip(machines, shots, times)]
    if table["record_id"].duplicated().any():
        repeated = table["record_id"][table["record_id"].duplicated()].tolist()
        raise ValueError(f"Different TC-26 records share a key: {repeated[:5]}")
    table["transition"] = "L_to_H"
    table["source_regime"] = "L_mode"
    table["target_regime"] = "H_mode"
    table["transition_observed"] = True
    table["source_phase"] = raw["PHASE"].to_numpy()
    table["selected"] = col("SELEC2024") == 1.0

    table["p_loss_W"] = col("PLTH")
    table["p_rad_W"] = col("PRADCORE")
    table["i_p_A"] = np.abs(col("IP"))
    table["b_t_T"] = np.abs(col("BT"))
    table["n_e_line_avg_m3"] = col("NEL")
    table["surface_area_m2"] = col("SPLASMA")
    table["r_geo_m"] = col("RGEO")
    table["a_m"] = col("AMIN")
    table["kappa"] = col("KAPPA")
    table["delta"] = np.nan
    table["q95"] = col("Q95")
    table["z_eff"] = col("ZEFF")
    mass = col("PGASA")
    table["main_ion"] = [_main_ion(m) for m in mass]
    table["main_ion_mass_amu"] = mass
    table["hydrogen_fraction"] = np.nan
    table["helium_fraction"] = np.nan
    table["hydrogenic_mix"] = [_mass_range(m, label) for m, label in zip(mass, table["main_ion"])]
    table["divertor_configuration"] = "LSN"
    table["divertor_closure"] = [
        " ".join(part for part in (name, config) if part) or None
        for name, config in zip(raw["DIVNAME"].fillna(""), raw["DIVCON"].fillna(""))
    ]
    igradb = col("IGRADB")
    table["grad_b_drift"] = np.where(
        igradb == 1.0, "toward_x_point", np.where(np.isfinite(igradb), "away_from_x_point", None)
    )
    table["first_wall"] = [
        f"main={wall}, divertor={divertor}" for wall, divertor in zip(raw["WALMAT"], raw["DIVMAT"])
    ]
    table["auxiliary_heating"] = raw["AUXHEAT"].to_numpy()
    selected = table["selected"].to_numpy()
    table["density_branch"] = np.where(selected, "high", None)
    table["n_e_min_m3"] = np.nan
    table["p_lh_scaling_W"] = np.nan
    for column, text in TC26_DEFINITIONS.items():
        table[column] = text
    table["source_database"] = "ITPA TC-26 metal-wall L-H threshold database"
    table["source_release"] = release or f"nfae39f2supp2.csv sha256:{digest[:12]}"
    table["source_reference"] = TC26_REFERENCE
    return validate_transition_table(table)
