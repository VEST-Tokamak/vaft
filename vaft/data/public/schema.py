"""Canonical tables shared by every public-database source.

* :data:`CONFINEMENT_COLUMNS` -- one row per confinement record, a
  steady-state window of one discharge (ITPA DB5.2.3, VEST summaries).
* :data:`TRANSITION_COLUMNS` -- one row per regime-transition record, with the
  event named explicitly (TCV L-H database, ITPA TC-26 metal-wall database).

Every analysis and plot downstream reads only these columns -- never a
source-specific name.

Rules
-----
* Quantities are strict SI (the units
  :func:`vaft.formula.confinement_time_from_engineering_parameters` takes);
  a missing label is null (``None`` / ``NaN``; test it with ``pandas.isna``).
* Missing in the source is ``NaN`` here.  Nothing is filled with a default.
* A quantity whose definition differs between sources (loss power, reference
  radius of the toroidal field, how ``W_th`` was measured, where the ion mass
  came from) carries a ``*_definition`` / ``*_source`` string on every row, so
  a population mixing sources can be split on it.
* Rows are identified by ``record_id`` (``"<machine>:<shot>:<time key>"``),
  not by position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CONFINEMENT_COLUMNS",
    "ColumnSpec",
    "TRANSITION_COLUMNS",
    "empty_confinement_table",
    "empty_transition_table",
    "make_record_id",
    "validate_confinement_table",
    "validate_transition_table",
]


@dataclass(frozen=True)
class ColumnSpec:
    """Unit and meaning of one canonical column.

    Attributes
    ----------
    unit : str
        SI unit, ``"1"`` for dimensionless, ``"str"`` / ``"bool"`` / ``"int"``
        for labels.
    description : str
        What the column holds.
    """

    unit: str
    description: str


CONFINEMENT_COLUMNS: dict[str, ColumnSpec] = {
    # identity
    "machine": ColumnSpec("str", "Canonical machine name, upper case (e.g. 'JET', 'VEST')."),
    "record_id": ColumnSpec("str", "Unique '<machine>:<shot>:<time key>' identifier; the time key is the source's own (DB5 TIME_ID) or milliseconds."),
    "shot": ColumnSpec("int", "Discharge number."),
    "time_s": ColumnSpec("s", "Time of the record within the discharge."),
    "regime": ColumnSpec("str", "Confinement phase as labelled by the source (e.g. 'H', 'HGELM', 'OHM', 'L')."),
    # engineering parameters
    "i_p_A": ColumnSpec("A", "Plasma current magnitude."),
    "b_t_T": ColumnSpec("T", "Toroidal field magnitude at r_geo_m (see b_t_definition)."),
    "n_e_line_avg_m3": ColumnSpec("m^-3", "Line-averaged electron density."),
    "p_loss_W": ColumnSpec("W", "Loss power (see p_loss_definition)."),
    "w_th_J": ColumnSpec("J", "Thermal stored energy (see w_th_definition)."),
    "tau_e_th_s": ColumnSpec("s", "Thermal energy confinement time (see tau_e_definition)."),
    "r_geo_m": ColumnSpec("m", "Geometric major radius of the last closed flux surface."),
    "a_m": ColumnSpec("m", "Minor radius of the last closed flux surface."),
    "epsilon": ColumnSpec("1", "Inverse aspect ratio a_m / r_geo_m."),
    "kappa": ColumnSpec("1", "Boundary (height / width) elongation."),
    "kappa_area": ColumnSpec("1", "Volume elongation V / (2 pi^2 a^2 R), the kappa of IPB98(y,2)."),
    "delta": ColumnSpec("1", "Average triangularity of the boundary."),
    "m_eff_amu": ColumnSpec("amu", "Effective ion mass (see m_eff_source)."),
    # definitions and provenance
    "n_e_definition": ColumnSpec("str", "Which chord or construction n_e_line_avg_m3 is."),
    "p_loss_definition": ColumnSpec("str", "How p_loss_W was formed, in the source's terms."),
    "w_th_definition": ColumnSpec("str", "How w_th_J was obtained."),
    "tau_e_definition": ColumnSpec("str", "How tau_e_th_s was formed."),
    "b_t_definition": ColumnSpec("str", "Which field b_t_T is and at which radius."),
    "m_eff_source": ColumnSpec("str", "Where m_eff_amu came from ('source' or 'user-specified')."),
    "selected": ColumnSpec("bool", "Source's own standard-dataset flag (DB5: SELDB5 == 1); False where the source has none."),
    "source_database": ColumnSpec("str", "Database or VAFT layer the row came from."),
    "source_release": ColumnSpec("str", "Release / version of that source."),
    "source_reference": ColumnSpec("str", "Citation or DOI for the source."),
}

#: One row per regime-transition record (#1205 section 5, #1066 section 17).
#: The event is explicit -- ``transition``, ``source_regime``,
#: ``target_regime`` -- and never hidden in an unrelated IDS.  A record where
#: the source looked for the transition and did not see it keeps the same
#: event columns with ``transition_observed = False``.
TRANSITION_COLUMNS: dict[str, ColumnSpec] = {
    # identity
    "machine": ColumnSpec("str", "Canonical machine name, upper case (e.g. 'TCV')."),
    "record_id": ColumnSpec("str", "Unique '<machine>:<shot>:<time_ms>' identifier, time_ms from time_s."),
    "shot": ColumnSpec("int", "Discharge number."),
    "time_s": ColumnSpec("s", "Time the record's conditions were taken at: normally L-mode just before the transition (TC-26), or the transition time itself (TCV); a source may place it after transition_time_s, kept as given."),
    "transition_time_s": ColumnSpec("s", "Time of the transition itself; missing when it was not observed or not given."),
    # event semantics
    "transition": ColumnSpec("str", "Event name, e.g. 'L_to_H'.  'H_to_L' is a different event, not its inverse."),
    "source_regime": ColumnSpec("str", "Regime before the event, e.g. 'L_mode'."),
    "target_regime": ColumnSpec("str", "Regime after the event, e.g. 'H_mode'."),
    "transition_observed": ColumnSpec("bool", "True: the event happened (at transition_time_s); False: sought but not observed at these conditions."),
    "source_phase": ColumnSpec("str", "The source's own phase / event label, verbatim (e.g. TC-26 'LH', 'LHL'; TCV 'ILH=1')."),
    "selected": ColumnSpec("bool", "Source's own standard-set flag (TC-26: SELEC2024 == 1); False where the source has none."),
    # conditions at the event
    "p_loss_W": ColumnSpec("W", "Loss power at the event (see p_loss_definition)."),
    "p_rad_W": ColumnSpec("W", "Radiated power at the event (see p_rad_definition: total or core)."),
    "i_p_A": ColumnSpec("A", "Plasma current magnitude."),
    "b_t_T": ColumnSpec("T", "Toroidal field magnitude (see b_t_definition)."),
    "n_e_line_avg_m3": ColumnSpec("m^-3", "Line-averaged electron density (see n_e_definition)."),
    "surface_area_m2": ColumnSpec("m^2", "Surface area of the last closed flux surface."),
    "r_geo_m": ColumnSpec("m", "Geometric major radius."),
    "a_m": ColumnSpec("m", "Minor radius."),
    "kappa": ColumnSpec("1", "Elongation."),
    "delta": ColumnSpec("1", "Average triangularity."),
    "q95": ColumnSpec("1", "Safety factor at 95 % poloidal flux."),
    "z_eff": ColumnSpec("1", "Effective charge."),
    "main_ion": ColumnSpec("str", "Fuelling / main ion species the source assigns, 'H', 'D', 'He' ...; not a purity statement (see hydrogenic_mix); missing when the source does not say."),
    "main_ion_mass_amu": ColumnSpec("amu", "Main ion mass number."),
    "hydrogen_fraction": ColumnSpec("1", "Hydrogen concentration of the hydrogenic ions, as the source measures it (see isotope_definition)."),
    "helium_fraction": ColumnSpec("1", "Helium concentration estimate as the source gives it (see isotope_definition)."),
    "hydrogenic_mix": ColumnSpec("str", "Hydrogenic mixture class, e.g. 'D-dominated', 'H-dominated', 'mixed H/D' (TCV, from cH) or 'M_eff 1-2' / 'M_eff 2-3' (TC-26, from mass); missing otherwise (see isotope_definition)."),
    "divertor_configuration": ColumnSpec("str", "Magnetic configuration (e.g. 'LSN', 'USN', 'DN', 'limited'); missing when the source does not say."),
    "divertor_closure": ColumnSpec("str", "Divertor geometry / closure / baffling as the source labels it."),
    "grad_b_drift": ColumnSpec("str", "Ion grad-B drift direction: 'toward_x_point' or 'away_from_x_point'; missing when not given."),
    "first_wall": ColumnSpec("str", "Plasma-facing materials as the source labels them (main chamber / divertor)."),
    "auxiliary_heating": ColumnSpec("str", "Auxiliary heating method(s) as the source labels them (e.g. 'NB', 'IC', 'NBEC')."),
    "density_branch": ColumnSpec("str", "'low' or 'high' relative to n_e_min_m3; missing without n_e_min_m3."),
    "n_e_min_m3": ColumnSpec("m^-3", "Density of minimum threshold power (see density_branch_definition)."),
    "p_lh_scaling_W": ColumnSpec("W", "Scaling-law threshold power for this record (see p_lh_scaling_definition)."),
    # definitions and provenance
    "p_loss_definition": ColumnSpec("str", "How p_loss_W was formed, in the source's terms."),
    "p_rad_definition": ColumnSpec("str", "Which radiated power p_rad_W is."),
    "b_t_definition": ColumnSpec("str", "Which field b_t_T is."),
    "n_e_definition": ColumnSpec("str", "Which chord or construction n_e_line_avg_m3 is."),
    "isotope_definition": ColumnSpec("str", "How main ion and concentrations were determined."),
    "density_branch_definition": ColumnSpec("str", "Where n_e_min_m3 came from and how the branch was assigned."),
    "p_lh_scaling_definition": ColumnSpec("str", "Which scaling p_lh_scaling_W is and who computed it."),
    "source_database": ColumnSpec("str", "Database the row came from."),
    "source_release": ColumnSpec("str", "Release / version of that source."),
    "source_reference": ColumnSpec("str", "Citation or DOI for the source."),
}

_LABEL_UNITS = {"str", "bool", "int"}
#: Signed quantities, exempt from the magnitude check.
_SIGNED = {"time_s", "delta"}


def make_record_id(machine: str, shot, time_s) -> str:
    """Canonical record identifier.

    Parameters
    ----------
    machine : str
        Canonical machine name [str].
    shot : int
        Discharge number [int].
    time_s : float
        Time of the record [s].

    Returns
    -------
    str
        ``"<machine>:<shot>:<time_ms>"`` with the time rounded to the
        millisecond [str].
    """
    return f"{machine}:{int(shot)}:{int(round(float(time_s) * 1000.0))}"


def _empty(columns: dict[str, ColumnSpec]) -> pd.DataFrame:
    data = {}
    for name, spec in columns.items():
        if spec.unit == "str":
            data[name] = pd.Series([], dtype=object)
        elif spec.unit == "bool":
            data[name] = pd.Series([], dtype=bool)
        elif spec.unit == "int":
            data[name] = pd.Series([], dtype="Int64")
        else:
            data[name] = pd.Series([], dtype=float)
    return _attach_attrs(pd.DataFrame(data), columns)


def _attach_attrs(table: pd.DataFrame, columns: dict[str, ColumnSpec]) -> pd.DataFrame:
    table.attrs["units"] = {k: v.unit for k, v in columns.items()}
    table.attrs["descriptions"] = {k: v.description for k, v in columns.items()}
    return table


def _validate(table: pd.DataFrame, columns: dict[str, ColumnSpec], kind: str) -> pd.DataFrame:
    missing = [name for name in columns if name not in table.columns]
    if missing:
        raise ValueError(f"{kind} table is missing columns: {missing}")
    duplicated = table["record_id"][table["record_id"].duplicated()]
    if len(duplicated):
        raise ValueError(f"record_id is not unique: {duplicated.unique()[:5].tolist()}")
    for name, spec in columns.items():
        if spec.unit in _LABEL_UNITS:
            continue
        values = pd.to_numeric(table[name], errors="raise").to_numpy(dtype=float)
        if np.any(np.isinf(values)):
            raise ValueError(f"Column {name!r} holds infinite values")
        if name not in _SIGNED and np.any(values[np.isfinite(values)] < 0.0):
            raise ValueError(f"Column {name!r} holds negative values; magnitudes are expected")
    out = table.loc[:, list(columns)].reset_index(drop=True)
    return _attach_attrs(out, columns)


def empty_confinement_table() -> pd.DataFrame:
    """An empty canonical confinement table with the right columns and dtypes.

    Returns
    -------
    pandas.DataFrame
        Zero rows, every column of :data:`CONFINEMENT_COLUMNS` [table].
    """
    return _empty(CONFINEMENT_COLUMNS)


def empty_transition_table() -> pd.DataFrame:
    """An empty canonical transition table with the right columns and dtypes.

    Returns
    -------
    pandas.DataFrame
        Zero rows, every column of :data:`TRANSITION_COLUMNS` [table].
    """
    return _empty(TRANSITION_COLUMNS)


def validate_confinement_table(table: pd.DataFrame) -> pd.DataFrame:
    """Check a table against the canonical schema and return it column-ordered.

    Parameters
    ----------
    table : pandas.DataFrame
        Candidate confinement table [table].

    Returns
    -------
    pandas.DataFrame
        The same rows with exactly the canonical columns, in canonical order,
        and the unit/description metadata in ``attrs`` [table].

    Raises
    ------
    ValueError
        A canonical column is missing, ``record_id`` is not unique, a numeric
        column holds an infinite value, or a magnitude column (all numeric
        columns except ``time_s`` and ``delta``) holds a negative one.
    """
    return _validate(table, CONFINEMENT_COLUMNS, "Confinement")


def validate_transition_table(table: pd.DataFrame) -> pd.DataFrame:
    """Check a table against the canonical transition schema.

    Parameters
    ----------
    table : pandas.DataFrame
        Candidate transition table [table].

    Returns
    -------
    pandas.DataFrame
        The same rows with exactly the columns of :data:`TRANSITION_COLUMNS`,
        in order, with unit/description metadata in ``attrs`` [table].

    Raises
    ------
    ValueError
        As :func:`validate_confinement_table`, and when ``transition`` is
        missing on a row -- an event record without its event is not a record.
    """
    out = _validate(table, TRANSITION_COLUMNS, "Transition")
    if out["transition"].isna().any() or (out["transition"].astype(str) == "").any():
        raise ValueError("Every transition record must name its transition")
    return out
