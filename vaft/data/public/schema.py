"""Canonical confinement-state table shared by every confinement source.

One row is one confinement record: a steady-state time window of one discharge
of one machine.  External databases (ITPA DB5.2.3) and VEST summaries are both
normalised into this table, and every analysis and plot downstream reads only
these columns -- never a source-specific name.

Rules
-----
* Quantities are strict SI, the units
  :func:`vaft.formula.confinement_time_from_engineering_parameters` takes.
* Missing in the source is ``NaN`` here.  Nothing is filled with a default.
* A quantity whose definition differs between sources (loss power, reference
  radius of the toroidal field, how ``W_th`` was measured, where the ion mass
  came from) carries a ``*_definition`` / ``*_source`` string on every row, so
  a population mixing sources can be split on it.
* Rows are identified by ``record_id`` (``"<machine>:<shot>:<time_ms>"``), not
  by position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CONFINEMENT_COLUMNS",
    "ColumnSpec",
    "empty_confinement_table",
    "make_record_id",
    "validate_confinement_table",
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
    "record_id": ColumnSpec("str", "Unique '<machine>:<shot>:<time_ms>' identifier."),
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

_LABEL_UNITS = {"str", "bool", "int"}


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


def empty_confinement_table() -> pd.DataFrame:
    """An empty canonical confinement table with the right columns and dtypes.

    Returns
    -------
    pandas.DataFrame
        Zero rows, every column of :data:`CONFINEMENT_COLUMNS` [table].
    """
    data = {}
    for name, spec in CONFINEMENT_COLUMNS.items():
        if spec.unit == "str":
            data[name] = pd.Series([], dtype=object)
        elif spec.unit == "bool":
            data[name] = pd.Series([], dtype=bool)
        elif spec.unit == "int":
            data[name] = pd.Series([], dtype="Int64")
        else:
            data[name] = pd.Series([], dtype=float)
    return _attach_attrs(pd.DataFrame(data))


def _attach_attrs(table: pd.DataFrame) -> pd.DataFrame:
    table.attrs["units"] = {k: v.unit for k, v in CONFINEMENT_COLUMNS.items()}
    table.attrs["descriptions"] = {k: v.description for k, v in CONFINEMENT_COLUMNS.items()}
    return table


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
        A canonical column is missing, ``record_id`` is not unique, or a
        numeric column holds a negative or infinite value.
    """
    missing = [name for name in CONFINEMENT_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"Confinement table is missing columns: {missing}")
    duplicated = table["record_id"][table["record_id"].duplicated()]
    if len(duplicated):
        raise ValueError(f"record_id is not unique: {duplicated.unique()[:5].tolist()}")
    for name, spec in CONFINEMENT_COLUMNS.items():
        if spec.unit in _LABEL_UNITS or name == "time_s":
            continue
        values = pd.to_numeric(table[name], errors="raise").to_numpy(dtype=float)
        if np.any(np.isinf(values)):
            raise ValueError(f"Column {name!r} holds infinite values")
        if np.any(values[np.isfinite(values)] < 0.0):
            raise ValueError(f"Column {name!r} holds negative values; magnitudes are expected")
    out = table.loc[:, list(CONFINEMENT_COLUMNS)].reset_index(drop=True)
    return _attach_attrs(out)
