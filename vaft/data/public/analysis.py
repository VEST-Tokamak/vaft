"""Scaling predictions and coverage on the canonical confinement table.

These functions evaluate the published scalings already in
:func:`vaft.formula.confinement_time_from_engineering_parameters` row-wise
over a canonical table; they add no scaling of their own.  A row with any
input the chosen scaling needs missing or non-positive gets ``NaN``, never a
substituted value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "SCALING_INPUTS",
    "confinement_coverage",
    "h_factor",
    "predict_confinement_time",
    "transition_margin",
]

#: Exponent variable of ``_SCALING_COEFS`` -> (formula keyword, canonical column).
#: ``kappa`` is resolved from the ``kappa_column`` argument.
SCALING_INPUTS: dict[str, tuple[str, str]] = {
    "Ip_MA": ("I_p", "i_p_A"),
    "Bt": ("B_t", "b_t_T"),
    "P_MW": ("P_loss", "p_loss_W"),
    "n_19": ("n_e", "n_e_line_avg_m3"),
    "Mi": ("M", "m_eff_amu"),
    "R": ("R", "r_geo_m"),
    "epsilon": ("epsilon", "epsilon"),
    "kappa": ("kappa", None),
}


def _used_variables(scaling: str) -> list[str]:
    from vaft.formula.constants import _SCALING_COEFS

    if scaling not in _SCALING_COEFS:
        raise ValueError(f"Unknown scaling {scaling!r}. Available: {list(_SCALING_COEFS)}")
    coefs = _SCALING_COEFS[scaling]
    return list(coefs["exponents"]) if "exponents" in coefs else [
        key for key in SCALING_INPUTS if key in coefs
    ]


def predict_confinement_time(
    table: pd.DataFrame,
    scaling: str = "H98y2",
    *,
    kappa_column: str = "kappa_area",
) -> pd.Series:
    """Scaling-law thermal confinement time for every row of a confinement table.

    Parameters
    ----------
    table : pandas.DataFrame
        Canonical confinement table [table].
    scaling : str, optional
        Name understood by
        :func:`vaft.formula.confinement_time_from_engineering_parameters`;
        default ``"H98y2"`` (IPB98(y,2)) [str].
    kappa_column : str, optional
        Elongation column fed to the scaling; default ``"kappa_area"``, the
        volume elongation IPB98(y,2) was regressed on.  ``"kappa"`` (boundary
        elongation) is a different quantity and must be chosen explicitly [str].

    Returns
    -------
    pandas.Series
        Predicted thermal confinement time indexed like ``table``, ``NaN``
        where an input the scaling uses is missing or non-positive [s].

    Notes
    -----
    Density is the line average (``n_e_line_avg_m3``), which is what every
    scaling in ``_SCALING_COEFS`` declares, so no density conversion happens.
    """
    from vaft.formula import confinement_time_from_engineering_parameters

    used = _used_variables(scaling)
    columns = {
        name: (kappa_column if name == "kappa" else SCALING_INPUTS[name][1])
        for name in used
    }
    for column in columns.values():
        if column not in table.columns:
            raise KeyError(f"Column {column!r} needed by {scaling} is not in the table")

    values = {name: pd.to_numeric(table[col], errors="coerce").to_numpy(float)
              for name, col in columns.items()}
    valid = np.ones(len(table), dtype=bool)
    for array in values.values():
        valid &= np.isfinite(array) & (array > 0.0)

    result = np.full(len(table), np.nan)
    # The formula is scalar-valued; evaluate it once per complete row.
    for row in np.flatnonzero(valid):
        kwargs = {
            keyword: (float(values[name][row]) if name in values else 1.0)
            # 1.0 stands in for a variable this scaling does not use; the
            # formula neither checks nor raises it to any power.
            for name, (keyword, _) in SCALING_INPUTS.items()
        }
        result[row] = confinement_time_from_engineering_parameters(
            scaling=scaling, input_density_definition="line_avg", **kwargs
        )
    return pd.Series(result, index=table.index, name=f"tau_e_{scaling}_s")


def h_factor(
    table: pd.DataFrame,
    scaling: str = "H98y2",
    *,
    kappa_column: str = "kappa_area",
) -> pd.Series:
    """Confinement enhancement factor ``tau_e_th_s / tau_scaling`` per row.

    Parameters
    ----------
    table : pandas.DataFrame
        Canonical confinement table [table].
    scaling : str, optional
        Scaling name, default ``"H98y2"`` [str].
    kappa_column : str, optional
        Elongation column, default ``"kappa_area"`` [str].

    Returns
    -------
    pandas.Series
        H-factor indexed like ``table``; ``NaN`` where the measurement or the
        prediction is missing [-].
    """
    predicted = predict_confinement_time(table, scaling, kappa_column=kappa_column)
    measured = pd.to_numeric(table["tau_e_th_s"], errors="coerce")
    return (measured / predicted).rename(f"h_{scaling}")


def confinement_coverage(
    table: pd.DataFrame,
    columns: tuple[str, ...] = (
        "i_p_A", "b_t_T", "n_e_line_avg_m3", "p_loss_W", "tau_e_th_s",
        "r_geo_m", "epsilon", "kappa", "kappa_area", "delta", "m_eff_amu",
    ),
    *,
    by: str = "machine",
) -> pd.DataFrame:
    """Count finite values per column and group.

    Parameters
    ----------
    table : pandas.DataFrame
        Canonical confinement table [table].
    columns : tuple of str, optional
        Columns to count; default the engineering and confinement columns
        [str].
    by : str, optional
        Grouping column, default ``"machine"`` [str].

    Returns
    -------
    pandas.DataFrame
        Rows per group, a ``rows`` column with the group size and one column
        per requested quantity holding its finite count [count].
    """
    finite = table[list(columns)].apply(pd.to_numeric, errors="coerce").notna()
    finite[by] = table[by].to_numpy()
    counts = finite.groupby(by).sum()
    counts.insert(0, "rows", table.groupby(by).size())
    return counts


def transition_margin(table: pd.DataFrame) -> pd.Series:
    """Loss power over the record's scaling threshold, ``p_loss_W / p_lh_scaling_W``.

    Parameters
    ----------
    table : pandas.DataFrame
        Canonical transition table [table].

    Returns
    -------
    pandas.Series
        Margin indexed like ``table``; ``NaN`` where either power is missing
        or the threshold is not positive [-].

    Notes
    -----
    The threshold is whatever ``p_lh_scaling_W`` holds -- for TCV the
    source-computed Martin 2008 value (``p_lh_scaling_definition`` says so).
    VAFT evaluates no L-H scaling of its own until one exists in
    :mod:`vaft.formula` (#670, #1066).
    """
    loss = pd.to_numeric(table["p_loss_W"], errors="coerce")
    threshold = pd.to_numeric(table["p_lh_scaling_W"], errors="coerce")
    margin = loss / threshold.where(threshold > 0.0)
    return margin.rename("transition_margin")
