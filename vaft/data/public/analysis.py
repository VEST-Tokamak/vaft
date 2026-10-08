"""Scaling predictions and coverage on the canonical confinement table.

These functions evaluate the published scalings already in
:func:`vaft.formula.confinement_time_from_engineering_parameters` row-wise
over a canonical table; they add no scaling of their own.  A row with any
input the chosen scaling needs missing or non-positive gets ``NaN``, never a
substituted value.
"""

from __future__ import annotations

import warnings

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

#: What each elongation column of the canonical table is, for the
#: ``kappa_definition`` provenance of a prediction (the ITER97-L and IPB98
#: sources do not settle which one they regressed on: ``elongation_note`` in
#: :data:`vaft.formula.constants._SCALING_COEFS`).
KAPPA_DEFINITIONS: dict[str, str] = {
    "kappa_area": "kappa_area: area elongation, cross-section area / (pi a^2)",
    "kappa": "kappa: boundary (LCFS) elongation b/a",
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
        ``attrs["kappa_definition"]`` states which elongation column the
        scaling was fed (or that it uses none).

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
    out = pd.Series(result, index=table.index, name=f"tau_e_{scaling}_s")
    out.attrs["kappa_definition"] = (KAPPA_DEFINITIONS.get(kappa_column, f"{kappa_column}: column as supplied")
                                     if "kappa" in used else "no elongation term")
    return out


def h_factor(
    table: pd.DataFrame,
    scaling: str = "H98y2",
    *,
    kappa_column: str = "kappa_area",
    thermal_as_global=False,
) -> pd.Series:
    """Confinement enhancement factor ``tau_observed / tau_scaling`` per row, on the scaling's energy basis.

    Parameters
    ----------
    table : pandas.DataFrame
        Canonical confinement table [table].
    scaling : str, optional
        Scaling name, default ``"H98y2"`` [str].
    kappa_column : str, optional
        Elongation column, default ``"kappa_area"`` [str].
    thermal_as_global : bool or collection of str, optional
        Where a global (or unaudited-basis) scaling finds no ``tau_e_global_s``,
        use ``tau_e_th_s`` instead: ``True`` on every row, a collection of
        machine names on those machines' rows only (e.g. ``{"VEST"}``, an
        ohmic machine without fast ions; one name may be given as a string);
        default ``False``, strict [-].

    Returns
    -------
    pandas.Series
        H-factor indexed like ``table``; ``NaN`` where the measurement or the
        prediction is missing, or where the scaling needs a global confinement
        time the row does not have.  ``attrs`` records ``energy_basis``,
        ``kappa_definition`` (the elongation column the prediction used) and,
        when thermal times stood in for global ones, ``approximation`` and
        ``substituted_rows`` [-].

    Raises
    ------
    ValueError
        ``thermal_as_global`` names machines and the table has no ``machine``
        column.
    TypeError
        ``thermal_as_global`` is a number that is not a bool (``numpy.bool_``
        is accepted as a bool).

    Warns
    -----
    UserWarning
        A global or unaudited scaling (ITER89-P, NSTX 2006 L, Kurskiev 2022,
        the Goldston 1984 forms) on rows that have ``tau_e_th_s`` but no
        ``tau_e_global_s`` while ``thermal_as_global`` does not cover them:
        those rows are ``NaN`` by the audit gate of #1713, and the warning
        names the scaling and the way out. Thermal scalings never warn.

    Notes
    -----
    A thermal scaling (IPB98(y,2), ITER97-L, NSTX 2006 H) is compared with
    ``tau_e_th_s``, a global one (ITER89-P, NSTX 2006 L) with
    ``tau_e_global_s`` (#1713): the basis comes from
    :func:`vaft.formula.equilibrium.confinement_scaling_basis` and the
    resolution from :func:`vaft.process.confinement.resolve_observed_confinement`.
    For thermal scalings the result is unchanged from before #1713.  The gate
    is deliberate: an H factor formed from a thermal time against a global
    scaling is not the conventional one unless the fast-ion energy is
    negligible, which only the caller can assert (``thermal_as_global``).
    """
    from vaft.formula.equilibrium import confinement_scaling_basis
    from vaft.process.confinement import resolve_observed_confinement

    predicted = predict_confinement_time(table, scaling, kappa_column=kappa_column)
    basis = confinement_scaling_basis(scaling).energy_basis
    thermal = pd.to_numeric(table["tau_e_th_s"], errors="coerce").to_numpy(float)
    glob = (pd.to_numeric(table["tau_e_global_s"], errors="coerce").to_numpy(float)
            if "tau_e_global_s" in table else None)
    strict = resolve_observed_confinement(thermal, basis, tau_global=glob)
    relaxed = resolve_observed_confinement(thermal, basis, tau_global=glob, thermal_as_global=True)
    if isinstance(thermal_as_global, (bool, np.bool_)):
        # A flag derived from data (``(table.machine == "VEST").all()``) is a
        # numpy bool, not a Python one; both mean every row.
        allow = np.full(len(table), bool(thermal_as_global))
    elif isinstance(thermal_as_global, (int, float, np.number)):
        raise TypeError("thermal_as_global is a bool or a collection of machine names, "
                        f"not {type(thermal_as_global).__name__}")
    else:
        # One machine name is one name, not a set of its letters.
        names = {thermal_as_global} if isinstance(thermal_as_global, str) else set(thermal_as_global)
        if "machine" not in table:
            raise ValueError("thermal_as_global names machines, but the table has no 'machine' column")
        allow = table["machine"].isin(names).to_numpy()
    tau = np.where(allow, relaxed.tau, strict.tau)
    # Rows the energy-basis gate (#1713) blanks although a thermal time is there:
    # say so once, with the scaling, instead of returning NaN silently.
    gated = ~allow & ~np.isfinite(strict.tau) & np.isfinite(relaxed.tau)
    if gated.any():
        warnings.warn(
            f"h_factor({scaling!r}): {strict.reason}; {int(gated.sum())} of {len(table)} rows are NaN "
            f"(energy basis {basis!r}, audit gate #1713). Pass thermal_as_global=True or the machine names "
            "to compare with tau_e_th_s (recorded in attrs), or supply tau_e_global_s.",
            UserWarning, stacklevel=2)
    out = (pd.Series(tau, index=table.index) / predicted).rename(f"h_{scaling}")
    substituted = allow & (relaxed.energy_basis_used == "thermal") & (basis != "thermal")
    out.attrs.update(energy_basis=basis, approximation=relaxed.approximation if substituted.any() else None,
                     substituted_rows=int(substituted.sum()), kappa_definition=predicted.attrs["kappa_definition"])
    return out


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
