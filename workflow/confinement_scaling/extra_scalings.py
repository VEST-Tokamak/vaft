"""Ohmic and L-mode confinement scalings for the Lane D figures (#548), evaluated on table columns.

The formulas live in ``vaft.formula`` since #670; this module only maps the
canonical confinement columns onto them, row by row, NaN where an input is
missing.

| name | formula | ``vaft.formula`` |
|---|---|---|
| ``NeoAlcator`` | Goldston, PPCF 26 (1984) 87, eq. (3) | ``neo_alcator_confinement_time_from_n_a_R_q`` |
| ``Goldston84L`` | same, eq. (6), L-mode, deuterium | ``goldston_l_mode_confinement_time_from_I_P_R_a_kappa`` |
| ``Goldston84OhmicL`` | eq. (11) applied to eqs. (3) and (6) | ``ohmic_l_mode_confinement_time_from_tau_ohmic_tau_aux`` |
| ``ITER97L`` | Kaye et al., NF 37 (1997) 1303 | ``confinement_time_from_engineering_parameters(scaling="ITER97L")`` |

Inputs come from the canonical confinement columns:
- n is the line average;
- q is the cylindrical q of ``vaft.formula.q_cyl_from_B_R_epsilon_kappa_I``, with the
  area elongation (``kappa_area``);
- kappa in Goldston and ITER97L is the boundary elongation ``kappa``;
- P is ``p_loss_W``;
- M is ``m_eff_amu``.

Caveats are in each formula's docstring: Goldston's eq. (11) used the <nT> form of
tau_AUX, his q is the limiter q of mostly circular plasmas, and ITER97L is the
thermal fit to the hydrogenic L-mode standard set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = {
    "NeoAlcator": "neo-Alcator ohmic (Goldston 1984)",
    "Goldston84L": "Goldston 1984 L-mode",
    "Goldston84OhmicL": "Goldston 1984 ohmic+L (quadrature)",
    "ITER97L": "ITER97-L (Kaye 1997)",
}
NAMES = tuple(LABELS)

#: The elongation these scalings are fed, recorded as ``attrs["kappa_definition"]`` of
#: :func:`predict`. ``vaft.data.public.predict_confinement_time`` feeds ``kappa_area``
#: by default, so an ITER97-L H factor from the two paths differs by (kappa/kappa_area)^0.64.
KAPPA_DEFINITION = "kappa: boundary (LCFS) elongation b/a"
_USES_KAPPA = frozenset({"Goldston84L", "Goldston84OhmicL", "ITER97L"})


def _col(table: pd.DataFrame, name: str) -> np.ndarray:
    arr = pd.to_numeric(table[name], errors="coerce").to_numpy(float)
    return np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)


def q_cyl(table: pd.DataFrame) -> np.ndarray:
    """Cylindrical q from the canonical columns, NaN where an input is missing."""
    from vaft.formula.equilibrium import q_cyl_from_B_R_epsilon_kappa_I

    b, r, eps, ka, ip = (_col(table, c) for c in ("b_t_T", "r_geo_m", "epsilon", "kappa_area", "i_p_A"))
    ok = np.isfinite(b) & np.isfinite(r) & np.isfinite(eps) & np.isfinite(ka) & np.isfinite(ip)
    out = np.full(len(table), np.nan)
    if ok.any():
        out[ok] = q_cyl_from_B_R_epsilon_kappa_I(b[ok], r[ok], eps[ok], ka[ok], ip[ok])
    return out


def _rowwise(func, table: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """``func`` on the rows where every column is finite and positive, NaN elsewhere."""
    args = [_col(table, c) for c in columns]
    ok = np.logical_and.reduce([np.isfinite(a) for a in args])
    out = np.full(len(table), np.nan)
    if ok.any():
        out[ok] = func(*(a[ok] for a in args))
    return out


def neo_alcator(table: pd.DataFrame) -> np.ndarray:
    """Goldston 1984 eq. (3) [s]."""
    from vaft.formula.equilibrium import neo_alcator_confinement_time_from_n_a_R_q

    n, a, r = (_col(table, c) for c in ("n_e_line_avg_m3", "a_m", "r_geo_m"))
    q = q_cyl(table)
    ok = np.isfinite(n) & np.isfinite(a) & np.isfinite(r) & np.isfinite(q)
    out = np.full(len(table), np.nan)
    if ok.any():
        out[ok] = neo_alcator_confinement_time_from_n_a_R_q(n[ok], a[ok], r[ok], q[ok])
    return out


def goldston_l(table: pd.DataFrame) -> np.ndarray:
    """Goldston 1984 eq. (6) [s]."""
    from vaft.formula.equilibrium import goldston_l_mode_confinement_time_from_I_P_R_a_kappa

    return _rowwise(goldston_l_mode_confinement_time_from_I_P_R_a_kappa, table,
                    ("i_p_A", "p_loss_W", "r_geo_m", "a_m", "kappa"))


def goldston_ohmic_l(table: pd.DataFrame) -> np.ndarray:
    """Eqs. (3) and (6) combined as Goldston 1984 eq. (11) [s]."""
    from vaft.formula.equilibrium import ohmic_l_mode_confinement_time_from_tau_ohmic_tau_aux

    t_oh, t_l = neo_alcator(table), goldston_l(table)
    ok = np.isfinite(t_oh) & np.isfinite(t_l)
    out = np.full(len(table), np.nan)
    if ok.any():
        out[ok] = ohmic_l_mode_confinement_time_from_tau_ohmic_tau_aux(t_oh[ok], t_l[ok])
    return out


def iter97_l(table: pd.DataFrame) -> np.ndarray:
    """Kaye et al. 1997, ITER97-L thermal [s]."""
    from vaft.formula.equilibrium import confinement_time_from_engineering_parameters

    def one(ip, b, p, n, m, r, eps, kappa):
        return np.array([confinement_time_from_engineering_parameters(*row, scaling="ITER97L")
                         for row in zip(ip, b, p, n, m, r, eps, kappa)])

    return _rowwise(one, table, ("i_p_A", "b_t_T", "p_loss_W", "n_e_line_avg_m3", "m_eff_amu",
                                 "r_geo_m", "epsilon", "kappa"))


_FUNCTIONS = {"NeoAlcator": neo_alcator, "Goldston84L": goldston_l,
              "Goldston84OhmicL": goldston_ohmic_l, "ITER97L": iter97_l}


def predict(table: pd.DataFrame, name: str) -> pd.Series:
    """Predicted tau_E of one of these scalings, indexed like ``table`` [s]."""
    if name not in _FUNCTIONS:
        raise KeyError(f"unknown scaling {name!r}; known: {NAMES}")
    out = pd.Series(_FUNCTIONS[name](table), index=table.index, name=f"tau_e_{name}_s")
    out.attrs["kappa_definition"] = KAPPA_DEFINITION if name in _USES_KAPPA else "no elongation term"
    return out
