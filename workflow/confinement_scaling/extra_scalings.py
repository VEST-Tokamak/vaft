"""Ohmic and L-mode confinement scalings missing from ``vaft.formula`` (Lane D figures, #548).

VEST is an ohmic L-mode plasma, so the ohmic and L-mode scalings matter for it as
much as IPB98 does. ``vaft.formula.constants._SCALING_COEFS`` carries pure power
laws only. Neo-Alcator goes through q, and Goldston's ohmic/auxiliary combination
is a quadrature, so neither fits that table. Generalizing the scaling API is #670,
and Lane D touches ``vaft.formula`` only for defects. These are therefore
workflow-local, transcribed from the original papers and evaluated in the
papers' own units, so the unit conversion can be checked. They should move into
``vaft.formula`` under #670.

| name | formula (paper units) | source |
|---|---|---|
| ``NeoAlcator`` | tau = 7.1e-22 n[cm^-3] a[cm]^1.04 R[cm]^2.04 q^0.5 | Goldston, PPCF 26 (1984) 87, eq. (3) |
| ``Goldston84L`` | tau = 6.4e-8 I_p[A] P_tot[W]^-1/2 R[cm]^1.75 a[cm]^-0.37 kappa^1/2 | same, eq. (6), L-mode, deuterium |
| ``Goldston84OhmicL`` | 1/tau^2 = 1/tau_(3)^2 + 1/tau_(6)^2 | same, eq. (11) applied to eqs. (3) and (6) |
| ``ITER97L`` | tau_th = 0.023 I_p[MA]^0.96 B_T^0.03 R^1.83 eps^-0.06 kappa^0.64 n[1e19 m^-3]^0.40 M^0.20 P[MW]^-0.73 | Kaye et al., NF 37 (1997) 1303 |

Inputs come from the canonical confinement columns:
- n is the line average;
- q is the cylindrical q of ``vaft.formula.q_cyl_from_B_R_epsilon_kappa_I``, with the
  area elongation (``kappa_area``);
- kappa in Goldston and ITER97L is the boundary elongation ``kappa``;
- P is ``p_loss_W``;
- M is ``m_eff_amu``.

Caveats:
- Goldston 1984 did eq. (11) with the <nT> form of tau_AUX (its eq. 8), not eq. (6).
  Combining eqs. (3) and (6) in quadrature is the common usage, and it is labelled
  as such.
- Goldston's q is the limiter q of mostly circular plasmas.
- ITER97L is the thermal fit to the hydrogenic L-mode standard set.
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


def neo_alcator(table: pd.DataFrame) -> np.ndarray:
    """Goldston 1984 eq. (3), in its CGS units [s]."""
    n_cm3 = _col(table, "n_e_line_avg_m3") * 1e-6
    a_cm = _col(table, "a_m") * 100.0
    r_cm = _col(table, "r_geo_m") * 100.0
    return 7.1e-22 * n_cm3 * a_cm**1.04 * r_cm**2.04 * q_cyl(table) ** 0.5


def goldston_l(table: pd.DataFrame) -> np.ndarray:
    """Goldston 1984 eq. (6): I_p in A, P_tot in W, R and a in cm [s]."""
    ip = _col(table, "i_p_A")
    p = _col(table, "p_loss_W")
    r_cm = _col(table, "r_geo_m") * 100.0
    a_cm = _col(table, "a_m") * 100.0
    kappa = _col(table, "kappa")
    return 6.4e-8 * ip * p**-0.5 * r_cm**1.75 * a_cm**-0.37 * kappa**0.5


def goldston_ohmic_l(table: pd.DataFrame) -> np.ndarray:
    """Eqs. (3) and (6) combined as Goldston 1984 eq. (11), 1/tau^2 = sum 1/tau_i^2 [s]."""
    return (neo_alcator(table) ** -2 + goldston_l(table) ** -2) ** -0.5


def iter97_l(table: pd.DataFrame) -> np.ndarray:
    """Kaye et al. 1997, ITER97-L thermal: I_p MA, n 1e19 m^-3, P MW [s]."""
    return (0.023 * (_col(table, "i_p_A") / 1e6) ** 0.96 * _col(table, "b_t_T") ** 0.03
            * _col(table, "r_geo_m") ** 1.83 * _col(table, "epsilon") ** -0.06 * _col(table, "kappa") ** 0.64
            * (_col(table, "n_e_line_avg_m3") / 1e19) ** 0.40 * _col(table, "m_eff_amu") ** 0.20
            * (_col(table, "p_loss_W") / 1e6) ** -0.73)


_FUNCTIONS = {"NeoAlcator": neo_alcator, "Goldston84L": goldston_l,
              "Goldston84OhmicL": goldston_ohmic_l, "ITER97L": iter97_l}


def predict(table: pd.DataFrame, name: str) -> pd.Series:
    """Predicted tau_E of a workflow-local scaling, indexed like ``table`` [s]."""
    if name not in _FUNCTIONS:
        raise KeyError(f"unknown scaling {name!r}; known: {NAMES}")
    return pd.Series(_FUNCTIONS[name](table), index=table.index, name=f"tau_e_{name}_s")
