"""Global dimensionless-similarity projections (#1624, Phase A).

Three zero-dimensional spaces from the ARC physics-basis comparison
(Hillesheim et al. 2026, Fig. 6): $\\nu_*$, $\\beta_N$ and
$\\Omega_{ci}\\tau_{E,\\mathrm{th}}$, each against $\\rho_*$. They answer *where
a plasma sits in similarity space*, separating a size / finite-gyroradius
extrapolation ($\\rho_*$) from a collisionality, a pressure or a confinement-time
extrapolation. They are not stability-limit diagrams and draw no boundary by
default.

Conventions
-----------
The $\\rho_*$ and $\\nu_*$ axes are the ITPA confinement-database convention
of Verdoolaege et al. (2021), Eqs. (1a) and (1c), which Hillesheim et al.
(2026, Sec. 4) state they use. VAFT has other $\\nu_*$ and $\\rho_*$
definitions (#353); each is a different axis quantity, so a Sauter, pedestal
or separatrix collisionality column cannot be drawn on these axes. #353 still
decides VAFT's default definitions; nothing here makes one canonical.

Reference points are only those whose numbers a source states in its text.
The SPARC, ITER and EU-DEMO points and the DB5.2.3-STD5 population of
Hillesheim's Fig. 6 are plotted markers, not stated numbers, and are listed in
:data:`UNAVAILABLE_REFERENCES` instead of being digitised.

Nothing here reads ODS, database or shot data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

from vaft.diagram import _op_space
from vaft.diagram._projection_interpretation import AxisConvention, ProjectionInterpretation
from vaft.formula import boundaries as _b

__all__ = [
    "SIMILARITY_QUANTITIES",
    "SIMILARITY_PROJECTIONS",
    "ReferencePoint",
    "REFERENCE_POINTS",
    "UNAVAILABLE_REFERENCES",
    "reference_points",
]

HILLESHEIM_2026 = _b.BoundarySource(
    citation="J. C. Hillesheim et al., J. Plasma Phys. 92 (2026) E69",
    equation="Sec. 4, paragraph introducing Fig. 6",
    doi="10.1017/S0022377826101706",
)
VERDOOLAEGE_2021 = _b.BoundarySource(
    citation="G. Verdoolaege et al., Nucl. Fusion 61 (2021) 076006",
    equation="Sec. 2.2.2, Eqs. (1a)-(1d)",
    doi="10.1088/1741-4326/abdb91",
)
TROYON_1984 = _b.get_boundary("troyon").sources[0]   # the registered Troyon source, DOI included
LUCE_2008 = _b.BoundarySource(
    citation="T. C. Luce, C. C. Petty and J. G. Cordey, Plasma Phys. Control. Fusion 50 (2008) 043001",
    doi="10.1088/0741-3335/50/4/043001",
    note="why dimensionless coordinates separate transport mechanisms and support projection",
)

_VOLUME_AVERAGE = (
    "volume averages of n and T; in the DB5.2.3 practice T follows from the thermal stored energy, "
    "W_th = 3 n e T V, and n = 0.88 times the line-averaged electron density (Verdoolaege 2021, Sec. 2.2.2)"
)
_DB_INPUTS = {
    "T": "volume-averaged temperature [eV], T_e = T_i assumed (Hillesheim writes <T_i>)",
    "n": "volume-averaged electron density [m^-3]",
    "M_eff": "effective ion mass [amu]",
    "B_t": "vacuum toroidal field at R_geo [T]",
    "R_geo": "geometric major radius [m]",
    "epsilon": "inverse aspect ratio a/R_geo [-]",
    "kappa_a": "area elongation [-]",
    "I_p": "plasma current [A]",
}

#: The axis quantities of the similarity projections, by name.
SIMILARITY_QUANTITIES: Dict[str, _b.BoundaryQuantity] = {
    q.name: q for q in (
        _b.BoundaryQuantity(
            "rho_star_verdoolaege_2021", "rho_*", "-",
            "Thermal ion gyroradius over minor radius, 1.44e-4 (M_eff T)^(1/2) / (B_t a), T volume-averaged in eV "
            "(Verdoolaege 2021 Eq. 1a). Not a local or pedestal rho*.",
        ),
        _b.BoundaryQuantity(
            "nu_star_verdoolaege_2021", "nu_*", "-",
            "Global collisionality 5e-11 lnL n B_t R_geo^2 eps^(1/2) kappa_a / (I_p T^2), q_cyl substituted, "
            "lnL = 30.9 - ln(n^(1/2)/T) (Verdoolaege 2021 Eq. 1c). Not Sauter's local nu*, nor a pedestal, edge or "
            "separatrix nu*.",
        ),
        _b.BoundaryQuantity(
            "omega_ci_tau_e_th", "Omega_ci tau_E,th", "-",
            "Ion cyclotron frequency e B_t / (M_eff m_p) times the thermal energy confinement time.",
        ),
    )
}
_clash = {name for name, q in SIMILARITY_QUANTITIES.items()
          if name in _op_space.AXIS_QUANTITIES and _op_space.AXIS_QUANTITIES[name] != q}
if _clash:
    raise ValueError(f"similarity quantities {sorted(_clash)} would redefine registered axis quantities")
_op_space.AXIS_QUANTITIES.update(SIMILARITY_QUANTITIES)

_RHO_STAR = AxisConvention(
    quantity="rho_star_verdoolaege_2021",
    expression="1.44e-4 (M_eff T)^(1/2) (B_t R_geo epsilon)^(-1)",
    species="thermal ion, hydrogenic; T_e = T_i assumed",
    radial_definition="whole plasma; normalised by the minor radius a = R_geo epsilon",
    averaging_definition=_VOLUME_AVERAGE,
    source=replace(VERDOOLAEGE_2021, equation="Sec. 2.2.2, Eq. (1a)"),
    inputs={k: _DB_INPUTS[k] for k in ("T", "M_eff", "B_t", "R_geo", "epsilon")},
    formula="vaft.formula.equilibrium.rho_star_from_M_T_B_R_epsilon",
    unresolved=(
        "Hillesheim's design points use <T_i> of the design profiles; whether that equals the database's "
        "W_th-derived T is not stated",
    ),
)
_NU_STAR = AxisConvention(
    quantity="nu_star_verdoolaege_2021",
    expression="5e-11 lnL n B_t R_geo^2 epsilon^(1/2) kappa_a I_p^(-1) T^(-2), lnL = 30.9 - ln(n^(1/2)/T)",
    species=("ion-ion collision frequency nu_ii over the trapped-particle bounce frequency (Verdoolaege Eq. 1c "
             "writes nu_ii); T_e = T_i assumed, and lnL is the NRL electron form"),
    radial_definition="whole plasma; Sauter's (R/a)^(3/2) q R form with q = q_cyl (Eq. 1d) substituted",
    averaging_definition=_VOLUME_AVERAGE,
    source=replace(VERDOOLAEGE_2021, equation="Sec. 2.2.2, Eqs. (1c) and (1d)"),
    inputs={**{k: _DB_INPUTS[k] for k in ("n", "T", "B_t", "R_geo", "epsilon", "kappa_a", "I_p")},
            "lnL": "30.9 - ln(n^(1/2)/T), n in m^-3 and T in eV"},
    formula=("vaft.formula.equilibrium.nu_star_from_n_T_B_R_epsilon_kappa_I "
             "(ln_lambda=None evaluates equilibrium.coulomb_logarithm_from_n_T, the same lnL)"),
    unresolved=(
        "#353: VAFT's own comparison (equilibrium.nu_star_from_n_T_B_R_epsilon_kappa_I) puts this form at 1.45 "
        "times Sauter's electron nu*_e (Eq. 18b) with q = q_cyl; neither source states that ratio. VAFT's default "
        "nu* is undecided, so this axis accepts only this convention",
    ),
)
_OMEGA_TAU = AxisConvention(
    quantity="omega_ci_tau_e_th",
    expression="(e B_t / (M_eff m_p)) tau_E,th = 9.58e7 B_t tau_E,th / M_eff",
    species="hydrogenic ion (charge e)",
    radial_definition="vacuum toroidal field at R_geo",
    averaging_definition="tau_E,th = W_th / P_loss,th, the global thermal energy confinement time",
    source=HILLESHEIM_2026,
    inputs={"B_t": _DB_INPUTS["B_t"], "M_eff": _DB_INPUTS["M_eff"],
            "tau_E,th": "thermal energy confinement time [s]"},
    formula="vaft.formula.equilibrium.omega_i_tau_E_from_B_tau_E_M (Z_i = 1)",
    unresolved=(
        "Hillesheim's design points take tau_E,th from IPB98(y,2) times each design's H98; measured points use "
        "the measured tau_E,th, so the two are different estimates of one quantity",
    ),
)
_BETA_N = AxisConvention(
    quantity="normalized_beta",
    expression="beta[%] a[m] B_T[T] / I_p[MA]",
    species="total plasma pressure (all species, thermal and fast unless the population says otherwise)",
    radial_definition="volume average over the plasma",
    averaging_definition="Troyon's beta is 2 int p dV / int B^2 dV; the toroidal beta 2 mu0 <p>/B_T^2 at low beta",
    source=TROYON_1984,
    inputs={"beta": "volume-averaged beta [%]", "a": "minor radius [m]", "B_T": "toroidal field [T]",
            "I_p": "plasma current [MA]"},
    formula="vaft.formula.stability.beta_N_from_beta_a_B0_Ip",
    unresolved=(
        "whether beta includes the fast-ion pressure differs between sources; the population must say which",
    ),
)

_COMMON = dict(
    category="dimensionless_similarity",
    quantity_scope="global",
    boundary_type="none",
    applicability=(
        "zero-dimensional, time-stationary states (the database averages over stationary H-mode phases); "
        "a transient or start-up slice is placed but is not a similarity state"
    ),
    species="hydrogenic ions with T_e = T_i assumed by the convention",
    radial_definition="global: minor-radius normalisation, no local radius",
    averaging_definition=_VOLUME_AVERAGE,
    references=(HILLESHEIM_2026, VERDOOLAEGE_2021, LUCE_2008),
    required_profiles=(
        "volume-averaged density n and temperature T (or the thermal stored energy W_th with the plasma volume)",
        "effective ion mass M_eff",
    ),
)
_NOT_LOCAL = "a local neoclassical regime (banana/plateau/Pfirsch-Schlueter) classification; that needs a local nu*"


def _cite(source: _b.BoundarySource) -> str:
    return f"{source.citation}, {source.equation}" if source.equation else source.citation


def _register(key, title, y, conventions, **meta):
    projection = _op_space.OperationalProjection(
        key=key, title=title, x=SIMILARITY_QUANTITIES["rho_star_verdoolaege_2021"], y=y,
        references=(f"{HILLESHEIM_2026.citation}, Fig. 6", _cite(VERDOOLAEGE_2021)),
        assumptions=tuple(f"{c.quantity}: {c.expression} ({_cite(c.source)})" for c in conventions),
        interpretation=ProjectionInterpretation(parameter_conventions=conventions, **{**_COMMON, **meta}),
    )
    existing = _op_space._PROJECTIONS.get(key)
    if existing is not None and existing == projection:   # a module reload re-registers the same projection
        return existing
    names = {c.quantity for c in conventions}
    if names != {projection.x.name, projection.y.name}:
        raise ValueError(f"projection {key!r} needs exactly one convention per axis, has {sorted(names)}")
    _op_space._register(projection)
    return projection


SIMILARITY_PROJECTIONS: Tuple[str, ...] = tuple(p.key for p in (
    _register(
        "rho_star_nu_star", "Collisionality against normalised gyroradius",
        SIMILARITY_QUANTITIES["nu_star_verdoolaege_2021"], (_RHO_STAR, _NU_STAR),
        physical_question=(
            "Is an extrapolation mainly in machine size (rho*), in collisionality (nu*), or both, and does an "
            "existing population already cover the target collisionality?"
        ),
        interpretation=(
            "1/rho* is roughly the number of ion gyroradii across the minor radius; 1/nu* roughly the number of "
            "trapped-ion bounce orbits before a collision. Two machines close here are close in global kinetic "
            "similarity whatever their engineering parameters. Low nu* reached by low density and high temperature "
            "is not the same route as reactor-like high density at high field (Hillesheim 2026, Sec. 4)."
        ),
        similarity_role="x: size / finite-Larmor-radius extrapolation; y: collisionality extrapolation",
        not_for=(_NOT_LOCAL, "a stability or density limit: the space has no boundary"),
    ),
    _register(
        "rho_star_beta_n", "Normalised beta against normalised gyroradius",
        _op_space.AXIS_QUANTITIES["normalized_beta"], (_RHO_STAR, _BETA_N),
        physical_question=(
            "How much of an extrapolation is MHD-normalised pressure and how much is machine scale?"
        ),
        interpretation=(
            "A population can overlap a reactor's beta_N while remaining far from it in rho*. beta_N is a "
            "community-normalised performance coordinate, not a member of a minimal dimensionless basis."
        ),
        similarity_role="x: size / finite-Larmor-radius extrapolation; y: normalised pressure (MHD performance)",
        not_for=(
            "a beta limit: the Troyon line is not drawn by default; request boundaries=['troyon'] to show it as "
            "an ideal-MHD reference",
        ),
    ),
    _register(
        "rho_star_omega_ci_tau_e", "Confinement in gyro-orbits against normalised gyroradius",
        SIMILARITY_QUANTITIES["omega_ci_tau_e_th"], (_RHO_STAR, _OMEGA_TAU),
        physical_question=(
            "Is a device near present experiments in effective size, in confinement time measured in gyro-orbits, "
            "or in neither?"
        ),
        interpretation=(
            "Omega_ci tau_E,th is roughly the number of gyro-orbits a thermal ion completes before its energy is "
            "lost. Together with rho* it separates a geometric extrapolation from a confinement-time one."
        ),
        similarity_role="x: size / finite-Larmor-radius extrapolation; y: confinement time in gyro-periods",
        not_for=(_NOT_LOCAL, "a confinement-scaling fit; #1621 owns the regression"),
        required_profiles=_COMMON["required_profiles"] + ("thermal energy confinement time tau_E,th",),
    ),
))


# ---------------------------------------------------------------------------
# Literature reference points
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferencePoint:
    """One machine or design state with values a source states in its text.

    ``values`` maps an axis-quantity name to its value in that quantity's
    unit. ``kind`` is ``"design"`` for a projected operating point or
    ``"experiment"`` for a measured one. A quantity the source does not state
    is absent, never estimated.
    """

    machine: str
    label: str
    kind: str
    values: Mapping[str, float] = field(hash=False)
    source: _b.BoundarySource
    note: str = ""

    def __post_init__(self):
        if self.kind not in ("design", "experiment"):
            raise ValueError(f"kind must be 'design' or 'experiment', not {self.kind!r}")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


REFERENCE_POINTS: Tuple[ReferencePoint, ...] = (
    ReferencePoint(
        machine="ARC", label="ARC V3A", kind="design",
        values={"normalized_beta": 1.8, "rho_star_verdoolaege_2021": 0.0017, "nu_star_verdoolaege_2021": 0.031,
                "omega_ci_tau_e_th": 4.1e8},
        source=HILLESHEIM_2026,
        note="nominal flat-top operating point; tau_E,th from IPB98(y,2) times the design H98",
    ),
)

#: References of Hillesheim's Fig. 6 that the text does not state as numbers, with the reason they are absent.
UNAVAILABLE_REFERENCES: Tuple[Tuple[str, str], ...] = (
    ("SPARC PRD", "plotted in Hillesheim 2026 Fig. 6 only; no stated values (Creely 2020, Body 2023 are the sources)"),
    ("ITER", "plotted in Hillesheim 2026 Fig. 6 only; no stated values (Doyle 2007 is the source)"),
    ("EU-DEMO", "plotted in Hillesheim 2026 Fig. 6 only; no stated values (Siccinio 2022 is the source)"),
    ("ITPA DB5.2.3-STD5", "a 7537-entry database (Verdoolaege 2021a, https://osf.io/drwcq/); pass it as a population"),
)


def reference_points(projection) -> Tuple[ReferencePoint, ...]:
    """The registered reference points that state both axis values of a projection."""
    proj = _op_space.get_projection(projection) if isinstance(projection, str) else projection
    return tuple(p for p in REFERENCE_POINTS if proj.x.name in p.values and proj.y.name in p.values)
