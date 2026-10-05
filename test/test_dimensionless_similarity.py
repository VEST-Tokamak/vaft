"""Dimensionless-similarity projections, their interpretation metadata and the population plot (#1624)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vaft.diagram import _op_space as ops  # noqa: E402
from vaft.diagram import _similarity_space as sim  # noqa: E402
from vaft.diagram._projection_interpretation import AxisConvention, ProjectionInterpretation  # noqa: E402
from vaft.formula import equilibrium as eq  # noqa: E402
from vaft.formula.boundaries import BoundarySource  # noqa: E402
from vaft.plot.dimensionless_space import AXIS_SCALES, dimensionless_similarity  # noqa: E402

PHASE_A = ("rho_star_nu_star", "rho_star_beta_n", "rho_star_omega_ci_tau_e")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _population(n=24, seed=1):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({
        "machine": np.repeat(["VEST", "KSTAR", "DIII-D"], n // 3),
        "rho_star_verdoolaege_2021": 10 ** rng.uniform(-2.7, -1.5, n),
        "nu_star_verdoolaege_2021": 10 ** rng.uniform(-1.5, 1.0, n),
        "normalized_beta": rng.uniform(0.5, 3.5, n),
        "omega_ci_tau_e_th": 10 ** rng.uniform(5, 8, n),
    })
    t.attrs["units"] = {"normalized_beta": "% m T/MA"}
    return t


# --- registry and metadata contract ------------------------------------------------


def test_phase_a_projections_are_registered_with_complete_interpretations():
    assert set(PHASE_A) == set(sim.SIMILARITY_PROJECTIONS)
    for key in PHASE_A:
        proj = ops.get_projection(key)
        meta = proj.interpretation
        assert isinstance(meta, ProjectionInterpretation)
        assert meta.category == "dimensionless_similarity" and meta.quantity_scope == "global"
        assert meta.boundary_type == "none" and proj.default_boundaries == ()
        assert {c.quantity for c in meta.parameter_conventions} == {proj.x.name, proj.y.name}
        assert meta.physical_question and meta.similarity_role and meta.required_profiles
        assert proj.x.name == "rho_star_verdoolaege_2021"
        assert "Hillesheim" in meta.describe() and "Not for:" in meta.describe()


def test_axis_conventions_cite_the_source_equation():
    meta = ops.get_projection("rho_star_nu_star").interpretation
    assert meta.convention("rho_star_verdoolaege_2021").source.equation.endswith("Eq. (1a)")
    assert "(1c)" in meta.convention("nu_star_verdoolaege_2021").source.equation
    for _, item in meta.unresolved:
        assert item  # reported, never empty


def test_similarity_axes_are_distinct_from_every_other_registered_quantity():
    # the database nu* and rho* must not share a name with any other axis or boundary quantity
    names = [q.name for q in ops.AXIS_QUANTITIES.values()]
    assert len(names) == len(set(names))
    for name in ("rho_star_verdoolaege_2021", "nu_star_verdoolaege_2021", "omega_ci_tau_e_th"):
        assert ops.AXIS_QUANTITIES[name] is sim.SIMILARITY_QUANTITIES[name]


def test_existing_projections_are_unchanged_by_the_new_field():
    assert ops.get_projection("hugill").interpretation is None
    assert ops.get_projection("troyon").default_boundaries == ("troyon",)


def test_interpretation_rejects_unknown_vocabulary_and_duplicate_conventions():
    src = BoundarySource("x")
    conv = AxisConvention("q", "e", "s", "r", "a", src)
    base = dict(category="dimensionless_similarity", physical_question="q", interpretation="i",
                quantity_scope="global", similarity_role="r", boundary_type="none", applicability="a",
                species="s", radial_definition="r", averaging_definition="a", parameter_conventions=(conv,),
                references=(src,))
    ProjectionInterpretation(**base)
    for field, bad in (("category", "misc"), ("quantity_scope", "core"), ("boundary_type", "limit")):
        with pytest.raises(ValueError):
            ProjectionInterpretation(**{**base, field: bad})
    with pytest.raises(ValueError):
        ProjectionInterpretation(**{**base, "parameter_conventions": (conv, conv)})
    with pytest.raises(ValueError):
        ProjectionInterpretation(**{**base, "references": ()})


# --- the named formula functions evaluate the source expressions -------------------


def test_formula_functions_reproduce_the_verdoolaege_expressions():
    M, T, B, R, eps, kappa, n, I = 2.5, 8.0e3, 11.5, 4.25, 1.2 / 4.25, 1.75, 1.8e20, 10.0e6
    a = R * eps
    assert eq.rho_star_from_M_T_B_R_epsilon(M, T, B, R, eps) == pytest.approx(1.44e-4 * np.sqrt(M * T) / (B * a))
    lnl = 30.9 - np.log(np.sqrt(n) / T)
    assert eq.coulomb_logarithm_from_n_T(n, T) == pytest.approx(lnl)
    expected = 5.0e-11 * lnl * n * B * R ** 2 * np.sqrt(eps) * kappa / (I * T ** 2)
    assert eq.nu_star_from_n_T_B_R_epsilon_kappa_I(n, T, B, R, eps, kappa, I) == pytest.approx(expected)
    # Hillesheim's 9.58e7 B tau / m_eff is e/m_p to three digits
    assert eq.omega_i_tau_E_from_B_tau_E_M(B, 1.0, M) == pytest.approx(9.58e7 * B / M, rel=1e-3)


# --- reference points -------------------------------------------------------------------


def test_arc_reference_point_is_the_stated_values():
    (arc,) = sim.REFERENCE_POINTS
    assert arc.label == "ARC V3A" and arc.kind == "design"
    assert dict(arc.values) == {"normalized_beta": 1.8, "rho_star_verdoolaege_2021": 0.0017,
                                "nu_star_verdoolaege_2021": 0.031, "omega_ci_tau_e_th": 4.1e8}
    assert arc.source.doi == "10.1017/S0022377826101706"
    assert {name for name, _ in sim.UNAVAILABLE_REFERENCES} >= {"SPARC PRD", "ITER", "EU-DEMO"}
    for key in PHASE_A:
        assert sim.reference_points(key) == (arc,)


# --- population plot --------------------------------------------------------------------


@pytest.mark.parametrize("key", PHASE_A)
def test_population_plot_sets_scales_and_carries_metadata(key):
    fig, ax = dimensionless_similarity(_population(), key, group="machine")
    assert (ax.get_xscale(), ax.get_yscale()) == AXIS_SCALES[key]
    assert ax.vaft_interpretation is ops.get_projection(key).interpretation
    assert ax.vaft_exclusions.excluded == 0
    assert [r[0] for r in ax.vaft_references] == ["ARC V3A"]
    x0, x1 = ax.get_xlim()
    assert x0 < 0.0017 < x1
    assert "Verdoolaege" in ax.get_xlabel()


def test_missing_inputs_are_counted_per_group_and_not_plotted():
    t = _population()
    t.loc[t.machine == "VEST", "nu_star_verdoolaege_2021"] = np.nan
    t.loc[t.index[-1], "rho_star_verdoolaege_2021"] = -1.0
    with pytest.warns(UserWarning, match="UNASSESSED"):
        fig, ax = dimensionless_similarity(t, "rho_star_nu_star", group="machine", references=False)
    report = ax.vaft_exclusions
    assert report.unassessed == {"VEST": 8, "DIII-D": 1}
    assert report.assessed == len(t) - 9
    placed = sum(len(c.get_offsets()) for c in ax.collections)
    assert placed == report.assessed
    assert any("VEST: 8 unassessed" in text.get_text() for text in ax.get_legend().get_texts())


def test_another_convention_is_refused_not_drawn():
    t = _population().rename(columns={"nu_star_verdoolaege_2021": "nu_star_sauter"})
    with pytest.raises(KeyError, match="another convention"):
        dimensionless_similarity(t, "rho_star_nu_star")


def test_undeclared_beta_n_unit_is_refused():
    t = _population()
    t.attrs["units"] = {}
    with pytest.raises(ValueError, match="normalized_beta"):
        dimensionless_similarity(t, "rho_star_beta_n")


def test_stability_limit_projection_is_not_a_similarity_space():
    with pytest.raises(ValueError, match="operational_space_population"):
        dimensionless_similarity(_population(), "hugill")


def test_reference_table_needs_a_source_for_every_row():
    refs = pd.DataFrame({"label": ["X"], "kind": ["design"], "source": [""],
                         "rho_star_verdoolaege_2021": [2e-3], "nu_star_verdoolaege_2021": [0.05]})
    with pytest.raises(ValueError, match="no source"):
        dimensionless_similarity(_population(), "rho_star_nu_star", reference_table=refs)
    refs["source"] = ["user design note"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig, ax = dimensionless_similarity(_population(), "rho_star_nu_star", reference_table=refs)
    assert [r[0] for r in ax.vaft_references] == ["ARC V3A", "X"]


def test_the_similarity_plot_imports_no_ods_or_database_layer():
    import subprocess
    import sys

    code = ("import sys, vaft.plot.dimensionless_space; "
            "bad = [m for m in ('omas', 'vaft.omas', 'vaft.database', 'h5pyd') if m in sys.modules]; "
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
