"""Coordinate, gradient coordinate, reference length and convention kept apart (#551).

The analytic reference is a family of nested, Shafranov-shifted circles: the
surface of minor radius ``r`` is centred at ``R0 + D0 (1 - r^2/a^2)`` and
carries ``psi_norm = r^2/a^2``.  Its midplane crossings are therefore known
exactly -- ``r_minor = r``, ``r_center = R0 + D0 (1 - psi_norm)`` -- and with
``q = 1 + 2 psi_norm`` the toroidal-flux radius is
``rho_tor_norm^2 = (psi_norm + psi_norm^2)/2``, so a profile given on
``rho_tor_norm`` has an analytic derivative in ``r_minor``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vaft.data.equilibrium import Contour, EquilibriumConvention, EquilibriumData
from vaft.process.profile_gradients import (
    CONVENTIONS,
    RADIAL_COORDINATES,
    profile_gradient,
    radial_coordinate_map,
    radial_coordinate_map_from_arrays,
    resolve_convention,
    resolve_reference_length,
)

R0, A, PSI_EDGE = 0.4, 0.25, 0.02
WIDTH = 0.15  # T = exp(-r^2/WIDTH^2), so -d ln T/dr = 2 r/WIDTH^2


def shifted_circles(shift: float, n: int = 161) -> EquilibriumData:
    r = np.linspace(R0 - 0.35, R0 + 0.4, n)
    z = np.linspace(-0.35, 0.35, n)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    if shift == 0.0:
        s = ((rm - R0) ** 2 + zm**2)
    else:
        # (D + shift s/a^2)^2 + Z^2 = s with D = R - R0 - shift and s = r^2: the smaller root
        d = rm - R0 - shift
        qa, qb, qc = (shift / A**2) ** 2, 2 * d * shift / A**2 - 1.0, d**2 + zm**2
        s = (-qb - np.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
    psi_n = np.linspace(0.0, 1.0, 101)
    theta = np.linspace(0.0, 2 * np.pi, 200)
    return EquilibriumData(
        r=r, z=z, psi=PSI_EDGE * s / A**2, psi_axis=0.0, psi_boundary=PSI_EDGE,
        magnetic_axis=(R0 + shift, 0.0), lcfs=Contour(R0 + A * np.cos(theta), A * np.sin(theta)),
        psi_1d=PSI_EDGE * psi_n, q=1.0 + 2.0 * psi_n, time=0.25,
        convention=EquilibriumConvention(cocos=11, psi_per_radian=False),
    )


def r_of_rho_tor(rho):
    """Analytic r for q = 1 + 2 psi_norm: rho^2 = (psi_n + psi_n^2)/2."""
    return A * np.sqrt((-1.0 + np.sqrt(1.0 + 8.0 * rho**2)) / 2.0)


@pytest.fixture(scope="module")
def circular():
    return radial_coordinate_map(shifted_circles(0.0))


@pytest.fixture(scope="module")
def shifted():
    return radial_coordinate_map(shifted_circles(0.03))


# --------------------------------------------------------------------------
# the coordinate map
# --------------------------------------------------------------------------


def test_the_module_name_does_not_shadow_the_function():
    import vaft.process
    from vaft.process import profile_gradients

    assert callable(profile_gradients.profile_gradient)
    assert profile_gradients.profile_gradient is profile_gradient
    namespace: dict = {}
    exec("from vaft.process import *", namespace)
    assert callable(namespace["profile_gradient"])
    assert not isinstance(namespace["profile_gradient"], type(vaft.process))
    assert namespace["profile_gradient"] is profile_gradient
    assert vaft.process.profile_gradient is profile_gradient


def test_the_registry_names_every_coordinate_the_issue_asks_for():
    assert set(RADIAL_COORDINATES) == {
        "psi_norm", "rho_pol_norm", "rho_tor_norm", "r_inboard", "r_outboard",
        "r_center", "r_minor", "r_minor_norm",
    }
    assert RADIAL_COORDINATES["r_minor"].unit == "m"
    assert RADIAL_COORDINATES["r_minor_norm"].kind == "geometric"
    assert RADIAL_COORDINATES["rho_tor_norm"].kind == "flux"


def test_circular_r_minor_is_the_geometric_radius(circular):
    psi_n = circular.values("psi_norm")
    r_minor = circular.values("r_minor")
    np.testing.assert_allclose(r_minor, A * np.sqrt(psi_n), atol=1e-6)
    np.testing.assert_allclose(
        r_minor, 0.5 * (circular.values("r_outboard") - circular.values("r_inboard")), rtol=0, atol=0)
    norm = circular.values("r_minor_norm")
    assert norm[0] == pytest.approx(0.0, abs=1e-12)
    assert norm[-1] == pytest.approx(1.0, abs=1e-12)
    assert circular.a_minor.value == pytest.approx(A, rel=1e-6)


def test_a_shafranov_shift_separates_r_minor_from_r_out_minus_r_axis(shifted):
    psi_n = shifted.values("psi_norm")
    r_minor = shifted.values("r_minor")
    np.testing.assert_allclose(r_minor, A * np.sqrt(psi_n), atol=1e-6)
    np.testing.assert_allclose(shifted.values("r_center"), R0 + 0.03 * (1 - psi_n), atol=1e-6)
    outboard_from_axis = shifted.values("r_outboard") - shifted.R_major_axis.value
    assert np.max(np.abs(outboard_from_axis - r_minor)) > 0.02  # the shift, 3 cm at the edge


def test_shaped_solovev_keeps_three_radii_distinct():
    from vaft.process.equilibrium import derive_global_descriptors, solovev_example

    eq = solovev_example()
    psi_n = (eq.psi_1d - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
    # solovev_example carries no q; any positive, rising q gives a real rho_tor
    cmap = radial_coordinate_map(replace(eq, q=1.0 + 2.5 * psi_n**2))
    assert set(cmap.available) == set(RADIAL_COORDINATES)
    r_norm, rho_pol, rho_tor = (cmap.values(n) for n in ("r_minor_norm", "rho_pol_norm", "rho_tor_norm"))
    for one, other in ((r_norm, rho_pol), (r_norm, rho_tor), (rho_pol, rho_tor)):
        assert np.max(np.abs(one - other)) > 1e-2
    # a is the midplane half-width; on this up-down symmetric boundary the R extremes
    # sit at the axis height, so it agrees with the contour half-width, by construction only
    contour = derive_global_descriptors(eq)["minor_radius"].value
    assert cmap.a_minor.value == pytest.approx(contour, rel=1e-3)
    assert "derive_global_descriptors" in cmap.a_minor.definition


def test_the_map_is_chosen_by_time_and_refuses_a_contradiction():
    eq = shifted_circles(0.0)
    assert radial_coordinate_map(eq, time=0.25).time == 0.25
    with pytest.raises(ValueError, match="not the requested"):
        radial_coordinate_map(eq, time=0.30)


def test_a_surface_beyond_the_grid_is_unavailable_not_extrapolated():
    eq = shifted_circles(0.0)
    # extend psi_1d past the last surface the grid crosses on the inboard side
    psi_1d = PSI_EDGE * np.linspace(0.0, 2.5, 101)
    cmap = radial_coordinate_map(replace(eq, psi_1d=psi_1d, q=np.ones(101), psi_boundary=PSI_EDGE))
    r_minor = cmap["r_minor"].value
    assert np.isnan(r_minor[-1]) and np.isfinite(r_minor[0])


# --------------------------------------------------------------------------
# the gradient
# --------------------------------------------------------------------------


def test_analytic_log_gradient_in_r(circular):
    r = np.linspace(0.0, A, 81)
    result = profile_gradient(np.exp(-(r**2) / WIDTH**2), r, "r_minor", equilibrium=circular,
                              reference_length=None)
    np.testing.assert_allclose(result.values, 2 * r / WIDTH**2, atol=1e-9)
    assert result.unit == "m^-1"
    assert result.metadata["mathematical_definition"] == "-d(log(f)) / d(r_minor)"


def test_the_chain_rule_from_rho_tor_to_r_minor(shifted):
    rho = np.linspace(0.0, 1.0, 51)
    r = r_of_rho_tor(rho)
    result = profile_gradient(np.exp(-(r**2) / WIDTH**2), rho, "rho_tor_norm", equilibrium=shifted,
                              gradient_coordinate="r_minor", reference_length=None)
    expected = 2 * r / WIDTH**2
    np.testing.assert_allclose(result.values, expected, rtol=0, atol=2e-3 * expected.max())
    # the derivative in rho_tor itself is a different number
    in_rho = profile_gradient(np.exp(-(r**2) / WIDTH**2), rho, "rho_tor_norm", equilibrium=shifted,
                              gradient_coordinate="rho_tor_norm", reference_length=None)
    assert np.max(np.abs(in_rho.values - result.values)) > 1.0
    assert in_rho.unit == "1"


def test_the_chain_rule_from_psi_norm_to_r_minor(shifted):
    # psi_norm is singular against a radius at the axis; the derivative must still be right there
    psi = np.linspace(0.0, 1.0, 101) ** 2
    r = A * np.sqrt(psi)
    result = profile_gradient(np.exp(-(r**2) / WIDTH**2), psi, "psi_norm", equilibrium=shifted,
                              gradient_coordinate="r_minor", reference_length=None)
    expected = 2 * r / WIDTH**2
    np.testing.assert_allclose(result.values, expected, rtol=0, atol=1e-3 * expected.max())


def test_a_flux_label_gradient_at_the_axis_is_refused_not_zero(shifted):
    r = np.linspace(0.0, A, 41)
    with pytest.raises(ValueError, match="stationary"):
        profile_gradient(np.exp(-(r**2) / WIDTH**2), r, "r_minor", equilibrium=shifted,
                         gradient_coordinate="psi_norm", reference_length=None)
    inside = profile_gradient(np.exp(-(r[1:] ** 2) / WIDTH**2), r[1:], "r_minor", equilibrium=shifted,
                              gradient_coordinate="psi_norm", reference_length=None)
    np.testing.assert_allclose(inside.values, A**2 / WIDTH**2, rtol=2e-3)


def test_coordinate_and_gradient_coordinate_are_independent(shifted):
    rho = np.linspace(0.0, 1.0, 51)
    r = r_of_rho_tor(rho)
    profile = np.exp(-(r**2) / WIDTH**2)
    on_rho = profile_gradient(profile, rho, "rho_tor_norm", equilibrium=shifted,
                              coordinate="rho_tor_norm", reference_length="a_minor")
    on_r = profile_gradient(profile, rho, "rho_tor_norm", equilibrium=shifted,
                            coordinate="r_minor_norm", reference_length="a_minor")
    np.testing.assert_array_equal(on_rho.values, on_r.values)
    np.testing.assert_allclose(on_r.coordinate, r / A, atol=1e-5)
    assert on_rho.metadata["plot_coordinate"] == "rho_tor_norm"
    assert on_rho.metadata["gradient_coordinate"] == "r_minor"


def test_a_over_l_and_r_over_l_differ_exactly_by_r0_over_a(shifted):
    r = np.linspace(0.0, A, 41)
    profile = 2.0 - (r / A) ** 2
    a_l = profile_gradient(profile, r, "r_minor", equilibrium=shifted, reference_length="a_minor")
    r_l = profile_gradient(profile, r, "r_minor", equilibrium=shifted, reference_length="R_major_axis")
    raw = profile_gradient(profile, r, "r_minor", equilibrium=shifted, reference_length=None)
    ratio = shifted.R_major_axis.value / shifted.a_minor.value
    np.testing.assert_allclose(r_l.values[1:], ratio * a_l.values[1:], rtol=1e-12)
    np.testing.assert_allclose(a_l.values, shifted.a_minor.value * raw.values, rtol=1e-12)
    assert r_l.metadata["mathematical_definition"] == "-R_0 * d(log(f)) / d(r_minor)"


def test_the_reference_length_keeps_its_definition_and_provenance(shifted):
    r = np.linspace(0.0, A, 21)
    result = profile_gradient(2.0 - (r / A) ** 2, r, "r_minor", equilibrium=shifted,
                              reference_length="a_minor", source_quantity="T_e",
                              quantity="electron_temperature_gradient")
    record = result.metadata["reference_length"]
    assert record["symbol"] == "a" and record["unit"] == "m"
    assert record["value"] == pytest.approx(A, rel=1e-6)
    assert "midplane" in record["definition"]
    assert record["provenance"]["source_time"] == 0.25
    assert record["provenance"]["convention"]["cocos"] == 11
    assert result.metadata["quantity"] == "electron_temperature_gradient"
    assert result.metadata["source_quantity"] == "T_e"
    assert result.metadata["mathematical_definition"] == "-a * d(log(T_e)) / d(r_minor)"


def test_the_local_major_radius_is_a_profile(shifted):
    r = np.linspace(0.0, A, 21)
    local = profile_gradient(2.0 - (r / A) ** 2, r, "r_minor", equilibrium=shifted,
                             reference_length="R_major_surface")
    raw = profile_gradient(2.0 - (r / A) ** 2, r, "r_minor", equilibrium=shifted, reference_length=None)
    centre = R0 + 0.03 * (1 - (r / A) ** 2)
    np.testing.assert_allclose(local.values, centre * raw.values, rtol=1e-5, atol=1e-12)


def test_an_explicit_l_ref_needs_its_definition(shifted):
    with pytest.raises(ValueError, match="definition"):
        resolve_reference_length("L_ref", shifted)
    with pytest.raises(ValueError, match="no definition"):
        resolve_reference_length("L_ref", shifted, {"reference_length": {"value": 0.3, "unit": "m"}})
    length = resolve_reference_length("L_ref", shifted, {"reference_length": {
        "value": 0.3, "unit": "m", "definition": "the run's Lref", "symbol": "L_ref"}})
    assert length.value == 0.3 and length.length.definition == "the run's Lref"


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_a_non_monotonic_mapping_is_refused():
    r_minor = np.array([0.0, 0.05, 0.10, 0.08, 0.12, 0.2])
    cmap = radial_coordinate_map_from_arrays(r_inboard=R0 - r_minor, r_outboard=R0 + r_minor,
                                             psi_norm=np.linspace(0, 1, 6))
    with pytest.raises(ValueError, match="monotonic"):
        profile_gradient(np.linspace(2, 1, 6), np.linspace(0, 1, 6), "psi_norm",
                         equilibrium=cmap, reference_length="a_minor")


def test_nothing_is_extrapolated(circular):
    rho = np.linspace(0.0, 1.1, 23)  # past the boundary
    with pytest.raises(ValueError, match="outside the radial support"):
        profile_gradient(2.0 - rho**2, rho, "rho_tor_norm", equilibrium=circular, reference_length=None)
    rho = np.linspace(0.0, 0.9, 19)
    with pytest.raises(ValueError, match="outside the radial support"):
        profile_gradient(2.0 - rho**2, rho, "rho_tor_norm", equilibrium=circular,
                         reference_length=None, at=[0.95])


def test_the_sqrt_psi_proxy_is_not_a_toroidal_radius():
    psi = np.linspace(0, 1, 11)
    r_minor = 0.2 * np.sqrt(psi)
    cmap = radial_coordinate_map_from_arrays(r_inboard=R0 - r_minor, r_outboard=R0 + r_minor,
                                             psi_norm=psi, rho_tor_norm=np.sqrt(psi))
    assert not cmap["rho_tor_norm"].available
    assert "proxy" in cmap["rho_tor_norm"].reason
    with pytest.raises(ValueError, match="proxy"):
        profile_gradient(2 - psi, psi, "psi_norm", equilibrium=cmap, coordinate="rho_tor_norm",
                         reference_length="a_minor")


@pytest.mark.parametrize("kwargs, match", [
    ({"reference_length": "a_minor", "profile": [1.0, 0.0, 0.5]}, "strictly positive"),
    ({}, "state reference_length"),
    ({"reference_length": "a_minor", "gradient_coordinate": "psi_norm"}, "dimensionless"),
    ({"reference_length": "R_major"}, "must be one of"),
    ({"reference_length": None, "coordinate": "a_minor"}, "not a radial coordinate"),
])
def test_ill_posed_requests_are_refused(circular, kwargs, match):
    profile = kwargs.pop("profile", [3.0, 2.0, 1.0])
    with pytest.raises(ValueError, match=match):
        profile_gradient(profile, [0.0, 0.1, 0.2], "r_minor", equilibrium=circular, **kwargs)


def test_a_time_the_record_cannot_confirm_is_refused(shifted):
    with pytest.raises(ValueError, match="cannot be confirmed"):
        radial_coordinate_map(replace(shifted_circles(0.0), time=None), time=0.25)
    with pytest.raises(ValueError, match="not the requested"):
        profile_gradient([3.0, 2.0, 1.0], [0.0, 0.1, 0.2], "r_minor", equilibrium=shifted,
                         reference_length=None, time=0.4)


def test_an_inward_or_unitless_l_ref_is_refused(shifted):
    with pytest.raises(ValueError, match="not increasing outward"):
        profile_gradient([3.0, 2.0, 1.0], [0.0, 0.1, 0.2], "r_minor", equilibrium=shifted,
                         gradient_coordinate="r_inboard", reference_length="a_minor")
    with pytest.raises(ValueError, match="no unit"):
        resolve_reference_length("L_ref", shifted, {"reference_length": {
            "value": 0.3, "definition": "the run's Lref"}})


def test_a_minor_without_an_lcfs_surface_is_refused():
    eq = shifted_circles(0.0)
    cmap = radial_coordinate_map(replace(eq, psi_1d=PSI_EDGE * np.linspace(0, 0.9, 101)))
    assert not cmap.a_minor.available
    with pytest.raises(ValueError, match="unavailable"):
        resolve_reference_length("a_minor", cmap)


def test_a_resampled_boundary_surface_still_gives_a_minor():
    eq = shifted_circles(0.0)
    reference = radial_coordinate_map(eq)
    # a psi_1d that lands 1e-7 short of the boundary flux, as a resampled grid does
    cmap = radial_coordinate_map(replace(eq, psi_1d=eq.psi_1d * (1.0 - 1e-7)))
    assert cmap.a_minor.available
    assert cmap.a_minor.value == pytest.approx(reference.a_minor.value, rel=1e-6)


def test_a_boundary_mismatch_is_reported_with_its_size():
    eq = shifted_circles(0.0)
    cmap = radial_coordinate_map(replace(eq, psi_1d=eq.psi_1d * (1.0 - 1e-4)))
    assert not cmap.a_minor.available
    assert "|psi_norm - 1| = 0.0001" in cmap.a_minor.reason
    assert "0.9999" in cmap.a_minor.reason


# --------------------------------------------------------------------------
# conventions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["tglf", "cgyro"])
def test_gacode_presets_equal_the_explicit_configuration(shifted, name):
    rho = np.linspace(0.0, 1.0, 31)
    profile = 2.0 - rho**2
    preset = profile_gradient(profile, rho, "rho_tor_norm", equilibrium=shifted, convention=name)
    explicit = profile_gradient(profile, rho, "rho_tor_norm", equilibrium=shifted,
                                gradient_coordinate="r_minor", reference_length="a_minor")
    np.testing.assert_array_equal(preset.values, explicit.values)
    assert preset.metadata["requested_convention"] == name
    assert preset.metadata["resolved_convention"]["gradient_coordinate"] == "r_minor"
    assert preset.metadata["reference_length"] == explicit.metadata["reference_length"]
    assert "gacode.io" in preset.metadata["resolved_convention"]["source"]


def test_a_preset_contradicted_by_an_explicit_argument_is_refused(shifted):
    with pytest.raises(ValueError, match="contradicts"):
        profile_gradient([3.0, 2.0, 1.0], [0.0, 0.1, 0.2], "r_minor", equilibrium=shifted,
                         convention="tglf", reference_length="R_major_axis")


@pytest.mark.parametrize("name", ["gs2", "gkw", "gene"])
def test_configuration_dependent_codes_refuse_without_metadata(name):
    with pytest.raises(ValueError, match="nothing is assumed"):
        resolve_convention(name)


def test_gene_is_never_a_fixed_major_radius():
    assert CONVENTIONS["gene"].reference_length is None
    resolved = resolve_convention("gene", {
        "gradient_coordinate": "r_minor",
        "reference_length": {"value": 0.25, "unit": "m", "definition": "Lref = minor_r * a"}})
    assert resolved["reference_length"] == "L_ref"


def test_gs2_resolves_only_what_its_documentation_fixes():
    assert resolve_convention("gs2", {"irho": 2, "local_eq": False})["reference_length"] == "a_minor"
    miller = resolve_convention("gs2", {"irho": 2, "local_eq": True, "reference_length": {
        "value": 0.3, "unit": "m", "definition": "the run's L_ref"}})
    assert (miller["gradient_coordinate"], miller["reference_length"]) == ("r_minor", "L_ref")
    for irho in (1, 3, 4):
        with pytest.raises(ValueError, match="not resolved"):
            resolve_convention("gs2", {"irho": irho, "local_eq": False})
    with pytest.raises(ValueError, match="L_ref"):
        resolve_convention("gs2", {"irho": 2, "local_eq": True})


def test_gkw_resolves_from_its_geometry(shifted):
    resolved = resolve_convention("gkw", {"geom_type": "circ"})
    assert (resolved["gradient_coordinate"], resolved["reference_length"]) == ("r_minor", "R_major_axis")
    r = np.linspace(0.0, A, 21)
    preset = profile_gradient(2 - (r / A) ** 2, r, "r_minor", equilibrium=shifted, convention="gkw",
                              metadata={"geom_type": "circ"})
    explicit = profile_gradient(2 - (r / A) ** 2, r, "r_minor", equilibrium=shifted,
                                reference_length="R_major_axis")
    np.testing.assert_array_equal(preset.values, explicit.values)
    assert preset.metadata["resolved_convention"]["metadata_used"] == {"geom_type": "circ"}
    for geometry in ("miller", "chease"):
        with pytest.raises(ValueError, match="not resolved"):
            resolve_convention("gkw", {"geom_type": geometry})


# --------------------------------------------------------------------------
# the locpargen oracle
# --------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from test_tglf_input import RADII, SAMPLE, read_oracle, requires_sample  # noqa: E402


@pytest.fixture(scope="module")
def gacode():
    from omas import load_omas_json

    from vaft.code.gacode.inputs import prepare_gacode_profile

    ods = load_omas_json(str(SAMPLE), consistency_check=False)
    profile = prepare_gacode_profile(ods, rho_max=0.95, z_eff=2.0, impurity="C")
    rmin, rmaj = np.asarray(profile.rmin), np.asarray(profile.rmaj)
    cmap = radial_coordinate_map_from_arrays(
        r_inboard=rmaj - rmin, r_outboard=rmaj + rmin, rho_tor_norm=profile.rho, source="input.gacode")
    return profile, cmap


@requires_sample
@pytest.mark.parametrize("rho", RADII)
def test_the_tglf_preset_reproduces_locpargen(gacode, rho):
    profile, cmap = gacode
    oracle = read_oracle(rho)
    rmin = np.asarray(profile.rmin)

    def at(values, grid=rmin, coordinate="r_minor"):
        return profile_gradient(values, grid, coordinate, equilibrium=cmap, coordinate="r_minor_norm",
                                convention="tglf", at=rho).values[0]

    assert at(profile.te) == pytest.approx(oracle["RLTS_1"], rel=1e-4)
    assert at(profile.ne) == pytest.approx(oracle["RLNS_1"], rel=1e-4)
    assert at(np.atleast_2d(profile.ti)[0]) == pytest.approx(oracle["RLTS_2"], rel=1e-4)
    assert at(np.atleast_2d(profile.ni)[0]) == pytest.approx(oracle["RLNS_2"], rel=1e-4)
    # the same T_i given on rho_tor goes through the chain rule and lands close by
    via_rho = at(np.atleast_2d(profile.ti)[0], grid=np.asarray(profile.rho), coordinate="rho_tor_norm")
    assert via_rho == pytest.approx(oracle["RLTS_2"], rel=5e-3, abs=2e-3)


def test_an_ods_slice_whose_own_time_disagrees_with_the_vector_is_refused():
    """``equilibrium.time`` chooses the slice index; the slice's own ``time``
    leaf must confirm it. A record whose vector says 0.320 s at index 1 while
    the slice says 0.330 s was rewritten out of step, and the ODS branch used
    to return that slice silently where the EquilibriumData branch refuses
    (cold review 0.8.0 delta-absorb-3 F1)."""
    import copy

    import numpy as np
    import pytest
    from omas import ODS

    from vaft.process.profile_gradients import _select_slice
    from vaft.process._equilibrium_parametric import as_equilibrium

    import gzip
    import tempfile

    from omas import load_omas_json

    from vaft.data.resources import data_path

    with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as raw:
        raw.write(gzip.open(str(data_path("samples/39915/omas.json.gz"))).read())
    ods = load_omas_json(raw.name, consistency_check=False)
    src = ODS(consistency_check=False)
    first = ods["equilibrium.time_slice.0"]
    src["equilibrium.time_slice.0"] = copy.deepcopy(first)
    src["equilibrium.time_slice.1"] = copy.deepcopy(first)
    src["equilibrium.time_slice.0.time"] = 0.300
    src["equilibrium.time_slice.1.time"] = 0.330
    src["equilibrium.time"] = np.array([0.300, 0.320])
    src["equilibrium.vacuum_toroidal_field.r0"] = ods["equilibrium.vacuum_toroidal_field.r0"]
    b0 = np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"]).reshape(-1)
    src["equilibrium.vacuum_toroidal_field.b0"] = np.array([b0[0], b0[0]])

    with pytest.raises(ValueError, match="equilibrium slice 1 is at t = 0.33"):
        _select_slice(src, 0.320, as_equilibrium)
    assert _select_slice(src, 0.300, as_equilibrium).time == pytest.approx(0.300)

