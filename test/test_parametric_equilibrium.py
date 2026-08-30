"""Regression tests for issue #65 equilibrium representations and algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.data.equilibrium import (
    Contour,
    EquilibriumConvention,
    EquilibriumData,
    MillerSurface,
    SolovevConstraint,
    SolovevEquilibrium,
    Topology,
)
from vaft.data.resources import sample_geqdsk
from vaft.process.equilibrium import (
    as_equilibrium,
    convert_cocos,
    derive_boundary_representation,
    derive_global_descriptors,
    derive_radial_coordinates,
    evaluate_miller,
    evaluate_solovev,
    fit_miller_sequence,
    fit_miller_surface,
    solve_solovev_constraints,
    validate_equilibrium,
)


def _analytic_equilibrium(kind: str = "limited") -> EquilibriumData:
    r = np.linspace(0.5, 1.5, 121)
    z = np.linspace(-0.75, 0.75, 151)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    if kind == "limited":
        psi = ((rm - 1.0) / 0.35) ** 2 + (zm / 0.5) ** 2
        boundary = 1.0
        theta = np.linspace(0, 2 * np.pi, 361, endpoint=False)
        lcfs = Contour(1.0 + 0.35 * np.cos(theta), 0.5 * np.sin(theta))
    else:
        a = 0.65
        psi = (rm - 1.0) ** 2 + zm**2 - zm**4 / a**2
        boundary = a**2 / 4
        # Extract the LCFS using the same orientation contract as production.
        from vaft.process.equilibrium import extract_flux_surface_contours

        raw = extract_flux_surface_contours(psi, r, z, 0.0, boundary, [1.0])[1.0]
        selected = max(raw, key=lambda pair: pair[0].size)
        lcfs = Contour(selected[0], selected[1])
    theta_wall = np.linspace(0, 2 * np.pi, 361, endpoint=False)
    limiter = Contour(1.0 + 0.48 * np.cos(theta_wall), 0.68 * np.sin(theta_wall))
    psi_1d = np.linspace(0, boundary, r.size)
    psi_n = psi_1d / boundary
    return EquilibriumData(
        r=r, z=z, psi=psi, psi_axis=0.0, psi_boundary=boundary,
        magnetic_axis=(1.0, 0.0), lcfs=lcfs, limiter=limiter,
        psi_1d=psi_1d, pressure=2e4 * (1 - psi_n**2),
        f=np.full(r.size, 2.0), q=1.0 + 2.0 * psi_n,
        ip=1e6, bt0=2.0, r0=1.0,
        convention=EquilibriumConvention(11, (11,), False, False, 1, 1, 1, "test"),
        metadata={"source_type": "synthetic"},
    )


def test_models_validate_shapes_and_missing_fields_are_actionable():
    with pytest.raises(ValueError, match="equal length"):
        Contour([1, 2], [0])
    report = validate_equilibrium(EquilibriumData(), required_for="global")
    assert not report.valid
    assert {item.code for item in report.issues} >= {"missing_grid", "missing_flux_bounds", "missing_lcfs"}


def test_geqdsk_and_ods_adapters_are_numerically_equivalent():
    source = sample_geqdsk()
    direct = as_equilibrium(source, convention=11)
    through_ods = as_equilibrium(source.to_omas(), convention=11)
    np.testing.assert_allclose(direct.r, through_ods.r)
    np.testing.assert_allclose(direct.z, through_ods.z)
    np.testing.assert_allclose(direct.psi, through_ods.psi)
    np.testing.assert_allclose(direct.q, through_ods.q)
    assert direct.ip == through_ods.ip


def test_native_ids_adapter_reads_the_repository_equilibrium_fixture():
    import vaft.data
    from vaft import imas

    source = vaft.data.sample(39915, representation="imas")
    with imas.load(source) as handle:
        native = handle.get("equilibrium")
        equilibrium = as_equilibrium(native, convention=11)
    assert equilibrium.psi.shape == (equilibrium.r.size, equilibrium.z.size)
    assert equilibrium.lcfs.r.size >= 3
    assert equilibrium.ip is not None


def test_ambiguous_cocos_is_preserved_and_conversion_round_trips():
    ambiguous = as_equilibrium(sample_geqdsk())
    assert ambiguous.convention.cocos is None
    assert len(ambiguous.convention.candidates) > 1
    with pytest.raises(ValueError, match="ambiguous"):
        convert_cocos(ambiguous, 2)
    original = as_equilibrium(sample_geqdsk(), convention=11)
    restored = convert_cocos(convert_cocos(original, 2), 11)
    np.testing.assert_allclose(restored.psi, original.psi)
    np.testing.assert_allclose(restored.f, original.f)
    np.testing.assert_allclose(restored.q, original.q)
    assert restored.ip == pytest.approx(original.ip)
    assert restored.bt0 == pytest.approx(original.bt0)


def test_global_descriptors_and_radial_coordinates_have_definitions():
    eq = _analytic_equilibrium()
    descriptors = derive_global_descriptors(eq, rational_q=(1.5, 2.0))
    assert descriptors.validation.valid
    assert descriptors["major_radius"].value == pytest.approx(1.0, rel=2e-3)
    assert descriptors["minor_radius"].value == pytest.approx(0.35, rel=2e-3)
    assert descriptors["elongation"].value == pytest.approx(0.5 / 0.35, rel=2e-3)
    assert descriptors["volume"].value == pytest.approx(2 * np.pi**2 * 1.0 * 0.35 * 0.5, rel=2e-3)
    assert descriptors["thermal_energy"].available
    assert descriptors["beta_t"].available
    assert len(descriptors.rational_surfaces[1.5]) == 1
    radial = derive_radial_coordinates(eq)
    np.testing.assert_allclose(radial["psi_n"].value[[0, -1]], [0, 1])
    np.testing.assert_allclose(radial["rho_pol_n"].value**2, radial["psi_n"].value)
    assert radial["rho_tor_n"].available


def test_nonmonotonic_toroidal_coordinate_is_unavailable():
    eq = _analytic_equilibrium()
    eq = EquilibriumData(**{**eq.__dict__, "q": np.linspace(-1, 1, eq.q.size)})
    value = derive_radial_coordinates(eq)["rho_tor_n"]
    assert not value.available
    assert "monotonic" in value.reason


def test_miller_exact_and_noisy_round_trip_and_edge_rejection():
    expected = MillerSurface(0.22, 0.9, -0.03, 1.7, 0.32)
    theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    r, z = evaluate_miller(expected, theta)
    exact = fit_miller_surface(Contour(r, z))
    assert exact.accepted
    assert exact.normalized_rms_error < 1e-4
    assert exact.surface.kappa == pytest.approx(expected.kappa, rel=2e-3)
    assert exact.surface.delta == pytest.approx(expected.delta, abs=2e-3)
    rng = np.random.default_rng(65)
    noisy = fit_miller_surface(Contour(r + rng.normal(0, 2e-4, r.size), z + rng.normal(0, 2e-4, z.size)))
    assert noisy.accepted and noisy.normalized_rms_error < 0.02
    edge = fit_miller_surface(Contour(r, z), radial_value=0.999)
    assert not edge.accepted and "psi_n" in edge.reason


def test_miller_sequence_recovers_radial_derivatives():
    # Nested exact Miller surfaces are supplied as an analytic psi grid only for
    # contour extraction; circular surfaces make the derivative expectation exact.
    eq = _analytic_equilibrium()
    sequence = fit_miller_sequence(eq, [0.16, 0.25, 0.36, 0.49, 0.64])
    accepted = [item for item in sequence.fits if item.accepted]
    assert len(accepted) >= 4
    assert sequence.derivative_reason is None
    assert all(item.surface.d_kappa_dr is not None for item in accepted)
    assert all(item.surface.magnetic_shear is not None for item in accepted)


def test_solovev_fields_satisfy_grad_shafranov_and_constraints_recover_coefficients():
    model = SolovevEquilibrium(np.array([0.03, -0.02, 0.015, 0.004, -0.001]), -1200.0, 0.08, 1.0)
    r = np.linspace(0.7, 1.3, 101); z = np.linspace(-0.4, 0.4, 91)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    values = evaluate_solovev(model, rm, zm)
    dpsi_dr = np.gradient(values["psi"], r, axis=0, edge_order=2)
    laplace_star = np.gradient(dpsi_dr, r, axis=0, edge_order=2) - dpsi_dr / rm + np.gradient(np.gradient(values["psi"], z, axis=1, edge_order=2), z, axis=1, edge_order=2)
    np.testing.assert_allclose(laplace_star[3:-3, 3:-3], values["grad_shafranov_source"][3:-3, 3:-3], rtol=3e-3, atol=2e-4)
    locations = [(0.75, -0.25), (0.8, 0.2), (0.95, -0.1), (1.05, 0.3), (1.2, -0.3), (1.25, 0.1), (1.0, 0.0)]
    constraints = [SolovevConstraint(rr, zz, "psi", float(evaluate_solovev(model, rr, zz)["psi"])) for rr, zz in locations]
    solved = solve_solovev_constraints(constraints, pprime=model.pprime, ffprime=model.ffprime, rref=model.rref)
    np.testing.assert_allclose(solved.coefficients, model.coefficients, rtol=1e-9, atol=1e-10)
    assert solved.rank == 5 and solved.residual_norm < 1e-10
    with pytest.raises(ValueError, match="five"):
        solve_solovev_constraints(constraints[:4], pprime=-1, ffprime=0, rref=1)


def test_boundary_limiter_and_double_null_classification_and_gaps():
    limited = derive_boundary_representation(_analytic_equilibrium("limited"), fourier_modes=12)
    assert limited.topology == Topology.LIMITER
    assert all(gap.distance.available for gap in limited.gaps)
    assert limited.fourier_reconstruction_error.value < 2e-3
    double = derive_boundary_representation(_analytic_equilibrium("double"), flux_tolerance=2e-2)
    assert double.topology == Topology.DOUBLE_NULL
    assert len([point for point in double.x_points if point.active]) >= 2
    assert double.d_r_sep.available
    assert abs(double.d_r_sep.value) < 5e-3


def test_descriptors_are_invariant_under_cocos_family_conversion():
    """Bugbot #204-1: psi-derived fields must honor the 2*pi weber convention.

    Converting COCOS 2 (psi per radian) to COCOS 12 (full weber) scales psi by
    2*pi; beta_p, the Shafranov integrals, and li are physical quantities and
    must not change with the bookkeeping convention.
    """
    base = as_equilibrium(sample_geqdsk(), convention=2)
    converted = convert_cocos(base, 12)
    d_base = derive_global_descriptors(base)
    d_conv = derive_global_descriptors(converted)
    for name in ("beta_p_boundary_average", "s1", "li_virial"):
        assert d_base[name].available and d_conv[name].available
        assert d_conv[name].value == pytest.approx(d_base[name].value, rel=1e-6), name


def test_miller_sequence_is_invariant_to_level_ordering():
    """Bugbot #204-2: q/shear/alpha must follow the radius sorting."""
    eq = _analytic_equilibrium()
    levels = [0.16, 0.25, 0.36, 0.49, 0.64]
    forward = fit_miller_sequence(eq, levels)
    reverse = fit_miller_sequence(eq, levels[::-1])
    assert forward.derivative_reason is None and reverse.derivative_reason is None

    def by_level(sequence):
        return {
            round(item.surface.radial_value, 6): (item.surface.q, item.surface.magnetic_shear)
            for item in sequence.fits
            if item.accepted
        }

    fwd, rev = by_level(forward), by_level(reverse)
    assert set(fwd) == set(rev)
    for level in fwd:
        assert fwd[level][0] == pytest.approx(rev[level][0], rel=1e-9), level
        assert fwd[level][1] == pytest.approx(rev[level][1], rel=1e-6), level
    # The fixture q profile rises with psi_n, so q must rise with radius too.
    ordered_q = [fwd[level][0] for level in sorted(fwd)]
    assert ordered_q == sorted(ordered_q)


def test_outboard_radius_uses_the_midplane_not_the_contour_maximum():
    """Bugbot #204-3: dRsep must sample R_out at z0, not max(R) of the contour."""
    from vaft.process._equilibrium_parametric import _outboard_radius_at_z

    contour = Contour(
        np.array([1.8, 2.2, 1.0, 0.4, 1.0, 1.8]),
        np.array([0.4, 0.6, 1.0, 0.0, -1.0, -0.6]),
    )
    assert float(np.max(contour.r)) == pytest.approx(2.2)  # off-midplane bulge
    assert _outboard_radius_at_z(contour, 0.0) == pytest.approx(1.8)
    assert _outboard_radius_at_z(Contour(np.array([1.0, 2.0]), np.array([1.0, 2.0])), 0.0) is None


def test_solovev_axis_is_the_o_point_not_a_grid_corner():
    """Bugbot #204-4: with the axis omitted, the O-point must be found even on
    a generous grid window where |psi - psi_boundary| peaks at a domain corner."""
    from vaft.process.equilibrium import solovev_to_equilibrium

    R0, a, b = 1.05, 0.35, 0.5
    constraints = [
        SolovevConstraint(R0 + a, 0.0, "psi", 0.0),
        SolovevConstraint(R0 - a, 0.0, "psi", 0.0),
        SolovevConstraint(R0, b, "psi", 0.0),
        SolovevConstraint(R0 + 0.6 * a, 0.75 * b, "psi", 0.0),
        SolovevConstraint(R0, 0.0, "dpsi_dr", 0.0),
    ]
    model = solve_solovev_constraints(constraints, pprime=-1200.0, ffprime=0.08, rref=1.0)
    r = np.linspace(0.3, 3.2, 161)
    z = np.linspace(-2.0, 2.0, 161)
    psi = evaluate_solovev(model, *np.meshgrid(r, z, indexing="ij"))["psi"]
    raw = np.unravel_index(np.argmax(np.abs(psi - model.psi_boundary)), psi.shape)
    assert raw[0] in (0, r.size - 1) or raw[1] in (0, z.size - 1)  # the old pick: a corner
    eq = solovev_to_equilibrium(model, r, z)
    assert eq.magnetic_axis[0] == pytest.approx(R0, abs=5e-3)
    assert eq.magnetic_axis[1] == pytest.approx(0.0, abs=5e-3)
