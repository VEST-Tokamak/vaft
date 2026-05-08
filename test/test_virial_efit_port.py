import numpy as np
import importlib.util
from pathlib import Path
from omas import load_omas_json

from vaft.formula.equilibrium import (
    virial_bongard_from_S_alpha_mu,
    virial_lao_from_S_alpha_mu_rt,
    virial_beta_pd_from_S_mu_rt,
)
from vaft.omas.process_wrapper import compute_virial_equilibrium_quantities_ods


_PROC_EQ_PATH = Path(__file__).resolve().parents[1] / "vaft" / "process" / "equilibrium.py"
_PROC_EQ_SPEC = importlib.util.spec_from_file_location("vaft_process_equilibrium", _PROC_EQ_PATH)
_PROC_EQ = importlib.util.module_from_spec(_PROC_EQ_SPEC)
assert _PROC_EQ_SPEC is not None and _PROC_EQ_SPEC.loader is not None
_PROC_EQ_SPEC.loader.exec_module(_PROC_EQ)

computed_diamagnetism_from_phi = _PROC_EQ.computed_diamagnetism_from_phi
efit_virial_volume_integrals = _PROC_EQ.efit_virial_volume_integrals
fractional_cell_weights_from_boundary = _PROC_EQ.fractional_cell_weights_from_boundary
shafranov_integrals = _PROC_EQ.shafranov_integrals


def _ellipse_boundary(npts: int = 361, center_r: float = 2.0, center_z: float = 0.3):
    th = np.linspace(0.0, 2.0 * np.pi, npts, endpoint=True)
    r = center_r + 0.4 * np.cos(th)
    z = center_z + 0.25 * np.sin(th)
    return r, z


def test_shafranov_is_orientation_invariant_after_normalization():
    r_bdry, z_bdry = _ellipse_boundary()
    bp_bdry = np.full_like(r_bdry, 0.8)

    r = np.linspace(1.2, 2.8, 61)
    z = np.linspace(-0.3, 0.9, 51)
    r_grid, z_grid = np.meshgrid(r, z, indexing="ij")

    # Keep B fields simple so alpha remains deterministic.
    b_r = np.zeros_like(r_grid)
    b_z = np.ones_like(r_grid)

    s1_ccw, s2_ccw, s3_ccw, alpha_ccw = shafranov_integrals(
        r_bdry,
        z_bdry,
        bp_bdry,
        r_grid,
        z_grid,
        b_r,
        b_z,
        R_0=1.6,
        Z_0=0.0,
        p_boundary=0.0,
        B_ref=1.0,
    )

    s1_cw, s2_cw, s3_cw, alpha_cw = shafranov_integrals(
        r_bdry[::-1],
        z_bdry[::-1],
        bp_bdry[::-1],
        r_grid,
        z_grid,
        b_r,
        b_z,
        R_0=1.6,
        Z_0=0.0,
        p_boundary=0.0,
        B_ref=1.0,
    )

    np.testing.assert_allclose(s1_cw, s1_ccw, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(s2_cw, s2_ccw, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(s3_cw, s3_ccw, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(alpha_cw, alpha_ccw, rtol=1e-12, atol=1e-12)


def test_shafranov_is_stable_to_boundary_downsampling():
    r_bdry, z_bdry = _ellipse_boundary(npts=721)
    # Mildly varying profile along contour to exercise Bp interpolation sensitivity.
    theta = np.linspace(0.0, 2.0 * np.pi, r_bdry.size, endpoint=True)
    bp_bdry = 0.7 + 0.08 * np.cos(2.0 * theta)

    r = np.linspace(1.2, 2.8, 61)
    z = np.linspace(-0.3, 0.9, 51)
    r_grid, z_grid = np.meshgrid(r, z, indexing="ij")
    b_r = np.zeros_like(r_grid)
    b_z = np.ones_like(r_grid)

    s1_ref, s2_ref, s3_ref, _ = shafranov_integrals(
        r_bdry,
        z_bdry,
        bp_bdry,
        r_grid,
        z_grid,
        b_r,
        b_z,
        R_0=1.6,
        Z_0=0.0,
        p_boundary=0.0,
        B_ref=1.0,
    )

    for step in (2, 4, 8):
        s1, s2, s3, _ = shafranov_integrals(
            r_bdry[::step],
            z_bdry[::step],
            bp_bdry[::step],
            r_grid,
            z_grid,
            b_r,
            b_z,
            R_0=1.6,
            Z_0=0.0,
            p_boundary=0.0,
            B_ref=1.0,
        )
        np.testing.assert_allclose(s1, s1_ref, rtol=3e-2, atol=1e-6)
        np.testing.assert_allclose(s2, s2_ref, rtol=3e-2, atol=1e-6)
        np.testing.assert_allclose(s3, s3_ref, rtol=3e-2, atol=1e-6)


def test_shafranov_is_invariant_to_open_or_duplicate_boundary_points():
    r_bdry, z_bdry = _ellipse_boundary(npts=241)
    bp_bdry = np.full_like(r_bdry, 0.8)

    r = np.linspace(1.2, 2.8, 61)
    z = np.linspace(-0.3, 0.9, 51)
    r_grid, z_grid = np.meshgrid(r, z, indexing="ij")
    b_r = np.zeros_like(r_grid)
    b_z = np.ones_like(r_grid)

    s1_ref, s2_ref, s3_ref, _ = shafranov_integrals(
        r_bdry,
        z_bdry,
        bp_bdry,
        r_grid,
        z_grid,
        b_r,
        b_z,
        R_0=1.6,
        Z_0=0.0,
        p_boundary=0.0,
        B_ref=1.0,
    )

    # Open boundary input (no duplicated closing point).
    s1_open, s2_open, s3_open, _ = shafranov_integrals(
        r_bdry[:-1],
        z_bdry[:-1],
        bp_bdry[:-1],
        r_grid,
        z_grid,
        b_r,
        b_z,
        R_0=1.6,
        Z_0=0.0,
        p_boundary=0.0,
        B_ref=1.0,
    )
    np.testing.assert_allclose([s1_open, s2_open, s3_open], [s1_ref, s2_ref, s3_ref], rtol=1e-6, atol=1e-8)

    # Duplicate points inserted.
    r_dup = np.insert(r_bdry, [5, 11, 23], [r_bdry[5], r_bdry[11], r_bdry[23]])
    z_dup = np.insert(z_bdry, [5, 11, 23], [z_bdry[5], z_bdry[11], z_bdry[23]])
    bp_dup = np.insert(bp_bdry, [5, 11, 23], [bp_bdry[5], bp_bdry[11], bp_bdry[23]])
    s1_dup, s2_dup, s3_dup, _ = shafranov_integrals(
        r_dup,
        z_dup,
        bp_dup,
        r_grid,
        z_grid,
        b_r,
        b_z,
        R_0=1.6,
        Z_0=0.0,
        p_boundary=0.0,
        B_ref=1.0,
    )
    np.testing.assert_allclose([s1_dup, s2_dup, s3_dup], [s1_ref, s2_ref, s3_ref], rtol=1e-6, atol=1e-8)


def test_shafranov_axis_fallback_is_finite_when_axis_inputs_missing():
    r_bdry, z_bdry = _ellipse_boundary(npts=241)
    bp_bdry = np.full_like(r_bdry, 0.8)

    r = np.linspace(1.2, 2.8, 61)
    z = np.linspace(-0.3, 0.9, 51)
    r_grid, z_grid = np.meshgrid(r, z, indexing="ij")
    b_r = np.zeros_like(r_grid)
    b_z = np.ones_like(r_grid)

    s1, s2, s3, alpha = shafranov_integrals(
        r_bdry,
        z_bdry,
        bp_bdry,
        r_grid,
        z_grid,
        b_r,
        b_z,
        R_0=np.nan,
        Z_0=np.nan,
        p_boundary=0.0,
        B_ref=1.0,
    )
    assert np.isfinite(s1)
    assert np.isfinite(s2)
    assert np.isfinite(s3)
    assert np.isfinite(alpha)


def test_efit_volume_integrals_and_xmui_are_si_consistent():
    r = np.array([1.0, 2.0], dtype=float)
    z = np.array([0.0, 1.0], dtype=float)
    r_grid, z_grid = np.meshgrid(r, z, indexing="ij")

    # Rectangle fully containing the two-by-two mesh centers.
    r_bdry = np.array([0.5, 2.5, 2.5, 0.5, 0.5], dtype=float)
    z_bdry = np.array([-0.5, -0.5, 1.5, 1.5, -0.5], dtype=float)

    b_r = np.zeros_like(r_grid)
    b_z = np.ones_like(r_grid)
    p_tot = np.zeros_like(r_grid)

    f_grid = np.ones_like(r_grid) * 1.0
    f_boundary = 2.0
    b_phi = f_grid / r_grid
    b_phi_vac = f_boundary / r_grid

    terms = efit_virial_volume_integrals(
        r_grid,
        z_grid,
        r_bdry,
        z_bdry,
        b_r,
        b_z,
        p_tot_grid=p_tot,
        B_phi_grid=b_phi,
        B_phi_vac_grid=b_phi_vac,
        F_grid=f_grid,
        F_boundary=f_boundary,
    )

    # alpha = 2*sum(R*Bz^2)/sum(R*Bp^2) with Br=0,Bz=1 -> 2 exactly
    np.testing.assert_allclose(terms["alpha"], 2.0, rtol=1e-12, atol=1e-12)
    # RT with G = Bp^2 + Bphi_vac^2 - Bphi^2 = 1 + (4-1)/R^2
    assert np.isfinite(terms["rt"])
    # phi_dia_comp = -sum((Fb-F)/R * dA) = -(1+1+0.5+0.5) = -3
    np.testing.assert_allclose(terms["phi_dia_comp"], -3.0, rtol=1e-12, atol=1e-12)
    # V = sum(2*pi*R*dA) = 12*pi
    np.testing.assert_allclose(terms["volume"], 12.0 * np.pi, rtol=1e-12, atol=1e-12)

    mui = computed_diamagnetism_from_phi(
        phi_dia_comp=terms["phi_dia_comp"],
        B_t0=2.0,
        R_0=1.5,
        volume=terms["volume"],
        B_ref=0.5,
    )
    expected_mui = (4.0 * np.pi * 2.0 * 1.5 * (-3.0)) / ((12.0 * np.pi) * (0.5**2))
    np.testing.assert_allclose(mui, expected_mui, rtol=1e-12, atol=1e-12)


def test_fractional_cell_weights_from_boundary_has_fractional_cells():
    r = np.linspace(1.0, 2.0, 11)
    z = np.linspace(-0.5, 0.5, 11)
    r_grid, z_grid = np.meshgrid(r, z, indexing="ij")

    # Slanted polygon to force boundary-cut cells.
    r_bdry = np.array([1.1, 1.9, 1.7, 1.0, 1.1], dtype=float)
    z_bdry = np.array([-0.4, -0.3, 0.4, 0.2, -0.4], dtype=float)
    w = fractional_cell_weights_from_boundary(
        r_grid, z_grid, r_bdry, z_bdry, samples_per_axis=5
    )

    assert np.all(np.isfinite(w))
    assert np.all((w >= 0.0) & (w <= 1.0))
    # At least one boundary cell should be partially filled.
    assert np.any((w > 0.0) & (w < 1.0))


def test_virial_closure_relations_match_manual_forms():
    s1 = 2.2
    s2 = -0.4
    s3 = 0.6
    alpha = 1.7
    mui = -0.12
    rt_over_r0 = 0.92

    beta_p_lao, li_lao = virial_lao_from_S_alpha_mu_rt(
        s1, s2, s3, alpha, mui, rt_over_r0
    )
    beta_p_bongard, li_bongard = virial_bongard_from_S_alpha_mu(s1, s2, s3, alpha, mui)
    beta_pd = virial_beta_pd_from_S_mu_rt(s1, s2, mui, rt_over_r0)

    expected_li_lao = (0.5 * s1 + 0.5 * s2 * (1.0 - rt_over_r0) - s3) / (alpha - 1.0)
    expected_beta_p_lao = 0.5 * s1 + 0.5 * s2 * (1.0 + rt_over_r0) + mui
    expected_beta_p_bongard = ((s1 + s2) * (alpha - 1.0) + alpha * mui + s3) / (3.0 * (alpha - 1.0) + 1.0)
    expected_li_bongard = (s1 + s2 - 2.0 * mui - 3.0 * s3) / (3.0 * alpha - 2.0)
    expected_beta_pd = 0.5 * s1 - mui + 0.5 * s2 * (1.0 - rt_over_r0)

    np.testing.assert_allclose(li_lao, expected_li_lao, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(beta_p_lao, expected_beta_p_lao, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(beta_p_bongard, expected_beta_p_bongard, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(li_bongard, expected_li_bongard, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(beta_pd, expected_beta_pd, rtol=1e-12, atol=1e-12)


def test_compute_virial_refreshes_and_fallbacks_boundary_axis():
    data_path = Path(__file__).resolve().parents[1] / "vaft" / "data" / "39915.json"
    ods = load_omas_json(str(data_path))
    eq_idx = 3
    ts = ods["equilibrium.time_slice"][eq_idx]

    # Intentionally poison geometric-axis values. The wrapper should refresh/fallback.
    ts["boundary.geometric_axis.r"] = 999.0
    ts["boundary.geometric_axis.z"] = np.nan

    out = compute_virial_equilibrium_quantities_ods(ods, time_slice=eq_idx)
    assert eq_idx in out
    assert np.isfinite(out[eq_idx]["s_1"])
    assert np.isfinite(out[eq_idx]["s_2"])
    assert np.isfinite(out[eq_idx]["s_3"])

    r0_after = float(ts["boundary.geometric_axis.r"])
    z0_after = float(ts["boundary.geometric_axis.z"])
    assert np.isfinite(r0_after)
    assert np.isfinite(z0_after)
    assert r0_after != 999.0


def test_compute_virial_recovers_when_geometric_axis_fields_deleted():
    data_path = Path(__file__).resolve().parents[1] / "vaft" / "data" / "39915.json"
    ods = load_omas_json(str(data_path))
    eq_idx = 3
    ts = ods["equilibrium.time_slice"][eq_idx]

    if "boundary.geometric_axis.r" in ts:
        del ts["boundary.geometric_axis.r"]
    if "boundary.geometric_axis.z" in ts:
        del ts["boundary.geometric_axis.z"]

    out = compute_virial_equilibrium_quantities_ods(ods, time_slice=eq_idx)
    assert eq_idx in out
    assert np.isfinite(out[eq_idx]["s_1"])
    assert np.isfinite(out[eq_idx]["s_2"])
    assert np.isfinite(out[eq_idx]["s_3"])

    r0_after = float(ts["boundary.geometric_axis.r"])
    z0_after = float(ts["boundary.geometric_axis.z"])
    assert np.isfinite(r0_after)
    assert np.isfinite(z0_after)
