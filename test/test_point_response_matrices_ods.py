"""Vectorized ODS response wrapper vs the legacy scalar path (issue #239)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from omas import ODS

from vaft.omas.process_wrapper import (
    compute_point_response_matrices_ods,
    compute_point_response_ods,
)


@pytest.fixture(scope="module")
def machine_ods() -> ODS:
    from vaft.machine_mapping.em_coupling import em_coupling
    from vaft.machine_mapping.pf_active import vfit_pf_active_static
    from vaft.machine_mapping.pf_passive import pf_passive

    ods = ODS(consistency_check=False)
    vfit_pf_active_static(ods, shot=45968)
    pf_passive(ods)
    em_coupling(ods, shot=45968)
    return ods


# Observation points away from coil/loop locations (the vectorized path
# does not shift-average near sources).
POINTS = [[0.35, 0.05], [0.45, -0.20], [0.55, 0.30], [0.25, 0.0]]


def test_matches_legacy_path_away_from_sources(machine_ods):
    psi_v, bz_v, br_v = compute_point_response_matrices_ods(machine_ods, POINTS)
    psi_l, bz_l, br_l = compute_point_response_ods(machine_ods, POINTS)
    assert psi_v.shape == psi_l.shape
    # Legacy uses polynomial elliptic approximations (~1e-6 relative).
    scale_psi = np.max(np.abs(psi_l))
    scale_b = np.max(np.abs(bz_l))
    np.testing.assert_allclose(psi_v, psi_l, atol=1e-5 * scale_psi)
    np.testing.assert_allclose(bz_v, bz_l, atol=1e-5 * scale_b)
    np.testing.assert_allclose(br_v, br_l, atol=1e-5 * scale_b)


def test_plasma_columns_appended(machine_ods):
    plasma = [[0.42, 0.03], [0.40, -0.10]]
    psi, bz, br = compute_point_response_matrices_ods(
        machine_ods, POINTS, plasma_points=plasma
    )
    n_coil = len(machine_ods["pf_active.coil"])
    n_loop = len(machine_ods["pf_passive.loop"])
    assert psi.shape == (len(POINTS), n_coil + n_loop + 2)
    psi_l, _, _ = compute_point_response_ods(machine_ods, POINTS, plasma)
    np.testing.assert_allclose(
        psi[:, -2:], psi_l[:, -2:], atol=1e-5 * np.max(np.abs(psi_l))
    )


def test_rejects_malformed_observation_points(machine_ods):
    """Transposed / flat inputs raise instead of returning garbage.

    The legacy path failed loudly on these; the vectorized path used to
    accept anything with >= 2 columns (PR #241 review finding 1).
    """
    r_list, z_list = [0.35, 0.45, 0.55], [0.05, -0.2, 0.3]
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        compute_point_response_matrices_ods(machine_ods, [r_list, z_list])
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        compute_point_response_matrices_ods(machine_ods, [0.35, 0.05, 0.45, -0.2])
    with pytest.raises(ValueError, match="plasma_points"):
        compute_point_response_matrices_ods(
            machine_ods, POINTS, plasma_points=[[0.4, 0.0, 1.0]]
        )


def test_coincident_observation_point_warns(machine_ods):
    """A point on a source yields finite clamp artifacts plus a warning."""
    r0 = float(machine_ods["pf_active.coil.0.element.0.geometry.rectangle.r"])
    z0 = float(machine_ods["pf_active.coil.0.element.0.geometry.rectangle.z"])
    with pytest.warns(UserWarning, match="coincide with a source"):
        psi, bz, br = compute_point_response_matrices_ods(machine_ods, [[r0, z0]])
    assert np.all(np.isfinite(psi)) and np.all(np.isfinite(bz)) and np.all(np.isfinite(br))


@pytest.mark.perf
def test_substantially_faster_than_legacy(machine_ods):
    compute_point_response_matrices_ods(machine_ods, POINTS)  # warm-up
    t0 = time.perf_counter()
    compute_point_response_matrices_ods(machine_ods, POINTS)
    fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    compute_point_response_ods(machine_ods, POINTS)
    slow = time.perf_counter() - t0
    assert fast < slow / 5, f"vectorized {fast:.3f}s vs legacy {slow:.3f}s"
