"""The wall eigenbasis on the packaged VEST machine (vaft #473, vfit #8/#9).

Regression values were measured on the packaged 39915 geometry with the
2303 resistance calibration; they are pinned together with that calibration
key so a recalibration (#308) fails here with a clear message rather than
silently moving every decay time.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.machine_mapping.wall_resistance import identify_calibration
from vaft.omas.process_wrapper import compute_impedance_matrices_ods, compute_wall_mode_basis_ods
from vaft.process.wall_modes import (
    WallModeError,
    check_wall_mode_basis,
    global_time_constants,
    project,
    reconstruct,
    select_slowest,
)

#: Slowest local decay time per segment [ms], packaged geometry, calibration 2303.
SEGMENT_SLOWEST_MS = {
    "W1": 1.983, "W2": 1.206, "W3": 0.304, "W4": 0.181, "W5": 0.277, "W6": 1.560,
    "W7": 2.873, "W8": 2.901, "W9": 2.051, "W10": 1.638, "W11": 0.148,
}
TOKAMAKER_TAU_WALL_MS = 6.88   # vaft #232, its own mesh with W11 excluded


@pytest.fixture(scope="module")
def packaged():
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    basis = compute_wall_mode_basis_ods(ods)
    R_mat, _L, M_mat = compute_impedance_matrices_ods(ods, [])
    return ods, basis, R_mat, M_mat


def test_39915_basis_has_full_rank_positive_finite_taus(packaged):
    _ods, basis, _R, _M = packaged
    assert sum(basis.n_modes()) == 950 and basis.n_elements == 950
    tau = basis.tau()
    assert np.all(np.isfinite(tau)) and tau.min() > 1e-7


def test_39915_per_segment_slowest_taus_match_reference(packaged):
    ods, basis, _R, _M = packaged
    assert identify_calibration(ods)["key"] == "2303", "reference values belong to calibration 2303"
    by_name: dict[str, list[float]] = {}
    for seg in basis.segments:
        by_name.setdefault(seg.id.split("_")[0], []).append(float(seg.tau[0]))
    for name, expected in SEGMENT_SLOWEST_MS.items():
        values = by_name[name]
        for value in values:
            assert value * 1e3 == pytest.approx(expected, rel=2e-3), name
        if len(values) == 2:  # mirrored halves see the same wall
            assert values[0] == pytest.approx(values[1], rel=1e-6), name


def test_39915_global_spectrum_from_the_basis_matches_the_pencil(packaged):
    _ods, basis, R_mat, M_mat = packaged
    tau = global_time_constants(basis, M_mat)
    assert tau[0] * 1e3 == pytest.approx(7.187, rel=1e-3)
    assert int(np.sum(tau > 1e-3)) == 11
    assert int(np.sum(tau > 1e-4)) == 98
    expected = np.sort(np.linalg.eigvals(np.linalg.solve(R_mat, M_mat)).real)[::-1]
    np.testing.assert_allclose(tau, expected, rtol=1e-8)


def test_39915_wall_time_constants_now_read_the_same_pencil(packaged):
    from vaft.validation.vacuum_benchmark import wall_time_constants

    ods, basis, _R, M_mat = packaged
    np.testing.assert_allclose(wall_time_constants(ods), global_time_constants(basis, M_mat), rtol=1e-10)


def test_39915_inter_segment_coupling_is_retained(packaged):
    _ods, basis, R_mat, M_mat = packaged
    metrics = check_wall_mode_basis(basis, R_mat, M_mat)
    assert min(metrics["coupling"].values()) > 1e-4
    assert metrics["coupling"]["W9_L-W8_L"] > 0.9
    assert metrics["r_r_identity_error"] < 1e-13
    assert metrics["max_segment_residual"] < 1e-12
    # The fast tail of a 230-loop segment is densely spaced (that is what a
    # continuum of skin-current patterns looks like); what must be
    # well-separated is the leading mode of every segment, and nothing may
    # sit inside the refusal band.
    assert metrics["min_relative_gap"] > 1e-6
    assert all(seg.min_relative_gap > 1e-6 for seg in basis.segments)
    assert all((seg.tau[0] - seg.tau[1]) / seg.tau[0] > 1e-2 for seg in basis.segments if seg.size > 1)


def test_39915_truncated_basis_reconstructs_a_current_it_contains(packaged):
    _ods, basis, R_mat, _M = packaged
    keep = select_slowest(basis, 40)
    a = np.linspace(-1.0, 1.0, 40)
    I = reconstruct(basis, a, keep)
    np.testing.assert_allclose(project(basis, I, R_mat, keep), a, atol=1e-10)


def test_39915_basis_is_deterministic_and_digest_is_pinned(packaged):
    ods, basis, _R, _M = packaged
    again = compute_wall_mode_basis_ods(ods)
    assert again.digest() == basis.digest()
    assert basis.provenance["segment_definition_version"] == "vest-name-zgap-1.5-v1"
    assert basis.provenance["resistance_calibration"] == "2303"
    assert basis.provenance["n_segments"] == "19"


def test_39915_slowest_tau_is_within_ten_percent_of_tokamaker(packaged):
    """TokaMaker's eig_wall is an independent global reference on its own
    mesh (W11 excluded); agreement to ~10% is what shared physics predicts,
    identity is not expected."""
    _ods, basis, _R, M_mat = packaged
    assert global_time_constants(basis, M_mat)[0] * 1e3 == pytest.approx(TOKAMAKER_TAU_WALL_MS, rel=0.10)


def test_record_writes_provenance_under_em_coupling_only(packaged):
    from vaft.machine_mapping.em_coupling import _validate_coordinate_order  # noqa: F401  (import guard)

    ods, _basis, _R, _M = packaged
    before = {str(k) for k in ods["pf_passive"].flat()}
    compute_wall_mode_basis_ods(ods, record=True)
    after = {str(k) for k in ods["pf_passive"].flat()}
    assert after == before
    lines = str(ods["em_coupling.code.parameters"]).splitlines()
    assert any(l.startswith("wall_mode_basis_digest=") for l in lines)
    assert any(l == "wall_mode_segment_definition_version=vest-name-zgap-1.5-v1" for l in lines)
    compute_wall_mode_basis_ods(ods, record=True)  # idempotent
    assert sum(l.startswith("wall_mode_basis_digest=") for l in str(ods["em_coupling.code.parameters"]).splitlines()) == 1


def test_41672_shipped_em_coupling_is_refused_until_remapped():
    import vaft
    import vaft.omas

    try:
        path = vaft.data.sample(41672, "imas")
    except (ValueError, FileNotFoundError):
        pytest.skip("sample 41672 is not available in this checkout")
    ods = vaft.omas.load(path)
    asymmetry = float(np.max(np.abs(ods["em_coupling.mutual_passive_passive"] - ods["em_coupling.mutual_passive_passive"].T))
                      / np.max(np.abs(ods["em_coupling.mutual_passive_passive"])))
    if asymmetry <= 1e-6:
        pytest.skip("the shipped coupling is already symmetric; nothing to refuse")
    with pytest.raises(WallModeError, match="asymmetric"):
        compute_wall_mode_basis_ods(ods)
    with pytest.warns(RuntimeWarning):
        basis = compute_wall_mode_basis_ods(ods, remap_em_coupling=True)
    assert sum(basis.n_modes()) == 950
