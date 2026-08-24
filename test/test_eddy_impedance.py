import numpy as np
from omas import ODS

from vaft.omas.process_wrapper import compute_impedance_matrices_ods
from vaft.process import (
    compute_impedance_matrices,
    compute_mutual_passive_active,
)


def test_impedance_uses_canonical_em_coupling_matrix():
    loop_resistances = np.array([2.0, 3.0])
    passive_loop_geometry = [
        ("W11", 0.3, 0.1, 1.0),
        ("W12", 0.4, -0.1, 1.04),
    ]
    mutual_pp = np.array([[10.0, 1.0], [1.0, 20.0]])
    mutual_pa = np.array([[0.4], [0.5]])

    R_mat, L_mat, M_mat = compute_impedance_matrices(
        loop_resistances,
        passive_loop_geometry,
        None,
        mutual_pp,
        mutual_pa,
        [(0.35, 0.0)],
    )

    np.testing.assert_array_equal(R_mat, np.diag(loop_resistances))
    np.testing.assert_array_equal(M_mat, mutual_pp)
    np.testing.assert_array_equal(L_mat[:, :1], mutual_pa)
    assert L_mat.shape == (2, 2)


def test_impedance_does_not_replace_canonical_coupling_from_geometry():
    canonical = np.array([[0.4], [0.5]])
    incompatible_geometry = [[(4.0, 3.0, 100)]]

    _, L_mat, _ = compute_impedance_matrices(
        np.array([2.0, 3.0]),
        [("W11", 0.3, 0.1, 1.0), ("W12", 0.4, -0.1, 1.04)],
        incompatible_geometry,
        np.eye(2),
        canonical,
        [],
    )

    np.testing.assert_array_equal(L_mat, canonical)


def test_impedance_wrapper_does_not_write_non_imas_cache_locations():
    ods = ODS(consistency_check=False)
    ods["pf_active.coil.0.name"] = "PF1"
    ods["pf_active.coil.0.element.0.geometry.rectangle.r"] = 0.5
    ods["pf_active.coil.0.element.0.geometry.rectangle.z"] = 0.2
    ods["pf_active.coil.0.element.0.turns_with_sign"] = 10

    for index, (name, resistance, r, z) in enumerate(
        [("W11", 2.0, 0.3, 0.1), ("W12", 3.0, 0.4, -0.1)]
    ):
        ods[f"pf_passive.loop.{index}.name"] = name
        ods[f"pf_passive.loop.{index}.resistance"] = resistance
        ods[
            f"pf_passive.loop.{index}.element.0.geometry.geometry_type"
        ] = 2
        ods[
            f"pf_passive.loop.{index}.element.0.geometry.rectangle.r"
        ] = r
        ods[
            f"pf_passive.loop.{index}.element.0.geometry.rectangle.z"
        ] = z

    mutual_pp = np.array([[10.0, 1.0], [1.0, 20.0]])
    passive_geometry = [
        ("W11", 0.3, 0.1, 1.0),
        ("W12", 0.4, -0.1, 1.04),
    ]
    coil_geometry = [[(0.5, 0.2, 10)]]
    mutual_pa = compute_mutual_passive_active(
        passive_geometry,
        coil_geometry,
    )
    ods["em_coupling.mutual_passive_passive"] = mutual_pp
    ods["em_coupling.mutual_passive_active"] = mutual_pa

    _, L_mat, _ = compute_impedance_matrices_ods(ods, [(0.35, 0.0)])

    np.testing.assert_array_equal(L_mat[:, :1], mutual_pa)
    for cache_key in ("R_mat", "L_mat", "M_mat"):
        assert f"pf_passive.{cache_key}" not in ods

    # The solver output can be merged into a consistency-checked ODS without
    # the invalid-location failure that blocked the pipeline.
    consistency_checked = ODS()
    consistency_checked.update(ods)
    assert "em_coupling.mutual_passive_active" in consistency_checked
