import numpy as np
from omas import ODS, load_omas_json

from vaft.data.resources import data_path
from vaft.machine_mapping.pf_active import vfit_pf_active_static
from vaft.omas.process_wrapper import compute_impedance_matrices_ods
from vaft.process import (
    compute_impedance_matrices,
    compute_mutual_passive_active,
)


def test_impedance_uses_em_coupling_when_geometry_is_not_provided():
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


def test_impedance_uses_current_geometry_when_reference_coupling_differs():
    reference = load_omas_json(
        str(data_path("omas/39915.json")),
        consistency_check=False,
    )
    passive_geometry = []
    for index in range(5):
        loop = reference[f"pf_passive.loop.{index}"]
        if loop["element.0.geometry.geometry_type"] == 1:
            r = np.mean(loop["element.0.geometry.outline.r"])
            z = np.mean(loop["element.0.geometry.outline.z"])
        else:
            r = loop["element.0.geometry.rectangle.r"]
            z = loop["element.0.geometry.rectangle.z"]
        passive_geometry.append(
            (
                loop["name"],
                r,
                z,
                1.0 if loop["name"] == "W11" else 1.04,
            )
        )

    current = ODS(consistency_check=False)
    vfit_pf_active_static(current, shot=48223)
    pf6_geometry = [
        [
            (
                current[
                    f"pf_active.coil.5.element.{index}.geometry.rectangle.r"
                ],
                current[
                    f"pf_active.coil.5.element.{index}.geometry.rectangle.z"
                ],
                current[
                    f"pf_active.coil.5.element.{index}.turns_with_sign"
                ],
            )
            for index in range(len(current["pf_active.coil.5.element"]))
        ]
    ]
    reference_coupling = np.asarray(
        reference["em_coupling.mutual_passive_active"],
    )[:5, 5:6]
    expected = compute_mutual_passive_active(
        passive_geometry,
        pf6_geometry,
    )
    assert not np.allclose(expected, reference_coupling, rtol=1e-3, atol=1e-12)

    _, L_mat, _ = compute_impedance_matrices(
        np.ones(5),
        passive_geometry,
        pf6_geometry,
        np.eye(5),
        reference_coupling,
        [],
    )
    np.testing.assert_allclose(L_mat, expected)


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
