import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.em_coupling import (
    calculate_em_coupling_from_raw_database,
    em_coupling,
)
from vaft.machine_mapping.pf_active import (
    pf_geometry_version_for_shot,
    vfit_pf_active_static,
)
from vaft.machine_mapping.pf_passive import pf_passive, wall_geometry_version_for_shot
from vaft.process import compute_mutual_passive_active


def _mapped_coupling(shot: int) -> ODS:
    ods = ODS(consistency_check=False)
    vfit_pf_active_static(ods, shot=shot)
    pf_passive(ods, shot=shot)
    em_coupling(ods, shot=shot)
    return ods


@pytest.fixture(scope="module")
def versioned_coupling() -> tuple[ODS, ODS]:
    return _mapped_coupling(45967), _mapped_coupling(45968)


def _passive_geometry(ods: ODS, indices: range):
    geometry = []
    for index in indices:
        loop = ods[f"pf_passive.loop.{index}"]
        if loop["element.0.geometry.geometry_type"] == 1:
            r = np.mean(loop["element.0.geometry.outline.r"])
            z = np.mean(loop["element.0.geometry.outline.z"])
        else:
            r = loop["element.0.geometry.rectangle.r"]
            z = loop["element.0.geometry.rectangle.z"]
        geometry.append(
            (loop["name"], r, z, 1.0 if loop["name"] == "W11" else 1.04)
        )
    return geometry


def _active_geometry(ods: ODS, indices: tuple[int, ...]):
    geometry = []
    for coil_index in indices:
        geometry.append(
            [
                (
                    ods[
                        f"pf_active.coil.{coil_index}.element.{element_index}.geometry.rectangle.r"
                    ],
                    ods[
                        f"pf_active.coil.{coil_index}.element.{element_index}.geometry.rectangle.z"
                    ],
                    ods[
                        f"pf_active.coil.{coil_index}.element.{element_index}.turns_with_sign"
                    ],
                )
                for element_index in range(
                    len(ods[f"pf_active.coil.{coil_index}.element"])
                )
            ]
        )
    return geometry


def test_coupling_uses_the_pf_active_geometry_boundary(versioned_coupling):
    historical, current = versioned_coupling

    # The PF6/PF7 rebuild falls in the week-long access after 45967 (#956).
    assert pf_geometry_version_for_shot(45967) == "1906"
    assert pf_geometry_version_for_shot(45968) == "2507"
    assert "PF geometry 1906" in historical["em_coupling.ids_properties.comment"]
    assert "PF geometry 2507" in current["em_coupling.ids_properties.comment"]

    historical_pa = np.asarray(historical["em_coupling.mutual_passive_active"])
    current_pa = np.asarray(current["em_coupling.mutual_passive_active"])
    unchanged_columns = [0, 1, 2, 3, 4, 7, 8, 9]
    np.testing.assert_array_equal(
        current_pa[:, unchanged_columns], historical_pa[:, unchanged_columns]
    )
    assert not np.allclose(current_pa[:, 5], historical_pa[:, 5])
    assert not np.allclose(current_pa[:, 6], historical_pa[:, 6])


@pytest.mark.parametrize("ods_index", [0, 1])
def test_pf6_pf7_coupling_matches_selected_geometry(versioned_coupling, ods_index):
    ods = versioned_coupling[ods_index]
    loop_indices = range(len(ods["pf_passive.loop"]))
    expected = compute_mutual_passive_active(
        _passive_geometry(ods, loop_indices),
        _active_geometry(ods, (5, 6)),
    )
    actual = np.asarray(ods["em_coupling.mutual_passive_active"])[
        list(loop_indices), 5:7
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-15)


def test_coupling_coordinates_identify_matrix_order(versioned_coupling):
    _, ods = versioned_coupling
    active_uris = list(ods["em_coupling.active_coils"])
    passive_uris = list(ods["em_coupling.passive_loops"])
    mutual_pa = np.asarray(ods["em_coupling.mutual_passive_active"])

    assert active_uris == [f"#pf_active/coil({index})" for index in range(1, 11)]
    assert len(passive_uris) == len(ods["pf_passive.loop"]) == 965
    assert passive_uris == [
        f"#pf_passive/loop({index})" for index in range(1, 966)
    ]
    assert mutual_pa.shape == (len(passive_uris), len(active_uris))
    assert np.shape(ods["em_coupling.mutual_active_active"]) == (
        len(active_uris),
        len(active_uris),
    )
    assert np.shape(ods["em_coupling.mutual_passive_passive"]) == (
        len(passive_uris),
        len(passive_uris),
    )
    consistency_checked = ODS()
    consistency_checked.update(ods)
    assert list(consistency_checked["em_coupling.active_coils"]) == active_uris


def test_coupling_rejects_active_coil_order_mismatch():
    ods = ODS(consistency_check=False)
    vfit_pf_active_static(ods, shot=45968)
    pf_passive(ods, shot=45968)
    ods["pf_active.coil.5.identifier"] = "PF7"

    with pytest.raises(ValueError, match="coil ordering"):
        em_coupling(ods, shot=45968)


def test_coupling_rejects_geometry_from_the_wrong_shot_era():
    ods = ODS(consistency_check=False)
    vfit_pf_active_static(ods, shot=45967)
    pf_passive(ods, shot=45967)

    with pytest.raises(ValueError, match="pf_active geometry"):
        em_coupling(ods, shot=45968)


def test_coupling_rejects_passive_loop_order_mismatch():
    ods = ODS(consistency_check=False)
    vfit_pf_active_static(ods, shot=45968)
    pf_passive(ods, shot=45968)
    ods["pf_passive.loop.0.name"] = "out-of-order"

    with pytest.raises(ValueError, match="loop ordering"):
        em_coupling(ods, shot=45968)


@pytest.mark.parametrize("legacy_call", [False, True])
def test_raw_mapping_entry_point_selects_coupling_for_shot(legacy_call):
    ods = ODS(consistency_check=False)
    vfit_pf_active_static(ods, shot=48223)
    pf_passive(ods, shot=48223)

    if legacy_call:
        calculate_em_coupling_from_raw_database(ods, {"shot": 48223})
    else:
        calculate_em_coupling_from_raw_database(ods, 48223)

    assert "PF geometry 2507" in ods["em_coupling.ids_properties.comment"]


# --- Wall history (issue #956) ------------------------------------------------


@pytest.mark.parametrize(("shot", "loops"), [(43016, 950), (43017, 965)])
def test_the_wall_boundary_selects_the_loops_and_the_coupling_together(shot, loops):
    ods = _mapped_coupling(shot)

    assert wall_geometry_version_for_shot(shot) == ("1512" if loops == 950 else "2409")
    assert len(ods["pf_passive.loop"]) == loops
    assert np.shape(ods["em_coupling.mutual_passive_passive"]) == (loops, loops)
    assert np.shape(ods["em_coupling.mutual_passive_active"]) == (loops, 10)
    assert f"wall_geometry={wall_geometry_version_for_shot(shot)}" in ods["em_coupling.code.parameters"]


def test_the_wall_2409_additions_extend_the_base_without_touching_it():
    before, after = _mapped_coupling(43016), _mapped_coupling(43017)
    base_pp = np.asarray(before["em_coupling.mutual_passive_passive"])
    pp = np.asarray(after["em_coupling.mutual_passive_passive"])

    np.testing.assert_array_equal(pp[:950, :950], base_pp)
    np.testing.assert_array_equal(
        np.asarray(after["em_coupling.mutual_passive_active"])[:950],
        np.asarray(before["em_coupling.mutual_passive_active"]),
    )
    assert np.array_equal(pp, pp.T)
    assert np.linalg.eigvalsh(pp).min() > 0.0
    names = {after[f"pf_passive.loop.{index}.name"] for index in range(950, 965)}
    assert names == {"W12"}
    z = [np.mean(after[f"pf_passive.loop.{index}.element.0.geometry.outline.z"]) for index in range(950, 965)]
    np.testing.assert_allclose(z, -1.164)


@pytest.mark.parametrize("shot", [43017, 45968])
def test_the_new_rows_follow_the_same_physics_as_the_base(shot):
    """Every added passive-active entry is VAFT's own filament coupling, as the
    950 base rows are, for the PF geometry the shot selects."""
    ods = _mapped_coupling(shot)
    loops = range(len(ods["pf_passive.loop"]))
    expected = compute_mutual_passive_active(
        _passive_geometry(ods, loops), _active_geometry(ods, tuple(range(10)))
    )
    np.testing.assert_allclose(
        np.asarray(ods["em_coupling.mutual_passive_active"]), expected, rtol=1e-10, atol=1e-15
    )


def test_a_passive_structure_from_another_wall_is_refused_with_the_fix():
    ods = ODS(consistency_check=False)
    vfit_pf_active_static(ods, shot=43017)
    pf_passive(ods)  # no shot: the historical 950-loop wall

    with pytest.raises(ValueError, match=r"pf_passive\(ods, shot=43017\)"):
        em_coupling(ods, shot=43017)


def test_a_wall_2409_product_as_reference_is_not_extended_twice(tmp_path):
    from vaft.machine_mapping.pf_passive import append_wall_loops

    ods = _mapped_coupling(43017)
    assert append_wall_loops(ods, "2409") == 0
    assert len(ods["pf_passive.loop"]) == 965
