"""Issue #478: one function moves a legacy Wb/rad equilibrium to the DD's Wb.

The packaged 39915 sample stored the g-file's psi per radian without saying
so, and every reader had to redetect that from the data.  The sample is now
regenerated through :func:`vaft.omas.equilibrium_psi_to_weber`, which scales
the flux leaves, the psi-derivative profiles and declares COCOS 11, so an ODS
that VAFT ships is self-describing.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from vaft.data.eqdsk import TWO_PI, ods_psi_to_wb_per_radian_factor, slice_flux_exponent
from vaft.data.resources import sample_geqdsk
from vaft.omas.general import equilibrium_psi_to_weber, ods_cocos, set_ods_cocos

PSI_LEAVES = (
    "global_quantities.psi_axis",
    "global_quantities.psi_boundary",
    "profiles_1d.psi",
    "profiles_2d.0.psi",
)
DERIVATIVE_LEAVES = ("profiles_1d.dpressure_dpsi", "profiles_1d.f_df_dpsi")


@pytest.fixture()
def weber():
    """A DD-conformant (Wb) ODS from the packaged g-file, undeclared."""
    return sample_geqdsk("efit/g039915.00319").to_omas()


@pytest.fixture()
def legacy(weber):
    """The same equilibrium the way the pre-#236 writer stored it (Wb/rad)."""
    ods = copy.deepcopy(weber)
    ts = ods["equilibrium.time_slice.0"]
    for leaf in PSI_LEAVES:
        ts[leaf] = np.asarray(ts[leaf], float) / TWO_PI if np.ndim(ts[leaf]) else float(ts[leaf]) / TWO_PI
    for leaf in DERIVATIVE_LEAVES:
        ts[leaf] = np.asarray(ts[leaf], float) * TWO_PI
    return ods


def _assert_same_equilibrium(candidate, reference):
    a = candidate["equilibrium.time_slice.0"]
    b = reference["equilibrium.time_slice.0"]
    for leaf in PSI_LEAVES + DERIVATIVE_LEAVES:
        np.testing.assert_allclose(np.asarray(a[leaf], float), np.asarray(b[leaf], float), rtol=1e-12)


def test_a_legacy_artifact_is_converted_and_declared(legacy, weber):
    assert ods_cocos(legacy) is None
    assert equilibrium_psi_to_weber(legacy, source="test") is True
    _assert_same_equilibrium(legacy, weber)
    assert ods_cocos(legacy) == 11
    assert legacy["equilibrium.code.parameters.cocos_source"] == "test"
    assert ods_psi_to_wb_per_radian_factor(legacy, 0) == pytest.approx(1.0 / TWO_PI)


def test_conversion_is_idempotent(legacy, weber):
    equilibrium_psi_to_weber(legacy)
    assert equilibrium_psi_to_weber(legacy) is False
    _assert_same_equilibrium(legacy, weber)


def test_a_weber_artifact_is_only_declared(weber):
    pristine = copy.deepcopy(weber)
    assert equilibrium_psi_to_weber(weber) is False
    _assert_same_equilibrium(weber, pristine)
    assert ods_cocos(weber) == 11


def test_a_declared_weber_artifact_is_left_alone(weber):
    set_ods_cocos(weber, 13)
    pristine = copy.deepcopy(weber)
    assert equilibrium_psi_to_weber(weber) is False
    _assert_same_equilibrium(weber, pristine)
    assert ods_cocos(weber) == 13


def test_a_declared_per_radian_artifact_is_converted_without_probing(legacy, weber):
    """The declaration wins: no phi, boundary or ip is consulted."""
    set_ods_cocos(legacy, 3)
    ts = legacy["equilibrium.time_slice.0"]
    for leaf in ("profiles_1d.phi", "boundary.outline.r", "boundary.outline.z", "global_quantities.ip"):
        if leaf in ts:
            del ts[leaf]
    assert slice_flux_exponent(ts) is None
    assert equilibrium_psi_to_weber(legacy) is True
    _assert_same_equilibrium(legacy, weber)
    assert ods_cocos(legacy) == 11


def test_inconsistent_slices_are_refused(legacy, weber):
    """A mixed ODS must not be half-converted."""
    mixed = copy.deepcopy(legacy)
    mixed["equilibrium.time"] = np.array([0.0, 1.0])
    mixed["equilibrium.time_slice.1"] = copy.deepcopy(weber["equilibrium.time_slice.0"])
    before = copy.deepcopy(mixed)
    with pytest.raises(ValueError, match="disagree"):
        equilibrium_psi_to_weber(mixed)
    _assert_same_equilibrium(mixed, before)
    assert ods_cocos(mixed) is None


def test_an_undecidable_artifact_is_refused(legacy):
    ts = legacy["equilibrium.time_slice.0"]
    for leaf in ("profiles_1d.phi", "boundary.outline.r", "boundary.outline.z"):
        if leaf in ts:
            del ts[leaf]
    with pytest.raises(ValueError, match="declare the COCOS index"):
        equilibrium_psi_to_weber(legacy)


def test_an_ods_without_equilibrium_is_untouched():
    from omas import ODS

    ods = ODS()
    ods["magnetics.flux_loop.0.flux.data"] = np.zeros(3)
    assert equilibrium_psi_to_weber(ods) is False
    assert "equilibrium" not in ods


def test_the_detector_honours_a_declared_cocos_before_probing(legacy):
    """A declaration outranks the data probes; a bare slice carries none."""
    assert ods_psi_to_wb_per_radian_factor(legacy, 0) == pytest.approx(1.0)
    set_ods_cocos(legacy, 11)  # deliberately wrong for the data: the label wins
    assert ods_psi_to_wb_per_radian_factor(legacy, 0) == pytest.approx(1.0 / TWO_PI)
    assert ods_psi_to_wb_per_radian_factor(legacy["equilibrium.time_slice.0"]) == pytest.approx(1.0)


def test_field_derivations_are_invariant_under_conversion(legacy):
    """The declared index must not divide by 2*pi a second time (#478).

    The derivations first scale psi to Wb/rad and then apply the Sauter
    prefactor; on a declared COCOS 11 ODS that prefactor used to carry its
    own 1/(2*pi), so B_pol came out 2*pi too small.
    """
    from vaft.omas.process_wrapper import (
        compute_magnetic_energy,
        compute_virial_equilibrium_quantities_ods,
    )

    converted = copy.deepcopy(legacy)
    equilibrium_psi_to_weber(converted)
    before = compute_virial_equilibrium_quantities_ods(copy.deepcopy(legacy), time_slice=0)[0]
    after = compute_virial_equilibrium_quantities_ods(converted, time_slice=0)[0]
    for key in ("B_pa", "beta_p", "li", "W_mag"):
        assert float(after[key]) == pytest.approx(float(before[key]), rel=1e-9), key
    assert compute_magnetic_energy(converted, time_slice=0) == pytest.approx(
        compute_magnetic_energy(copy.deepcopy(legacy), time_slice=0), rel=1e-9
    )


def test_the_packaged_sample_is_self_describing():
    """39915 ships in DD weber and says so on both representations."""
    import vaft

    ods = vaft.omas.load(vaft.data.sample(39915, representation="omas"))
    assert ods_cocos(ods) == 11
    assert "phi" not in ods["equilibrium.time_slice.0.profiles_1d"]
    verdicts = set()
    for index in range(len(ods["equilibrium.time_slice"])):
        assert ods_psi_to_wb_per_radian_factor(ods, index) == pytest.approx(1.0 / TWO_PI)
        verdicts.add(slice_flux_exponent(ods[f"equilibrium.time_slice.{index}"]))
    # Every slice that can answer says Wb; the degenerate last slice abstains.
    assert verdicts - {None} == {1}

    with vaft.imas.load(vaft.data.sample(39915, representation="imas"), imas_version="3.41.0") as handle:
        back = handle.to_omas()
    assert ods_cocos(back) == 11
    np.testing.assert_allclose(
        back["equilibrium.time_slice.0.global_quantities.psi_axis"],
        ods["equilibrium.time_slice.0.global_quantities.psi_axis"],
    )


def test_reading_the_declaration_leaves_no_empty_node_behind():
    """``ods_cocos`` on an unlabelled ODS must not create ``ids_properties.cocos``.

    ``flat()`` hides the empty node an OMAS read materializes, but ``save``
    writes it, which is how a DD 4-only leaf reached a DD 3.41 artifact.
    """
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = np.array([0.0])
    assert ods_cocos(ods) is None
    assert list(ods["equilibrium"].keys()) == ["time"]
    assert equilibrium_psi_to_weber(ods) is False  # no time_slice: untouched
    assert list(ods["equilibrium"].keys()) == ["time"]
    set_ods_cocos(ods, 11)
    assert ods_cocos(ods) == 11 and "ids_properties" not in ods["equilibrium"]


def test_a_declared_cocos_survives_the_imas_adapter(tmp_path):
    """JSON artifacts reach the IMAS adapter through an in-memory entry; the
    code.parameters block used to be dropped there because the JSON load
    left it as a plain branch rather than a CodeParameters object."""
    import vaft
    from omas import ODS
    from omas.omas_core import CodeParameters

    ods = ODS(imas_version="3.41.0")
    ods["equilibrium.time"] = np.array([0.0])
    ods["equilibrium.time_slice.0.global_quantities.ip"] = 1.0e5
    set_ods_cocos(ods, 11, source="test")
    path = tmp_path / "declared.json.gz"
    vaft.omas.save(ods, path)

    loaded = vaft.omas.load(path)
    assert isinstance(loaded["equilibrium.code.parameters"], CodeParameters)
    assert ods_cocos(loaded) == 11
    with vaft.imas.load(path, imas_version="3.41.0") as handle:
        assert ods_cocos(handle.to_omas()) == 11


def test_a_nested_code_parameters_cache_is_left_as_it_was(tmp_path):
    """Promotion is for flat declarations only; EFIT's parser cache keeps its
    nested shape (it is not representable as an XML parameters string)."""
    import vaft
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = np.array([0.0])
    ods["equilibrium.code.parameters.efit_collection.status"] = "completed"
    ods["equilibrium.code.parameters.efit_collection.slice_statuses.0.status"] = "ok"
    path = tmp_path / "cache.json.gz"
    vaft.omas.save(ods, path)
    loaded = vaft.omas.load(path)
    assert loaded["equilibrium.code.parameters.efit_collection.status"] == "completed"
    assert loaded["equilibrium.code.parameters.efit_collection.slice_statuses.0.status"] == "ok"
    assert ods_cocos(loaded) is None


def test_core_profiles_psi_grids_scale_with_the_equilibrium(legacy):
    legacy["core_profiles.profiles_1d.0.grid.psi"] = np.array([-0.002, -0.001, 0.0])
    legacy["core_profiles.profiles_1d.0.electrons.density"] = np.array([3e18, 2e18, 1e18])
    equilibrium_psi_to_weber(legacy)
    np.testing.assert_allclose(
        legacy["core_profiles.profiles_1d.0.grid.psi"], np.array([-0.002, -0.001, 0.0]) * TWO_PI
    )
    np.testing.assert_allclose(legacy["core_profiles.profiles_1d.0.electrons.density"], [3e18, 2e18, 1e18])


def test_slice_callers_see_the_declaration_through_the_ods(weber):
    """A bare slice carries no declaration; callers pass the ODS and the index."""
    from vaft.omas.process_wrapper import compute_diamagnetism

    set_ods_cocos(weber, 11)
    labelled = compute_diamagnetism(weber, time_index=0)
    del weber["equilibrium.code.parameters"]
    unlabelled = compute_diamagnetism(weber, time_index=0)
    assert np.isfinite(float(labelled))
    assert float(labelled) == pytest.approx(float(unlabelled), rel=1e-9)
