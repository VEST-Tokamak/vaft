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
    """The declaration wins: no phi, boundary, ip or q is consulted."""
    set_ods_cocos(legacy, 1)
    ts = legacy["equilibrium.time_slice.0"]
    # q goes too: the contour-q rung (7111a8e3) decides from it alone
    for leaf in ("profiles_1d.phi", "boundary.outline.r", "boundary.outline.z", "global_quantities.ip",
                 "profiles_1d.q"):
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


def test_an_artifact_without_phi_or_outline_is_decided_from_q(legacy, weber):
    ts = legacy["equilibrium.time_slice.0"]
    for leaf in ("profiles_1d.phi", "boundary.outline.r", "boundary.outline.z"):
        if leaf in ts:
            del ts[leaf]
    assert slice_flux_exponent(ts) == 0
    assert equilibrium_psi_to_weber(legacy) is True
    _assert_same_equilibrium(legacy, weber)


def test_an_undecidable_artifact_is_refused(legacy):
    ts = legacy["equilibrium.time_slice.0"]
    # without phi and an outline the contour-q rung still decides; without q
    # as well, nothing can
    for leaf in ("profiles_1d.phi", "boundary.outline.r", "boundary.outline.z", "profiles_1d.q"):
        if leaf in ts:
            del ts[leaf]
    assert slice_flux_exponent(ts) is None
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


# -- issue #1371: a declared 2-8 is fully transformed, not only scaled by 2*pi --

#: Leaves the synthetic records carry, with the OMAS ``cocos_transform`` key
#: each is scaled by.  Written out here rather than read from the table the
#: implementation walks, so the test does not build its input with the code
#: under test.
TRANSFORMED_LEAVES = {
    "global_quantities.psi_axis": "PSI",
    "global_quantities.psi_boundary": "PSI",
    "profiles_1d.psi": "PSI",
    "profiles_2d.0.psi": "PSI",
    "profiles_1d.dpressure_dpsi": "PPRIME",
    "profiles_1d.f_df_dpsi": "F_FPRIME",
    "profiles_1d.f": "F",
    "profiles_1d.q": "Q",
    "global_quantities.ip": "IP",
}
#: core_profiles leaves the synthetic records carry, with their factor keys.
CORE_PROFILES_LEAVES = {
    "core_profiles.profiles_1d.0.grid.psi": "PSI",
    "core_profiles.profiles_1d.0.q": "Q",
    "core_profiles.profiles_1d.0.j_tor": "TOR",
    "core_profiles.global_quantities.ip": "TOR",
    "core_profiles.vacuum_toroidal_field.b0": "TOR",
}
PSI_ERROR_LEAF = "profiles_1d.psi_error_upper"
FIELD_POINTS = ((0.40, 0.00), (0.45, 0.10), (0.30, -0.12))
ALL_PER_RADIAN = [1, 2, 3, 4, 5, 6, 7, 8]


def _declared_record(reference, cocos):
    """``reference`` (COCOS 11) re-expressed in ``cocos`` and declared so."""
    from omas.omas_physics import cocos_transform

    factors = cocos_transform(11, cocos)
    ods = copy.deepcopy(reference)
    ts = ods["equilibrium.time_slice.0"]
    for leaf, key in TRANSFORMED_LEAVES.items():
        scaled = np.asarray(ts[leaf], float) * factors[key]
        ts[leaf] = float(scaled) if scaled.ndim == 0 else scaled
    ods["equilibrium.vacuum_toroidal_field.b0"] = (
        np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float) * factors["BT"]
    )
    for leaf, key in CORE_PROFILES_LEAVES.items():
        ods[leaf] = np.asarray(reference[leaf], float) * factors[key]
    ts[PSI_ERROR_LEAF] = np.asarray(reference[f"equilibrium.time_slice.0.{PSI_ERROR_LEAF}"]) * abs(factors["PSI"])
    set_ods_cocos(ods, cocos)
    return ods


def _field(ods):
    from vaft.process.equilibrium import as_equilibrium, make_equilibrium_field_interpolator

    field = make_equilibrium_field_interpolator(as_equilibrium(ods))
    return np.array([field(r, z) for r, z in FIELD_POINTS], float).reshape(len(FIELD_POINTS), 3)


@pytest.fixture()
def cocos11(weber):
    """The packaged equilibrium taken as a COCOS 11 record (its signs fit 11)."""
    ts = weber["equilibrium.time_slice.0"]
    psi = np.asarray(ts["profiles_1d.psi"], float)
    ts[PSI_ERROR_LEAF] = np.full(psi.shape, 1.0e-4)
    weber["core_profiles.profiles_1d.0.grid.psi"] = psi
    weber["core_profiles.profiles_1d.0.q"] = np.asarray(ts["profiles_1d.q"], float)
    weber["core_profiles.profiles_1d.0.j_tor"] = np.linspace(2.0e5, 0.0, psi.size)
    weber["core_profiles.global_quantities.ip"] = np.array([float(ts["global_quantities.ip"])])
    weber["core_profiles.vacuum_toroidal_field.b0"] = np.asarray(
        weber["equilibrium.vacuum_toroidal_field.b0"], float
    )
    set_ods_cocos(weber, 11)
    return weber


@pytest.mark.parametrize("cocos", ALL_PER_RADIAN)
def test_a_declared_per_radian_record_is_fully_transformed_to_cocos_11(cocos11, cocos):
    record = _declared_record(cocos11, cocos)
    assert equilibrium_psi_to_weber(record, source="test") is True
    assert ods_cocos(record) == 11
    assert record["equilibrium.code.parameters.cocos_source"] == "test"

    a = record["equilibrium.time_slice.0"]
    b = cocos11["equilibrium.time_slice.0"]
    for leaf in TRANSFORMED_LEAVES:
        np.testing.assert_allclose(np.asarray(a[leaf], float), np.asarray(b[leaf], float), rtol=1e-12, err_msg=leaf)
    np.testing.assert_allclose(
        record["equilibrium.vacuum_toroidal_field.b0"], cocos11["equilibrium.vacuum_toroidal_field.b0"], rtol=1e-12
    )
    for leaf in CORE_PROFILES_LEAVES:
        np.testing.assert_allclose(record[leaf], cocos11[leaf], rtol=1e-12, err_msg=leaf)
    # error bars scale by the magnitude of the flux factor, never by its sign
    np.testing.assert_allclose(a[PSI_ERROR_LEAF], b[PSI_ERROR_LEAF], rtol=1e-12)


@pytest.mark.parametrize("cocos", ALL_PER_RADIAN)
def test_the_converted_record_gives_the_original_physical_field(cocos11, cocos):
    """B_Z (and B_R, B_phi) from the converted psi is the field the record described.

    The interpolator reads the declaration, so a 2*pi-only scale relabelled as
    11 reversed B_Z for a declared 2 and 3 (issue #1371).
    """
    reference = _field(cocos11)
    assert np.all(np.abs(reference[:, 1]) > 0)
    record = _declared_record(cocos11, cocos)
    # Before conversion the declared record already describes the same field
    # in (R, Z); B_phi carries the declared index's toroidal orientation.
    np.testing.assert_allclose(_field(record)[:, :2], reference[:, :2], rtol=1e-9)
    equilibrium_psi_to_weber(record)
    np.testing.assert_allclose(_field(record), reference, rtol=1e-9)


@pytest.mark.parametrize("cocos", ALL_PER_RADIAN)
def test_ip_and_q_signs_follow_cocos_11(cocos11, cocos):
    from vaft.data.cocos import cocos_spec

    record = _declared_record(cocos11, cocos)
    equilibrium_psi_to_weber(record)
    ts = record["equilibrium.time_slice.0"]
    ip = float(ts["global_quantities.ip"])
    b0 = float(np.ravel(record["equilibrium.vacuum_toroidal_field.b0"])[0])
    sigma_ip, sigma_b0 = int(np.sign(ip)), int(np.sign(b0))
    assert sigma_ip == np.sign(float(cocos11["equilibrium.time_slice.0.global_quantities.ip"]))
    spec = cocos_spec(11)
    assert np.all(np.sign(ts["profiles_1d.q"]) == spec.expected_sign("q", sigma_ip=sigma_ip, sigma_b0=sigma_b0))
    dpsi = float(ts["global_quantities.psi_boundary"]) - float(ts["global_quantities.psi_axis"])
    assert np.sign(dpsi) == spec.expected_sign("dpsi", sigma_ip=sigma_ip, sigma_b0=sigma_b0)
    assert np.all(np.sign(ts["profiles_1d.f"]) == spec.expected_sign("f", sigma_ip=sigma_ip, sigma_b0=sigma_b0))


def test_a_declared_weber_record_keeps_its_own_index(cocos11):
    """COCOS 12-18 already store weber and declare honestly: left as they are."""
    record = _declared_record(cocos11, 12)
    pristine = copy.deepcopy(record)
    assert equilibrium_psi_to_weber(record) is False
    assert ods_cocos(record) == 12
    for leaf in TRANSFORMED_LEAVES:
        np.testing.assert_allclose(
            np.asarray(record["equilibrium.time_slice.0"][leaf], float),
            np.asarray(pristine["equilibrium.time_slice.0"][leaf], float),
        )
    np.testing.assert_allclose(_field(record)[:, :2], _field(cocos11)[:, :2], rtol=1e-9)


def test_every_transformed_leaf_has_an_omas_factor():
    """The fixed leaf set must stay inside OMAS's table, with a real factor key."""
    from omas.omas_physics import cocos_signals, cocos_transform

    from vaft.omas.general import _COCOS_TRANSFORMED_LEAVES

    factors = cocos_transform(2, 11)
    for leaf in _COCOS_TRANSFORMED_LEAVES:
        assert leaf in cocos_signals, leaf
        assert cocos_signals[leaf] in factors, leaf
        assert ".constraints." not in leaf


def test_a_labelled_and_an_unlabelled_copy_convert_alike(legacy):
    """The probe path uses the same leaf set as a declared COCOS 1."""
    legacy["core_profiles.profiles_1d.0.grid.psi"] = np.array([-0.002, -0.001, 0.0])
    legacy["core_profiles.profiles_1d.0.grid.psi_boundary"] = 0.0
    labelled = copy.deepcopy(legacy)
    set_ods_cocos(labelled, 1)
    assert equilibrium_psi_to_weber(labelled) is True
    assert equilibrium_psi_to_weber(legacy) is True
    flat_a = labelled.flat()
    flat_b = legacy.flat()
    for key, value in flat_a.items():
        if key.startswith(("equilibrium.time_slice", "core_profiles")):
            np.testing.assert_allclose(np.asarray(flat_b[key], float), np.asarray(value, float), err_msg=key)


def test_efit_constraints_on_the_41524_sample_are_left_untouched():
    """EFIT constraints are copied from DD-conformant magnetics: no 2*pi, no sign.

    Walking the whole OMAS table scaled flux_loop.measured by 2*pi (and left
    its error bar alone), so it no longer matched magnetics.flux_loop.
    """
    import vaft

    ods = vaft.omas.load(vaft.data.sample(41524, representation="omas"))
    assert ods_cocos(ods) == 1
    before = copy.deepcopy(ods)
    assert equilibrium_psi_to_weber(ods) is True
    assert ods_cocos(ods) == 11

    constraints = {
        key: value
        for key, value in before.flat().items()
        if key.startswith("equilibrium.time_slice.") and ".constraints." in key
    }
    assert any(".flux_loop." in key for key in constraints)
    after = ods.flat()
    for key, value in constraints.items():
        if np.asarray(value).dtype.kind in "fiu":
            np.testing.assert_array_equal(np.asarray(after[key]), np.asarray(value), err_msg=key)

    time = float(ods["equilibrium.time"][0])
    flux = ods["magnetics.flux_loop.0.flux"]
    flux_time = np.asarray(flux["time"] if "time" in flux else ods["magnetics.time"], float)
    expected = float(np.interp(time, flux_time, np.asarray(flux["data"], float)))
    measured = float(ods["equilibrium.time_slice.0.constraints.flux_loop.0.measured"])
    assert measured == pytest.approx(expected, rel=0.05)
