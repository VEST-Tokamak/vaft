"""Issue #236: equilibrium psi is stored in Wb (IMAS DD), not the g-file's Wb/rad.

``to_omas`` multiplies psi-like leaves by 2*pi and divides the psi-derivative
profiles by it; ``from_omas`` inverts using
:func:`vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor`, which tells the two
storage families apart from the dphi/dpsi-vs-q slope so that legacy
VAFT-native artifacts (Wb/rad) and DD-conformant/OMFIT artifacts (Wb) both
read back correctly. The gradient-based physics in ``vaft.omas`` converts to
Wb/rad at read time, so its outputs are invariant to the storage convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.data.eqdsk import (
    TWO_PI,
    from_omas,
    ods_psi_to_wb_per_radian_factor,
    read_geqdsk,
)
from vaft.data.resources import data_path, sample_geqdsk


@pytest.fixture()
def gfile():
    return sample_geqdsk("efit/g039915.00319")


def _legacy_style(ods):
    """Rewrite a DD-correct ODS the way the pre-#236 writer did (Wb/rad)."""
    import copy

    legacy = copy.deepcopy(ods)
    for index in range(len(legacy["equilibrium.time_slice"])):
        ts = legacy[f"equilibrium.time_slice.{index}"]
        for leaf in ("global_quantities.psi_axis", "global_quantities.psi_boundary"):
            ts[leaf] = float(ts[leaf]) / TWO_PI
        ts["profiles_1d.psi"] = np.asarray(ts["profiles_1d.psi"], float) / TWO_PI
        ts["profiles_2d.0.psi"] = np.asarray(ts["profiles_2d.0.psi"], float) / TWO_PI
        ts["profiles_1d.f_df_dpsi"] = np.asarray(ts["profiles_1d.f_df_dpsi"], float) * TWO_PI
        ts["profiles_1d.dpressure_dpsi"] = np.asarray(ts["profiles_1d.dpressure_dpsi"], float) * TWO_PI
    return legacy


def test_to_omas_writes_dd_conformant_weber_psi(gfile):
    ods = gfile.to_omas()
    ts = ods["equilibrium.time_slice.0"]
    assert float(ts["global_quantities.psi_axis"]) == pytest.approx(float(gfile["SIMAG"]) * TWO_PI)
    assert float(ts["global_quantities.psi_boundary"]) == pytest.approx(float(gfile["SIBRY"]) * TWO_PI)
    np.testing.assert_allclose(
        np.asarray(ts["profiles_2d.0.psi"]), np.asarray(gfile["PSIRZ"], float) * TWO_PI
    )
    # psi-derivatives transform inversely.
    np.testing.assert_allclose(
        np.asarray(ts["profiles_1d.f_df_dpsi"]), np.asarray(gfile["FFPRIM"], float) / TWO_PI
    )
    np.testing.assert_allclose(
        np.asarray(ts["profiles_1d.dpressure_dpsi"]), np.asarray(gfile["PPRIME"], float) / TWO_PI
    )
    # phi = integral q dpsi_Wb was already written in Wb and must not change.
    slope = np.diff(np.asarray(ts["profiles_1d.phi"], float)) / np.diff(np.asarray(ts["profiles_1d.psi"], float))
    q_mid = 0.5 * (np.asarray(ts["profiles_1d.q"], float)[1:] + np.asarray(ts["profiles_1d.q"], float)[:-1])
    assert float(np.nanmedian(np.abs(slope / q_mid))) == pytest.approx(1.0, rel=0.05)


def test_family_detection_distinguishes_weber_and_per_radian(gfile):
    ods = gfile.to_omas()
    assert ods_psi_to_wb_per_radian_factor(ods) == pytest.approx(1.0 / TWO_PI)
    assert ods_psi_to_wb_per_radian_factor(_legacy_style(ods)) == pytest.approx(1.0)


def test_round_trip_and_legacy_read_recover_the_gfile(gfile):
    ods = gfile.to_omas()
    for source in (ods, _legacy_style(ods)):
        back = from_omas(source)
        for key in ("SIMAG", "SIBRY"):
            assert float(back[key]) == pytest.approx(float(gfile[key]), rel=1e-12), key
        np.testing.assert_allclose(np.asarray(back["PSIRZ"]), np.asarray(gfile["PSIRZ"], float), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(back["FFPRIM"]), np.asarray(gfile["FFPRIM"], float), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(back["PPRIME"]), np.asarray(gfile["PPRIME"], float), rtol=1e-12)


def test_omfit_produced_ods_now_reads_back_to_the_gfile():
    """The issue's own measurement: the committed OMFIT ODS holds Wb and used
    to come back 2*pi too large through from_omas."""
    from omas import load_omas_json

    ods = load_omas_json(str(data_path("kineticEfit/ods_48224_300ms.json")), consistency_check=False)
    reference = read_geqdsk(data_path("kineticEfit/g048224.00300"))
    back = from_omas(ods)
    assert float(back["SIMAG"]) == pytest.approx(float(reference["SIMAG"]), rel=2e-2)
    assert float(back["SIBRY"]) == pytest.approx(float(reference["SIBRY"]), rel=2e-2, abs=5e-4)


def test_gradient_physics_is_invariant_to_the_storage_convention(gfile):
    from vaft.omas.process_wrapper import (
        compute_diamagnetism,
        compute_magnetic_energy,
        compute_virial_equilibrium_quantities_ods,
    )

    new = gfile.to_omas()
    legacy = _legacy_style(new)

    virial_new = compute_virial_equilibrium_quantities_ods(new)[0]
    virial_legacy = compute_virial_equilibrium_quantities_ods(legacy)[0]
    for key in ("beta_p", "li", "B_pa", "s_1"):
        assert virial_new[key] == pytest.approx(virial_legacy[key], rel=1e-9), key

    assert compute_magnetic_energy(new) == pytest.approx(compute_magnetic_energy(legacy), rel=1e-9)
    mu_new = compute_diamagnetism(new)
    mu_legacy = compute_diamagnetism(legacy)
    assert mu_new == pytest.approx(mu_legacy, rel=1e-9)


def test_descriptor_path_agrees_between_gfile_and_ods(gfile):
    from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

    eq_ods = as_equilibrium(gfile.to_omas())
    assert eq_ods.convention.psi_per_radian is False  # slope-detected Wb family
    d_ods = derive_global_descriptors(eq_ods)
    d_g = derive_global_descriptors(as_equilibrium(gfile))
    for name in ("beta_p_boundary_average", "li_virial", "s1", "q95"):
        assert d_ods[name].value == pytest.approx(d_g[name].value, rel=1e-9), name


def test_loop_voltage_is_correct_and_storage_invariant(gfile):
    """Post-merge review finding B1: loop_voltage_from_total_flux multiplies by
    2*pi (a Wb/rad contract), so compute_voltage_consumption must convert the
    stored psi first. Two slices with d(psi_boundary-psi_axis) = 0.01 Wb over
    1 ms must give V_loop = 10 V, not 2*pi times that."""
    import copy

    from vaft.omas.formula_wrapper import compute_voltage_consumption

    ods = gfile.to_omas()
    second = copy.deepcopy(ods["equilibrium.time_slice.0"])
    ods["equilibrium.time_slice.1"] = second
    ods["equilibrium.time_slice.0.time"] = 0.0
    ods["equilibrium.time_slice.1.time"] = 1.0e-3
    base = float(ods["equilibrium.time_slice.0.global_quantities.psi_boundary"])
    ods["equilibrium.time_slice.1.global_quantities.psi_boundary"] = base + 0.01  # +0.01 Wb

    _t, v_loop, _v_ind, _v_res = compute_voltage_consumption(ods)
    v_loop = np.asarray(v_loop, float)
    assert float(np.nanmax(np.abs(v_loop))) == pytest.approx(10.0, rel=1e-6)

    legacy = _legacy_style(ods)
    _t2, v_legacy, _vi2, _vr2 = compute_voltage_consumption(legacy)
    v_legacy = np.asarray(v_legacy, float)
    np.testing.assert_allclose(v_legacy, v_loop, rtol=1e-9)


def _strip_phi(ods):
    """Drop ``profiles_1d.phi`` from every slice, as the EFIT pipeline once did."""
    for index in range(len(ods["equilibrium.time_slice"])):
        ts = ods[f"equilibrium.time_slice.{index}"]
        if "profiles_1d.phi" in ts:
            del ts["profiles_1d.phi"]
    return ods


def test_storage_family_is_detected_without_phi():
    """The slope test needs ``profiles_1d.phi``; the EFIT-pipeline ODS written
    before issue #236 holds Wb/rad and carries none, so the DD default used to
    rescale it by 2*pi. Ampere's law round the LCFS answers without phi."""
    import copy

    gfile = sample_geqdsk("efit/g039915.00319")
    legacy = _strip_phi(_legacy_style(gfile.to_omas()))
    assert ods_psi_to_wb_per_radian_factor(legacy) == pytest.approx(1.0)

    weber = _strip_phi(copy.deepcopy(gfile.to_omas()))
    assert ods_psi_to_wb_per_radian_factor(weber) == pytest.approx(1.0 / TWO_PI)


def test_storage_family_survives_a_degenerate_slice():
    """The convention is a property of the file. A slice EFIT failed on --
    psi_axis == psi_boundary, no boundary outline, which the packaged samples do
    contain -- must not drag the whole ODS onto the default."""
    import copy

    gfile = sample_geqdsk("efit/g039915.00319")
    ods = _strip_phi(_legacy_style(gfile.to_omas()))
    ods["equilibrium.time_slice.1"] = copy.deepcopy(ods["equilibrium.time_slice.0"])
    broken = ods["equilibrium.time_slice.1"]
    broken["global_quantities.psi_boundary"] = float(broken["global_quantities.psi_axis"])
    del broken["boundary.outline.r"]
    del broken["boundary.outline.z"]
    del broken["profiles_1d.q"]

    assert ods_psi_to_wb_per_radian_factor(ods, 1) == pytest.approx(1.0)


def test_virial_quantities_are_physical_on_the_packaged_sample():
    """Regression for the issue #278 default: on sample 39915 (Wb/rad, no phi)
    the virial path returned B_pa 2*pi too small and beta_p = 30.5."""
    from vaft.omas.sample import sample_ods

    try:
        ods = sample_ods()
    except Exception as exc:  # pragma: no cover - sample not packaged in this build
        pytest.skip(f"39915 sample unavailable: {exc}")

    from vaft.omas.process_wrapper import compute_virial_equilibrium_quantities_ods

    virial = compute_virial_equilibrium_quantities_ods(ods, time_slice=0)[0]
    assert 0.01 < float(virial["B_pa"]) < 0.5
    assert 0.0 < float(virial["beta_p"]) < 10.0
