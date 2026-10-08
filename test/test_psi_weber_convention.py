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
from vaft.omas.formula_wrapper import compute_voltage_consumption


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


def test_real_vest_shot_loop_voltage_consumption():
    """Issue #652: Real-data regression test for the voltage consumption path.

    Validates that compute_voltage_consumption correctly handles real VEST EFIT
    reconstructions (shot 39915), automatically detects the DD Weber flux convention,
    yields loop voltages and volt-second consumption within physical bounds for VEST,
    maintains partition consistency (V_loop = V_ind + V_res), and remains invariant
    under legacy Wb/rad storage.
    """
    g1 = sample_geqdsk("efit/g039915.00317")
    g2 = sample_geqdsk("efit/g039915.00319")

    ods = g1.to_omas()
    ods["equilibrium.time_slice.1"] = g2.to_omas()["equilibrium.time_slice.0"]
    ods["equilibrium.time_slice.0.time"] = 0.317
    ods["equilibrium.time_slice.1.time"] = 0.319

    t, v_loop, v_ind, v_res = compute_voltage_consumption(ods)
    t = np.asarray(t, float)
    v_loop = np.asarray(v_loop, float)
    v_ind = np.asarray(v_ind, float)
    v_res = np.asarray(v_res, float)

    # 1. Loop voltage physical bounds during flat-top
    assert np.all(v_loop > 0.5)
    assert np.all(v_loop < 5.0)
    assert float(np.mean(v_loop)) == pytest.approx(2.1733, rel=1e-3)

    # 2. Volt-second consumption over dt = 2 ms
    dt = t[1] - t[0]
    volt_seconds = float(v_loop[0] * dt)
    assert 1.0e-3 < volt_seconds < 1.0e-2
    # Exact consistency with the *boundary* flux change 2*pi * Delta(psi_boundary).
    # This used to pin Delta(psi_boundary - psi_axis), the flux change inside
    # the plasma, which is not a loop voltage; it happened to be within 17 %
    # here and is nine-fold off on 41672 (see the real per-radian test below).
    d_psi_wb = (float(g2["SIBRY"]) - float(g1["SIBRY"])) * TWO_PI
    assert volt_seconds == pytest.approx(d_psi_wb, rel=1e-6)

    # 3. Energy partition consistency: V_loop = V_ind + V_res, and V_res is a
    # plasma resistance's worth -- about 19 micro-ohm at 80 kA.  With the
    # vacuum toroidal field in W_mag it was not.
    np.testing.assert_allclose(v_res + v_ind, v_loop, rtol=1e-12)
    resistance = v_res / float(g1["CURRENT"])
    assert np.all((resistance > 5e-6) & (resistance < 5e-5))

    # 4. Storage convention invariance: DD Wb vs legacy Wb/rad
    legacy = _legacy_style(ods)
    _t_leg, v_loop_leg, v_ind_leg, v_res_leg = compute_voltage_consumption(legacy)
    np.testing.assert_allclose(v_loop_leg, v_loop, rtol=1e-9)
    np.testing.assert_allclose(v_ind_leg, v_ind, rtol=1e-9)
    np.testing.assert_allclose(v_res_leg, v_res, rtol=1e-9)


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


def test_an_inconsistent_phi_abstains_instead_of_overriding_ampere():
    """Review finding: the slope rung classified every finite ratio, so a phi
    that disagrees with q pre-empted the physically decisive Ampere test and
    produced a 2*pi error. A ratio near neither 1 nor 2*pi is evidence the input
    is broken, not evidence of a convention."""
    import copy

    from vaft.data.eqdsk import _ampere_flux_exponent, _slope_flux_exponent

    ods = sample_geqdsk("efit/g039915.00319").to_omas()  # weber family
    broken = copy.deepcopy(ods)
    ts = broken["equilibrium.time_slice.0"]
    ts["profiles_1d.phi"] = np.asarray(ts["profiles_1d.phi"], float) * 3.0

    assert _slope_flux_exponent(ts) is None, "a ratio of ~3 must abstain"
    assert _ampere_flux_exponent(ts) == 1, "Ampere's law still knows the family"
    assert ods_psi_to_wb_per_radian_factor(broken) == pytest.approx(1.0 / TWO_PI)

    # The healthy file still answers from the slope, without needing Ampere.
    assert _slope_flux_exponent(ods["equilibrium.time_slice.0"]) == 1


def test_an_out_of_range_time_index_still_consults_the_other_slices():
    """Review finding: a missing index made the walk yield the whole ODS as if it
    were a bare time slice and stop, so a file that could answer fell back to the
    DD default."""
    legacy = _strip_phi(_legacy_style(sample_geqdsk("efit/g039915.00319").to_omas()))

    assert ods_psi_to_wb_per_radian_factor(legacy, 0) == pytest.approx(1.0)
    assert ods_psi_to_wb_per_radian_factor(legacy, 7) == pytest.approx(1.0)

    # A caller handing over a bare time slice is still served.
    assert ods_psi_to_wb_per_radian_factor(
        legacy["equilibrium.time_slice.0"]
    ) == pytest.approx(1.0)
def test_legacy_artifact_without_phi_is_detected_by_ampere_law():
    """A phi-less legacy artifact must not be misread as Wb (release review).

    The packaged 39915 reference sample carries q and psi but no
    ``profiles_1d.phi``, so the dphi/dpsi slope test cannot run.  Falling
    back to the DD convention there silently divided a genuine Wb/rad
    artifact by 2*pi.  Ampere's law over the stored boundary settles it
    from the file's own data: the loop integral of B_pol reproduces the
    stored plasma current only under the correct convention.
    """
    import vaft

    ods = vaft.omas.load(vaft.data.sample(39915, representation="omas"))
    assert "phi" not in ods["equilibrium.time_slice.0.profiles_1d"]

    # Since issue #478 the packaged sample ships in DD weber and declares
    # COCOS 11; strip the declaration so Ampere's law alone must answer.
    del ods["equilibrium.code.parameters"]
    assert ods_psi_to_wb_per_radian_factor(ods, 0) == pytest.approx(1.0 / TWO_PI)

    stored_axis = float(ods["equilibrium.time_slice.0.global_quantities.psi_axis"])
    recovered = from_omas(ods, 0)
    simag = (recovered.data if hasattr(recovered, "data") else recovered)["SIMAG"]
    assert float(simag) == pytest.approx(stored_axis / TWO_PI, rel=1e-9)

    # And the pre-#478 artifact, rebuilt from the sample, is still detected.
    legacy = _legacy_style(ods)
    assert ods_psi_to_wb_per_radian_factor(legacy, 0) == pytest.approx(1.0)


def test_ampere_law_fallback_still_reports_weber_for_dd_conformant_data():
    """The fallback must not drag DD-conformant (Wb) artifacts backwards."""
    import copy

    import vaft

    # Rebuild the undeclared Wb/rad artifact the sample used to be (#478).
    ods = _legacy_style(vaft.omas.load(vaft.data.sample(39915, representation="omas")))
    del ods["equilibrium.code.parameters"]
    weber = copy.deepcopy(ods)
    ts = weber["equilibrium.time_slice.0"]
    for path in (
        "profiles_1d.psi",
        "profiles_2d.0.psi",
        "global_quantities.psi_axis",
        "global_quantities.psi_boundary",
    ):
        ts[path] = np.asarray(ts[path], dtype=float) * 2.0 * np.pi

    assert ods_psi_to_wb_per_radian_factor(weber, 0) == pytest.approx(1.0 / (2.0 * np.pi))


def test_detector_accepts_a_bare_time_slice_as_its_docstring_promises():
    """Callers that pass a time slice must get the same answer as the ODS.

    On an OMAS ODS a missing path returns an empty auto-vivified branch
    instead of raising, so the `is None` fallback never fired for a bare
    slice: the empty branch carried no psi, and the detector returned the
    Weber default for a Wb/rad artifact. Four production call sites pass a
    slice (vaft/database/_summary.py, vaft/omas/process_wrapper.py x2,
    vaft/omas/formula_wrapper.py), so this silently scaled B_R/B_Z, loop
    voltage and the psi_axis_Wb summary column by 1/(2*pi).
    """
    import copy

    import vaft

    # The packaged sample is DD weber since #478; its legacy form is Wb/rad.
    ods = _legacy_style(vaft.omas.load(vaft.data.sample(39915, representation="omas")))
    del ods["equilibrium.code.parameters"]
    assert ods_psi_to_wb_per_radian_factor(ods, 0) == pytest.approx(1.0)
    assert ods_psi_to_wb_per_radian_factor(
        ods["equilibrium.time_slice.0"]
    ) == pytest.approx(1.0)

    weber = copy.deepcopy(ods)
    slice_ = weber["equilibrium.time_slice.0"]
    for path in (
        "profiles_1d.psi",
        "profiles_2d.0.psi",
        "global_quantities.psi_axis",
        "global_quantities.psi_boundary",
    ):
        slice_[path] = np.asarray(slice_[path], dtype=float) * 2.0 * np.pi

    expected = 1.0 / (2.0 * np.pi)
    assert ods_psi_to_wb_per_radian_factor(weber, 0) == pytest.approx(expected)
    assert ods_psi_to_wb_per_radian_factor(
        weber["equilibrium.time_slice.0"]
    ) == pytest.approx(expected)


# Flux-closure windows of the packaged multi-slice equilibria; 39915 stores psi
# in weber and 41672 per radian, so the two land on different branches of the
# storage detector -- genuinely, not through _legacy_style.
_WINDOWS = {39915: (0.316, 0.326), 41672: (0.322, 0.341)}


@pytest.mark.parametrize("shot, per_radian", [(39915, False), (41672, True)])
def test_real_shots_on_both_storage_branches_match_the_inboard_flux_loop(shot, per_radian):
    """Issue #652: the voltage-consumption path on two real shots that take
    different branches of ods_psi_to_wb_per_radian_factor, checked against an
    independent measurement -- the inboard midplane flux loop -- rather than
    against a parity.  A 2 pi misclassification on either branch would put the
    ratio near 6 or 0.16; the path's old psi_boundary - psi_axis flux put
    41672 at 0.11."""
    import logging

    from vaft.omas.sample import sample_ods

    logging.disable(logging.WARNING)
    try:
        ods = sample_ods(shot)
        times = np.asarray(ods["equilibrium.time"], float)
        window = [i for i, t_i in enumerate(times) if _WINDOWS[shot][0] <= t_i <= _WINDOWS[shot][1]]
        assert ods_psi_to_wb_per_radian_factor(ods, window[0]) == pytest.approx(
            1.0 if per_radian else 1.0 / TWO_PI
        )
        t, v_loop, v_ind, v_res = (
            np.asarray(x, float) for x in compute_voltage_consumption(ods, time_slice=window)
        )
    finally:
        logging.disable(logging.NOTSET)

    loops = ods["magnetics.flux_loop"]
    inboard = min(
        range(len(loops)),
        key=lambda i: (abs(float(loops[i]["position.0.z"])), float(loops[i]["position.0.r"])),
    )
    loop_time = np.asarray(
        loops[inboard]["flux.time"] if "flux.time" in loops[inboard] else ods["magnetics.time"],
        dtype=float,
    )
    flux = np.interp(t, loop_time, np.asarray(loops[inboard]["flux.data"], float))
    ratio = abs(np.trapezoid(v_loop, t) / (flux[-1] - flux[0]))
    assert 0.7 < ratio < 1.3, ratio

    current = np.array(
        [float(ods[f"equilibrium.time_slice.{i}.global_quantities.ip"]) for i in window]
    )
    resistance = v_res / current
    assert np.all((resistance > 1e-6) & (resistance < 1e-4))
