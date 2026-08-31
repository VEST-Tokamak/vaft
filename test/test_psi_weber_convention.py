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
    ts = legacy["equilibrium.time_slice.0"]
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
