"""Equilibrium-native Mercier / GGJ interchange indices against DCON (#1902).

``test/data/gpec/mercier_reference_1902/dcon_di_dr.json`` holds DCON's own
``di`` and ``dr`` (mpsi 512, mtheta 1024, GPEC e68d7ac2) for the packaged VEST
g-file ``vaft/data/efit/g039915.00319`` and, when GPEC's example tree is
present, the upstream DIII-D-like example.  Nothing here runs a solver.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from vaft.process.local_mhd_criteria import interchange_comparison, mercier_ggj_from_ods, mercier_ggj_profile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "test" / "data" / "gpec" / "mercier_reference_1902" / "dcon_di_dr.json").read_text())


def _gfile(path):
    from vaft.data.eqdsk import read_geqdsk

    m = read_geqdsk(path).mapping
    nw, nh = int(m["NW"]), int(m["NH"])
    grid = dict(
        psi_grid=np.asarray(m["PSIRZ"]),  # the reader returns it indexed (R, Z)
        R=m["RLEFT"] + np.linspace(0.0, m["RDIM"], nw),
        Z=m["ZMID"] - m["ZDIM"] / 2 + np.linspace(0.0, m["ZDIM"], nh),
        psi_axis=float(m["SIMAG"]), psi_boundary=float(m["SIBRY"]),
    )
    profiles = dict(profile_psi_n=np.linspace(0.0, 1.0, nw), f=np.asarray(m["FPOL"]), pressure=np.asarray(m["PRES"]),
                    q=np.asarray(m["QPSI"]), axis_rz=(float(m["RMAXIS"]), float(m["ZMAXIS"])))
    return grid, profiles


VEST = ROOT / "vaft" / "data" / "efit" / "g039915.00319"
LEVELS = np.round(np.arange(0.15, 0.951, 0.05), 2)


@pytest.fixture(scope="module")
def vest():
    grid, profiles = _gfile(VEST)
    return grid, profiles, mercier_ggj_profile(**grid, levels_norm=LEVELS, **profiles)


def test_vest_slice_matches_dcon(vest):
    _, _, profile = vest
    case = REFERENCE["cases"]["vest_39915_319"]
    c = interchange_comparison(profile, case["psi_n"], case["di"], case["dr"], label="DCON 39915@319")
    outer = profile.psi_n >= 0.4
    assert np.nanmax(np.abs(c["dD_I"][outer])) < 2e-3 and np.nanmax(np.abs(c["dD_R"][outer])) < 2e-3
    inner = ~outer  # D_I runs to -3.5 here: compare relatively
    assert np.nanmax(np.abs(c["dD_I"][inner] / c["di"][inner])) < 0.02
    assert profile.provenance["unresolved_levels"] == 0
    # Universal criterion only: this slice is Mercier stable everywhere evaluated.
    assert not profile.mercier_unstable.any()


def test_the_derivative_and_contour_resolution_are_converged(vest):
    grid, profiles, profile = vest
    finer = mercier_ggj_profile(**grid, levels_norm=LEVELS, **profiles, oversample=4)
    outer = LEVELS >= 0.3
    # measured: 0.35 % at psi_N 0.3-0.35 between oversample 2 and 4, less outside
    np.testing.assert_allclose(finer.D_I[outer], profile.D_I[outer], rtol=5e-3)
    np.testing.assert_allclose(finer.H[outer], profile.H[outer], atol=2e-3)


def test_the_indices_do_not_depend_on_cocos_signs(vest):
    grid, profiles, profile = vest
    flipped = mercier_ggj_profile(
        **{**grid, "psi_grid": -grid["psi_grid"], "psi_axis": -grid["psi_axis"], "psi_boundary": -grid["psi_boundary"]},
        levels_norm=LEVELS, **{**profiles, "f": -profiles["f"], "q": -profiles["q"]},
    )
    np.testing.assert_allclose(flipped.D_I, profile.D_I, rtol=1e-10)
    np.testing.assert_allclose(flipped.D_R, profile.D_R, rtol=1e-10)


def test_without_a_pressure_gradient_only_the_shear_term_is_left(vest):
    grid, profiles, _ = vest
    flat = mercier_ggj_profile(**grid, levels_norm=LEVELS, **{**profiles, "pressure": np.full_like(profiles["pressure"], 1e3)})
    np.testing.assert_allclose(flat.D_I, -0.25, atol=1e-12)  # D_I = -1/4: shear alone is stabilizing
    np.testing.assert_allclose(flat.H, 0.0, atol=1e-12)
    np.testing.assert_allclose(flat.D_R, 0.0, atol=1e-12)  # marginal for resistive interchange
    np.testing.assert_allclose(flat.E_plus_F, 0.0, atol=1e-12)


def test_the_ods_path_is_the_array_path(vest):
    from vaft.data.eqdsk import read_geqdsk

    _, _, profile = vest
    ods = read_geqdsk(VEST).to_omas()
    from_ods = mercier_ggj_from_ods(ods, 0, levels_norm=LEVELS)
    np.testing.assert_allclose(from_ods.D_I, profile.D_I, rtol=1e-6)
    assert from_ods.provenance["time_slice"] == 0


def test_the_code_h_is_the_root_nearer_the_equilibrium():
    from vaft.process.local_mhd_criteria import InterchangeProfile

    profile = InterchangeProfile(psi_n=np.array([0.5]), D_I=np.array([-0.3]), D_R=np.array([-0.05]), H=np.array([0.0]),
                                 E_plus_F=np.array([-0.05]), dV_dchi=np.array([1.0]), averages={})
    c = interchange_comparison(profile, [0.0, 1.0], [-0.3, -0.3], [-0.05, -0.05], label="synthetic")
    assert c["H_code"][0] == pytest.approx(0.0)  # (H - 1/2)^2 = 0.25: roots 0 and 1, 0 is nearer


def test_inputs_are_refused_rather_than_guessed(vest):
    grid, profiles, _ = vest
    with pytest.raises(ValueError, match="inside"):
        mercier_ggj_profile(**grid, levels_norm=[0.0, 0.2, 0.4, 0.6, 0.8], **profiles)
    with pytest.raises(ValueError, match="indexed"):
        mercier_ggj_profile(**{**grid, "psi_grid": grid["psi_grid"][:, :-1]}, levels_norm=LEVELS, **profiles)
    with pytest.raises(ValueError, match="oversample"):
        mercier_ggj_profile(**grid, levels_norm=LEVELS, **profiles, oversample=0)


DIIID = Path(os.environ.get("GPECHOME", Path.home() / "git" / "GPEC")) / "docs" / "examples" / "DIIID_ideal_example" / "TkMkr_D3Dlike_Hmode.geqdsk"


@pytest.mark.skipif(not DIIID.is_file(), reason="GPEC's DIII-D-like example is not installed here")
def test_diiid_like_example_matches_dcon():
    grid, profiles = _gfile(DIIID)
    levels = np.round(np.arange(0.12, 0.91, 0.02), 2)
    profile = mercier_ggj_profile(**grid, levels_norm=levels, **profiles)
    case = REFERENCE["cases"]["diiid_like"]
    c = interchange_comparison(profile, case["psi_n"], case["di"], case["dr"], label="DCON DIII-D-like")
    assert np.nanmax(np.abs(c["dD_I"])) < 2e-3 and np.nanmax(np.abs(c["H"] - c["H_code"])) < 1e-3
