"""The CGYRO input translation, held against CGYRO's own projection (#1354).

The fixture ``data/gacode/cgyro_oracle_48224/r<r/a>/`` is CGYRO itself (GACODE b493397,
TDST_GNU) run in test mode (``cgyro -t``) with ``PROFILE_MODEL=2`` on the
``input.gacode`` that :func:`prepare_gacode_profile` writes for the packaged 48224
kinetic sample (``rho_max=0.95, z_eff=2, impurity="C"``). CGYRO projects the surface
with its own ``expro_locsim`` and writes what it resolved to ``out.cgyro.equilibrium``;
:func:`compare_with_oracle` holds VAFT's renaming of the TGLF local input against that,
the way ``test_tglf_input.py`` holds the TGLF projection against ``locpargen``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vaft.code.gacode import cgyro
from vaft.code.gacode.cgyro.outputs import collect_cgyro_outputs, parse_equilibrium

ORACLE = Path(__file__).parent / "data" / "gacode" / "cgyro_oracle_48224"
RADII = (0.6, 0.7, 0.8)

SAMPLE = None
try:  # pragma: no cover - depends on the repository-only sample being present
    from vaft.data.resources import data_path

    _candidate = Path(data_path("kineticEfit/ods_48224_300ms.json"))
    SAMPLE = _candidate if _candidate.exists() else None
except Exception:
    SAMPLE = None

requires_sample = pytest.mark.skipif(
    SAMPLE is None, reason="the packaged 48224 kinetic sample is a repository-only asset"
)


@pytest.fixture(scope="module")
def profile():
    from omas import load_omas_json

    from vaft.code.gacode.inputs import prepare_gacode_profile

    ods = load_omas_json(str(SAMPLE), consistency_check=False)
    return prepare_gacode_profile(ods, rho_max=0.95, z_eff=2.0, impurity="C")


@requires_sample
@pytest.mark.parametrize("rho", RADII)
def test_every_local_parameter_matches_cgyros_own_projection(profile, rho):
    local = cgyro.prepare_cgyro_input(profile, rho)
    equilibrium = parse_equilibrium(ORACLE / f"r{rho}" / "out.cgyro.equilibrium", 3)
    report = cgyro.compare_with_oracle(local, equilibrium)
    expected = set(cgyro.inputs.ORACLE_KEYS) | {"nu_ee"} | {
        f"{key}_{i}" for i in (1, 2, 3) for key in ("DENS", "TEMP", "MASS", "DLNNDR", "DLNTDR")
    }
    assert set(report) == expected
    failures = {key: row for key, row in report.items() if not row["ok"]}
    assert not failures, failures


@requires_sample
@pytest.mark.parametrize("rho", RADII)
def test_the_field_orientation_matches_cgyros_own_signs(profile, rho):
    """CGYRO signs q by IPCCW*BTCCW and B_unit by -BTCCW (cgyro_make_profiles), so
    the oracle's signed q and b_unit pin both orientation flags of the translation --
    a swap of SIGN_IT and SIGN_BT, invisible to |q|, fails here."""
    import numpy as np

    local = cgyro.prepare_cgyro_input(profile, rho)
    run = collect_cgyro_outputs(ORACLE / f"r{rho}")
    assert np.sign(run.equilibrium["q"]) == local.ipccw * local.btccw
    assert np.sign(run.equilibrium["b_unit"]) == -local.btccw
    # and the ion direction CGYRO printed is the one the sign rule gives: omega > 0
    # exactly when q*rho < 0, i.e. when IPCCW = +1
    assert run.ion_direction == (1 if local.ipccw > 0 else -1)


def test_test_mode_writes_an_exit_line_but_is_not_a_solved_run():
    """``cgyro -t`` ends with 'Linear terminated at max time' and no frequency record."""
    run = collect_cgyro_outputs(ORACLE / "r0.7")
    assert run.exit_message == "Linear terminated at max time"
    assert run.frequency is None and not run.solved


def test_b_gs2_is_the_field_ratio_the_imas_mapping_needs():
    equilibrium = parse_equilibrium(ORACLE / "r0.7" / "out.cgyro.equilibrium", 3)
    assert equilibrium["b_gs2"] == pytest.approx(0.48495)
    assert equilibrium["hiprec_flag"] == 0
    assert equilibrium["z_eff"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# a real linear run: the translated 48224 r/a=0.7 input, EM (N_FIELD=2), ky=0.3,
# adaptive step, 8 MPI tasks on tdst (2026-10-01). Restart file not kept.
# --------------------------------------------------------------------------

LINEAR = Path(__file__).parent / "data" / "gacode" / "cgyro_linear_48224_r0.7_em"


def test_a_real_linear_run_parses_in_cgyros_own_layouts():
    run = collect_cgyro_outputs(LINEAR)
    assert run.exit_message == "Linear converged" and run.solved and run.converged
    # out.cgyro.freq is ASCII on this build (no bin.cgyro.freq); [omega, gamma] per step
    assert run.frequency.shape == (1, run.time.size)
    assert run.final_growth_rate[0] == pytest.approx(0.28338, rel=1e-4)
    assert run.final_frequency[0] == pytest.approx(-1.0227, rel=1e-4)
    assert set(run.ballooning) == {"phi", "a_parallel"}
    assert run.ballooning["phi"].shape == run.grid["thetab"].shape
    assert run.flux.shape[:3] == (3, 3, 2)
    assert run.version["commit"] == "b493397"


def test_cgyro_writes_ky_signed_and_the_container_reports_its_magnitude():
    run = collect_cgyro_outputs(LINEAR)
    assert run.grid["ky"][0] == pytest.approx(-0.3)
    assert run.ky[0] == pytest.approx(0.3)


def test_the_real_mode_is_in_the_electron_direction():
    """Native omega < 0 with CGYRO's ion direction omega > 0: an electron-direction mode,
    so TGLF's convention (ion negative) gives a positive frequency."""
    run = collect_cgyro_outputs(LINEAR)
    assert run.ion_direction == 1
    assert run.frequency_ion_negative[0] == pytest.approx(1.0227, rel=1e-4)


@requires_sample
def test_the_real_run_maps_into_gyrokinetics_local(profile, tmp_path):
    from omas import ODS, load_omas_json, save_omas_json

    from vaft.machine_mapping.gyrokinetics import gyrokinetics_local_from_cgyro

    local = cgyro.prepare_cgyro_input(profile, 0.7)
    run = collect_cgyro_outputs(LINEAR)
    parameters = cgyro.cgyro_parameters(local, cgyro.CGYROConfig(field_model="em-aperp"))
    ods = ODS()
    report = gyrokinetics_local_from_cgyro(
        ods, local, run, provenance={"parameters": parameters}, time=0.3)
    assert report["written"]
    mode = "gyrokinetics_local.linear.wavevector.0.eigenmode.0"
    rate = local.geometry["RMAJ"] / 2 ** 0.5
    assert ods[f"{mode}.growth_rate_norm"] == pytest.approx(0.28338 * rate, rel=1e-4)
    assert ods[f"{mode}.frequency_norm"] > 0  # electron direction, ion-negative convention
    assert ods["gyrokinetics_local.model.include_a_field_parallel"] == 1
    save_omas_json(ods, str(tmp_path / "gk.json"))
    assert load_omas_json(str(tmp_path / "gk.json"))["gyrokinetics_local.code.name"] == "CGYRO"
