"""TGLF -> gyrokinetics_local, per TGLF_MAPPING_AUDIT (#1591 PR-B).

Uses the real #1482 runs (39915 r/a 0.7; SAT0 ES and SAT2 with A_parallel). The local
input is rebuilt from each run's own ``input.tglf``, so the mapping sees exactly the
numbers TGLF solved with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omas import ODS

from vaft.code.gacode.tglf.inputs import TGLFInput
from vaft.code.gacode.tglf.outputs import collect_tglf_outputs
from vaft.machine_mapping import gyrokinetics as gk

DATA = Path(__file__).parent / "data" / "gacode"
SAT0 = DATA / "tglf_vest_39915_r0.70_sat0-es"
SAT2 = DATA / "tglf_vest_39915_r0.70_sat2-em-bper"


def read_input_tglf(directory: Path) -> tuple[TGLFInput, dict]:
    raw = {}
    for line in (directory / "input.tglf").read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            value = value.strip()
            if value in (".true.", ".false."):
                raw[key.strip()] = value == ".true."
            else:
                try:
                    raw[key.strip()] = float(value)
                except ValueError:
                    raw[key.strip()] = value
    ns = int(raw["NS"])
    species = lambda key: np.array([raw[f"{key}_{i}"] for i in range(1, ns + 1)])
    local = TGLFInput(
        rho=raw["RMIN_LOC"], rmin_loc=raw["RMIN_LOC"], rmaj_loc=raw["RMAJ_LOC"],
        drmajdx_loc=raw["DRMAJDX_LOC"], zmaj_loc=raw["ZMAJ_LOC"], dzmajdx_loc=raw["DZMAJDX_LOC"],
        q_loc=raw["Q_LOC"], q_prime_loc=raw["Q_PRIME_LOC"], p_prime_loc=raw["P_PRIME_LOC"],
        kappa_loc=raw["KAPPA_LOC"], s_kappa_loc=raw["S_KAPPA_LOC"], delta_loc=raw["DELTA_LOC"],
        s_delta_loc=raw["S_DELTA_LOC"], zeta_loc=raw["ZETA_LOC"], s_zeta_loc=raw["S_ZETA_LOC"],
        zs=species("ZS"), mass=species("MASS"), as_=species("AS"), taus=species("TAUS"),
        rlns=species("RLNS"), rlts=species("RLTS"), betae=raw["BETAE"], xnue=raw["XNUE"],
        zeff=raw["ZEFF"], debye=raw["DEBYE"], sign_bt=raw["SIGN_BT"], sign_it=raw["SIGN_IT"],
        names=("e", "H+", "C6+")[:ns],
    )
    return local, raw


@pytest.fixture(scope="module", params=[SAT0, SAT2], ids=["sat0-es", "sat2-em"])
def mapped(request):
    local, parameters = read_input_tglf(request.param)
    outputs = collect_tglf_outputs(request.param)
    ods = ODS()
    report = gk.gyrokinetics_local_from_tglf(ods, local, outputs, parameters=parameters, time=0.317)
    return local, parameters, outputs, ods, report


def test_the_rescaling_uses_tglfs_own_bt0_which_is_cgyros_b_gs2(mapped):
    local, _, outputs, ods, _ = mapped
    b = outputs.saturation_parameters["Bt0_out"]
    assert b == pytest.approx(0.54748, rel=1e-4)       # CGYRO's b_gs2 on this surface
    ky0 = ods["gyrokinetics_local.linear.wavevector.0.binormal_wavevector_norm"]
    assert ky0 == pytest.approx(outputs.ky_spectrum[0] * np.sqrt(2) / b)


def test_every_found_mode_is_written_and_empty_slots_are_not(mapped):
    local, _, outputs, ods, report = mapped
    rate = local.rmaj_loc / np.sqrt(2)
    gamma, omega = outputs.growth_rate, outputs.frequency
    for k in range(outputs.ky_spectrum.size):
        found = [m for m in range(gamma.shape[1]) if not (gamma[k, m] == 0 and omega[k, m] == 0)]
        modes = ods[f"gyrokinetics_local.linear.wavevector.{k}"].get("eigenmode", None)
        written = 0 if modes is None else len(modes)
        assert written == len(found)
        for slot, m in enumerate(found):
            mode = f"gyrokinetics_local.linear.wavevector.{k}.eigenmode.{slot}"
            assert ods[f"{mode}.growth_rate_norm"] == pytest.approx(gamma[k, m] * rate)
            assert ods[f"{mode}.frequency_norm"] == pytest.approx(omega[k, m] * rate)
            assert ods[f"{mode}.initial_value_run"] == 0


def test_quasilinear_fluxes_sum_to_tglfs_totals_in_the_species_order_written(mapped):
    local, parameters, outputs, ods, _ = mapped
    factors = gk._factors(local.rmaj_loc, outputs.saturation_parameters["Bt0_out"])
    energy = ods["gyrokinetics_local.non_linear.fluxes_1d.energy_phi_potential"]
    if parameters.get("USE_BPER"):
        energy = energy + ods["gyrokinetics_local.non_linear.fluxes_1d.energy_a_field_parallel"]
    # gbflux is electrons first; species[:] is electrons last
    expected = np.roll(outputs.energy_flux, -1) * factors["flux"]
    assert energy == pytest.approx(expected, rel=2e-3)
    assert ods["gyrokinetics_local.species.2.charge_norm"] == -1.0
    assert ods["gyrokinetics_local.non_linear.quasi_linear"] == 1


def test_the_field_model_and_preset_are_recorded(mapped):
    _, parameters, outputs, ods, _ = mapped
    assert ods["gyrokinetics_local.code.name"] == "TGLF"
    assert ods["gyrokinetics_local.model.include_a_field_parallel"] == int(bool(parameters.get("USE_BPER")))
    text = ods["gyrokinetics_local.code.parameters"]
    assert f"<xnu_model>{outputs.saturation_parameters['XNU_MODEL']}</xnu_model>" in text
    assert "ion_diamagnetic_negative" in text


def test_unsupported_quantities_are_reported_and_absent(mapped):
    _, _, _, ods, report = mapped
    assert any("collisions_*" in r for r in report["skipped"])
    assert any("fluctuation spectra" in r for r in report["skipped"])
    assert "collisions" not in ods["gyrokinetics_local"]
    assert "fields" not in ods["gyrokinetics_local.linear.wavevector.0.eigenmode.0"]


def test_a_run_without_saturation_parameters_writes_nothing(tmp_path):
    local, parameters = read_input_tglf(SAT0)
    for path in SAT0.iterdir():
        if path.name != "out.tglf.scalar_saturation_parameters":
            (tmp_path / path.name).write_bytes(path.read_bytes())
    ods = ODS()
    report = gk.gyrokinetics_local_from_tglf(ods, local, collect_tglf_outputs(tmp_path),
                                             parameters=parameters)
    assert not report["written"] and "gyrokinetics_local" not in ods


def test_the_ods_round_trips(mapped, tmp_path):
    from omas import load_omas_json, save_omas_json

    *_, ods, _ = mapped
    save_omas_json(ods, str(tmp_path / "gk.json"))
    back = load_omas_json(str(tmp_path / "gk.json"))
    assert back["gyrokinetics_local.code.name"] == "TGLF"


def test_every_tglf_audit_class_is_declared():
    kinds = {"exact", "unit", "coordinate", "derived", "convention", "unsupported"}
    assert {kind for kind, _ in gk.TGLF_MAPPING_AUDIT.values()} <= kinds


def test_ky_resolved_fluxes_sum_to_the_totals(mapped):
    local, parameters, outputs, ods, _ = mapped
    nl = "gyrokinetics_local.non_linear"
    per_ky = ods[f"{nl}.fluxes_2d_k_x_sum.energy_phi_potential"]
    assert per_ky.shape == (3, outputs.ky_spectrum.size)
    assert per_ky.sum(axis=1) == pytest.approx(ods[f"{nl}.fluxes_1d.energy_phi_potential"])
    assert ods[f"{nl}.binormal_wavevector_norm"].size == outputs.ky_spectrum.size


# -- merge_linear_scan (cold review 0.8.0 delta-absorb-17 transport F3/F4) ------------

def _single_ky_runs():
    import copy

    from _synthetic_inputs import make_gyrokinetics_local

    one = make_gyrokinetics_local(None)
    two = copy.deepcopy(one)
    wave = "gyrokinetics_local.linear.wavevector.0"
    two[f"{wave}.binormal_wavevector_norm"] = one[f"{wave}.binormal_wavevector_norm"] / 2
    return one, two


def test_merge_linear_scan_refuses_mixed_conventions():
    """normalisation / frequency_sign_convention live only in code.parameters, which
    the physics-leaf comparison skips; the merge must compare them itself."""
    import copy
    import re

    one, two = _single_ky_runs()
    assert gk.merge_linear_scan([one, two])["gyrokinetics_local.linear.wavevector"]
    parameters = str(one["gyrokinetics_local.code.parameters"])
    for key in ("frequency_sign_convention", "normalisation"):
        assert f"<{key}>" in parameters
        other = copy.deepcopy(two)
        other["gyrokinetics_local.code.parameters"] = re.sub(
            fr"<{key}>.*?</{key}>", f"<{key}>something else</{key}>", parameters)
        with pytest.raises(ValueError, match=key):
            gk.merge_linear_scan([one, other])
    foreign_code = copy.deepcopy(two)
    foreign_code["gyrokinetics_local.code.name"] = "GS2"
    with pytest.raises(ValueError, match="code.name"):
        gk.merge_linear_scan([one, foreign_code])
    with pytest.raises(ValueError, match="repeats binormal_wavevector_norm"):
        gk.merge_linear_scan([one, copy.deepcopy(one)])


def test_merge_linear_scan_names_a_run_without_a_wavevector():
    """A run that produced no eigenmode is refused by position wherever it sits;
    first in the list it used to die with KeyError, later it was silently merged."""
    import copy

    one, two = _single_ky_runs()
    empty = copy.deepcopy(two)
    del empty["gyrokinetics_local.linear.wavevector"]
    with pytest.raises(ValueError, match="run 0 carries no linear.wavevector"):
        gk.merge_linear_scan([empty, one])
    with pytest.raises(ValueError, match="run 1 carries no linear.wavevector"):
        gk.merge_linear_scan([one, empty])
