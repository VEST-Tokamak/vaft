"""Advisory ordering checks in adapter run provenance: EFIT, GPEC suite, CGYRO."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from vaft.validation.model_contracts import adapter_mode, ordering_provenance


def test_the_record_is_strict_json_and_none_without_a_formalism():
    record = ordering_provenance(adapter_mode("gpec:rdcon_rmatch").formalism, {"lundquist_number": 1e5})
    text = json.dumps(record, allow_nan=False)                     # a missing margin is null, never NaN
    assert json.loads(text)["contracts"][0]["margins"]["k_perp_rho_i"] is None
    assert [(c["contract"], c["role"]) for c in record["contracts"]] == [
        ("ideal_single_fluid_mhd", "outer_solution"), ("resistive_mhd", "layer_solution")]
    assert record["advisory"] is True and record["solver"] == "rdcon"
    assert ordering_provenance(None, {}) is None


# -- EFIT --

def test_efit_records_the_check_without_changing_its_scientific_identity(tmp_path):
    from vaft.code.efit.magnetic import EFITConfig, _write_efit_configuration_manifest, resolved_efit_configuration

    state = {"tau_evolution_over_tau_alfven": 3e4, "sonic_mach_number": 0.01, "alfven_mach_number": 0.002}
    plain = resolved_efit_configuration(EFITConfig(shot=39915))
    checked = resolved_efit_configuration(EFITConfig(shot=39915, ordering_state=state))
    assert plain["scientific_sha256"] == checked["scientific_sha256"]
    (contract,) = checked["ordering_check"]["contracts"]
    assert (contract["contract"], contract["role"], contract["status"]) == (
        "quasi_static_equilibrium", "governing", "SUPPORTED")
    assert plain["ordering_check"]["contracts"][0]["status"] == "UNASSESSED"   # EFIT's own inputs carry none
    # the strict manifest writer accepts it, NaN inputs included
    manifest = _write_efit_configuration_manifest(
        EFITConfig(shot=39915, ordering_state={"tau_evolution_over_tau_alfven": math.nan}), [], tmp_path / "m.json")
    assert json.loads(manifest.read_text())["resolved"]["ordering_check"]["contracts"][0]["status"] == "UNASSESSED"


# -- GPEC suite --

def test_gpec_cells_record_the_check_of_their_resolved_formalism(tmp_path):
    from vaft.code import gpec

    geqdsk = tmp_path / "g039915.00325"
    geqdsk.write_text("  EFITD   01/01/2024   #  39915  325ms        3  65  65\n 1.0 2.0 3.0\n", encoding="utf-8")
    case = gpec.GPECCaseInputs(shot=39915, time_ms=325, geqdsk=geqdsk, workdir=tmp_path / "run")
    state = {"lundquist_number": 1e5, "ion_skin_depth_over_a": 0.3}
    config = gpec.GPECSuiteConfig(modules=("dcon",), modes=(1,), run_mode="prepare_only", ordering_state=state)
    result = gpec.run_gpec_suite_case(case, config)
    (record,) = [r for r in result.records if r.module == "dcon"]
    assert record.plasma_formalism is not None
    check = record.ordering_check
    assert [(c["contract"], c["role"]) for c in check["contracts"]] == [("ideal_single_fluid_mhd", "governing")]
    assert check["contracts"][0]["margins"]["lundquist_number"] == pytest.approx(5.0)
    # carried by the asdict run manifest
    import dataclasses

    assert dataclasses.asdict(record)["ordering_check"] == check


# -- CGYRO --

SAMPLE = None
try:  # pragma: no cover - depends on the repository-only sample being present
    from vaft.data.resources import data_path

    _candidate = Path(data_path("kineticEfit/ods_48224_300ms.json"))
    SAMPLE = _candidate if _candidate.exists() else None
except Exception:
    SAMPLE = None


@pytest.mark.skipif(SAMPLE is None, reason="the packaged 48224 kinetic sample is a repository-only asset")
def test_cgyro_local_quantities_match_the_formula_kernels():
    from omas import load_omas_json

    from vaft.code.gacode import cgyro
    from vaft.code.gacode.cgyro.runner import _ordering_check
    from vaft.code.gacode.inputs import prepare_gacode_profile
    from vaft.formula.constants import QE
    from vaft.formula.ordering import debye_length

    profile = prepare_gacode_profile(load_omas_json(str(SAMPLE), consistency_check=False),
                                     rho_max=0.95, z_eff=2.0, impurity="C")
    local = cgyro.prepare_cgyro_input(profile, 0.7)
    state = cgyro.local_ordering_state(local)
    n, t = local.tglf.normalisation, local.tglf
    # independently, in SI: m_D from the normalisation's own rho_s = c_s m_D / (e B_unit)
    b = abs(n.b_unit)
    m_d = QE * b * abs(n.gyroradius) / n.sound_speed
    t_i = float(t.taus[1]) * n.electron_temperature                   # J
    rho_i = math.sqrt(t_i * float(t.mass[1]) * m_d) / (abs(float(t.zs[1])) * QE * b)
    assert state["rho_i_over_LTi"] == pytest.approx(rho_i * abs(float(t.rlts[1])) / n.minor_radius, rel=1e-4)  # constants rounded in GACODE units
    lambda_d = debye_length(n.electron_density, n.electron_temperature / QE)
    assert state["debye_length_over_L"] == pytest.approx(
        lambda_d * abs(float(t.rlts[0])) / n.minor_radius, rel=2e-3)        # TGLF's DEBYE uses NRL's 7.43
    # the run provenance records the gyrokinetic contract on them, with the caller's extra state
    record = _ordering_check(SimpleNamespace(local=local), cgyro.CGYROConfig(), {"sonic_mach_number": 0.02})
    (contract,) = record["contracts"]
    assert (contract["contract"], contract["role"]) == ("gyrokinetic_delta_f", "governing")
    assert contract["margins"]["rho_i_over_LTi"] is not None and contract["margins"]["sonic_mach_number"] is not None
    assert contract["margins"]["k_par_over_k_perp"] is None            # not an input: UNASSESSED
    assert contract["status"] == "UNASSESSED"


def test_cgyro_without_a_local_input_records_only_the_callers_state():
    from vaft.code.gacode import cgyro
    from vaft.code.gacode.cgyro.runner import _ordering_check

    assert cgyro.local_ordering_state(None) == {}
    record = _ordering_check(SimpleNamespace(local=None), cgyro.CGYROConfig(), None)
    assert record["contracts"][0]["status"] == "UNASSESSED"


def test_unknown_keys_are_listed_and_labels_keep_their_type():
    from vaft.validation.model_contracts import _json_label

    record = ordering_provenance(adapter_mode("efit:reconstruction").formalism,
                                 {"tau_evolution_over_tau_alfven": 3e4, "tau_evol_over_tau_a": 3e4})
    assert record["ignored"] == ["tau_evol_over_tau_a"]
    assert _json_label(3) == 3 and isinstance(_json_label(3), int)
    assert _json_label(0.7) == 0.7 and _json_label(("s", 1)) == "('s', 1)" and _json_label("mid") == "mid"


def test_a_failing_check_is_recorded_not_raised(monkeypatch):
    import vaft.validation.model_contracts as mc

    def boom(*args, **kwargs):
        raise RuntimeError("broken contract")

    monkeypatch.setattr(mc, "ordering_provenance", boom)
    record = mc.safe_ordering_provenance(adapter_mode("efit:reconstruction").formalism, {})
    assert record["advisory"] is True and "broken contract" in record["error"]
    json.dumps(record, allow_nan=False)


def test_cgyro_caller_state_precedence_and_species_overrides(monkeypatch):
    from vaft.code.gacode import cgyro
    from vaft.code.gacode.cgyro import runner

    monkeypatch.setattr(runner, "local_ordering_state", lambda local: {"rho_i_over_LTi": 0.01})

    def value(record, name):
        return record["contracts"][0]["values"][name]

    local = SimpleNamespace(local=object())
    # a finite caller value wins; a caller NaN does not hide the derived one
    assert value(runner._ordering_check(local, cgyro.CGYROConfig(), {"rho_i_over_LTi": 0.5}), "rho_i_over_LTi") == 0.5
    assert value(runner._ordering_check(local, cgyro.CGYROConfig(), {"rho_i_over_LTi": math.nan}),
                 "rho_i_over_LTi") == 0.01
    # a species override in extra_parameters means the staged input is not what ran
    scanned = runner._ordering_check(local, cgyro.CGYROConfig(extra_parameters={"dlntdr_1": 3.0}), None)
    assert value(scanned, "rho_i_over_LTi") is None and "DLNTDR_1" in scanned["notes"][0]
