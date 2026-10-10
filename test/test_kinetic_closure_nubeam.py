"""The analytic fast-ion closure against NUBEAM on the packaged VEST case (#1606).

``test/data/nubeam/vest_fast_ion_reference.json`` is one NUBEAM run of
``vaft/data/nubeam/vest_case`` (5 x 1 ms, 2000 markers), written by
``workflow/kinetic_closure/nubeam_fast_ion_reference.py``.  The closure is given
NUBEAM's own plasma and birth rate, so only the slowing-down physics differs:
steady, local, isotropic and lossless here, Monte Carlo orbits with bad-orbit
and charge-exchange losses there.  The bands below document where the analytic
baseline sits, not a tolerance to be tuned (#1606 Validation).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from vaft.process.kinetic_closure import fast_ion_slowing_down_estimate

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "test" / "data" / "nubeam" / "vest_fast_ion_reference.json"


@pytest.fixture(scope="module")
def case():
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    dv = np.diff(data["volume_edges_m3"])
    ions = {i["label"]: (np.asarray(i["density_m3"]), i["Z"], i["A"]) for i in data["ions"]}
    beam = data["beam"]
    estimate = fast_ion_slowing_down_estimate(
        np.asarray(data["n_e_m3"]), np.asarray(data["T_e_eV"]), ions,
        np.asarray(data["birth_rate_per_zone_s"]) / dv, beam["energy_eV"], A_b=beam["A_b"], Z_b=beam["Z_b"])
    return data, dv, estimate


def _losses(data):
    balance = data["nubeam"]["power_balance"]["H beam ion"]
    lost = -(balance["bad orbit loss"] + balance["internal cx loss"] + balance["external cx loss"])
    return lost / data["nubeam"]["deposited_W"]


def test_the_reference_is_a_steady_state(case):
    data, _, estimate = case
    balance = data["nubeam"]["power_balance"]["H beam ion"]
    heating = -(balance["electron heating"] + balance["ion heating"] + balance["thermalization"])
    assert abs(balance["d/dt(f.i. energy)"]) < 0.01 * heating               # stored energy no longer growing
    # particle balance closes: what is born either thermalises or is lost (0.867 vs 0.868)
    nubeam = data["nubeam"]
    assert nubeam["thermalisation_rate_s"] / nubeam["birth_rate_s"] == pytest.approx(1 - _losses(data), abs=0.01)
    window = data["provenance"]["steps"] * data["provenance"]["step_s"]
    assert np.nanmax(estimate.tau_thermalisation) < window                  # the run outlasts slowing down
    # the births carry what was injected less the shine-through
    assert nubeam["deposited_W"] == pytest.approx(
        balance["injected power (W)"] + balance["shine-through"], rel=0.005)


def test_the_lossless_closure_bounds_nubeam_from_above(case):
    data, dv, estimate = case
    n_ratio = np.nansum(estimate.n_fast * dv) / np.sum(np.asarray(data["nubeam"]["n_fast_m3"]) * dv)
    w_ratio = np.nansum(estimate.W_fast * dv) / np.sum(np.asarray(data["nubeam"]["W_fast_J_m3"]) * dv)
    # 1.32 and 1.26 on this run: the closure keeps the 13 % NUBEAM loses to bad orbits and CX
    assert 1.0 < w_ratio < n_ratio < 1.6
    loss = _losses(data)
    assert 0.10 < loss < 0.20
    # with NUBEAM's own loss fraction taken out the totals are within 15 % (1.14, 1.10);
    # the rest is orbit width (local deposition here) and Monte Carlo noise
    assert n_ratio * (1 - loss) == pytest.approx(1.0, abs=0.15)
    assert w_ratio * (1 - loss) == pytest.approx(1.0, abs=0.15)


def test_the_beam_slows_mostly_on_electrons(case):
    data, _, estimate = case
    # 10 keV H into a 10-20 eV plasma: E_c is a few percent of E_b, so the drag is electron drag
    assert np.nanmax(estimate.E_c_eV) < 0.05 * data["beam"]["energy_eV"]
    heating = data["nubeam"]
    assert heating["electron_heating_W"] > 10 * heating["ion_heating_W"]


def test_the_builder_reads_the_last_step_by_number():
    spec = importlib.util.spec_from_file_location(
        "nubeam_fast_ion_reference", ROOT / "workflow" / "kinetic_closure" / "nubeam_fast_ion_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = [Path(f"R_birth_cpu0_1.cdf_{k}") for k in (1, 10, 2)]
    assert [p.name for p in sorted(names, key=module._step_of)][-1].endswith("_10")
