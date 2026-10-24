"""The #1887 direct GPEC -> PENTRC reference: the quadratic NTV model, replayed without a solver.

``test/data/gpec/ntv_diiid_like_1887/scan.json`` holds the signed PENTRC
torques of 18 direct runs (9 K^2 probes, a reference with scaled, reversed and
rotated copies, and 4 held-out points) on the upstream *artificial*
TokaMaker DIII-D-like example (not DIII-D #147131), at converged and at
default PENTRC tolerances, plus 4 confirmation runs of the model's optima.
These tests pin what the study found: converged data qualify ``Q``, default
tolerances do not, and the optima the qualified ``Q`` predicts are what GPEC
-> PENTRC returned.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vaft.process.ntv_coil_response import (
    fixed_amplitude_phase_extrema,
    hermitian_torque_matrix,
    ntv_operating_space,
    quadratic_invariance_checks,
    quadratic_torque,
    qualify_torque_matrix,
    two_group_phase_extrema,
)

DATA = json.loads((Path(__file__).parent / "data" / "gpec" / "ntv_diiid_like_1887" / "scan.json").read_text())
SAMPLES = {row["sample_id"]: row for row in DATA["samples"]}
VALIDATION = ("held0", "held1", "held2", "held3", "ref", "ref_x2", "ref_rot1.2")
ATOL, RTOL = 1e-3, 0.02  # N m, and 2 %


def _c(row):
    return np.asarray(row["c_re"]) + 1j * np.asarray(row["c_im"])


def _qualify(key, method="fgar"):
    torque = {k: row[key][method] for k, row in SAMPLES.items()}
    q = hermitian_torque_matrix(torque, 3)
    invariance = quadratic_invariance_checks(
        torque["ref"], scaled=[(0.5, torque["ref_x0.5"]), (2.0, torque["ref_x2"])],
        reversed_torque=torque["ref_neg"], rotated=[(1.2, torque["ref_rot1.2"])], atol=ATOL, rtol=RTOL,
    )
    phasors = np.array([_c(SAMPLES[k]) for k in VALIDATION])
    return qualify_torque_matrix(q, phasors, [torque[k] for k in VALIDATION], atol=ATOL, rtol=RTOL,
                                 invariance_checks=invariance)


@pytest.mark.parametrize("method", ["fgar", "tgar", "pgar"])
def test_converged_pentrc_qualifies_a_positive_definite_q(method):
    result = _qualify("torque_n_m", method)
    assert result.status == "qualified", result.reasons
    assert result.eigenvalues.min() > 0  # positive definite on this background
    held_out = next(check for check in result.checks if check.name == "held_out")
    assert held_out.max_rel_error < 5e-3  # study: fgar 0.03 %, pgar 0.22 % (smaller torques)


def test_default_tolerances_reject_it_and_fake_an_indefinite_q():
    result = _qualify("torque_n_m_default_tolerance")
    assert result.status == "rejected"
    assert {"amplitude_scaling", "held_out"} <= {c.name for c in result.checks if not c.passed}
    assert result.eigenvalues.min() < 0 < result.eigenvalues.max()


def test_the_predicted_optima_are_what_gpec_pentrc_returned():
    q = _qualify("torque_n_m").Q
    three = fixed_amplitude_phase_extrema(q, (1, 1, 1), grid_points=72, refine=3)
    two = two_group_phase_extrema(q[:2, :2], (1, 1))
    predicted = {"opt3_max": three["T_max"], "opt3_min": three["T_min"], "opt2_max": two["T_max"],
                 "opt2_min": two["T_min"]}
    for row in DATA["confirmations"]:
        direct = row["torque_n_m"]["fgar"]
        assert predicted[row["sample_id"]] == pytest.approx(row["predicted_fgar_n_m"], abs=1e-6)
        # The claim: Q at the confirmed phasor is what GPEC -> PENTRC returned (worst: opt3_min,
        # 1.5e-4 N m on 0.025 N m).
        assert float(quadratic_torque(q, _c(row))) == pytest.approx(direct, abs=3e-4, rel=1e-3)
    # Relative phase alone moves the *direct* torque by two orders of magnitude at fixed currents.
    direct = {row["sample_id"]: row["torque_n_m"]["fgar"] for row in DATA["confirmations"]}
    assert direct["opt3_max"] / direct["opt3_min"] > 100


def test_the_reference_lands_in_the_1178_operating_space_result():
    qualification = _qualify("torque_n_m")
    rows = list(SAMPLES.values())
    result = ntv_operating_space(
        DATA["case"]["groups"], DATA["case"]["n"], np.array([_c(r) for r in rows]),
        [r["torque_n_m"]["fgar"] for r in rows],
        sample_ids=[r["sample_id"] for r in rows], roles=[r["role"] for r in rows],
        amplitude_unit=DATA["case"]["phasor_unit_a_per_turn"], frame=DATA["case"]["sector_angles"],
        provenance={"method": "fgar", **DATA["pentrc"]["tolerances"]["converged"]},
        qualification=qualification,
    )
    frame = result.to_frame()
    assert set(frame["metric"]) == {"ntv_torque", "ntv_torque_model"}
    assert (frame["status"] == "valid").all()
    assert frame.loc[frame["sample_id"] == "e0", "amplitude_c"].iloc[0] == pytest.approx(1000.0)
    assert result.provenance["ntv_torque_model"]["qualification"] == "qualified"
    with pytest.raises(ValueError, match="rejected"):
        ntv_operating_space(
            ("c", "il", "iu"), 1, np.ones((1, 3)), [1.0], sample_ids=["x"], roles=["scan"], amplitude_unit=1.0,
            frame="-", provenance={}, qualification=_qualify("torque_n_m_default_tolerance"),
        )


def test_a_failed_direct_run_is_a_failed_row_not_a_zero():
    result = ntv_operating_space(("a", "b"), 1, np.array([[1, 0], [0, 1j]]), [0.5, np.nan],
                                 sample_ids=["p", "q"], roles=["scan", "scan"], amplitude_unit=1.0,
                                 frame="synthetic", provenance={"synthetic": True})
    values = result.values.set_index("sample_id")
    assert values.loc["q", "status"] == "failed" and np.isnan(values.loc["q", "value"])
    assert result.samples.set_index("sample_id").loc["q", "phase_b"] == pytest.approx(np.pi / 2)


def _driver():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "workflow" / "gpec_ntv_coil_scan" / "scan_ntv_coil_torque.py"
    spec = importlib.util.spec_from_file_location("scan_ntv_coil_torque", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_driver_confirms_only_a_qualified_q(tmp_path):
    driver = _driver()
    source = Path(__file__).parent / "data" / "gpec" / "ntv_diiid_like_1887" / "scan.json"
    q = driver.qualified_q(source, 3, ATOL, RTOL)
    np.testing.assert_allclose(q, _qualify("torque_n_m").Q)
    stale = json.loads(source.read_text())
    for row in stale["samples"]:
        row["torque_n_m"] = row["torque_n_m_default_tolerance"]
    (tmp_path / "default.json").write_text(json.dumps(stale))
    with pytest.raises(SystemExit, match="rejected"):
        driver.qualified_q(tmp_path / "default.json", 3, ATOL, RTOL)
    names = [name for name, _, _ in driver.candidates(["confirm"], ["c", "il", "iu"], 0, 0, source)]
    assert names == ["opt3_max", "opt3_min", "opt2_max", "opt2_min"]  # the table's own names


def test_the_driver_rewrites_every_coil_current_or_refuses(tmp_path):
    driver = _driver()
    template = tmp_path / "coil.in"
    lines = ['&COIL_CONTROL', '    data_dir="../coil"', '    coil_name(1)="a"', '    coil_name(2)="b"']
    lines += [f"    coil_cur(1,{j})= 1.0D3" for j in (1, 2)] + [f"    coil_cur(2,{j})=-5e2 ! note" for j in (1, 2)]
    template.write_text("\n".join(lines + ["/"]) + "\n")
    currents = {"a": np.array([10.0, -20.0]), "b": np.array([3.5, 0.0])}
    driver.write_coil_in(template, tmp_path / "out.in", tmp_path, currents, ("a", "b"))
    text = (tmp_path / "out.in").read_text()
    assert "coil_cur(1,1)= 10.000000\n" in text and "D3" not in text  # no Fortran exponent left behind
    assert "coil_cur(2,2)= 0.000000 ! note" in text
    # A Windows coil directory is written verbatim, not read as regex escapes (\U).
    windows_dir = r"C:\Users\runner\coil"
    driver.write_coil_in(template, tmp_path / "out.in", windows_dir, currents, ("a", "b"))
    assert f'data_dir="{windows_dir}"' in (tmp_path / "out.in").read_text()
    template.write_text(template.read_text().replace("coil_cur(2,2)=", "coil_cur(2, 2)="))
    with pytest.raises(ValueError, match="scalar"):
        driver.write_coil_in(template, tmp_path / "out.in", tmp_path, currents, ("a", "b"))


def test_driver_currents_are_coil_excitation_from_mode():
    from vaft.machine_mapping.coils_non_axisymmetric_geometry import CoilExcitation

    driver = _driver()
    angles = {"a": np.radians([0.0, 60.0, 120.0, 180.0, 240.0, 300.0])}
    currents = driver.sector_currents(np.array([1.3 * np.exp(0.4j)]), angles, ("a",), 1, 1000.0)
    expected = CoilExcitation.from_mode("a", 1300.0, 1, np.degrees(0.4), np.degrees(angles["a"])).currents_a
    np.testing.assert_allclose(currents["a"], expected)

