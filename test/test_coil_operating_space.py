"""The #1178 coil operating-space result: schema, refusals and the long form (#1886)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coil_operating_space_fixtures import GROUPS, synthetic_operating_space
from vaft.process.coil_operating_space import (
    COIL_EXCITATION_CONVENTION,
    OPERATING_SPACE_ROLES,
    OPERATING_SPACE_STATUSES,
    CoilOperatingSpace,
    relative_coil_coordinates,
)


@pytest.mark.parametrize("kind", ["regular", "irregular"])
def test_the_long_form_is_one_row_per_sample_and_metric(kind):
    result = synthetic_operating_space(kind=kind, points=6)
    frame = result.to_frame()
    assert len(frame) == len(result.values) == 2 * len(result.samples)
    for column in ("sample_id", "metric", "value", "unit", "status", "role",
                   "amplitude_upper", "phase_lower", "amplitude_ratio_middle", "phase_rel_lower"):
        assert column in frame.columns, column
    assert "amplitude_ratio_upper" not in frame.columns  # group 0 is the reference
    assert result.metrics() == ("phi_res_rms_edge", "ntv_torque")


def test_relative_phase_is_wrapped_and_undefined_at_zero_amplitude():
    rel = relative_coil_coordinates([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]], [[0.5, 0.2, 1.0], [0.0, 1.0, 2.0]])
    assert rel["phase_rel"][0, 1] == pytest.approx(2 * np.pi - 0.3)
    assert rel["amplitude_ratio"][0, 1] == 2.0
    assert np.isnan(rel["phase_rel"][0, 2])  # A_g = 0
    assert np.isnan(rel["amplitude_ratio"][1]).all() and np.isnan(rel["phase_rel"][1]).all()  # A_0 = 0


def test_failed_and_uncomputed_values_are_nan_never_zero():
    frame = synthetic_operating_space(points=4).to_frame()
    torque = frame[frame["metric"] == "ntv_torque"]
    assert set(torque["status"]) == {"valid", "failed", "not_computed"}
    assert torque.loc[torque["status"] != "valid", "value"].isna().all()
    assert np.isfinite(torque.loc[torque["status"] == "valid", "value"]).all()


def test_the_phasors_follow_the_declared_convention():
    result = synthetic_operating_space(points=4)
    c = result.phasors()
    row = result.samples.iloc[5]
    assert c[5, 1] == pytest.approx(row["amplitude_middle"] * np.exp(1j * row["phase_middle"]))
    assert result.convention["phasor"] == "c = A exp(+i delta)"


def _parts():
    result = synthetic_operating_space(points=4)
    return dict(n=1, groups=GROUPS, samples=result.samples.copy(), values=result.values.copy(),
                convention=dict(result.convention), grid=dict(result.grid))


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["values"].loc.__setitem__((0, "value"), 0.0) or p["values"].loc.__setitem__((0, "status"), "failed"),
     "only a valid row"),
    (lambda p: p["values"].loc.__setitem__((0, "status"), "ok"), "unknown statuses"),
    (lambda p: p["samples"].loc.__setitem__((0, "role"), "best"), "unknown roles"),
    (lambda p: p["samples"].loc.__setitem__((0, "amplitude_upper"), -1.0), "not negative"),
    (lambda p: p["convention"].pop("frame"), "convention lacks"),
    (lambda p: p.__setitem__("grid", {"kind": "mesh"}), "grid kind"),
    (lambda p: p.__setitem__("values", pd.concat([p["values"], p["values"].iloc[:1]])), "one value per"),
    (lambda p: p["values"].loc.__setitem__((0, "sample_id"), "nowhere"), "unknown samples"),
    (lambda p: p.__setitem__("values", pd.concat([p["values"].assign(m=np.nan), p["values"].assign(m=np.nan).iloc[:1]])),
     "one value per"),
    (lambda p: p["values"].__setitem__("role", "scan"), "both carry"),
    (lambda p: p["values"].loc.__setitem__((0, "unit"), None), "names its unit"),
    (lambda p: p["values"].loc.__setitem__((0, "unit"), " "), "names its unit"),
    (lambda p: p.__setitem__("values", p["values"].astype({"value": object}).assign(value="1.5")), "numeric"),
    (lambda p: p.__setitem__("n", 0), "positive integer"),
    (lambda p: p.__setitem__("n", 1.5), "positive integer"),
    (lambda p: p.__setitem__("grid", {"kind": "regular"}), "names its axes"),
    (lambda p: p.__setitem__("values", p["values"].iloc[:0]), "at least one"),
    (lambda p: p["values"].loc.__setitem__((0, "metric"), None), "names its metric"),
    (lambda p: p["samples"].loc.__setitem__((0, "role"), None) or p["samples"].loc.__setitem__((1, "role"), "best"),
     "unknown roles"),
])
def test_a_result_that_breaks_the_contract_is_refused(mutate, message):
    parts = _parts()
    mutate(parts)
    with pytest.raises(ValueError, match=message):
        CoilOperatingSpace(**parts)


def test_vocabularies_are_the_ones_agreed_on_1886():
    assert OPERATING_SPACE_STATUSES == ("valid", "infeasible", "failed", "not_computed", "undefined")
    assert {"scan", "reference", "analytic_optimum", "numerical_optimum", "gpec_confirmation"} <= set(OPERATING_SPACE_ROLES)
    assert {"probe", "held_out", "invariance"} <= set(OPERATING_SPACE_ROLES)
    assert COIL_EXCITATION_CONVENTION["fourier_coefficient"].startswith("C_n = c / 2 (coefficient of exp(+i n phi)")


def test_the_result_owns_copies_of_its_tables():
    parts = _parts()
    result = CoilOperatingSpace(**parts)
    parts["values"].loc[0, "value"] = 123.0
    assert result.values.loc[0, "value"] != 123.0


def test_the_phase_convention_is_the_one_coil_excitation_draws():
    """delta is recovered from from_mode currents: peak at n phi = -delta, C_n = c/2 on exp(+i n phi)."""
    from vaft.machine_mapping.coils_non_axisymmetric_geometry import CoilExcitation

    n, amplitude, delta = 1, 2.0, 0.7
    angles = np.arange(12) * 30.0
    currents = np.asarray(CoilExcitation.from_mode("c", amplitude, n, np.degrees(delta), angles).currents_a)
    phi = np.radians(angles)
    c_n = np.mean(currents * np.exp(-1j * n * phi))  # coefficient of exp(+i n phi)
    assert c_n == pytest.approx(amplitude * np.exp(1j * delta) / 2)
    # GPEC's exp(-i n phi) coefficient is the conjugate.
    assert np.mean(currents * np.exp(1j * n * phi)) == pytest.approx(np.conj(amplitude * np.exp(1j * delta)) / 2)
    # The peak moves toward -phi as delta grows.
    fine = np.linspace(-np.pi, np.pi, 3601)
    peak = fine[np.argmax(np.cos(n * fine + delta))]
    assert peak == pytest.approx(-delta / n, abs=1e-3)
    assert "toward -phi" in COIL_EXCITATION_CONVENTION["phase_direction"]

