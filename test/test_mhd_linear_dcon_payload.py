"""DCON's stability payload in mhd_linear.code.parameters (#940), on real GPEC output.

The fixtures are the #792 reference: real DCON (GPEC e68d7ac2) for shot 39915 at
319 ms, n=1, with ``bal_flag=t``, full edge and peak-dW truncated.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.code.gpec import read_dcon_output
from vaft.machine_mapping.mhd_linear import DCON_FRAGMENT_VERSION, extract_dcon_stability, mhd_linear

REFERENCE = Path(__file__).resolve().parent / "data" / "gpec" / "dcon_edge_792"


def _mapped(tmp_path: Path, treatment: str) -> tuple[ODS, Path]:
    source = tmp_path / treatment
    shutil.copytree(REFERENCE / treatment, source)
    ods = ODS(consistency_check=False)
    mhd_linear(ods, str(source), {"module": "dcon", "modes": [1]})
    return ods, source


def test_full_edge_payload_round_trips_the_reader(tmp_path):
    ods, source = _mapped(tmp_path, "full_edge")
    native = read_dcon_output(source, mode=1)
    [row] = extract_dcon_stability(ods)
    assert (row["n_tor"], row["time_slice"], row["position"]) == (1, 0, 0)
    assert row["W_t"] == pytest.approx(native.total1)
    assert row["W_p"] == pytest.approx(native.plasma1) and row["W_v"] == pytest.approx(native.vacuum1)
    np.testing.assert_allclose(row["W_t_spectrum"], native.W_t_eigenvalue)
    assert row["edge_treatment"] == "full_edge" and row["edge_scan"] is None
    assert row["requested_psiedge"] == 1.0 and row["psilim"] == pytest.approx(native.psilim)
    np.testing.assert_array_equal(row["psi_n"], native.psi_n)
    np.testing.assert_array_equal(row["D_I"], native.di)


def test_truncated_payload_carries_the_edge_scan(tmp_path):
    ods, source = _mapped(tmp_path, "peak_dw_truncated")
    native = read_dcon_output(source, mode=1)
    [row] = extract_dcon_stability(ods)
    assert row["edge_treatment"] == "peak_dw_truncated" and row["requested_psiedge"] == 0.95
    np.testing.assert_allclose(row["edge_scan"]["dW"], native.edge_scan.dW)
    # DCON truncates at the peak of Re dW_edge; the payload keeps that visible.
    peak = int(np.argmax(row["edge_scan"]["dW"].real))
    assert row["psilim"] == pytest.approx(row["edge_scan"]["psi_n"][peak])


def test_local_criteria_summaries_respect_the_evaluation_flags(tmp_path):
    ods, source = _mapped(tmp_path, "full_edge")
    native = read_dcon_output(source, mode=1)
    [row] = extract_dcon_stability(ods)
    assert row["mercier_evaluated"] is True and row["ballooning_evaluated"] is True
    assert row["max_D_I"] == pytest.approx(float(np.nanmax(native.di)))
    assert row["min_C_A"] == pytest.approx(float(np.nanmin(native.ca1[native.ca1_evaluated])))
    assert 0.0 <= row["psi_n_at_max_D_I"] <= 1.0


def test_values_survive_a_string_round_trip_of_code_parameters(tmp_path):
    # Multi-fragment code.parameters comes back from a product reload as the string.
    ods, _ = _mapped(tmp_path, "full_edge")
    text = str(ods["mhd_linear.code.parameters"])
    reloaded = ODS(consistency_check=False)
    reloaded["mhd_linear.code.parameters"] = text
    [a] = extract_dcon_stability(ods)
    [b] = extract_dcon_stability(reloaded)
    assert a["W_t"] == b["W_t"] and np.array_equal(a["C_A"], b["C_A"], equal_nan=True)


def test_reading_does_not_create_mhd_linear_and_v1_fragments_are_skipped():
    ods = ODS(consistency_check=False)
    assert extract_dcon_stability(ods) == []
    assert "mhd_linear" not in ods.keys()
    ods["mhd_linear.code.parameters"] = '<parameters><solver name="dcon" n_tor="1"><mlow>-8</mlow></solver></parameters>'
    assert extract_dcon_stability(ods) == []


def test_malformed_fragments_warn_instead_of_raising():
    ods = ODS(consistency_check=False)
    ods["mhd_linear.code.parameters"] = f'<parameters><solver name="dcon" version="{DCON_FRAGMENT_VERSION}"/></parameters>'
    with pytest.warns(RuntimeWarning):
        assert extract_dcon_stability(ods) == []
