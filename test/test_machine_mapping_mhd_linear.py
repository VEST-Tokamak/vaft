"""Unit tests for parsing GPEC-suite `.nc`/`solutions.bin` output into
`mhd_linear`/`ntms`.

No real DCON/RDCON/STRIDE binary is required: these use synthetic fixtures
shaped like the real GPEC netCDF conventions (global attrs `mlow`/`mhigh`/
`mpert`/`mband`/`n`; an `i` dim of size 2 for real/imag) so the parsing logic
is exercised hermetically but against the actual source-verified layout.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import xarray as xr
from omas import ODS

from vaft.machine_mapping.mhd_linear import mhd_linear


def _write_dcon_output(path, *, n: int, w_t: float, mlow: int = -8, mhigh: int = 16) -> None:
    mpert = mhigh - mlow + 1
    ds = xr.Dataset(
        {"W_t_eigenvalue": (("mode", "i"), [[w_t, 0.0]])},
        coords={"i": [0, 1], "mode": [1]},
        attrs={"mlow": mlow, "mhigh": mhigh, "mpert": mpert, "mband": 0, "n": n},
    )
    ds.to_netcdf(path / f"dcon_output_n{n}.nc")


def _write_resistive_output(
    module: str,
    path,
    *,
    n: int,
    m_values: list[int],
    delta_prime_diag: list[complex],
    mlow: int = -8,
    mhigh: int = 16,
) -> None:
    msing = len(m_values)
    mpert = mhigh - mlow + 1
    delta_prime = np.zeros((msing, msing, 2), dtype=float)
    for i, value in enumerate(delta_prime_diag):
        delta_prime[i, i, 0] = value.real
        delta_prime[i, i, 1] = value.imag
    ds = xr.Dataset(
        {
            "Delta_prime": (("r", "r_prime", "i"), delta_prime),
            "r": (("r",), m_values),
            "psi_n_rational": (("r",), [0.1 * (i + 1) for i in range(msing)]),
            "q_rational": (("r",), [float(m) / n for m in m_values]),
        },
        coords={"i": [0, 1]},
        attrs={"mlow": mlow, "mhigh": mhigh, "mpert": mpert, "mband": 0, "n": n},
    )
    ds.to_netcdf(path / f"{module}_output_n{n}.nc")


def test_dcon_module_writes_energy_perturbed_and_n_tor(tmp_path):
    _write_dcon_output(tmp_path, n=1, w_t=-0.42)
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    # `toroidal_mode` is an AOS: array position is insertion order, not the
    # physical mode number, which is only ever recovered via `n_tor`.
    assert ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["n_tor"] == 1
    assert ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["energy_perturbed"] == pytest.approx(-0.42)
    assert extras == {1: {"module": "dcon", "variable": "W_t_eigenvalue", "value": pytest.approx(-0.42)}}


def test_dcon_records_mode_range_provenance_in_code_parameters(tmp_path):
    _write_dcon_output(tmp_path, n=1, w_t=-0.1, mlow=-3, mhigh=5)
    ods = ODS()

    mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    params = ods["mhd_linear"]["code"]["parameters"]
    assert '<solver name="dcon" n_tor="1">' in params
    assert "<mlow>-3</mlow>" in params
    assert "<mhigh>5</mhigh>" in params
    assert "normalized" in params  # energy_perturbed units caveat


def test_rdcon_module_has_no_mhd_linear_slot_for_delta_prime_but_populates_ntms(tmp_path):
    _write_resistive_output("rdcon", tmp_path, n=2, m_values=[3, 4], delta_prime_diag=[1.23 + 0.1j, -0.5 + 0j])
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path), {"module": "rdcon", "time_slice": 0})

    assert ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["n_tor"] == 2
    assert ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["ballooning_type"]["name"] == "Tearing"
    assert "energy_perturbed" not in ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]

    # The diagonal Delta-prime per rational surface reaches `ntms`, which has
    # a genuine field (`deltaw`) for it; `mhd_linear` has none.
    ntms_modes = ods["ntms"]["time_slice"][0]["mode"]
    assert len(ntms_modes) == 2
    assert {ntms_modes[i]["m_pol"] for i in range(2)} == {3, 4}
    for i in range(2):
        assert ntms_modes[i]["n_tor"] == 2
        assert ntms_modes[i]["deltaw"][0]["name"] == "classical"

    # The full complex per-surface breakdown (real part only lands in ntms)
    # survives in the manifest-facing `extras`/`value`.
    values = extras[2]["value"]
    assert len(values) == 2
    assert {v["m"] for v in values} == {3, 4}
    assert any(v["delta_prime_imag"] == pytest.approx(0.1) for v in values)


def test_stride_module_extracts_delta_prime_the_same_way_as_rdcon(tmp_path):
    _write_resistive_output("stride", tmp_path, n=1, m_values=[2], delta_prime_diag=[-0.05 + 0j])
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path), {"module": "stride", "time_slice": 0})

    assert extras[1]["value"][0]["delta_prime_real"] == pytest.approx(-0.05)
    assert ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["ballooning_type"]["name"] == "Tearing"
    assert ods["ntms"]["time_slice"][0]["mode"][0]["m_pol"] == 2


def test_missing_variable_is_skipped_not_fatal(tmp_path):
    # Present file, no global attrs / expected variable -- must not raise.
    ds = xr.Dataset({"some_other_variable": (("i",), [1.0])})
    ds.to_netcdf(tmp_path / "dcon_output_n1.nc")
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    assert extras == {}
    assert len(ods["mhd_linear"]) == 0


def test_no_matching_files_is_skipped_not_fatal(tmp_path):
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path), {"module": "rdcon", "time_slice": 0})

    assert extras == {}


def test_unsupported_module_raises():
    ods = ODS()
    with pytest.raises(ValueError, match="gpec"):
        mhd_linear(ods, "/nonexistent", {"module": "gpec"})


def test_a_mode_that_fails_to_parse_does_not_leave_a_gap_for_the_next_one(tmp_path):
    """A matched filename that fails to parse (e.g. missing required netCDF
    attributes) must not consume an AOS position: the next successfully
    parsed mode has to land at the next *actually written* index, not at
    `existing + its position in the raw filename list`, or indexing skips
    straight past the end of the AOS and raises."""
    # n=1: matches the filename pattern but has none of the required global
    # attributes (mlow/mhigh/mpert/mband/n) -- read_dcon_output raises.
    xr.Dataset({"W_t_eigenvalue": (("mode", "i"), [[0.1, 0.0]])}, coords={"i": [0, 1], "mode": [1]}).to_netcdf(
        tmp_path / "dcon_output_n1.nc"
    )
    _write_dcon_output(tmp_path, n=2, w_t=-0.5)
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    assert extras == {2: {"module": "dcon", "variable": "W_t_eigenvalue", "value": pytest.approx(-0.5)}}
    modes = ods["mhd_linear"]["time_slice"][0]["toroidal_mode"]
    assert len(modes) == 1
    assert modes[0]["n_tor"] == 2


def _write_solutions_bin(path, *, blocks: list[list[list[float]]]) -> None:
    """``blocks[i]`` is a list of 7-float steps for poloidal-harmonic block ``i``."""
    with open(path / "solutions.bin", "wb") as f:
        for steps in blocks:
            for vec7 in steps:
                payload = struct.pack("<7f", *vec7)
                f.write(struct.pack("<i", len(payload)))
                f.write(payload)
                f.write(struct.pack("<i", len(payload)))
            f.write(struct.pack("<i", 0))  # zero-length record ends this block


def test_dcon_solutions_bin_recovers_true_m_labels_and_stays_out_of_mhd_linear_ids(tmp_path):
    """`solutions.bin`'s poloidal-harmonic blocks are labeled via the run's
    real `mlow` (read from the netCDF global attribute, never hardcoded), and
    the recovered Fourier-space eigenfunction never reaches
    `displacement_perpendicular` -- it is a Fourier-harmonic array, not the
    real-space grid that field requires (see the #170 investigation)."""
    _write_dcon_output(tmp_path, n=1, w_t=-0.42, mlow=-3, mhigh=-1)  # mpert=3, m in {-3,-2,-1}
    _write_solutions_bin(
        tmp_path,
        blocks=[
            [[0.1, 0.3, 1.0, 1.5, -2.5, 0.0, 0.0]],  # ipert=0 -> m = mlow+0 = -3
            [[0.1, 0.3, 1.0, 2.5, -3.5, 0.0, 0.0]],  # ipert=1 -> m = -2
            [[0.1, 0.3, 1.0, 3.5, -4.5, 0.0, 0.0]],  # ipert=2 -> m = -1
        ],
    )
    ods = ODS()

    mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    modes = ods["mhd_linear"]["time_slice"][0]["toroidal_mode"]
    # Exactly one toroidal_mode entry for this run (n_tor=1), not one per
    # poloidal harmonic -- solutions.bin describes multiple harmonics of the
    # *same* toroidal mode, never separate toroidal modes.
    assert len(modes) == 1
    assert modes[0]["n_tor"] == 1
    assert modes[0]["energy_perturbed"] == pytest.approx(-0.42)
    assert "displacement_perpendicular" not in modes[0].get("plasma", {})

    native_path = tmp_path / "dcon_native_n1.json"
    assert native_path.exists()
    from vaft.code.gpec import DconOutput

    result = DconOutput.read_json(native_path)
    assert result.eigenfunction is not None
    assert result.eigenfunction.m.tolist() == [-3, -2, -1]
    assert result.eigenfunction.xi_psi_real[1, 0] == pytest.approx(2.5)


def test_defaults_to_dcon_module_when_unspecified(tmp_path):
    _write_dcon_output(tmp_path, n=1, w_t=0.1)
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path))

    assert extras[1]["module"] == "dcon"
