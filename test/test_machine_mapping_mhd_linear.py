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

from vaft.code.gpec import DconOutput
from vaft.machine_mapping.mhd_linear import MAX_RADIAL_POINTS, mhd_linear


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


def test_dcon_solutions_bin_recovers_true_m_labels_and_reaches_the_ids(tmp_path):
    """`solutions.bin`'s harmonic blocks are labeled by the run's real `mlow`
    (read from the netCDF global attribute, never hardcoded) and reach the IDS
    on a declared `(psi, m)` grid.

    `displacement_perpendicular` is a closest fit, not an exact one -- the
    values are `xi.grad(psi)`, a contravariant flux component under an
    arbitrary eigenvector normalization, and the field documents metres -- so
    what is asserted here is that the mismatch is *recorded*, in
    `code.parameters`, rather than left for a reader to infer from the field
    name."""
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

    # One toroidal mode, three poloidal harmonics, one radial sample.
    plasma = modes[0]["plasma"]
    assert plasma["grid"]["dim2"].tolist() == [-3.0, -2.0, -1.0]
    assert plasma["grid"]["dim1"].tolist() == pytest.approx([0.1])
    assert plasma["displacement_perpendicular"]["real"].shape == (1, 3)
    assert plasma["displacement_perpendicular"]["real"][0].tolist() == pytest.approx([1.5, 2.5, 3.5])
    # A private (negative) grid_type index, because the IMAS identifier's
    # Fourier grids name angles DCON is not using here.
    assert plasma["grid_type"]["index"] < 0
    assert "hamada" in plasma["grid_type"]["name"]

    # b = i(m - n q) xi -- match/ideal.f:372, recomputed because match forms it
    # internally and never writes it out.
    expected_b = [(m - 1 * 1.0) * xi for m, xi in zip((-3.0, -2.0, -1.0), (1.5, 2.5, 3.5))]
    assert plasma["b_field_perturbed"]["coordinate1"]["imaginary"][0].tolist() == pytest.approx(
        expected_b
    )

    # The dominant harmonic is the largest |xi|, by true m rather than block index.
    assert modes[0]["m_pol_dominant"] == pytest.approx(-1.0)

    # Neither array is in its field's documented units, and both say so where a
    # consumer can test it instead of parsing prose.
    parameters = ods["mhd_linear"]["code"]["parameters"]
    assert 'normalization="dcon_eigenvector_arbitrary"' in parameters
    assert 'imas_documented_units="m"' in parameters
    assert 'imas_documented_units="T"' in parameters
    assert 'definition="i*(m - n*q)*xi.grad(psi)"' in parameters

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


# --- regressions found in review of the #170 implementation --------------------

def test_an_unreadable_output_warns_rather_than_silently_looking_unrun(tmp_path):
    """The readers are stricter than the old best-effort scalar extraction, so a
    file they reject must be visible: without a warning it is indistinguishable
    downstream from a cell that was never run."""
    xr.Dataset({"some_other_variable": (("i",), [1.0])}).to_netcdf(tmp_path / "dcon_output_n1.nc")
    ods = ODS()

    with pytest.warns(RuntimeWarning, match="skipping unreadable output"):
        extras = mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    assert extras == {}


def test_a_solver_succeeding_only_at_a_later_time_slice_is_not_an_error(tmp_path):
    """An OMAS AOS only auto-vivifies at its current length, so writing time
    slice 2 while 0 and 1 produced nothing used to raise IndexError -- which the
    caller then misreported as a *failed* solver run."""
    _write_dcon_output(tmp_path, n=1, w_t=-0.42)
    ods = ODS()

    extras = mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 2})

    assert extras[1]["value"] == pytest.approx(-0.42)
    assert len(ods["mhd_linear.time_slice"]) == 3
    assert ods["mhd_linear"]["time_slice"][2]["toroidal_mode"][0]["n_tor"] == 1
    # Slices this solver did not produce are flagged negative ("shall not be
    # used"), not 0, which would falsely claim a successful run.
    assert list(ods["mhd_linear.code.output_flag"]) == [-1, -1, 0]


def test_the_energy_perturbed_units_caveat_is_machine_readable(tmp_path):
    """A consumer must be able to detect programmatically that the stored value
    is not in the Joules `energy_perturbed`'s IMAS documentation promises."""
    import xml.etree.ElementTree as ET

    _write_dcon_output(tmp_path, n=1, w_t=-0.42)
    ods = ODS()
    mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    root = ET.fromstring(ods["mhd_linear.code.parameters"])
    element = root.find(".//energy_perturbed")
    assert element is not None
    assert element.get("units") == "1"
    assert element.get("imas_documented_units") == "J"
    assert element.get("units") != element.get("imas_documented_units")


def test_the_eigenfunction_is_strided_into_the_ids_and_says_by_how_much(tmp_path):
    """The IDS gets a radially strided view; the sidecar keeps full resolution.

    DCON integrates on thousands of steps, and at a realistic mpert the
    full-resolution arrays would make this one stage product larger than an
    entire packaged sample shot -- so the IDS carries every harmonic but fewer
    radial samples, with the stride recorded rather than silently applied.
    Striding, not interpolating: every number in the IDS is one DCON computed.
    """
    n_psi = MAX_RADIAL_POINTS * 3
    psi = np.linspace(0.01, 0.99, n_psi)
    _write_dcon_output(tmp_path, n=1, w_t=-0.42, mlow=-1, mhigh=0)
    _write_solutions_bin(
        tmp_path,
        blocks=[
            [[p, 0.3, 1.0 + p, float(block) + p, 0.0, 0.0, 0.0] for p in psi]
            for block in range(2)
        ],
    )
    ods = ODS()

    mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    plasma = ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["plasma"]
    assert plasma["grid"]["dim1"].size <= MAX_RADIAL_POINTS
    assert plasma["displacement_perpendicular"]["real"].shape == (plasma["grid"]["dim1"].size, 2)
    # Strided values are exact samples of the original grid, not interpolants.
    assert plasma["grid"]["dim1"][0] == pytest.approx(psi[0], rel=1e-6)
    assert 'radial_stride="3"' in ods["mhd_linear"]["code"]["parameters"]

    full_resolution = DconOutput.read_json(tmp_path / "dcon_native_n1.json")
    assert full_resolution.eigenfunction.psi.shape[1] == n_psi


def test_ragged_harmonic_blocks_keep_the_eigenfunction_out_of_the_ids(tmp_path):
    """Blocks that do not share one psi grid have no honest `grid.dim1`.

    `read_solutions_bin` pads short blocks with NaN, so a ragged file would
    otherwise be written against whichever block happened to be first. The run
    is still mapped -- only the subtree that cannot be described is skipped.
    """
    _write_dcon_output(tmp_path, n=1, w_t=-0.42, mlow=-1, mhigh=0)
    _write_solutions_bin(
        tmp_path,
        blocks=[
            [[0.1, 0.3, 1.0, 1.5, 0.0, 0.0, 0.0], [0.2, 0.3, 1.0, 1.6, 0.0, 0.0, 0.0]],
            [[0.1, 0.3, 1.0, 2.5, 0.0, 0.0, 0.0]],  # one step short
        ],
    )
    ods = ODS()

    with pytest.warns(RuntimeWarning, match="do not share one psi grid"):
        mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    mode_entry = ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]
    assert mode_entry["n_tor"] == 1
    assert mode_entry["energy_perturbed"] == pytest.approx(-0.42)
    assert "displacement_perpendicular" not in mode_entry.get("plasma", {})
    assert (tmp_path / "dcon_native_n1.json").exists()


def test_a_run_without_solutions_bin_writes_no_eigenfunction_paths(tmp_path):
    """No `match` means no eigenfunction, which is a normal state, not a failure."""
    _write_dcon_output(tmp_path, n=1, w_t=-0.42, mlow=-1, mhigh=0)
    ods = ODS()

    mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})

    mode_entry = ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]
    assert mode_entry["energy_perturbed"] == pytest.approx(-0.42)
    assert "plasma" not in mode_entry
    assert "m_pol_dominant" not in mode_entry
    assert "displacement_perpendicular" not in ods["mhd_linear"]["code"]["parameters"]


def test_the_mapped_eigenfunction_survives_a_save_load_round_trip(tmp_path):
    """Written with `consistency_check` on, and still valid coming back.

    Issue #170's acceptance criterion is that `mhd_linear` carries no subtree
    written with validation disabled; a round trip is what makes that claim
    testable rather than asserted.
    """
    _write_dcon_output(tmp_path, n=1, w_t=-0.42, mlow=-2, mhigh=0)
    _write_solutions_bin(
        tmp_path,
        blocks=[
            [[0.1, 0.3, 1.0, 1.0 + block, 0.5, 0.0, 0.0], [0.5, 0.4, 2.0, 2.0 + block, 0.5, 0.0, 0.0]]
            for block in range(3)
        ],
    )
    ods = ODS()
    mhd_linear(ods, str(tmp_path), {"module": "dcon", "time_slice": 0})
    assert ods.consistency_check

    target = tmp_path / "mhd_linear.json"
    ods.save(str(target))
    restored = ODS()
    restored.load(str(target))

    original = ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["plasma"]
    round_tripped = restored["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["plasma"]
    np.testing.assert_allclose(
        round_tripped["displacement_perpendicular"]["real"],
        original["displacement_perpendicular"]["real"],
    )
    assert round_tripped["grid_type"]["index"] == original["grid_type"]["index"]
