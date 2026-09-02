"""Config-level shot-boundary coverage for issue #195's vest.yaml revisions.

These tests only exercise YAML resolution (`resolve_vest_diagnostic` /
`resolve_shot_revisions`). The behavioral coverage of the processing paths
themselves lives in test_plasma_current_processing.py,
test_pf_active_processing.py and test_equilibrium_magnetics_processing.py.
"""

import pytest

from vaft.machine_mapping.utils import resolve_shot_revisions, resolve_vest_diagnostic


def _resolve_nested(diagnostic: str, path: tuple[str, ...], shot: int) -> dict:
    config = resolve_vest_diagnostic(shot, diagnostic)
    node = config["processing"]
    for key in path[:-1]:
        node = node[key]
    item = node[path[-1]]
    return resolve_shot_revisions(
        {key: value for key, value in item.items() if key != "revisions"},
        item.get("revisions"),
        shot,
        context=f"{diagnostic} {'.'.join(path)}",
    )


@pytest.mark.parametrize(
    ("shot", "expected_start", "expected_end"),
    [
        (43760, 7250, 8750),
        (43761, 6000, 7251),
    ],
)
def test_plasma_current_baseline_boundary_43761(shot, expected_start, expected_end):
    resolved = _resolve_nested("plasma_current", ("reference",), shot)  # sanity: reference resolves too
    assert resolved is not None
    baseline = _resolve_nested("plasma_current", ("baseline",), shot)
    assert baseline["analysis_start"] == expected_start
    assert baseline["analysis_end"] == expected_end


@pytest.mark.parametrize(
    ("shot", "expected_mode"),
    [
        (46402, "subtract"),
        (46403, "subtract_fl10_windowed"),
        (47116, "subtract_fl10_windowed"),
        (47117, "disabled"),
    ],
)
def test_plasma_current_fl10_mode_boundaries(shot, expected_mode):
    reference = _resolve_nested("plasma_current", ("reference",), shot)
    assert reference["mode"] == expected_mode


def test_plasma_current_fl10_windowed_config_is_complete():
    reference = _resolve_nested("plasma_current", ("reference",), 46403)
    fl10 = reference["fl10"]
    # 0.0, not the donor's 0.26: VAFT's loader already applies the DAQ
    # trigger correction that `vest_ip.m` adds by hand. Re-applying it puts
    # FL10 outside the compensation window and silently disables it.
    assert fl10["time_offset_s"] == pytest.approx(0.0)
    assert fl10["decimate_factor"] == 10
    assert fl10["gain_numerator"] == pytest.approx(11.0)
    assert fl10["smooth_span"] == 10
    assert fl10["subtract_window"] == [0.26, 0.36]
    assert fl10["reference_offset_index"] == 175


@pytest.mark.parametrize("shot", [17455, 46402, 46403, 47116, 47117, 60000])
def test_plasma_current_effective_resistance_is_late_era_value_post_17455(shot):
    # Renamed from `mutual_inductance` in issue #214: dividing a loop voltage
    # by an inductance yields A/s, not the current this term is subtracted
    # from. The value is unchanged from the donor.
    reference = _resolve_nested("plasma_current", ("reference",), shot)
    assert reference["effective_resistance_ohm"] == pytest.approx(5.0e-4)


@pytest.mark.parametrize(
    ("shot", "expected_pf1_gain"),
    [
        (45964, -5.0e4),
        (45965, -1.0e4),
    ],
)
def test_pf1_gain_boundary_45965_regression(shot, expected_pf1_gain):
    coil_gains = resolve_vest_diagnostic(shot, "pf_active")["processing"]["coil_gains"]
    assert float(coil_gains[0]) == pytest.approx(expected_pf1_gain)


@pytest.mark.parametrize(
    ("shot", "expected_pf5_gain"),
    [
        (48371, -1.0e4),
        (48372, -5.0e3),
    ],
)
def test_pf5_gain_boundary_48372(shot, expected_pf5_gain):
    coil_gains = resolve_vest_diagnostic(shot, "pf_active")["processing"]["coil_gains"]
    assert float(coil_gains[4]) == pytest.approx(expected_pf5_gain)


def test_pf_active_saturation_repair_config_targets_pf6_by_zero_based_index():
    processing = resolve_vest_diagnostic(45965, "pf_active")["processing"]
    repair = processing["saturation_repair"]
    assert set(repair.keys()) == {5}
    assert repair[5]["value"] == pytest.approx(-5000.0)
    assert repair[5]["tolerance"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("shot", "expected_probe_baseline", "expected_flux_window"),
    [
        (43684, 5000, None),
        (43685, 1750, [0.24, 0.26]),
    ],
)
def test_equilibrium_magnetics_baseline_boundary_43685(
    shot, expected_probe_baseline, expected_flux_window
):
    """The 0.26--0.36 s output window is encoded by index_start/index_end
    (6500/9000 on the 4e-5 s grid); what changes at 43685 is baseline policy."""
    window = _resolve_nested("equilibrium_magnetics", ("window",), shot)
    assert window["index_start"] == 6500
    assert window["index_end"] == 9000
    assert window["probe_baseline_end"] == expected_probe_baseline
    assert window["flux_baseline_window"] == expected_flux_window


@pytest.mark.parametrize(
    ("shot", "expected_daq_mode"),
    [
        (46402, "legacy"),
        (46403, "legacy"),
        (46404, "native_daq"),
    ],
)
def test_equilibrium_magnetics_daq_mode_boundary_46404(shot, expected_daq_mode):
    window = _resolve_nested("equilibrium_magnetics", ("window",), shot)
    assert window["daq_mode"] == expected_daq_mode


def test_magnetics_and_plasma_current_boundaries_differ_by_one_shot():
    """The off-by-one is real and deliberate, confirmed against the donor:
    `vest_ip.m` switches the FL10 path at `shot >= 46403`, while the
    equilibrium-magnetics dispatch in Batch_FiniteElementFitting_v11_Header.m
    and VEST_PrepareFilamentaryFitInput.m sends `shot <= 46403` to the legacy
    function. Shot 46403 therefore takes the NEW plasma-current path and the
    OLD magnetics path."""
    reference = _resolve_nested("plasma_current", ("reference",), 46403)
    window = _resolve_nested("equilibrium_magnetics", ("window",), 46403)
    assert reference["mode"] == "subtract_fl10_windowed"
    assert window["daq_mode"] == "legacy"

    assert _resolve_nested("equilibrium_magnetics", ("window",), 46404)["daq_mode"] == "native_daq"


@pytest.mark.parametrize(
    "shot",
    [
        0, 17454, 17455, 19286, 19287, 20258, 20259, 38109, 38110, 38360, 38361,
        38400, 38401, 41445, 41446, 41451, 41452, 41659, 41660, 42850, 42851,
        43684, 43685, 43760, 43761, 45964, 45965, 46402, 46403, 46404, 47116, 47117,
        48371, 48372, 60000,
    ],
)
def test_no_revision_overlap_across_declared_boundaries(shot):
    """Cheap insurance: resolving every diagnostic at every declared boundary
    shot (across plasma_current, pf_active, equilibrium_magnetics) must not
    raise VestConfigurationError for overlapping revisions."""
    for diagnostic in ("plasma_current", "pf_active", "equilibrium_magnetics"):
        resolve_vest_diagnostic(shot, diagnostic)
    for path, diagnostic in (
        (("reference",), "plasma_current"),
        (("baseline",), "plasma_current"),
        (("sign",), "plasma_current"),
        (("window",), "equilibrium_magnetics"),
    ):
        _resolve_nested(diagnostic, path, shot)
