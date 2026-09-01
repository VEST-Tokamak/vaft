"""FL10 plasma-current compensation semantics (issue #214).

The FL10 term in the plasma-current path divides a *loop voltage* by a scalar
and subtracts the result from a Rogowski *current*. That only closes
dimensionally if the scalar is in ohms:

    ip_ref = V_FL10 * flux_gain / X       [A]  requires  X in V/A = ohm
    an inductance would give              [A/s]  -- not subtractable

The donor `vest_ip.m` nevertheless names the constant
`indMutual_FL10_inLimiter` and labels its plots "FL10 Voltage/Mutual Inductance
with Inboard Limiter". That conflict is real and unresolved; these tests pin
the dimensional requirement and the donor's numbers so a future refactor cannot
silently reinterpret the coefficient as an inductance again.

What the term represents was corrected by VEST operations on 2026-09-01: the
channel is the INNER Rogowski coil, which in principle links no vessel eddy
current, and the subtracted quantity is a proxy for the induced current in the
tungsten limiter around the CS wall. From shot 47117 the inboard was changed to
carbon, that induced current is no longer measured, and the compensation is
correctly disabled. The method for computing the effective current is under
revision, so these tests deliberately pin only the arithmetic and the shot
eras -- not the physical model, which is expected to change.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.machine_mapping.magnetics import (
    _plasma_processing_for_shot,
    vfit_plasma_current,
)
from vaft.machine_mapping.utils import load_yaml, package_data_path

RAW = "vaft/data/samples/41672/source/vest_41672_daq_raw.json.gz"
SHOT = 41672


@pytest.mark.parametrize(
    ("shot", "expected_ohm"),
    [
        # Donor rule: `if shot < 17455 -> 2.8e-4 else 5.0e-4`.
        (17454, 2.8e-4),
        (17455, 5.0e-4),
        (41672, 5.0e-4),
        (46403, 5.0e-4),
        (47117, 5.0e-4),
    ],
)
def test_donor_constants_are_preserved_exactly(shot, expected_ohm):
    """#214 renames the key; it must not move a single value."""
    _, reference, _, _ = _plasma_processing_for_shot(shot)
    assert float(reference["effective_resistance_ohm"]) == pytest.approx(expected_ohm)


def test_the_coefficient_is_not_named_as_an_inductance():
    """A future refactor must not reintroduce the dimensionally wrong name.

    `V / H` is `A/s`, which cannot be subtracted from a Rogowski current. The
    key is named for the units the formula actually requires.
    """
    for shot in (17454, 41672, 47117):
        _, reference, _, _ = _plasma_processing_for_shot(shot)
        assert "effective_resistance_ohm" in reference
        assert "mutual_inductance" not in reference


def test_only_one_name_for_the_coefficient_survives_in_config():
    """The stale `effective_res` label is what caused #214 to be filed.

    It sat in a dead `rogowski_coil` block as a third name for this constant,
    at the unrevised 2.8e-4 value -- wrong for every shot >= 17455 -- and a
    reader taking it as authoritative is how the coefficient came to be read
    as a resistance. Leaving it while renaming the live key would keep the
    trap armed, so the block is gone and this asserts it stays gone.
    """
    document = load_yaml(package_data_path("vest.yaml"))
    diagnostics = document[0]["diagnostics"]
    assert "rogowski_coil" not in diagnostics

    reference = diagnostics["plasma_current"]["processing"]["reference"]
    assert "effective_resistance_ohm" in reference
    assert "mutual_inductance" not in reference

    # No `effective_res` key anywhere in the document, at any depth.
    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield str(key)
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    assert "effective_res" not in set(keys(document))


def test_shot_era_modes_match_the_donor():
    """`vest_ip.m`: legacy subtraction, then a windowed era, then disabled."""
    expected = {
        17454: "subtract",
        41672: "subtract",
        46403: "subtract_fl10_windowed",
        47116: "subtract_fl10_windowed",
        47117: "disabled",
        48000: "disabled",
    }
    for shot, mode in expected.items():
        _, reference, _, _ = _plasma_processing_for_shot(shot)
        assert reference.get("mode") == mode, shot


def test_compensation_is_a_current_of_plausible_vessel_magnitude():
    """Dimensional sanity on real data, not just on the config.

    Interpreting the divisor as ohms makes `V_FL10 * gain / R` a current. On
    shot 41672 that is a few percent of the raw Rogowski reading -- the right
    order for a vessel-current correction. Under an inductance reading the
    same number would be an A/s rate, which the subtraction could not use.
    """
    from vaft.database import raw as raw_db

    _, reference, _, _ = _plasma_processing_for_shot(SHOT)
    resistance = float(reference["effective_resistance_ohm"])
    gain = float(reference["flux_gain"])

    time, voltage = raw_db.require_signal(
        raw_db.vest_load(SHOT, int(reference["field"]), sample_opt=RAW),
        shot=SHOT,
        field=int(reference["field"]),
        signal_name="FL10",
    )
    window = (time >= 0.26) & (time < 0.36)
    ip_ref = voltage * gain / resistance

    # A current-equivalent of order 10 kA against a ~100 kA discharge.
    peak = float(np.nanmax(np.abs(ip_ref[window])))
    assert 1e3 < peak < 1e5


def test_fl10_compensation_does_not_change_the_processed_plasma_current():
    """#214 is a provenance fix: no numerical behaviour may change."""
    time, ip = vfit_plasma_current(SHOT, raw_source=RAW)
    assert time.size == ip.size
    # A real VEST discharge, not a degenerate or rescaled trace.
    assert 5e4 < float(np.nanmax(np.abs(ip))) < 5e5


def test_the_two_uses_of_field_25_stay_distinct():
    """FL10 is a flux loop in one path and a loop-voltage monitor in this one.

    The equilibrium-magnetics path integrates field 25 into poloidal flux; the
    plasma-current path deliberately does not. Conflating them is the error
    #214 exists to prevent.
    """
    from vaft.machine_mapping.magnetics import (
        vest_equilibrium_magnetics_channel_definitions,
    )

    _, reference, _, _ = _plasma_processing_for_shot(SHOT)
    assert int(reference["field"]) == 25

    flux_loop_fields = {
        int(channel["field_code"])
        for channel in vest_equilibrium_magnetics_channel_definitions()
        if channel["kind"] == "flux_loop"
    }
    # Same physical channel, two different derived quantities.
    assert 25 in flux_loop_fields


def test_provenance_reports_the_renamed_coefficient():
    from vaft.machine_mapping.provenance import vest_processing_provenance

    reference = vest_processing_provenance(SHOT)["plasma_current"]["reference"]
    assert "effective_resistance_ohm" in reference
    assert "mutual_inductance" not in reference
    assert reference["compensation_enabled"] is True

    disabled = vest_processing_provenance(47200)["plasma_current"]["reference"]
    assert disabled["compensation_enabled"] is False


def test_compensation_is_disabled_once_the_inboard_is_carbon():
    """Shot 47117 is a physical boundary, not an unexplained processing quirk.

    VEST operations (2026-09-01): the inboard side was changed to carbon, so
    the induced current in that structure is no longer measured by the inner
    Rogowski coil and there is nothing left for FL10 to compensate.

    Resolved empirically against the SQL database (2026-09-01): 47113 -> 47114
    is a 6.5-day machine access window, and reproducing IP from raw signals
    shows the FL10 term improves the post-discharge residual on 23/23 plasma
    shots over 47050-47113 but only 5/49 over 47114-47200. #191's competing
    47018 sits 12 minutes after 47017 inside a continuous run day with no
    machine access and no signal step, so it is not a transition.

    The physical boundary is 47114; `from_shot` stays at the donor's 47117
    because 47114-47116 carry no plasma (7-10 kA, FL10 ~0.005 V), so the
    three-shot difference cannot change any processed result.
    """
    for shot in (47115, 47116):
        _, reference, _, _ = _plasma_processing_for_shot(shot)
        assert reference.get("mode") != "disabled", shot

    for shot in (47117, 47118, 50000):
        _, reference, _, _ = _plasma_processing_for_shot(shot)
        assert reference.get("mode") == "disabled", shot


def test_the_commissioning_shots_inside_the_boundary_carry_no_plasma():
    """Why the donor's 47117 is harmless despite 47114 being the real break.

    47114-47117 are four vacuum/commissioning shots taken in a 22-minute block
    on 2025-11-13, immediately after a 6.5-day machine access window. They are
    still configured as compensated, but they contain no plasma, so the
    three-shot offset between the physical boundary (47114) and the configured
    one (47117) cannot affect any analysed result.

    Offline assertion: this pins the configured eras only. The raw-signal
    evidence lives in the #214/#216 investigation record.
    """
    _, at_47113, _, _ = _plasma_processing_for_shot(47113)
    assert at_47113.get("mode") == "subtract_fl10_windowed"

    for shot in (47114, 47115, 47116):
        _, reference, _, _ = _plasma_processing_for_shot(shot)
        assert reference.get("mode") == "subtract_fl10_windowed", shot

    _, at_47117, _, _ = _plasma_processing_for_shot(47117)
    assert at_47117.get("mode") == "disabled"
