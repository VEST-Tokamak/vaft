"""Where the magnetic diagnostics sit toroidally (issue #718).

Pinned against the VEST magnetic diagnostic layout figure. Its top view places
the four toroidal probes at coordinates that convert to 45.6 / 164.8 / 225.0 /
283.8 degrees of VEST clock angle -- 1:30, 5:30, 7:30 and 9:30, matching all
four channel names -- and its toroidal/poloidal panel gives the three poloidal
families at 30, 120 and 165 degrees of clock angle.

Clock angle and IMAS phi are asserted separately on purpose: they are different
coordinates running opposite ways, and a test that checked only one would not
notice the other being written into the IDS.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.machine_mapping.conventions import port_toroidal_angle, vest_clock_angle
from vaft.machine_mapping.magnetics import (
    EQUILIBRIUM_PROBE_CLOCK,
    INBOARD_PROBE_MAX_R,
    OUTBOARD_PROBE_MIN_R,
    SIDE_PROBE_MIN_ABS_Z,
    TOROIDAL_MIRNOV_REFERENCE_CHANNELS,
    _load_static_channels,
    vfit_magnetics_static,
)
from vaft.machine_mapping.registry import port_phi
from vaft.machine_mapping.utils import get_path, path_exists


#: Read off the layout figure's top view: (x, y) in metres, per probe. These
#: are taken by eye from a plot, so the tolerance below is degrees, not
#: decimals -- the test is asking "is this the same position", not "is this the
#: same number".
LAYOUT_FIGURE_TOP_VIEW = {
    "OutMirnov_130_Bz": (0.50, 0.51),
    "OutMirnov_530_Bz": (-0.70, 0.19),
    "OutMirnov_730_Bz": (-0.50, -0.50),
    "MagneticFieldProbe_C2-05_Bz": (0.17, -0.69),
}


def test_reference_mirnov_clock_positions_match_the_layout_figure():
    """The figure's coordinates and the channel names say the same thing."""
    for channel in TOROIDAL_MIRNOV_REFERENCE_CHANNELS:
        x, y = LAYOUT_FIGURE_TOP_VIEW[channel["name"]]
        measured = np.rad2deg(np.arctan2(y, x)) % 360.0
        assert vest_clock_angle(channel["clock"]) == pytest.approx(measured, abs=2.0)


def test_reference_mirnov_names_encode_their_own_clock_position():
    """``OutMirnov_130`` is at 1:30. The name is the provenance."""
    for channel in TOROIDAL_MIRNOV_REFERENCE_CHANNELS:
        _, _, suffix = channel["name"].partition("OutMirnov_")
        if not suffix:
            continue  # C2-05 does not carry its position in its name
        digits = suffix.split("_")[0]
        hour, minute = int(digits[:-2]), int(digits[-2:])
        assert channel["clock"] == pytest.approx(hour + minute / 60.0)


@pytest.mark.parametrize(
    "name,clock_angle_deg,phi_deg",
    [
        ("OutMirnov_130_Bz", 45.0, 315.0),
        ("OutMirnov_530_Bz", 165.0, 195.0),
        ("OutMirnov_730_Bz", 225.0, 135.0),
        ("MagneticFieldProbe_C2-05_Bz", 285.0, 75.0),
    ],
)
def test_reference_mirnov_resolves_in_both_coordinates(name, clock_angle_deg, phi_deg):
    """The stored value is phi, not the clock angle.

    Before #718 these were 0 / 120 / 180 / 240 -- a relative frame anchored on
    the first channel, a uniform -45 deg from the clock angles the names state,
    and left-handed besides.
    """
    channel = next(c for c in TOROIDAL_MIRNOV_REFERENCE_CHANNELS if c["name"] == name)
    assert vest_clock_angle(channel["clock"]) == pytest.approx(clock_angle_deg)
    assert np.rad2deg(port_toroidal_angle(channel["clock"])) == pytest.approx(phi_deg)


@pytest.mark.parametrize(
    "family,clock_angle_deg,phi_deg",
    [("inboard", 30.0, 330.0), ("side", 120.0, 240.0), ("outboard", 165.0, 195.0)],
)
def test_equilibrium_probe_families_sit_where_the_figure_puts_them(family, clock_angle_deg, phi_deg):
    clock = EQUILIBRIUM_PROBE_CLOCK[family]
    assert vest_clock_angle(clock) == pytest.approx(clock_angle_deg)
    assert np.rad2deg(port_toroidal_angle(clock)) == pytest.approx(phi_deg)


def test_the_side_family_agrees_with_the_port_document():
    """4 o'clock is ``4ML10 : Magnetic probe 1`` -- an independent source."""
    assert port_toroidal_angle(EQUILIBRIUM_PROBE_CLOCK["side"]) == pytest.approx(port_phi("4ML10"))


def test_the_outboard_family_shares_its_angle_with_the_530_reference():
    """Both are on the wall at 5:30; the figure draws them overlapping."""
    reference = next(c for c in TOROIDAL_MIRNOV_REFERENCE_CHANNELS if c["name"] == "OutMirnov_530_Bz")
    assert EQUILIBRIUM_PROBE_CLOCK["outboard"] == pytest.approx(reference["clock"])


def test_every_equilibrium_probe_classifies_into_a_family():
    """No probe falls through to "no phi" in the packaged geometry."""
    unclassified = [
        channel for channel in _load_static_channels()
        if channel["kind"] == "b_field_pol_probe"
        and not (
            abs(float(channel["z"])) > SIDE_PROBE_MIN_ABS_Z
            or float(channel["r"]) < INBOARD_PROBE_MAX_R
            or float(channel["r"]) > OUTBOARD_PROBE_MIN_R
        )
    ]
    assert unclassified == []


# ---------------------------------------------------------------------------
# What the mapper actually writes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def static_ods():
    payload = {}
    vfit_magnetics_static(payload)
    return payload


def test_probes_are_written_at_their_family_angle(static_ods):
    written = []
    for index in range(len(get_path(static_ods, "magnetics.b_field_pol_probe"))):
        path = f"magnetics.b_field_pol_probe.{index}.position.phi"
        if path_exists(static_ods, path):
            written.append(np.rad2deg(get_path(static_ods, path)))
    written = np.asarray(written)
    for clock in EQUILIBRIUM_PROBE_CLOCK.values():
        expected = np.rad2deg(port_toroidal_angle(clock))
        assert np.isclose(written, expected).any(), f"no probe written at {expected} deg"


def test_no_probe_is_left_at_the_zero_placeholder(static_ods):
    """0.0 is 12 o'clock, a real place -- it cannot double as "unknown".

    Every probe family is away from 12 o'clock, so nothing legitimately lands
    on 0 here; a probe that did would be the old placeholder coming back.
    """
    for index in range(len(get_path(static_ods, "magnetics.b_field_pol_probe"))):
        path = f"magnetics.b_field_pol_probe.{index}.position.phi"
        if path_exists(static_ods, path):
            assert get_path(static_ods, path) != 0.0


def test_flux_loops_get_no_toroidal_angle(static_ods):
    """A flux loop encircles the axis, so it has no scalar phi to write."""
    loops = get_path(static_ods, "magnetics.flux_loop")
    assert len(loops) > 0
    for index in range(len(loops)):
        assert not path_exists(static_ods, f"magnetics.flux_loop.{index}.position.phi")
        assert not path_exists(static_ods, f"magnetics.flux_loop.{index}.position.0.phi")


# ---------------------------------------------------------------------------
# Thomson: phi varies along the laser chord
# ---------------------------------------------------------------------------


def test_thomson_volumes_lie_on_the_laser_chord():
    """Each polychromator's phi follows from its own radius, not a constant."""
    from vaft.machine_mapping.thomson_scattering import (
        _CHANNEL_META,
        _chord_geometry,
        scattering_volume_phi,
    )

    _, tangency = _chord_geometry()
    angles = [np.rad2deg(scattering_volume_phi(r)) for _, r, _, _ in _CHANNEL_META]

    # Not one angle for the whole diagnostic.
    assert len(set(np.round(angles, 3))) == len(angles)
    # Every radius is outside the chord's closest approach, as it must be.
    assert all(r > tangency for _, r, _, _ in _CHANNEL_META)
    # The volumes straddle the 9MM10 viewing port, which is what picks this
    # branch of the chord over the one near 12 o'clock.
    viewing = np.rad2deg(port_phi("9MM10"))
    assert min(angles) < viewing < max(angles)


def test_a_bad_thomson_radius_costs_one_channel_not_the_shot():
    """Dynamic radii come from each shot's file, so they can be defective."""
    from vaft.machine_mapping.thomson_scattering import (
        _scattering_volume_phi_or_none,
        scattering_volume_phi,
    )

    assert _scattering_volume_phi_or_none(0.10, 0) is None      # inside the tangency radius
    assert _scattering_volume_phi_or_none(float("nan"), 1) is None
    assert _scattering_volume_phi_or_none(0.475, 2) is not None

    # The public helper still refuses rather than inventing a location.
    with pytest.raises(ValueError, match="not on the"):
        scattering_volume_phi(0.10)


def test_thomson_static_writes_a_distinct_angle_per_channel():
    import omas

    from vaft.machine_mapping.thomson_scattering import vfit_thomson_scattering_static

    ods = omas.ODS()
    vfit_thomson_scattering_static(ods)
    written = [
        round(float(np.rad2deg(ods[f"thomson_scattering.channel.{i}.position.phi"])), 3)
        for i in range(5)
    ]
    assert written == sorted(written, reverse=True)
    assert len(set(written)) == 5
