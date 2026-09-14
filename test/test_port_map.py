"""The VEST port table and the clock-position-to-toroidal-angle convention.

Issue #718. Two coordinates live here and they run opposite ways round the
machine, so the tests are deliberately split: those that pin the *clock angle*
(the hardware coordinate, clockwise-positive) and those that pin *phi* (IMAS,
counter-clockwise). Mixing them in one assertion is how the two came to be
conflated in the first place.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vaft.machine_mapping.conventions import (
    VEST_PORT_CLOCK_DEGREES,
    VEST_PORT_CLOCK_TO_PHI_SIGN,
    clock_angle_to_toroidal_angle,
    port_toroidal_angle,
    vest_clock_angle,
)
from vaft.machine_mapping.registry import (
    PortMapError,
    load_port_map,
    port_clock,
    port_clock_angle,
    port_phi,
    validate_port_map,
)


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------

#: The table the convention was fixed against. 12 o'clock is phi = 0 and clock
#: number advances clockwise, which is decreasing phi.
EXPECTED_PHI_DEG = {
    12: 0.0, 1: 330.0, 2: 300.0, 3: 270.0, 4: 240.0, 5: 210.0,
    6: 180.0, 7: 150.0, 8: 120.0, 9: 90.0, 10: 60.0, 11: 30.0,
}


@pytest.mark.parametrize("clock,phi_deg", sorted(EXPECTED_PHI_DEG.items()))
def test_port_toroidal_angle_matches_the_agreed_table(clock, phi_deg):
    assert np.rad2deg(port_toroidal_angle(clock)) == pytest.approx(phi_deg, abs=1e-9)


def test_the_clock_coordinate_is_not_phi():
    """The distinction the whole issue turns on.

    Only 12 o'clock and 6 o'clock survive the reflection unchanged; every other
    port has two different numbers, and writing one where the other belongs is
    the defect #718 found.
    """
    same = {h for h in range(1, 13) if math.isclose(vest_clock_angle(h), np.rad2deg(port_toroidal_angle(h)))}
    assert same == {6, 12}


def test_clock_angle_advances_clockwise_and_phi_counter_clockwise():
    assert VEST_PORT_CLOCK_TO_PHI_SIGN == -1
    assert VEST_PORT_CLOCK_DEGREES == 30.0
    # One hour later is +30 deg of clock angle and -30 deg of phi.
    assert vest_clock_angle(4) - vest_clock_angle(3) == pytest.approx(30.0)
    assert np.rad2deg(port_toroidal_angle(4) - port_toroidal_angle(3)) == pytest.approx(-30.0)


def test_fractional_hours_place_the_between_port_hardware():
    """The outboard magnetic probes mount on the wall between ports."""
    assert vest_clock_angle(1.5) == pytest.approx(45.0)
    assert np.rad2deg(port_toroidal_angle(1.5)) == pytest.approx(315.0)


def test_clock_angle_and_hour_entry_points_agree():
    for hour in (1.0, 1.5, 4.0, 5.5, 7.5, 9.5, 12.0):
        assert clock_angle_to_toroidal_angle(vest_clock_angle(hour)) == pytest.approx(
            port_toroidal_angle(hour)
        )


def test_angles_are_wrapped_into_one_turn():
    for hour in range(1, 13):
        assert 0.0 <= port_toroidal_angle(hour) < 2 * math.pi
        assert 0.0 <= vest_clock_angle(hour) < 360.0


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_the_port_table_loads_and_validates():
    ports = load_port_map()
    assert len(ports) == 88
    assert len({record["name"] for record in ports.values()}) == len(ports)


def test_every_clock_position_is_represented():
    clocks = {float(record["clock"]) for record in load_port_map().values()}
    assert clocks == {float(hour) for hour in range(1, 13)}


def test_empty_ports_are_recorded_rather_than_omitted():
    """"Available" and "No port" are facts about the machine, not absences."""
    uses = [record["use"] for record in load_port_map().values()]
    assert uses.count("Available") > 0
    assert uses.count("No port") > 0


@pytest.mark.parametrize(
    "name,clock_angle_deg,phi_deg",
    [
        ("12MM10", 0.0, 0.0),      # soft X-ray
        ("10MR", 300.0, 60.0),     # interferometry / ion Doppler
        ("7T6", 210.0, 150.0),     # 280 GHz interferometer
        ("2MR", 60.0, 300.0),      # NBI
        ("11M12", 330.0, 30.0),    # triple probe and IMPA
        ("4ML10", 120.0, 240.0),   # magnetic probe 1
        ("8MM10", 240.0, 120.0),   # Thomson laser entry
        ("1MM10", 30.0, 330.0),    # Thomson laser dump
        ("9MM10", 270.0, 90.0),    # Thomson viewing
        ("6MR", 180.0, 180.0),     # entrance: camera, filterscope, HXR
    ],
)
def test_named_ports_resolve_in_both_coordinates(name, clock_angle_deg, phi_deg):
    assert port_clock_angle(name) == pytest.approx(clock_angle_deg)
    assert np.rad2deg(port_phi(name)) == pytest.approx(phi_deg)


def test_an_unknown_port_raises_rather_than_defaulting():
    with pytest.raises(PortMapError):
        port_phi("13ZZ99")


# ---------------------------------------------------------------------------
# Validation: the name and the fields must agree
# ---------------------------------------------------------------------------


def _record(**overrides):
    base = {"name": "12MM10", "clock": 12, "chamber": "main", "tier": "middle",
            "size_inch": 10, "use": "Soft X-ray"}
    base.update(overrides)
    return {base["name"]: base}


@pytest.mark.parametrize(
    "label,record",
    [
        ("clock disagrees with the name", _record(clock=4)),
        ("chamber disagrees with the name", _record(chamber="top")),
        ("tier disagrees with the name", _record(tier="lower")),
        ("tier outside the main chamber", {"7T6": {
            "name": "7T6", "clock": 7, "chamber": "top", "tier": "middle", "use": "x"}}),
        ("name is not a VEST port", {"13ZZ99": {
            "name": "13ZZ99", "clock": 1, "chamber": "main", "use": "x"}}),
        ("clock out of range", {"12MM10": {
            "name": "12MM10", "clock": 13, "chamber": "main", "tier": "middle", "use": "x"}}),
        ("empty use", _record(use="   ")),
        ("missing use", {"12MM10": {
            "name": "12MM10", "clock": 12, "chamber": "main", "tier": "middle"}}),
    ],
)
def test_validation_rejects(label, record):
    with pytest.raises(PortMapError):
        validate_port_map(record)


def test_validation_accepts_the_packaged_table():
    validate_port_map(load_port_map())


# ---------------------------------------------------------------------------
# Loading: cached, isolated, and on one error type
# ---------------------------------------------------------------------------


def _table_with(replacement: str, tmp_path):
    """The packaged table with its first port entry replaced."""
    from vaft.machine_mapping.registry import registry_path

    source = registry_path().read_text(encoding="utf-8")
    original = "    - {name: 1MU10,  clock: 1,"
    assert source.count(original) == 1
    target = tmp_path / "vest.yaml"
    target.write_text(source.replace(original, replacement, 1), encoding="utf-8")
    return target


def test_repeated_lookups_do_not_reparse_the_table():
    """port_phi is called per diagnostic channel; parsing there is not free."""
    from vaft.machine_mapping.registry import _load_port_map_cached, registry_path

    load_port_map()
    before = _load_port_map_cached.cache_info().misses
    for _ in range(50):
        port_phi("11M12")
        port_clock("10MR")
    assert _load_port_map_cached.cache_info().misses == before


def test_a_returned_table_can_be_mutated_without_poisoning_the_cache():
    first = load_port_map()
    first["11M12"]["clock"] = 99
    first["11M12"]["use"] = "corrupted"
    assert load_port_map()["11M12"]["clock"] == 11
    assert port_clock("11M12") == 11


def test_a_port_entry_without_a_name_raises_the_documented_error(tmp_path):
    """Not a bare KeyError: callers are told to catch PortMapError."""
    table = _table_with("    - {clock: 1,", tmp_path)
    with pytest.raises(PortMapError, match="missing name"):
        load_port_map(table)


def test_a_non_mapping_port_entry_is_named_for_what_is_wrong(tmp_path):
    table = _table_with("    - not-a-mapping\n    - {name: 1MU10, clock: 1,", tmp_path)
    with pytest.raises(PortMapError, match="must be a mapping"):
        load_port_map(table)


def test_a_duplicate_port_name_is_reported_as_a_duplicate(tmp_path):
    table = _table_with(
        "    - {name: 10MR, clock: 10, chamber: main, size: rectangular, use: Duplicate}\n"
        "    - {name: 1MU10,  clock: 1,",
        tmp_path,
    )
    with pytest.raises(PortMapError, match="duplicate port name"):
        load_port_map(table)
