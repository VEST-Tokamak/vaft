"""Toroidal content of a non-axisymmetric coil excitation (migration slice V5).

Two figures over `coils_non_axisymmetric`: the sector currents against
toroidal angle, and the mode content the process layer reduces them to.  The
fixtures are synthetic patterns with a known answer, because the point of the
spectrum is that it returns the amplitude and phase that were put in.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.plot.backend.recipes import RECIPES, _coil_set_excitation

SECTORS = 6


def _coil_ods(
    pattern, *, sets=("MID",), times=(0.0,), sectors=SECTORS, phi0=0.0,
    provenance=True,
):
    """A coil set of equally spaced sectors carrying ``pattern(phi)``.

    Shaped like a mapped ODS rather than like a convenient test: the sets are
    declared in ``code.parameters`` the way
    :mod:`vaft.machine_mapping.coils_non_axisymmetric` declares them, because
    that block is what the grouping is read from.
    """
    ods = ODS(consistency_check=False)
    if provenance:
        ods["coils_non_axisymmetric.code.parameters"] = "<parameters>" + "".join(
            f'<coil_set name="{label}" identifier="{label}" turns="20" '
            f'sectors="{sectors}" dat_file="{label}.dat">geometry</coil_set>'
            for label in sets
        ) + "</parameters>"
    index = 0
    angles = phi0 + np.arange(sectors) * (2.0 * np.pi / sectors)
    for label in sets:
        for sector, phi in enumerate(angles):
            base = f"coils_non_axisymmetric.coil.{index}"
            ods[f"{base}.name"] = f"{label} sector {sector + 1}"
            ods[f"{base}.identifier"] = f"{label}_{sector + 1:02d}"
            # A short filament straddling the sector angle, so the mean of
            # its `phi` is the sector angle.
            span = np.linspace(phi - 0.05, phi + 0.05, 5)
            elements = f"{base}.conductor.0.elements"
            ods[f"{elements}.start_points.phi"] = span
            ods[f"{elements}.start_points.r"] = np.full(span.size, 0.6)
            ods[f"{elements}.start_points.z"] = np.zeros(span.size)
            ods[f"{base}.current.time"] = np.asarray(times, dtype=float)
            ods[f"{base}.current.data"] = np.asarray(
                [pattern(phi, t) for t in times], dtype=float
            )
            index += 1
    return ods, angles


def test_the_sector_currents_are_read_at_the_angles_they_act_at():
    ods, angles = _coil_ods(lambda phi, t: 1000.0 * np.cos(2.0 * phi))

    (row,) = _coil_set_excitation(ods)

    np.testing.assert_allclose(row["phi"], angles, atol=1e-12)
    np.testing.assert_allclose(row["current"], 1000.0 * np.cos(2.0 * angles), atol=1e-9)


def test_a_coil_across_the_branch_cut_keeps_its_own_angle():
    """A filament that crosses phi = 0 has samples at both ends of the range,
    and their arithmetic mean is on the far side of the machine."""
    ods, _ = _coil_ods(lambda phi, t: 1.0, sectors=4, phi0=0.0)
    # Straddle the cut deliberately: -0.05 .. +0.05 written as 6.23 .. 0.05.
    span = np.linspace(-0.05, 0.05, 5)
    ods["coils_non_axisymmetric.coil.0.conductor.0.elements.start_points.phi"] = (
        np.mod(span, 2.0 * np.pi)
    )

    (row,) = _coil_set_excitation(ods)

    assert min(row["phi"]) == pytest.approx(0.0, abs=1e-9)


def test_the_spectrum_returns_the_amplitude_and_phase_that_were_put_in():
    """`toroidal_mode_decomposition`'s convention: a pattern A cos(n phi + d)
    has |C_n| = A/2 and arg(C_n) = d.  The figure must not quietly rescale."""
    amplitude, delta = 1000.0, 0.7
    ods, _ = _coil_ods(lambda phi, t: amplitude * np.cos(2.0 * phi + delta))

    model = RECIPES["coil_3d_spectrum_current"].builder(ods, unit="A")

    (series,) = model.series
    values = dict(zip(series.x.astype(int), series.y))
    assert values[2] == pytest.approx(amplitude / 2.0, rel=1e-9)
    assert values[0] == pytest.approx(0.0, abs=1e-9)
    assert values[1] == pytest.approx(0.0, abs=1e-9)


def test_the_spectrum_stops_where_the_sectors_stop_resolving():
    """Six sectors resolve |n| <= 2; a bar at n = 3 would report a
    neighbour's amplitude as its own."""
    ods, _ = _coil_ods(lambda phi, t: 1000.0 * np.cos(2.0 * phi))

    model = RECIPES["coil_3d_spectrum_current"].builder(ods)

    (series,) = model.series
    assert series.x.max() == 2
    assert "|n| <= 2 resolved" in series.label


def test_asking_past_the_resolved_band_is_answered_and_labelled():
    """The coefficient is still computable, and still aliased; the figure
    says so rather than refusing or staying quiet."""
    ods, _ = _coil_ods(lambda phi, t: 1000.0 * np.cos(2.0 * phi))

    model = RECIPES["coil_3d_spectrum_current"].builder(ods, modes=[1, 2, 3, 5])

    assert "aliased above the resolved band" in model.title
    assert list(model.series[0].x.astype(int)) == [1, 2, 3, 5]


def test_each_coil_set_is_its_own_series():
    ods, _ = _coil_ods(lambda phi, t: 1000.0 * np.cos(2.0 * phi), sets=("MID", "UP"))

    model = RECIPES["coil_3d_profile_current"].builder(ods)

    assert {series.label.split(" (")[0] for series in model.series} == {"MID", "UP"}
    assert model.x_limits == (0.0, 360.0)


def test_the_instant_drawn_is_on_the_title_and_defaults_to_the_strongest():
    """A run's excitation is a flat-top with zeros around it, and the sample
    a reader wants is the one that is on."""
    ods, _ = _coil_ods(
        lambda phi, t: t * 1000.0 * np.cos(2.0 * phi), times=(0.0, 0.5, 1.0)
    )

    default = RECIPES["coil_3d_profile_current"].builder(ods)
    chosen = RECIPES["coil_3d_profile_current"].builder(ods, time=0.5)

    assert "t=1 s" in default.title
    assert "t=0.5 s" in chosen.title


def test_every_set_is_drawn_at_the_same_instant():
    """A figure whose curves come from different times is not a toroidal
    pattern. Two sets that peak at different times must still be read at one
    instant, and the title names it.
    """
    ods, angles = _coil_ods(lambda phi, t: 0.0, sets=("MID", "UP"), times=(0.0, 1.0))
    # MID peaks at t = 0, UP at t = 1.
    for sector in range(SECTORS):
        ods[f"coils_non_axisymmetric.coil.{sector}.current.data"] = np.array([900.0, 1.0])
        ods[f"coils_non_axisymmetric.coil.{SECTORS + sector}.current.data"] = np.array(
            [1.0, 100.0]
        )

    rows = _coil_set_excitation(ods)

    assert {row["time"] for row in rows} == {0.0}
    assert "t=0 s" in RECIPES["coil_3d_profile_current"].builder(ods).title


def test_a_time_outside_the_record_is_refused():
    """Snapping to the nearest end would draw one instant under another's
    label, which is the failure the spectrum's psi_n guard exists to prevent.
    """
    ods, _ = _coil_ods(lambda phi, t: 1000.0 * np.cos(2.0 * phi), times=(0.0, 1.0, 2.0))

    with pytest.raises(ValueError, match="outside the recorded current"):
        _coil_set_excitation(ods, time=99.0)


def test_coils_that_match_no_declared_set_are_refused():
    """The coil `name` is a display string. Splitting it on " sector" turns
    any other naming into one one-sector set per coil, and a mode number read
    off that grouping means nothing.
    """
    ods, _ = _coil_ods(lambda phi, t: 1.0, provenance=False)

    with pytest.raises(ValueError, match="match no <coil_set> block"):
        _coil_set_excitation(ods)


def test_the_grouping_is_the_mappers_own_not_the_display_name():
    """Rename every coil and the sets must survive, because the identifier
    and the `<coil_set>` block are what the mapper actually grouped by.
    """
    ods, _ = _coil_ods(lambda phi, t: 1000.0 * np.cos(2.0 * phi), sets=("MID", "UP"))
    for index in range(2 * SECTORS):
        ods[f"coils_non_axisymmetric.coil.{index}.name"] = f"winding {index}"

    rows = _coil_set_excitation(ods)

    assert {row["label"] for row in rows} == {"MID", "UP"}
    assert all(row["phi"].size == SECTORS for row in rows)


def test_a_current_with_no_geometry_is_refused():
    """The toroidal angle a current acts at is the whole point; without it
    the mode content would be assembled from array order."""
    ods, _ = _coil_ods(lambda phi, t: 1.0)
    del ods["coils_non_axisymmetric.coil.0.conductor.0.elements.start_points.phi"]

    with pytest.raises(ValueError, match="toroidal angle"):
        _coil_set_excitation(ods)


def test_geometry_with_no_excitation_says_what_to_overlay():
    ods, _ = _coil_ods(lambda phi, t: 1.0)
    for index in range(SECTORS):
        del ods[f"coils_non_axisymmetric.coil.{index}.current.data"]

    with pytest.raises(ValueError, match="apply_coil_excitation"):
        _coil_set_excitation(ods)


def test_the_unit_is_shown_and_the_values_carry_it():
    ods, _ = _coil_ods(lambda phi, t: 1000.0 * np.cos(2.0 * phi))

    amps = RECIPES["coil_3d_spectrum_current"].builder(ods, unit="A")
    kiloamps = RECIPES["coil_3d_spectrum_current"].builder(ods, unit="kA")

    assert "[A]" in amps.y_label and "[kA]" in kiloamps.y_label
    np.testing.assert_allclose(kiloamps.series[0].y, amps.series[0].y * 1e-3)
