"""Rendering GPEC's resonant response from the IDS that carries it.

These quantities have no IMAS slot -- `mhd_linear` has no per-surface numeric
field, and `ntms.mode[].deltaw` is in m^-1 where GPEC's `Delta` is unitless --
so the mapper stores the spectral field and the surface geometry, and the
adapter derives the table. The fixtures here are built by the same mapper a
real run goes through, because the thing under test is precisely whether what
it wrote is enough.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from gpec_nc_fixtures import write_control_nc, write_cylindrical_nc, write_profile_nc
from vaft.machine_mapping.gpec_ideal import gpec_ideal
from vaft.plot.backend.recipes import RECIPES

NAMES = ("mhd_linear_profile_resonant_flux", "mhd_linear_profile_island_width")


@pytest.fixture
def mapped(tmp_path):
    """An ODS built the way a real ideal-GPEC run reaches one."""
    write_control_nc(tmp_path, n=1)
    write_cylindrical_nc(tmp_path, n=1)
    write_profile_nc(tmp_path, n=1, rational_q=(2.0, 3.0))
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1]})
    return ods


@pytest.mark.parametrize("name", NAMES)
def test_a_mapped_run_renders_one_point_per_rational_surface(mapped, name):
    model = RECIPES[name].builder(mapped)
    assert len(model.series) == 1
    series = model.series[0]
    assert series.x.size == 2  # the fixture's two rational surfaces
    assert series.y.size == series.x.size
    assert np.all(np.isfinite(series.y))
    assert np.all(series.y >= 0.0)
    assert "n=1" in model.title


@pytest.mark.parametrize("name", NAMES)
def test_the_title_says_the_values_were_derived(mapped, name):
    """A reader must not take these for numbers the IDS carried: they are the
    adapter's, and they agree with GPEC's own to about a per cent rather than
    exactly."""
    assert "derived" in RECIPES[name].builder(mapped).title


def test_the_surfaces_come_back_at_the_positions_the_mapper_recorded(mapped):
    import re

    recorded = [
        float(value)
        for value in re.findall(r'<surface psi_n="([^"]+)"', mapped["mhd_linear.code.parameters"])
    ]
    for name in NAMES:
        np.testing.assert_allclose(RECIPES[name].builder(mapped).series[0].x, recorded)


def test_the_two_figures_share_one_derivation(mapped, monkeypatch):
    """Each surface costs a pair of one-sided cubic fits. Two renderers over
    one table must not pay for them twice, and a future refactor that moved
    the derivation into the renderer would."""
    from vaft.plot.backend import recipes

    calls = []
    original = recipes.__dict__["_gpec_resonant_table"]
    monkeypatch.setitem(
        recipes.__dict__, "_gpec_resonant_table",
        lambda ods, **options: (calls.append(1), original(ods, **options))[1],
    )
    for name in NAMES:
        RECIPES[name].builder(mapped)
    assert len(calls) == len(NAMES), "one derivation per figure, not per trace"


def test_a_dcon_only_product_says_what_is_missing(tmp_path):
    """`mhd_linear` also holds DCON's eigenfunction on a (psi, m) grid, so the
    grid alone does not mean the resonant table can be derived -- the surface
    geometry only an ideal-GPEC mapping writes is what decides it."""
    write_control_nc(tmp_path, n=1)
    write_cylindrical_nc(tmp_path, n=1)
    write_profile_nc(tmp_path, n=1)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1]})
    ods["mhd_linear.code.parameters"] = "<parameters/>"
    with pytest.raises(ValueError, match="no rational-surface geometry"):
        RECIPES[NAMES[0]].builder(ods)


def test_each_mode_reads_its_own_geometry(tmp_path):
    """`code.parameters` holds one <solver> per mode and their surfaces differ,
    so an adapter that took the first block would draw n=1's surfaces on n=2's
    figure."""
    ods = ODS(consistency_check=False)
    for mode, surfaces in ((1, (2.0, 3.0)), (2, (2.0, 2.5, 3.0))):
        write_control_nc(tmp_path, n=mode)
        write_cylindrical_nc(tmp_path, n=mode)
        write_profile_nc(tmp_path, n=mode, rational_q=surfaces)
        gpec_ideal(ods, str(tmp_path), {"mode": mode, "modes": [1, 2]})
    counts = {
        mode: RECIPES[NAMES[0]].builder(ods, n_tor=mode).series[0].x.size
        for mode in (1, 2)
    }
    assert counts == {1: 2, 2: 3}


def test_a_surface_outside_the_mapped_harmonic_band_is_skipped_not_guessed(tmp_path):
    """The jump is taken in the resonant harmonic's own column. If m = nq is
    outside the band the mapper wrote, there is no column to take it in, and
    the nearest one belongs to a different helicity."""
    write_control_nc(tmp_path, n=1)
    write_cylindrical_nc(tmp_path, n=1)
    # The fixture widens its harmonic band to cover its own rational surfaces,
    # as the real file does; turning that off is how a narrow band -- a
    # trimmed extract, or a run with mhigh below its own outermost surface --
    # is reproduced. m_out then spans -2..2, so q = 2 (m = 2) is inside and
    # q = 40 (m = 40) is far outside.
    write_profile_nc(
        tmp_path, n=1, rational_q=(2.0, 40.0), cover_rational_harmonics=False
    )
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1]})
    assert RECIPES[NAMES[0]].builder(ods).series[0].x.size == 1


def test_a_run_whose_surfaces_are_all_outside_the_band_is_an_error(tmp_path):
    write_control_nc(tmp_path, n=1)
    write_cylindrical_nc(tmp_path, n=1)
    write_profile_nc(
        tmp_path, n=1, rational_q=(40.0, 50.0), cover_rational_harmonics=False
    )
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1]})
    with pytest.raises(ValueError, match="resonates at a harmonic the mapped band covers"):
        RECIPES[NAMES[0]].builder(ods)
