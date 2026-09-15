"""The resistive half of the stability suite is drawable (#170).

The stage's validation set covered DCON alone -- perturbed energy, the
eigenfunction, and a run-status coverage panel. What RDCON and STRIDE actually
compute, the classical tearing index, went unplotted: it had no slot under
`toroidal_mode` and survived only in the run manifest.

It now lives in `ntms.time_slice[t].mode[i].deltaw[:].value`, so it can be drawn
from the stage's own product. These tests pin the two decisions that make the
figure mean something -- what a trace is, and which `deltaw` contribution is
read -- rather than that a PNG appeared.
"""

import numpy as np
import pytest
from omas import ODS

from vaft.database.production_qa import STAGE_VALIDATION_PLOTS, stage_plot_filenames
from vaft.plot.backend.recipes import RECIPES
from vaft.plot.registry import canonical_names


pytestmark = pytest.mark.core


def _ods(surfaces_by_time, *, times=(0.316, 0.317), name="classical"):
    """`surfaces_by_time[t] = [(n_tor, m_pol, delta_prime), ...]`."""
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 39915
    ods["ntms.ids_properties.homogeneous_time"] = 1
    ods["ntms.time"] = np.asarray(times[: len(surfaces_by_time)], dtype=float)
    for index, surfaces in enumerate(surfaces_by_time):
        for position, (n_tor, m_pol, value) in enumerate(surfaces):
            entry = ods["ntms"]["time_slice"][index]["mode"][position]
            entry["n_tor"] = int(n_tor)
            entry["m_pol"] = int(m_pol)
            entry["deltaw"][0]["name"] = name
            entry["deltaw"][0]["value"] = float(value)
    return ods


def _build(ods, **options):
    return RECIPES["ntms_time_delta_prime"].builder(ods, **options)


def test_the_renderer_is_registered_and_owned_by_its_own_subject():
    """`ntms` is a subject of its own, not a view of `mhd_linear`.

    An `ntms.mode` entry is a rational *surface* the solver located, not one of
    the toroidal modes the caller asked for, so the two IDS are indexed by
    different things and a plot of one is not a plot of the other.
    """
    assert "ntms_time_delta_prime" in canonical_names()
    assert "ntms_time_delta_prime" in RECIPES


def test_a_trace_is_one_rational_surface_not_one_toroidal_mode():
    """Several surfaces share an `n_tor`; collapsing them would average together
    physically distinct tearing layers."""
    ods = _ods([[(1, 2, 0.5), (1, 3, -1.5)], [(1, 2, 0.8), (1, 3, -1.2)]])

    model = _build(ods)

    labels = sorted(series.label for series in model.series)
    assert labels == ["m/n = 2/1", "m/n = 3/1"]
    by_label = {series.label: series for series in model.series}
    assert list(by_label["m/n = 2/1"].y) == pytest.approx([0.5, 0.8])
    assert list(by_label["m/n = 3/1"].y) == pytest.approx([-1.5, -1.2])


def test_a_surface_the_solver_did_not_find_is_absent_rather_than_padded():
    """How many surfaces exist is itself a result.

    Padding a missing surface with a value would put a number in the figure that
    no solver produced.
    """
    ods = _ods([[(1, 2, 0.5)], [(1, 2, 0.8), (1, 3, -1.0)]])

    model = _build(ods)

    by_label = {series.label: series for series in model.series}
    assert len(by_label["m/n = 2/1"].x) == 2
    # The 3/1 surface exists at the second time only, and is drawn there alone.
    assert len(by_label["m/n = 3/1"].x) == 1
    assert by_label["m/n = 3/1"].x[0] == pytest.approx(0.317)


def test_the_classical_contribution_is_selected_by_name_not_by_position():
    """`deltaw` is an array of contributions.

    Reading slot 0 would let a future contribution be plotted as the classical
    index simply because it was appended first.
    """
    ods = _ods([[(1, 2, 0.5)]])
    entry = ods["ntms"]["time_slice"][0]["mode"][0]
    # Insert a different contribution ahead of the classical one.
    entry["deltaw"][0]["name"] = "neoclassical"
    entry["deltaw"][0]["value"] = 99.0
    entry["deltaw"][1]["name"] = "classical"
    entry["deltaw"][1]["value"] = 0.5

    model = _build(ods)

    assert list(model.series[0].y) == pytest.approx([0.5])


def test_an_ods_with_no_tearing_index_says_which_codes_write_one():
    ods = ODS(consistency_check=False)
    ods["ntms.ids_properties.homogeneous_time"] = 1
    ods["ntms"]["time_slice"][0]

    with pytest.raises(ValueError, match="only RDCON and STRIDE"):
        _build(ods)

    empty = ODS(consistency_check=False)
    with pytest.raises(ValueError, match="no ntms time slices"):
        _build(empty)


def test_the_axes_say_which_sign_is_unstable():
    """The DCON figure says negative is unstable; this one is the opposite.

    A reader moving between the two stability figures has to be told, or the
    convention is something they carry in their head.
    """
    model = _build(_ods([[(1, 2, 0.5)]]))

    assert "Delta" in model.y_label
    assert "positive is unstable" in model.title


def test_the_stage_draws_the_resistive_codes_as_well_as_dcon():
    """#170's third acceptance criterion.

    Optional rather than required: a DCON-only configuration is a legitimate
    run, so a shot with no resistive solver has nothing to draw here rather than
    a missing figure.
    """
    entries = {entry.plot: entry for entry in STAGE_VALIDATION_PLOTS["mhd_linear"]}

    assert "ntms_time_delta_prime" in entries
    assert entries["ntms_time_delta_prime"].required is False
    assert "stability_delta_prime.png" in stage_plot_filenames("mhd_linear")
    assert "stability_delta_prime.png" not in stage_plot_filenames(
        "mhd_linear", required_only=True
    )
