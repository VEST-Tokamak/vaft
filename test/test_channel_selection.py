"""Semantic channel selection and the shared region classifier (issue #259).

The inboard/outboard divider is inferred per diagnostic family from that
family's own geometry, so it cannot be moved by how densely one side is
instrumented, by an outlying channel, or by a hard-coded machine radius.  Every
plotting consumer resolves through the same definition.  Policy:
``notebooks/plotting_sample_using_vaft_plot_module.ipynb``.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import omas
import pytest

import vaft.omas
from vaft.plot.selection import (
    INBOARD,
    OUTBOARD,
    UNCLASSIFIED,
    classify_regions,
    radial_divider,
    representative_index,
)


# ---------------------------------------------------------------------------
# The divider rule
# ---------------------------------------------------------------------------

def test_channel_density_cannot_move_the_divider():
    """Twenty inboard sensors against two outboard must split in the same place.

    A mean-based rule fails this: it would sit at 0.17 and call every outboard
    channel inboard.
    """
    dense = radial_divider([0.1] * 20 + [0.9] * 2)
    sparse = radial_divider([0.1] * 2 + [0.9] * 20)
    balanced = radial_divider([0.1] * 5 + [0.9] * 5)
    assert dense.divider == pytest.approx(0.5)
    assert sparse.divider == pytest.approx(0.5)
    assert balanced.divider == pytest.approx(0.5)


def test_an_outlying_channel_does_not_drag_the_divider():
    """VEST's probe array carries an IMPA scan out to R = 1.26 m.

    The extent midpoint would be dragged from 0.44 to 0.67 by those eight
    channels; the widest gap ignores them.
    """
    probes = [0.089] * 27 + [0.1377] * 16 + [0.796] * 25
    with_impa = probes + [0.91, 0.96, 1.01, 1.06, 1.11, 1.16, 1.21, 1.26]
    assert radial_divider(probes).divider == pytest.approx(
        radial_divider(with_impa).divider
    )
    # And the IMPA channels themselves still read as outboard.
    assert classify_regions(with_impa)[-1] == OUTBOARD


def test_families_are_classified_independently():
    """Flux loops are split against flux loops, probes against probes."""
    loops = [0.091, 0.106, 0.130, 0.138, 0.592, 0.792]
    probes = [0.089, 0.1377, 0.796, 1.26]
    assert radial_divider(loops).divider != pytest.approx(
        radial_divider(probes).divider
    )
    # Pooling them would produce a single divider that serves neither.
    pooled = radial_divider(loops + probes).divider
    assert pooled != pytest.approx(radial_divider(loops).divider)


def test_a_family_without_a_gap_gets_no_split():
    """Every fluctuation Mirnov sits at one radius; there is no inboard there."""
    assert not radial_divider([0.796] * 12)
    assert classify_regions([0.796] * 12) == [UNCLASSIFIED] * 12
    # A single loose cluster is not two sides either.
    assert not radial_divider([0.50, 0.51, 0.52, 0.53])


def test_a_channel_on_the_divider_is_left_unclassified():
    """Classified against the family's divider, a borderline channel abstains.

    The split is passed in rather than re-inferred, because a channel sitting
    mid-gap is a third cluster and would legitimately move an inferred divider.
    """
    split = radial_divider([0.1, 0.9])
    regions = classify_regions([0.1, split.divider, 0.9], split=split)
    assert regions == [INBOARD, UNCLASSIFIED, OUTBOARD]
    # Just outside the tolerance it is assigned a side again.
    outside = split.divider + 10 * split.tolerance
    assert classify_regions([outside], split=split) == [OUTBOARD]


def test_classification_survives_a_floating_point_perturbation():
    """No channel may be identified by exact coordinate equality."""
    loops = np.array([0.091, 0.106, 0.130, 0.138, 0.592, 0.792])
    nudged = np.nextafter(loops, loops + 1.0)
    assert classify_regions(loops) == classify_regions(nudged)


def test_the_representative_is_the_channel_nearest_the_midplane():
    z = [0.685, 0.46, -0.46, -0.685, -0.805, 0.04]
    assert representative_index(z, [4, 5]) == 5
    # Ties resolve to the lowest index, so the answer is reproducible.
    assert representative_index([0.5, -0.5, 0.5], [0, 1, 2]) == 0
    assert representative_index([0.1], []) is None


# ---------------------------------------------------------------------------
# The public selection contract
# ---------------------------------------------------------------------------

@pytest.fixture()
def loop_ods():
    ods = omas.ODS()
    ods["magnetics.time"] = np.linspace(0.0, 0.1, 4)
    for index, (name, r, z) in enumerate(
        (("FL01", 0.091, 0.04), ("FL02", 0.138, 0.805), ("FL03", 0.792, 0.46))
    ):
        ods[f"magnetics.flux_loop.{index}.name"] = name
        ods[f"magnetics.flux_loop.{index}.position.0.r"] = r
        ods[f"magnetics.flux_loop.{index}.position.0.z"] = z
        ods[f"magnetics.flux_loop.{index}.flux.data"] = np.ones(4) * (index + 1)
    return ods


def test_selection_accepts_all_indices_and_identifiers(loop_ods):
    for request, expected in (
        (None, 3),
        ("all", 3),
        (1, 1),
        ([0, 2], 2),
        ("FL02", 1),
        (["FL01", "FL03"], 2),
    ):
        figure, axes = vaft.omas.plot_flux_loop_time_flux(loop_ods, selection=request)
        assert len(axes.lines) == expected, request
        matplotlib.pyplot.close(figure)


def test_an_unknown_selection_raises_and_names_what_exists(loop_ods):
    with pytest.raises(ValueError, match="unknown selection"):
        vaft.omas.plot_flux_loop_time_flux(loop_ods, selection="FL99")
    # The message lists the real identifiers rather than guessing at a match.
    with pytest.raises(ValueError, match="FL01"):
        vaft.omas.plot_flux_loop_time_flux(loop_ods, selection="fl01")


def test_channels_is_deprecated_in_favour_of_selection(loop_ods):
    with pytest.warns(DeprecationWarning, match="selection="):
        figure, axes = vaft.omas.plot_flux_loop_time_flux(loop_ods, channels=[0, 1])
    assert len(axes.lines) == 2
    matplotlib.pyplot.close(figure)

    with pytest.raises(TypeError, match="not both"):
        vaft.omas.plot_flux_loop_time_flux(loop_ods, selection=[0], channels=[1])


# ---------------------------------------------------------------------------
# One definition, every consumer
# ---------------------------------------------------------------------------

def _sample():
    from vaft.data import sample

    return vaft.omas.load(sample(39915, "omas"))


def test_the_packaged_shot_classifies_as_the_machine_is_built():
    ods = _sample()
    loops = classify_regions(np.asarray(ods["magnetics.flux_loop.:.position.0.r"], float))
    assert loops.count(INBOARD) == 7
    assert loops.count(OUTBOARD) == 4


def test_legacy_selectors_agree_with_the_shared_classifier():
    """plot.time, plot.analysis and the new selector share one definition."""
    from vaft.plot.time import (
        _find_flux_loop_inboard_indices,
        _find_flux_loop_outboard_indices,
    )

    ods = _sample()
    regions = classify_regions(
        np.asarray(ods["magnetics.flux_loop.:.position.0.r"], float)
    )
    expected_in = [i for i, name in enumerate(regions) if name == INBOARD]
    expected_out = [i for i, name in enumerate(regions) if name == OUTBOARD]

    assert sorted(_find_flux_loop_inboard_indices(ods)[0].tolist()) == expected_in
    assert sorted(_find_flux_loop_outboard_indices(ods)[0].tolist()) == expected_out


def test_the_midplane_loop_is_found_by_geometry_not_by_a_literal():
    from vaft.plot.time import _find_flux_loop_inboard_midplane_indices

    ods = _sample()
    chosen = _find_flux_loop_inboard_midplane_indices(ods)[0]
    assert len(chosen) == 1
    z = float(ods[f"magnetics.flux_loop.{int(chosen[0])}.position.0.z"])
    inboard_z = [
        abs(float(ods[f"magnetics.flux_loop.{i}.position.0.z"]))
        for i, name in enumerate(
            classify_regions(np.asarray(ods["magnetics.flux_loop.:.position.0.r"], float))
        )
        if name == INBOARD
    ]
    assert abs(z) == pytest.approx(min(inboard_z))
