"""Ordering spaces and ordering margins (#1627 §3): registry, labels and the two plots."""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from vaft.plot.ordering_space import ordering_margins_plot, ordering_space  # noqa: E402
from vaft.validation.orderings import (  # noqa: E402
    CONTRACTS,
    MARGIN_LABELS,
    ORDERING_QUANTITIES,
    ORDERING_SPACES,
    SYMBOLS,
    OrderingSpace,
    margin_label,
    required_ordering,
)


def test_every_quantity_has_a_symbol_and_one_required_side():
    assert set(SYMBOLS) == set(ORDERING_QUANTITIES)
    for q in ORDERING_QUANTITIES:
        sides = {a.ordering for c in CONTRACTS.values() for a in c.assumptions if a.quantity == q}
        assert required_ordering(q) == sides.pop() and not sides


def test_the_issue_spaces_are_registered_with_their_axes():
    axes = {name: (s.x, s.y) for name, s in ORDERING_SPACES.items()}
    assert axes["resistive_two_fluid"] == ("ion_skin_depth_over_a", "lundquist_number")
    assert axes["closure_magnetization"] == ("electron_parallel_knudsen_number", "electron_magnetization")
    assert axes["flr_collisional"] == ("ion_parallel_knudsen_number", "rho_i_over_LTi")
    assert axes["flr_collisional_electron"] == ("electron_parallel_knudsen_number", "rho_s_over_LTe")
    assert axes["equilibrium_relaxation"] == ("tau_age_over_tau_resistive", "tau_evolution_over_tau_alfven")
    with pytest.raises(KeyError):
        OrderingSpace("bad", "not_a_quantity", "beta", "t", ("a", "b"), ("c", "d"))


@pytest.mark.parametrize("margin, label", [(3.0, "over 2 decades"), (2.0, "1-2 decades"), (1.5, "1-2 decades"),
                                           (0.5, "0-1 decade"), (0.0, "outside"), (-0.01, "outside"),
                                           (math.nan, "unassessed")])
def test_margin_labels_are_documented_bands(margin, label):
    assert margin_label(margin) == label
    assert [b for b, _ in MARGIN_LABELS] == sorted([b for b, _ in MARGIN_LABELS], reverse=True)
    # band names, not statuses: the only shared word is "outside", whose band (m <= 0) is exactly OUTSIDE
    from vaft.validation.applicability import APPLICABILITY_STATUSES

    names = {label.lower() for _, label in MARGIN_LABELS}
    assert names & {s.lower() for s in APPLICABILITY_STATUSES} == {"outside"} and "marginal" not in names


def _states():
    return pd.DataFrame({"ion_skin_depth_over_a": [0.2, 0.5, 2.0, np.nan, -1.0],
                         "lundquist_number": [1e5, 3e4, 1e5, 1e5, 1e5],
                         "efit_quality": ["good", "good", "admissible", "good", "good"]})


def test_ordering_space_places_assessable_rows_and_counts_the_rest():
    fig, ax = ordering_space(_states(), "resistive_two_fluid", group="efit_quality")
    assert ax.ordering_exclusions == {"total": 5, "placed": 3, "unassessed": 2}
    assert ax.get_xscale() == ax.get_yscale() == "log"
    # the order-unity lines are drawn on both axes, and the axis names say which side is required
    assert "\\ll 1" in ax.get_xlabel() and "\\gg 1" in ax.get_ylabel()
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "ideal\nsingle-fluid" in texts and "resistive\nHall / two-fluid" in texts
    assert "3 of 5 placed, 2 unassessed" in texts
    plt.close(fig)


def test_ordering_space_uses_given_axes_and_refuses_unknown_inputs():
    fig, ax = plt.subplots()
    assert ordering_space(_states(), "resistive_two_fluid", ax=ax) == (fig, ax)
    with pytest.raises(KeyError, match="no ordering space"):
        ordering_space(_states(), "stability")
    with pytest.raises(KeyError, match="lundquist_number"):
        ordering_space(_states().drop(columns="lundquist_number"), "resistive_two_fluid")
    plt.close(fig)


@pytest.mark.parametrize("x, y", [("ion_skin_depth_over_a", "lundquist_number"),        # small, large
                                  ("electron_parallel_knudsen_number", "rho_s_over_LTe"),  # small, small
                                  ("tau_age_over_tau_resistive", "tau_evolution_over_tau_alfven"),  # large, large
                                  ("lundquist_number", "ion_skin_depth_over_a")])       # large, small
def test_the_both_hold_label_sits_in_the_shaded_permitted_region(x, y):
    space = OrderingSpace("t", x, y, "t", ("xh", "xf"), ("yh", "yf"))
    fig, ax = ordering_space(pd.DataFrame({x: [0.5], y: [0.5]}), space, decades=(1.0, 3.0))
    shaded = ax.collections[0].get_paths()[0].get_extents()
    holds = [t for t in ax.texts if t.get_text() == "yh\nxh"][0]
    px, py = holds.get_position()
    assert shaded.x0 <= px <= shaded.x1 and shaded.y0 <= py <= shaded.y1
    for value, side in ((px, required_ordering(x)), (py, required_ordering(y))):
        assert (value < 1.0) if side == "small" else (value > 1.0)
    # every requested margin line is inside the axes
    lo, hi = ax.get_xlim()
    assert lo < (1e-3 if required_ordering(x) == "small" else 1e3) < hi
    plt.close(fig)


def test_margins_plot_reports_fraction_and_count_per_ordering():
    fig, ax = ordering_margins_plot(_states(), quantities=["lundquist_number", "ion_skin_depth_over_a"])
    texts = [t.get_text() for t in ax.texts]
    assert "100% hold, n=5" in texts           # S = 3e4..1e5 >> 1 everywhere
    assert "67% hold, n=3" in texts            # d_i/a: 0.2 and 0.5 hold, 2.0 fails, two unassessable
    assert {"over 2 decades", "1-2 decades", "0-1 decade", "outside"} <= set(texts)
    with pytest.raises(ValueError):
        ordering_margins_plot(pd.DataFrame({"x": [1.0]}))
    plt.close(fig)


def test_margins_plot_counts_global_orderings_per_state_like_ordering_margins():
    from vaft.process.ordering_state import ordering_margins

    states = pd.DataFrame({"shot": [1, 2], "lundquist_number": [1e5, 0.5]})
    # three profile points per state, each carrying its state's S as a joined table would
    points = pd.DataFrame({"shot": [1, 1, 1, 2, 2, 2], "lundquist_number": [1e5] * 3 + [0.5] * 3,
                           "rho_s_over_LTe": [0.01, 0.02, 2.0, 0.05, 0.1, 0.3]})
    fig, ax = ordering_margins_plot(states, points, quantities=["lundquist_number", "rho_s_over_LTe"])
    texts = [t.get_text() for t in ax.texts]
    reference = ordering_margins(CONTRACTS, states, points)
    s_row = reference.loc["lundquist_number (large)"]
    rho_row = reference.loc["rho_s_over_LTe (small)"]
    assert (s_row.per, int(s_row.n)) == ("state", 2)
    assert f"{100 * s_row.permitted:.0f}% hold, n={int(s_row.n)}" in texts        # 50%, n=2, not n=6
    assert f"{100 * rho_row.permitted:.0f}% hold, n={int(rho_row.n)}" in texts    # per point
    plt.close(fig)


def test_a_sourced_threshold_moves_the_lines_and_the_margin(monkeypatch):
    import vaft.validation.orderings as orderings

    monkeypatch.setattr(orderings, "ordering_threshold",
                        lambda q: 0.1 if q == "ion_skin_depth_over_a" else 1.0)
    fig, ax = ordering_space(_states(), "resistive_two_fluid")
    assert 0.1 in [line.get_xdata()[0] for line in ax.lines]       # the threshold line, not x = 1
    assert "\\ll 0.1" in ax.get_xlabel()
    plt.close(fig)
    fig, ax = ordering_margins_plot(_states(), quantities=["ion_skin_depth_over_a"])
    assert "0% hold, n=3" in [t.get_text() for t in ax.texts]      # 0.2, 0.5 and 2.0 all exceed 0.1
    plt.close(fig)
