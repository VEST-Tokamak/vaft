"""Direct labels instead of legends on the multi-machine figures (#1946)."""

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from vaft.plot._direct_labels import LABEL_GID, LEADER_GID, DirectLabel, add_direct_labels  # noqa: E402

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")


class _Boundary:
    def __init__(self, r, z):
        self.r, self.z = r, z


class _Equilibrium:
    def __init__(self, r0, a, kappa, z0=0.0):
        theta = np.linspace(0, 2 * np.pi, 181)
        self.lcfs = _Boundary(r0 + a * np.cos(theta + 0.3 * np.sin(theta)), z0 + kappa * a * np.sin(theta))
        self.limiter = _Boundary(r0 + 1.15 * a * np.cos(theta), z0 + 1.15 * kappa * a * np.sin(theta))
        self.magnetic_axis = (r0 + 0.05 * a, z0)


#: Seven machines, five of them nested within 1.5 m of the origin.
MACHINES = {
    "VEST": _Equilibrium(0.4, 0.3, 1.6),
    "TCV": _Equilibrium(0.88, 0.25, 1.4),
    "MAST-U": _Equilibrium(0.85, 0.6, 1.8),
    "NSTX": _Equilibrium(0.9, 0.62, 2.0),
    "DIII-D": _Equilibrium(1.7, 0.6, 1.8),
    "JET": _Equilibrium(2.9, 1.0, 1.7, 0.3),
    "ITER": _Equilibrium(6.2, 2.0, 1.8, 0.5),
}


def _labels(ax):
    return [t for t in ax.texts if str(t.get_gid() or "").startswith(LABEL_GID)]


def _leaders(ax):
    return {line.get_gid()[len(LEADER_GID):]: line for line in ax.lines
            if str(line.get_gid() or "").startswith(LEADER_GID)}


def _extents(fig, texts):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [t.get_window_extent(renderer) for t in texts]


def _assert_no_overlap(boxes):
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b), (a, b)


def _assert_inside(ax, boxes):
    frame = ax.bbox
    for box in boxes:
        assert box.x0 >= frame.x0 - 0.5 and box.x1 <= frame.x1 + 0.5
        assert box.y0 >= frame.y0 - 0.5 and box.y1 <= frame.y1 + 0.5


def _crowded():
    """Seven clusters, five piled near the origin, labelled through the helper."""
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(5, 4))
    specs = []
    centres = [(0.1, 0.1), (0.15, 0.12), (0.12, 0.2), (0.2, 0.15), (0.18, 0.08), (2.0, 2.0), (3.0, 1.0)]
    for i, (cx, cy) in enumerate(centres):
        x, y = cx + 0.05 * rng.standard_normal(40), cy + 0.05 * rng.standard_normal(40)
        ax.scatter(x, y, color=COLORS[i], s=8)
        specs.append(DirectLabel(f"machine {i}", COLORS[i], x, y, "above"))
    ax.set_xlim(-0.2, 3.5)
    ax.set_ylim(-0.2, 2.5)
    texts = add_direct_labels(ax, specs)
    return fig, ax, texts


def test_labels_take_the_series_colour_and_name_it():
    fig, ax, texts = _crowded()
    assert [t.get_text() for t in texts] == [f"machine {i}" for i in range(7)]
    for i, text in enumerate(texts):
        assert to_rgba(text.get_color()) == to_rgba(COLORS[i])
        assert text.get_gid() == f"{LABEL_GID}machine {i}"
    plt.close(fig)


def test_crowded_labels_do_not_overlap_and_stay_inside_the_axes():
    fig, ax, texts = _crowded()
    boxes = _extents(fig, texts)
    _assert_no_overlap(boxes)
    _assert_inside(ax, boxes)
    plt.close(fig)


def test_placement_is_deterministic():
    first = _crowded()
    second = _crowded()
    a = [t.get_position() for t in first[2]]
    first[0].canvas.draw()
    second[0].canvas.draw()
    assert [t.get_position() for t in first[2]] == [t.get_position() for t in second[2]]
    assert a != [t.get_position() for t in first[2]]  # the draw did place them
    plt.close(first[0])
    plt.close(second[0])


def test_a_displaced_label_gets_a_leader_in_its_colour_from_its_anchor():
    fig, ax = plt.subplots(figsize=(4, 3))
    # One point boxed in by a dense grid of other data: the label must move away.
    gx, gy = np.meshgrid(np.linspace(0.3, 0.7, 40), np.linspace(0.3, 0.7, 40))
    ax.scatter(gx.ravel(), gy.ravel(), s=2, color="grey")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_direct_labels(ax, [DirectLabel("target", "#eb6834", [0.5], [0.5]),
                           DirectLabel("grid", "grey", gx.ravel(), gy.ravel())])
    fig.canvas.draw()
    leader = _leaders(ax)["target"]
    assert leader.get_visible()
    assert to_rgba(leader.get_color()) == to_rgba("#eb6834")
    start = ax.transAxes.transform(leader.get_xydata()[0])
    assert np.allclose(start, ax.transData.transform((0.5, 0.5)), atol=1.0)
    plt.close(fig)


def test_layout_after_labelling_keeps_labels_apart():
    # The renderer is called first, the figure is laid out afterwards.
    fig, ax, texts = _crowded()
    fig.set_size_inches(3.0, 2.6)
    fig.tight_layout()
    boxes = _extents(fig, texts)
    _assert_no_overlap(boxes)
    _assert_inside(ax, boxes)
    plt.close(fig)


def test_a_label_wider_than_the_axes_is_shrunk_inside_them():
    fig, ax = plt.subplots(figsize=(2, 2))
    text, = add_direct_labels(ax, [DirectLabel("a very long label that cannot fit in two inches at all",
                                               "k", [0.5], [0.5])])
    _assert_inside(ax, _extents(fig, [text]))
    plt.close(fig)


def test_machine_lcfs_comparison_names_every_machine_without_a_legend():
    from vaft.plot.population import CATEGORICAL, machine_lcfs_comparison

    fig, ax = machine_lcfs_comparison(MACHINES, figsize=(6, 5))
    assert ax.get_legend() is None
    texts = _labels(ax)
    assert [t.get_text() for t in texts] == list(MACHINES)
    for i, text in enumerate(texts):
        assert to_rgba(text.get_color()) == to_rgba(CATEGORICAL[i % len(CATEGORICAL)])
    boxes = _extents(fig, texts)
    _assert_no_overlap(boxes)
    _assert_inside(ax, boxes)
    # Machines nest near the origin: at least one label needs a leader.
    assert any(line.get_visible() for line in _leaders(ax).values())
    plt.close(fig)


def test_machine_lcfs_comparison_normalized_view_and_legend_option():
    from vaft.plot.population import machine_lcfs_comparison

    scales = {"VEST": (0.4, 0.3), "JET": (2.9, 1.0)}
    fig, ax = machine_lcfs_comparison(MACHINES, normalized_by=scales)
    assert [t.get_text() for t in _labels(ax)] == ["VEST", "JET"]
    assert "a" in ax.get_xlabel()
    plt.close(fig)
    fig, ax = machine_lcfs_comparison(MACHINES, labels="legend")
    assert ax.get_legend() is not None and not _labels(ax)
    plt.close(fig)
    with pytest.raises(ValueError):
        machine_lcfs_comparison(MACHINES, labels="sideways")


def _population():
    rng = np.random.default_rng(3)
    rows = []
    for machine, n, ip, tau in (("JET", 300, 2e6, 0.3), ("AUG", 200, 8e5, 0.08), ("D3D", 150, 1.2e6, 0.1),
                                ("NSTX", 80, 8e5, 0.04), ("MAST", 60, 7e5, 0.03), ("START", 20, 2e5, 0.004),
                                ("TCV", 20, 3e5, 0.01), ("VEST", 30, 1e5, 1e-3)):
        for _ in range(n):
            rows.append({"machine": machine, "i_p_A": ip * np.exp(0.3 * rng.standard_normal()),
                         "b_t_T": 1.0, "tau_e_th_s": tau * np.exp(0.3 * rng.standard_normal())})
    table = pd.DataFrame(rows)
    predicted = table["tau_e_th_s"] * np.exp(0.2 * rng.standard_normal(len(table)))
    return table, predicted


@pytest.mark.parametrize("renderer", ["confinement_population", "confinement_predicted_vs_measured"])
def test_population_scatters_label_clusters_instead_of_a_legend(renderer):
    from vaft.plot import population

    table, predicted = _population()
    draw = getattr(population, renderer)
    args = (table, predicted) if renderer == "confinement_predicted_vs_measured" else (table,)
    fig, ax = draw(*args, max_groups=4)
    assert ax.get_legend() is None
    names = [t.get_text() for t in _labels(ax)]
    # Four coloured groups, the folded "Other" once, and the highlighted VEST.
    assert len(names) == 6
    assert sum(name.startswith("Other") for name in names) == 1
    assert any(name.startswith("VEST") for name in names)
    other = next(t for t in _labels(ax) if t.get_text().startswith("Other"))
    assert to_rgba(other.get_color()) == to_rgba(population.OTHER_TEXT_COLOR)
    boxes = _extents(fig, _labels(ax))
    _assert_no_overlap(boxes)
    _assert_inside(ax, boxes)
    plt.close(fig)
    fig, ax = draw(*args, max_groups=4, labels="legend")
    assert ax.get_legend() is not None and not _labels(ax)
    plt.close(fig)


def _base(key):
    from matplotlib.font_manager import FontProperties

    return FontProperties(size=matplotlib.rcParams[key]).get_size_in_points()


def test_population_axis_and_tick_text_is_larger_than_the_rc_base():
    from vaft.plot import population

    table, predicted = _population()
    fig, ax = population.confinement_predicted_vs_measured(table, predicted)
    assert ax.xaxis.label.get_fontsize() > _base("axes.labelsize")
    assert ax.yaxis.label.get_fontsize() > _base("axes.labelsize")
    fig.canvas.draw()
    assert ax.get_xticklabels()[0].get_fontsize() > _base("xtick.labelsize")
    plt.close(fig)


def test_coverage_strip_text_is_larger_and_scales_with_the_presentation_format():
    from vaft.plot import population
    from vaft.plot.presentation import resolve_presentation

    table, _ = _population()
    table["r_geo_m"] = 1.0
    fig, axes = population.confinement_coverage_strip(table, columns=("i_p_A", "r_geo_m"))
    screen_label = axes[0].xaxis.label.get_fontsize()
    row = axes[0].get_yticklabels()[0].get_fontsize()
    assert screen_label > _base("axes.labelsize") and row > _base("ytick.labelsize")
    fig.canvas.draw()
    assert axes[1].get_xticklabels()[0].get_fontsize() > _base("xtick.labelsize")
    plt.close(fig)
    with resolve_presentation("slide", None).context():
        fig, axes = population.confinement_coverage_strip(table, columns=("i_p_A", "r_geo_m"))
        assert axes[0].xaxis.label.get_fontsize() > 1.5 * screen_label
        plt.close(fig)
