"""Stability atlas population reader and its 3x1 histogram figure (#1852).

Synthetic atlas tables with known answers; no atlas file is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaft.process.mhd_stability import load_stability_atlas, stability_atlas_populations

KEY = {"efit_lineage": "magnetics"}


def _atlas():
    rows, surfaces = [], []
    for shot, w in ((1, [1.0, -0.5]), (2, [2.0, 0.3])):
        for n, w_t in zip((1, 2), w):
            rows.append({"shot": shot, "time_efit_s": 0.3, **KEY, "n_tor": n, "dcon_full_512_W_t": w_t,
                         "dcon_full_256_W_t": 99.0,  # never read: the population is at mpsi 512
                         "dcon_full_sign_class": "robust_unstable" if w_t < 0 else (
                             "numerically_unresolved" if shot == 2 and n == 2 else "robust_stable"),
                         "max_D_I": -0.2 if shot == 1 else 0.4, "min_C_A": 0.1 if shot == 1 else -0.3,
                         "mercier_evaluated": True, "ballooning_evaluated": shot == 1})
    # Slice 1 n=1 RDCON: the largest Delta' is the second (inconsistent) surface.
    for shot, solver, n, values, consistent in (
        (1, "rdcon", 1, (-5.0, 12.0), (True, False)),
        (2, "rdcon", 1, (-3.0, -1.0), (True, True)),
        (1, "stride", 2, (7.0,), (True,)),
    ):
        for m, (dp, ok) in enumerate(zip(values, consistent), start=2):
            surfaces.append({"shot": shot, "time_efit_s": 0.3, **KEY, "n_tor": n, "solver": solver, "m": m,
                             "delta_prime": 1e9, "delta_prime_mpsi512": dp, "two_resolution_consistent": ok})
    return pd.DataFrame(rows), pd.DataFrame(surfaces)


@pytest.fixture()
def populations():
    return stability_atlas_populations(*_atlas())


def test_ideal_is_every_row_at_mpsi_512_with_qa_only_counted(populations):
    np.testing.assert_array_equal(np.sort(populations.ideal_w_t[1]), [1.0, 2.0])
    np.testing.assert_array_equal(np.sort(populations.ideal_w_t[2]), [-0.5, 0.3])
    row = populations.summary.set_index("series").loc["n=2"]
    # The unresolved row (slice 2) stays in the values and is only missing from the QA count.
    assert (row.slices, row.qa, row.unstable) == (2, 1, 1)
    assert populations.summary.set_index("series").loc["n=1"].unstable == 0


def test_resistive_takes_the_largest_delta_prime_of_each_slice(populations):
    np.testing.assert_array_equal(np.sort(populations.delta_prime_max[("rdcon", 1)]), [-1.0, 12.0])
    row = populations.summary.set_index("series").loc["RDCON n=1"]
    # The maximizing surface of slice 1 is not two-resolution consistent.
    assert (row.slices, row.qa, row.unstable) == (2, 1, 1)
    assert ("stride", 2) in populations.delta_prime_max and ("stride", 1) not in populations.delta_prime_max


def test_local_criteria_come_from_the_lowest_n_and_only_where_evaluated(populations):
    np.testing.assert_array_equal(np.sort(populations.mercier_max_d_i), [-0.2, 0.4])
    np.testing.assert_array_equal(populations.ballooning_min_c_a, [0.1])  # slice 2 not evaluated
    summary = populations.summary.set_index("series")
    assert summary.loc["max D_I"].unstable == 1 and summary.loc["min C_A"].unstable == 0


def test_the_summary_keeps_the_interpretation_out_of_the_values(populations):
    assert list(populations.summary.columns) == ["panel", "series", "criterion", "slices", "qa", "unstable"]
    assert populations.provenance["resolution"] == "mpsi512"


def test_load_reads_both_tables_and_names_a_missing_one(tmp_path):
    atlas_n, atlas_surfaces = _atlas()
    atlas_n.to_csv(tmp_path / "atlas_n.csv", index=False)
    with pytest.raises(FileNotFoundError, match="atlas_surfaces"):
        load_stability_atlas(tmp_path)
    atlas_surfaces.to_csv(tmp_path / "atlas_surfaces.csv", index=False)
    read_n, read_surfaces = load_stability_atlas(tmp_path)
    assert len(read_n) == len(atlas_n) and len(read_surfaces) == len(atlas_surfaces)


def test_the_figure_has_three_panels_with_zero_lines_and_zones(populations):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from vaft.plot.stability_atlas import stability_atlas_population

    fig, axes = stability_atlas_population(populations)
    assert isinstance(axes, np.ndarray) and len(axes) == 3
    texts = [[t.get_text() for t in ax.texts] for ax in axes]
    assert {"stable", "unstable"} <= set(texts[0]) and {"stable", "unstable"} <= set(texts[1])
    assert any("Mercier" in t for t in texts[2]) and any("ballooning" in t for t in texts[2])
    for ax in axes:
        assert ax.get_xscale() == "symlog"
        assert any(np.allclose(line.get_xdata(), 0.0) for line in ax.get_lines())  # the stability boundary
        assert ax.get_title() == "" and ax.get_legend() is None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_the_figure_refuses_the_wrong_number_of_axes(populations):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vaft.plot.stability_atlas import stability_atlas_population

    fig, axes = plt.subplots(2, 1)
    with pytest.raises(ValueError, match="three panels"):
        stability_atlas_population(populations, ax=axes)
    plt.close(fig)


def test_a_value_beyond_the_bins_is_still_drawn(populations, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import dataclasses

    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    from vaft.plot.stability_atlas import stability_atlas_population

    drawn = []
    hist = Axes.hist

    def recording_hist(self, x, bins=None, **kwargs):
        result = hist(self, x, bins=bins, **kwargs)
        drawn.append((len(x), result[0].sum()))
        return result

    monkeypatch.setattr(Axes, "hist", recording_hist)
    # 2e15 is past the last resistive bin; every counted slice must still be drawn.
    wide = dataclasses.replace(populations, delta_prime_max={("rdcon", 1): np.array([2e15, -1.0])})
    fig, _ = stability_atlas_population(wide)
    plt.close(fig)
    assert drawn and all(given == counted for given, counted in drawn)
