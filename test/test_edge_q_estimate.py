"""Edge-q estimates for shots without an equilibrium (#1583)."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula import boundaries as B
from vaft.formula.equilibrium import (
    estimated_q95,
    normalized_plasma_current,
    q_star_cylindrical,
    q_star_kink,
)
from vaft.machine_mapping.edge_q_estimate import SHAPE_KEYS, vest_edge_q_estimate_policy
from vaft.machine_mapping.utils import VestConfigurationError

# ITER design point (Post et al. 1991): R = 6.2 m, a = 2.0 m, B = 5.3 T, I_p = 15 MA,
# kappa_95 = 1.7, delta_95 = 0.33 -> q95 = 3.0.
ITER = dict(a=2.0, R0=6.2, B0=5.3, kappa=1.70, delta=0.33, I_p=15e6)
VEST = dict(a=0.274, R0=0.379, B0=0.15, kappa=1.51, delta=0.30, I_p=100e3)


def _f_start(A, c=1.0):
    return 1.17 * c * np.sqrt(A / (A - 1.0))


def _f_iter(A):
    return (1.17 - 0.65 / A) / (1.0 - 1.0 / A**2) ** 2


def test_the_iter_design_point_gives_q95_three():
    assert estimated_q95(**ITER, scaling="iter") == pytest.approx(3.0, abs=0.01)


@pytest.mark.parametrize("point", [ITER, VEST])
def test_start_over_iter_is_the_ratio_of_the_aspect_ratio_functions(point):
    A = point["R0"] / point["a"]
    ratio = estimated_q95(**point, scaling="start") / estimated_q95(**point, scaling="iter")
    assert ratio == pytest.approx(_f_start(A) / _f_iter(A), rel=1e-12)


def test_double_null_scales_the_start_estimate_by_c_077():
    limiter = estimated_q95(**VEST, configuration="limiter")
    double_null = estimated_q95(**VEST, configuration="double_null")
    assert double_null / limiter == pytest.approx(0.77, rel=1e-12)


def test_the_si_front_delegates_to_the_registered_coordinates_in_ma():
    ma = VEST["I_p"] * 1e-6
    shape = (VEST["a"], VEST["R0"], VEST["B0"], VEST["kappa"])
    assert estimated_q95(**VEST) == pytest.approx(B.start_q95_coordinates(*shape, VEST["delta"], ma))
    assert estimated_q95(**VEST, scaling="iter") == pytest.approx(B.iter_q95_coordinates(*shape, VEST["delta"], ma))
    assert q_star_cylindrical(*shape, VEST["I_p"]) == pytest.approx(B.cylindrical_kink_coordinates(*shape, ma))
    assert q_star_kink(*shape, VEST["I_p"]) == pytest.approx(B.kink_coordinates(*shape, ma))


def test_the_current_sign_is_dropped_and_arrays_broadcast():
    q = estimated_q95(VEST["a"], VEST["R0"], VEST["B0"], VEST["kappa"], VEST["delta"], np.array([-1e5, 1e5, 2e5]))
    assert q[0] == pytest.approx(q[1])
    assert q[2] == pytest.approx(q[1] / 2)


@pytest.mark.parametrize("kwargs, match", [
    (dict(scaling="uckan"), "scaling"),
    (dict(configuration="single_null"), "configuration"),
    (dict(scaling="iter", configuration="double_null"), "START scaling only"),
])
def test_unknown_options_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        estimated_q95(**VEST, **kwargs)


def test_the_vest_policy_comes_from_the_machine_description():
    policy = vest_edge_q_estimate_policy()
    assert (policy.scaling, policy.configuration) == ("start", "limiter")
    assert set(policy.default_shape) == set(SHAPE_KEYS)
    assert policy.default_shape["minor_radius"] == pytest.approx(0.274)
    assert policy.default_shape["major_radius"] == pytest.approx(0.379)
    assert policy.default_shape["elongation"] == pytest.approx(1.51)
    assert policy.default_shape["triangularity"] == pytest.approx(0.30)
    assert "Akers" in policy.provenance["scaling"]


def test_a_policy_without_its_shape_is_rejected(tmp_path):
    path = tmp_path / "machine.yaml"
    path.write_text("edge_q_estimate:\n  scaling: start\n  configuration: limiter\n"
                    "  status: assumed\n  provenance: test\n")
    with pytest.raises(VestConfigurationError, match="default_shape"):
        vest_edge_q_estimate_policy(info_file=str(path))


# ------------------------------------------------------------------
# ODS extraction on the packaged sample
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample():
    from vaft.omas.sample import sample_ods

    return sample_ods(39915)


def test_with_an_equilibrium_the_estimate_tracks_its_q95(sample):
    from vaft.omas.edge_q import edge_q_estimate

    result = edge_q_estimate(sample)
    assert result.source == "equilibrium"
    assert result.label == "q95 (START estimate)"
    assert result.equilibrium_q95 is not None
    ratio = result.estimated_q95 / np.interp(result.time, result.equilibrium_time, result.equilibrium_q95)
    ratio = ratio[np.isfinite(ratio)]
    assert ratio.size >= 5
    # #1580 measured 0.99 (IQR 0.95-1.03) over 133 Tier A states; 39915's shrinking
    # late slices sit above that, so the band is the population's, not the IQR.
    assert 0.9 < np.median(ratio) < 1.15
    assert np.all((ratio > 0.85) & (ratio < 1.3))


def test_without_an_equilibrium_the_default_shape_is_named(sample):
    from vaft.omas.edge_q import edge_q_estimate

    result = edge_q_estimate(sample, source="magnetics")
    assert result.source == "magnetics"
    assert "vest.yaml:edge_q_estimate.default_shape" in result.shape_source
    assert "default_shape" in result.provenance
    assert result.current_source == "magnetics.ip.0"
    finite = np.isfinite(result.estimated_q95)
    assert finite.any()
    assert np.all(np.abs(result.plasma_current[finite]) >= result.current_threshold)


def test_a_caller_shape_is_reported_and_must_be_complete(sample):
    from vaft.omas.edge_q import edge_q_estimate

    shape = dict(minor_radius=0.25, major_radius=0.36, elongation=1.6, triangularity=0.2)
    result = edge_q_estimate(sample, source="magnetics", shape=shape)
    assert result.shape_source == "the caller's shape"
    assert np.all(result.shape["elongation"] == 1.6)
    with pytest.raises(ValueError, match="triangularity"):
        edge_q_estimate(sample, source="magnetics", shape={k: shape[k] for k in SHAPE_KEYS[:3]})


def test_the_other_proxies_keep_their_own_names_and_values(sample):
    from vaft.omas.edge_q import edge_q_estimate

    result = edge_q_estimate(sample, scaling="iter")
    assert result.label == "q95 (ITER estimate)"
    a, R, kappa = (result.shape[k] for k in ("minor_radius", "major_radius", "elongation"))
    ok = np.isfinite(result.estimated_q95)
    np.testing.assert_allclose(result.q_star_kink[ok],
                               q_star_kink(a, R, result.toroidal_field, kappa, result.plasma_current)[ok])
    np.testing.assert_allclose(result.normalized_current[ok],
                               np.abs(normalized_plasma_current(result.plasma_current, R, a,
                                                                result.toroidal_field))[ok])
    assert not hasattr(result, "q_a")


# ------------------------------------------------------------------
# Plot layer: registry, facade and labels
# ------------------------------------------------------------------

EDGE_Q_PLOTS = ("summary_time_estimated_q95", "summary_time_q_star_cylindrical",
                "summary_time_q_star_kink", "summary_time_normalized_current")


def test_the_views_are_registered_and_reach_the_omas_facade():
    import vaft.omas as vomas
    from vaft.plot import registry

    for name in EDGE_Q_PLOTS:
        spec = registry.get_spec(name)
        assert (spec.subject, spec.view) == ("summary", "time")
        assert callable(getattr(vomas, f"plot_{name}"))
        assert "q_a" not in name


def test_the_estimate_is_labelled_and_the_equilibrium_q95_overlaid(sample):
    import vaft.omas as vomas

    model = vomas.extract_summary_time_estimated_q95(sample)
    labels = [series.label for series in model.series]
    assert model.title.startswith("q95 (START estimate)")
    assert labels[0].startswith("q95 (START estimate)")
    assert "q95 (equilibrium)" in labels


def test_without_an_equilibrium_the_legend_says_default_shape(sample):
    import vaft.omas as vomas

    model = vomas.extract_summary_time_estimated_q95(sample, estimate_from="magnetics", scaling="iter")
    assert model.series[0].label == "q95 (ITER estimate), from magnetics, default shape"


def test_the_renderer_returns_figure_and_axes(sample):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import vaft.omas as vomas

    for name in EDGE_Q_PLOTS:
        figure, axes = getattr(vomas, f"plot_{name}")(sample)
        assert axes.get_ylabel()
        plt.close(figure)
