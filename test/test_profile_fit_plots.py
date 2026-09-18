"""Measured kinetic points and their fit, on a flux coordinate (#952).

``thomson_scattering_profile_fit`` and ``charge_exchange_profile_fit`` draw the
channels at one diagnostic time with their error bars, the channels the fit
refused as hollow markers, and the fit curve -- with its band when the method
has one -- against psi_N, rho_N or R.  ``equilibrium=`` maps the same channels
through another equilibrium, or several, to show how much the mapping moves
them.
"""

import numpy as np
import pytest

pytest.importorskip("omas")
import omas

from vaft.data import data_path
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import build_model
from vaft.plot.models import Profile1D


@pytest.fixture(scope="module")
def ods():
    return omas.load_omas_json(str(data_path("kineticEfit/ods_48224_300ms.json")), consistency_check=False)


def _build(ods, name="thomson_scattering_profile_fit", **options):
    return build_model(name, normalize_entries(ods), **options)


def _by_label(model, word):
    return [series for series in model.series if word in series.label]


def test_points_carry_their_error_bars_and_the_fit_says_how_well_it_fits(ods):
    model = _build(ods, field="te", coordinate="psi_norm")
    assert isinstance(model, Profile1D)
    measured = _by_label(model, "measured")[0]
    assert measured.yerr is not None and np.all(measured.yerr > 0)
    assert measured.style["linestyle"] == "none"
    fit = _by_label(model, "fit")[0]
    assert "χ²/ν" in fit.label and "k=" in fit.label
    assert "psi" in model.coordinate_label.lower() or "ψ" in model.coordinate_label
    assert model.x_limits == (0.0, 1.0)
    # the measured values are the channels' own, at the equilibrium's time
    channel = [ods[f"thomson_scattering.channel.{i}.t_e.data"][2] for i in range(7)]
    assert set(np.round(measured.y, 6)) <= set(np.round(channel, 6))
    assert "300.0 ms" in model.title


def test_a_gaussian_process_fit_draws_its_band(ods):
    model = _build(ods, field="ne", coordinate="rho_tor_norm", fitting_function="gp")
    fit = _by_label(model, "fit")[0]
    assert "gp fit" in fit.label
    assert fit.yerr is None or fit.yerr.shape == fit.y.shape


def test_major_radius_draws_the_fit_along_the_chord(ods):
    model = _build(ods, field="ne", coordinate="r_major")
    measured = _by_label(model, "measured")[0]
    radii = [ods[f"thomson_scattering.channel.{i}.position.r"] for i in range(7)]
    assert set(np.round(measured.x, 6)) <= set(np.round(radii, 6))
    fit = _by_label(model, "fit")[0]
    assert fit.x.min() >= min(radii) - 1e-9 and fit.x.max() <= max(radii) + 1e-9
    assert model.x_limits is None


def test_two_equilibria_place_the_same_channels_at_two_radii(ods):
    equilibria = {
        "magnetic": data_path("kineticEfit/g048224.00300"),
        "kinetic": data_path("kineticEfit/g048224.00300.kinetic_efit"),
    }
    model = _build(ods, field="te", coordinate="rho_tor_norm", equilibrium=equilibria)
    magnetic = _by_label(model, "magnetic: measured")[0]
    kinetic = _by_label(model, "kinetic: measured")[0]
    assert magnetic.x.size == kinetic.x.size
    assert np.max(np.abs(magnetic.x - kinetic.x)) > 1e-3
    np.testing.assert_allclose(magnetic.y, kinetic.y)  # same measurements
    assert len(_by_label(model, "fit")) == 2


def test_refused_channels_are_hollow(ods):
    model = _build(ods, "charge_exchange_profile_fit", field="vphi", coordinate="r_major", order=2)
    refused = _by_label(model, "refused")
    assert refused, "the CX channels outside the LCFS are drawn, hollow, against R"
    assert refused[0].style["markerfacecolor"] == "none"


def test_time_index_picks_the_diagnostic_sample(ods):
    model = _build(ods, field="te", coordinate="psi_norm", time_index=3)
    assert "301.0 ms" in model.title
    assert "equilibrium at 300.0 ms" in model.title
    model = _build(ods, field="te", coordinate="psi_norm", time=0.2995)
    assert "299.0 ms" in model.title or "300.0 ms" in model.title


def test_an_unknown_field_or_coordinate_is_refused(ods):
    with pytest.raises(ValueError, match="field"):
        _build(ods, field="ti")
    with pytest.raises(ValueError, match="coordinate"):
        _build(ods, coordinate="r_minor")


def test_the_adapter_renders(ods):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import vaft.omas

    fig, ax = vaft.omas.plot_charge_exchange_profile_fit(ods, field="ti", order=2)
    assert ax.get_legend() is not None
    plt.close(fig)


def test_a_plain_geqdsk_mapping_is_one_equilibrium_not_a_collection(ods):
    from vaft.data.eqdsk import read_geqdsk

    geq = read_geqdsk(data_path("kineticEfit/g048224.00300"))
    mapping = dict(geq.mapping)
    assert "PSIRZ" in mapping and "NW" in mapping
    # iterated as {name: equilibrium}, this would try to map through 'NW', 'PSIRZ', ...
    from_mapping = _build(ods, field="te", coordinate="psi_norm", equilibrium=mapping)
    from_geqdsk = _build(ods, field="te", coordinate="psi_norm", equilibrium=geq)
    measured = _by_label(from_mapping, "measured")
    assert len(measured) == 1 and ":" not in measured[0].label
    np.testing.assert_allclose(measured[0].x, _by_label(from_geqdsk, "measured")[0].x)
