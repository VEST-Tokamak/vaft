"""The registered NBI plots, read from an ODS rather than a NUBEAM directory.

`vaft/plot/nubeam.py` draws a live NUBEAM result; these draw a *mapped* one, so
a saved ODS plots without the run that produced it. The catalog contract itself
(returns, ax/show, model type) is covered for every registered plot by
test_plot_contract.py -- what is here is the NBI-specific behaviour.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

import vaft.omas as vomas
from vaft.code.nubeam.outputs import NUBEAMOutputs, NUBEAMRadialGrid
from vaft.machine_mapping.core_sources import core_sources_from_nubeam
from vaft.plot import registry

ZONES = 4
NBI_PLOTS = (
    "nbi_profile_electron_heating",
    "nbi_profile_ion_heating",
    "nbi_profile_current_drive",
)


@pytest.fixture
def ods(tmp_path):
    outputs = NUBEAMOutputs(
        workdir=tmp_path,
        runid="TESTRUN",
        profiles={
            "pbe": np.array([100.0, 200.0, 300.0, 400.0]),
            "pbi": np.array([10.0, 20.0, 30.0, 40.0]),
            "curbeam": np.array([1.0, 2.0, 3.0, 4.0]),
        },
        grid=NUBEAMRadialGrid(
            rho=np.linspace(0.0, 1.0, ZONES + 1),
            volume=np.linspace(0.0, 8.0, ZONES + 1),
            area=np.linspace(0.0, 2.0, ZONES + 1),
        ),
    )
    out = ODS()
    core_sources_from_nubeam(out, outputs)
    return out


def test_every_nbi_plot_is_registered():
    for name in NBI_PLOTS:
        assert name in registry.canonical_names()
        assert registry.get_spec(name).subject == "nbi"


def test_a_mapped_ods_offers_the_nbi_plots(ods):
    offered = {row["name"] for row in vomas.available_plots(ods)}
    assert set(NBI_PLOTS) <= offered


def test_an_ods_without_a_beam_does_not_offer_them():
    """Discovery must not advertise a plot the data cannot support."""
    empty = ODS()
    empty["equilibrium.time_slice.0.global_quantities.ip"] = -1.0e5
    offered = {row["name"] for row in vomas.available_plots(empty)}
    assert not (set(NBI_PLOTS) & offered)


@pytest.mark.parametrize("name", NBI_PLOTS)
def test_each_plot_draws_the_mapped_profile(ods, name):
    figure, axes = getattr(vomas, f"plot_{name}")(ods)
    assert len(axes.lines) == 1
    assert len(axes.lines[0].get_xdata()) == ZONES
    plt.close(figure)


def test_labels_carry_the_imas_unit_not_nubeams(ods):
    """core_sources is a density; the per-zone watts stayed in the adapter."""
    figure, axes = vomas.plot_nbi_profile_electron_heating(ods)
    assert "W" in axes.get_ylabel()
    assert "m" in axes.get_ylabel()  # per cubic metre
    assert r"$\rho_{tor,norm}$" in axes.get_xlabel()
    plt.close(figure)


def test_the_beam_is_found_by_identifier_not_position(ods):
    """core_sources holds every source; nothing fixes the beam's index."""
    # Push the NBI entry to index 1 by inserting another source ahead of it.
    shifted = ODS()
    shifted["core_sources.source.0.identifier.index"] = 7  # ohmic
    shifted["core_sources.source.0.identifier.name"] = "ohmic"
    base = "core_sources.source.1"
    for key, value in ods.flat().items():
        if key.startswith("core_sources.source.0."):
            shifted[base + key[len("core_sources.source.0") :]] = value
    figure, axes = vomas.plot_nbi_profile_electron_heating(shifted)
    assert len(axes.lines) == 1
    plt.close(figure)


def test_an_ods_with_sources_but_no_beam_says_which_it_has():
    ods = ODS()
    ods["core_sources.source.0.identifier.index"] = 7
    ods["core_sources.source.0.identifier.name"] = "ohmic"
    with pytest.raises(ValueError) as error:
        vomas.plot_nbi_profile_electron_heating(ods)
    message = str(error.value)
    assert "no neutral-beam entry" in message
    assert "ohmic" in message


def test_plotting_does_not_mutate_the_ods(ods):
    before = set(ods.flat().keys())
    for name in NBI_PLOTS:
        figure, _ = getattr(vomas, f"plot_{name}")(ods)
        plt.close(figure)
    assert set(ods.flat().keys()) == before
