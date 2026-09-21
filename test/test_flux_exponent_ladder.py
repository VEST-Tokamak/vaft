"""The flux-exponent ladder at ODS level: precedence, and one ladder everywhere.

``psi`` is stored in weber by the Data Dictionary and in weber per radian by
VAFT's legacy artifacts, and nothing in most files says which.  Three probes
can answer, in decreasing order of evidence, and *which slice* a probe answers
on matters as much as which probe answers -- a weak probe on one slice must not
overrule a strong one on the other eighteen.

Everything here runs on the packaged samples and on transformations of them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import vaft
from vaft.data.eqdsk import (
    _ampere_detail,
    _contour_q_exponent,
    _slice_reader,
    flux_exponent_tier,
    ods_flux_exponent,
    ods_psi_to_wb_per_radian_factor,
    slice_flux_exponent,
)
from vaft.data.resources import available_samples
from vaft.plot.backend.convention import flux_exponent as plot_flux_exponent
from vaft.plot.backend.convention import psi_convention

TWO_PI = 2.0 * math.pi


@pytest.fixture(scope="module")
def ods():
    """A packaged sample with many time slices."""
    return vaft.omas.load(vaft.data.sample(41672, representation="omas"))


def _undeclared(source):
    """A copy with its COCOS declaration removed, so the probes have to answer."""
    import copy

    ods = copy.deepcopy(source)
    for path in ("equilibrium.ids_properties.cocos", "equilibrium.code.parameters.cocos"):
        try:
            del ods[path]
        except Exception:
            pass
    return ods


def _blind(ods, index):
    """Strip one slice of everything but the contour-q probe's inputs."""
    ts = ods[f"equilibrium.time_slice.{index}"]
    for path in ("profiles_1d.phi", "boundary.outline.r", "boundary.outline.z"):
        try:
            del ts[path]
        except Exception:
            pass
    return ods


# --------------------------------------------------------------------------
# The probes, one slice at a time
# --------------------------------------------------------------------------


def test_the_contour_q_probe_answers_a_slice_the_others_cannot(ods):
    """It reads q, F and the psi map, so neither the toroidal flux nor the
    current nor the boundary outline is one of its inputs."""
    blind = _blind(_undeclared(ods), 0)
    read = _slice_reader(blind["equilibrium.time_slice.0"])
    assert _ampere_detail(read) == (None, None)
    assert _contour_q_exponent(read) == 0
    assert slice_flux_exponent(blind["equilibrium.time_slice.0"]) == 0


def test_an_ampere_contradiction_stops_the_probing(ods):
    """A slice whose ip and psi map disagree has a real defect.  The contour-q
    probe reads neither of them, so it must not be allowed to answer over the
    top of a contradiction it cannot see."""
    import copy

    broken = _undeclared(ods)
    for index in range(len(broken["equilibrium.time_slice"])):
        ts = broken[f"equilibrium.time_slice.{index}"]
        try:
            del ts["profiles_1d.phi"]
        except Exception:
            pass
        ts["global_quantities.ip"] = float(ts["global_quantities.ip"]) * 0.2

    readers = [
        _slice_reader(broken[f"equilibrium.time_slice.{i}"])
        for i in range(len(broken["equilibrium.time_slice"]))
    ]
    ran = [_ampere_detail(read) for read in readers]
    assert any(exponent is None and ratio is not None for exponent, ratio in ran), (
        "the fixture assumes Ampere's law runs and rejects both families"
    )
    tier, decided = flux_exponent_tier(readers)
    assert tier is None and decided == {}
    # On its own the contour-q probe is perfectly happy: ip is not its input.
    assert _contour_q_exponent(readers[0]) is not None


# --------------------------------------------------------------------------
# Precedence across slices
# --------------------------------------------------------------------------


def test_a_weak_probe_on_one_slice_does_not_overrule_a_strong_one_on_the_rest(ods):
    """Slice-first ordering let the requested slice decide the whole file from
    the weakest evidence available anywhere.  With slice 0 blinded and its q
    scaled by 2*pi -- so contour-q reads it as weber while every other slice
    measures per radian -- slice-first gave 1/(2*pi) at time index 0 and 1 at
    index 1, and equilibrium_psi_to_weber then refused the file as
    self-contradictory.
    """
    corrupted = _blind(_undeclared(ods), 0)
    ts0 = corrupted["equilibrium.time_slice.0"]
    ts0["profiles_1d.q"] = np.asarray(ts0["profiles_1d.q"], dtype=float) * TWO_PI

    # The blinded slice on its own really does read as the other family.
    assert slice_flux_exponent(ts0) == 1
    # The file does not follow it.
    tier, decided = flux_exponent_tier(
        [
            _slice_reader(corrupted[f"equilibrium.time_slice.{i}"])
            for i in range(len(corrupted["equilibrium.time_slice"]))
        ]
    )
    assert tier == "ampere"
    assert 0 not in decided, "the blinded slice cannot be reached by this tier"
    assert set(decided.values()) == {0}
    for index in (0, 1):
        assert ods_psi_to_wb_per_radian_factor(corrupted, index) == pytest.approx(1.0)


def test_every_index_of_one_file_answers_alike(ods):
    """The convention is a property of the file, so asking about a different
    time slice must not change it."""
    undeclared = _undeclared(ods)
    total = len(undeclared["equilibrium.time_slice"])
    factors = {
        round(ods_psi_to_wb_per_radian_factor(undeclared, index), 12)
        for index in range(total)
    }
    assert len(factors) == 1


def test_the_tier_records_which_probe_answered(ods):
    readers = [
        _slice_reader(ods[f"equilibrium.time_slice.{i}"])
        for i in range(len(ods["equilibrium.time_slice"]))
    ]
    tier, decided = flux_exponent_tier(readers)
    assert tier in {"slope", "ampere", "contour_q"}
    assert decided
    assert set(decided.values()) <= {0, 1}
    assert ods_flux_exponent(ods, 0) == decided[min(decided)]


# --------------------------------------------------------------------------
# One ladder, two readers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shot", sorted(available_samples()))
def test_the_plot_label_never_disagrees_with_the_computed_factor(shot):
    """The plotting layer used to run two of the three probes while claiming to
    follow ``ods_psi_to_wb_per_radian_factor``.  A label two pi away from the
    value plotted under it is worse than no label."""
    ods = vaft.omas.load(vaft.data.sample(shot, representation="omas"))
    total = len(ods["equilibrium.time_slice"]) if "equilibrium.time_slice" in ods else 1
    for index in range(max(total, 1)):
        factor = ods_psi_to_wb_per_radian_factor(ods, index)
        label = psi_convention(ods, index)
        assert label == ("Wb/rad" if factor == pytest.approx(1.0) else "Wb")


def test_the_two_readers_agree_where_only_the_third_probe_can_answer(ods):
    """The case that used to diverge: no toroidal flux and no boundary outline
    anywhere, so only contour-q answers -- and only the eqdsk ladder had it."""
    blind = _undeclared(ods)
    for index in range(len(blind["equilibrium.time_slice"])):
        _blind(blind, index)
    assert ods_psi_to_wb_per_radian_factor(blind, 0) == pytest.approx(1.0)
    assert psi_convention(blind, 0) == "Wb/rad"
    assert plot_flux_exponent(blind, 0) == ods_flux_exponent(blind, 0)


# --------------------------------------------------------------------------
# No packaged sample changes its answer
# --------------------------------------------------------------------------

#: What every packaged sample reads as, measured on ``develop`` before the
#: contour-q probe existed and unchanged by it.  A new probe must widen what
#: *can* be decided; it must never move a file that was already decided.
#: 40600 and 45531 carry no equilibrium at all and reach the Data Dictionary
#: default, which is what ``1/(2*pi)`` means for them.
PACKAGED_FACTORS = {
    39915: 1.0 / TWO_PI,
    40600: 1.0 / TWO_PI,
    41524: 1.0,
    41672: 1.0,
    45531: 1.0 / TWO_PI,
    48224: 1.0 / TWO_PI,
}


def test_the_registered_samples_are_all_covered():
    """A sample added without a line here is a sample whose flux convention
    nobody recorded, so the pin below would quietly stop covering it."""
    missing = set(available_samples()) - set(PACKAGED_FACTORS)
    extra = set(PACKAGED_FACTORS) - set(available_samples())
    assert not missing, (
        f"packaged samples {sorted(missing)} have no recorded flux convention; "
        "measure ods_psi_to_wb_per_radian_factor on them and add a line"
    )
    assert not extra, f"{sorted(extra)} are no longer packaged"


@pytest.mark.parametrize("shot", sorted(PACKAGED_FACTORS))
def test_no_packaged_sample_changes_its_flux_convention(shot):
    ods = vaft.omas.load(vaft.data.sample(shot, representation="omas"))
    total = len(ods["equilibrium.time_slice"]) if "equilibrium.time_slice" in ods else 1
    for index in range(max(total, 1)):
        assert ods_psi_to_wb_per_radian_factor(ods, index) == pytest.approx(
            PACKAGED_FACTORS[shot]
        )
