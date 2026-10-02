"""What compute_magnetic_energy integrates, over what, and whether it writes.

Found while fixing the voltage-consumption path (#652): the default total is
dominated by the vacuum toroidal field over the plasma volume, the flux mask
admitted cells outside the boundary, and the call wrote B fields into the
caller's ODS.  The poloidal part is pinned against an independent route --
L_i I_p^2 / 2 from li_3, which the update layer takes from a surface line
integral, not from this grid -- on the packaged multi-slice shots.
"""

import copy
import logging

import numpy as np
import pytest

pytest.importorskip("omas")
pytest.importorskip("skimage")

from vaft.formula.constants import MU0
from vaft.omas.process_wrapper import compute_magnetic_energy
from vaft.omas.sample import sample_ods
from vaft.omas.update import (
    resolve_reference_major_radius,
    update_equilibrium_global_quantities_beta_li,
)

# Early, mid and late slices; the late ones are where the flux mask used to
# admit cells by the solenoid (41672 slice 10: 23.6 kJ against 0.63 kJ).
SLICES = [(39915, 0), (39915, 6), (41672, 0), (41672, 5), (41672, 10)]


@pytest.fixture(scope="module")
def samples():
    logging.disable(logging.WARNING)
    try:
        yield {shot: sample_ods(shot) for shot in {39915, 41672}}
    finally:
        logging.disable(logging.NOTSET)


def _internal_energy(ods, idx):
    work = copy.deepcopy(ods)
    slice_node = work["equilibrium.time_slice"][idx]
    if "global_quantities.li_3" in slice_node:
        del slice_node["global_quantities.li_3"]
    update_equilibrium_global_quantities_beta_li(work, time_slice=[idx])
    li_3 = float(work[f"equilibrium.time_slice.{idx}.global_quantities.li_3"])
    ip = float(work[f"equilibrium.time_slice.{idx}.global_quantities.ip"])
    return 0.25 * MU0 * resolve_reference_major_radius(work) * li_3 * ip**2


@pytest.mark.parametrize("shot, idx", SLICES)
def test_the_poloidal_part_is_the_internal_inductance_energy(samples, shot, idx):
    ods = samples[shot]
    poloidal = compute_magnetic_energy(ods, time_slice=idx, components="poloidal")
    assert poloidal == pytest.approx(_internal_energy(ods, idx), rel=0.03)


@pytest.mark.parametrize("shot, idx", SLICES)
def test_the_total_is_poloidal_plus_a_much_larger_toroidal_part(samples, shot, idx):
    ods = samples[shot]
    total = compute_magnetic_energy(ods, time_slice=idx)
    poloidal = compute_magnetic_energy(ods, time_slice=idx, components="poloidal")
    toroidal = compute_magnetic_energy(ods, time_slice=idx, components="toroidal")
    assert total == pytest.approx(poloidal + toroidal, rel=1e-12)
    assert toroidal > 10.0 * poloidal


def test_the_default_call_leaves_the_ods_alone(samples):
    ods = copy.deepcopy(samples[39915])
    before = set(ods["equilibrium.time_slice.0.profiles_2d.0"].keys())
    compute_magnetic_energy(ods, time_slice=0)
    compute_magnetic_energy(ods, time_slice=0, components="poloidal")
    assert set(ods["equilibrium.time_slice.0.profiles_2d.0"].keys()) == before


def test_writing_the_fields_is_an_explicit_request(samples):
    ods = copy.deepcopy(samples[39915])
    compute_magnetic_energy(ods, time_slice=0, write_fields=True)
    grid = ods["equilibrium.time_slice.0.profiles_2d.0"]
    for leaf in ("b_field_r", "b_field_z", "b_field_tor"):
        assert leaf in grid


def test_an_unknown_component_is_refused(samples):
    with pytest.raises(ValueError, match="components"):
        compute_magnetic_energy(samples[39915], time_slice=0, components="plasma")
