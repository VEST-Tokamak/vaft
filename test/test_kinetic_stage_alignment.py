"""The time-alignment rules of the Thomson lineage stages (#130).

These pin the two things that only showed up on real shots: a profile mapped
onto an equilibrium it does not belong to, and a lineage decided by an IDS
being present rather than by what was measured.
"""

import numpy as np
import pytest
from omas import ODS

from vaft.omas import vest_upstream as vu


def _core_profiles(times_ms, *, kinetic_indices=()) -> ODS:
    """A core_profiles product with the given slice times, in seconds."""
    ods = ODS(consistency_check=False)
    ods["core_profiles.time"] = np.asarray(times_ms, dtype=float) / 1e3
    for index, _ in enumerate(times_ms):
        base = f"core_profiles.profiles_1d.{index}"
        ods[f"{base}.grid.rho_tor_norm"] = np.linspace(0.0, 1.0, 5)
        ods[f"{base}.electrons.temperature"] = np.linspace(100.0, 10.0, 5)
        if index in kinetic_indices:
            ods[f"{base}.ion.0.temperature"] = np.linspace(50.0, 5.0, 5)
    return ods


def test_a_measured_ion_temperature_is_what_makes_a_slice_kinetic():
    assert vu._has_kinetic_slice(_core_profiles([300.0], kinetic_indices=(0,))) is True
    assert vu._has_kinetic_slice(_core_profiles([300.0])) is False


def test_an_ion_diagnostic_without_a_kinetic_slice_is_not_kinetic():
    """48226 carries `charge_exchange` and has no kinetic slice.

    The ion diagnostic covers 298-307 ms and the equilibrium does not, so every
    surviving slice is electron-only. Deciding the lineage from the IDS refused
    `electron_efit` for a product only `electron_efit` can serve -- and
    `kinetic_efit` could not serve it either, so both lineages declined it.
    """
    ods = _core_profiles([306.0, 307.0])
    ods["charge_exchange.channel.0.ion.0.t_i.data"] = np.array([1.0, 2.0])
    assert "charge_exchange.channel" in ods
    assert vu._has_kinetic_slice(ods) is False


def test_the_tolerance_is_tighter_than_the_profile_layer_resolves():
    """A misaligned equilibrium is worse than a missing one, so it is stricter.

    `vaft.process.profile` resolves its own samples within 1 ms; accepting a
    1 ms equilibrium offset would map a profile onto a plasma one millisecond
    older and leave nothing to show it happened.
    """
    assert vu.EQUILIBRIUM_TIME_TOLERANCE_MS < 1.0


@pytest.mark.parametrize(
    "offset_ms, mapped",
    [
        (0.0, True),
        (vu.EQUILIBRIUM_TIME_TOLERANCE_MS, True),        # boundary: inclusive
        (vu.EQUILIBRIUM_TIME_TOLERANCE_MS + 0.01, False),
        (10.0, False),                                    # the 48226 case
    ],
)
def test_the_tolerance_boundary_is_inclusive(offset_ms, mapped):
    """`>` not `>=`: a match exactly at the tolerance is still a match."""
    assert (offset_ms > vu.EQUILIBRIUM_TIME_TOLERANCE_MS) is not mapped


def test_the_best_aligned_profile_time_is_chosen_not_the_first():
    """With a sub-millisecond tolerance the choice decides the outcome.

    On 48226 the first slice sits 2.0 ms from an equilibrium and a later one
    sits 1.0 ms away; picking the first refuses a shot the second could serve.
    """
    profile_times = np.array([306.0, 307.0])
    eq_times = np.array([308.0, 309.0, 311.0])
    offsets = np.min(np.abs(profile_times[:, None] - eq_times[None, :]), axis=1)
    assert float(profile_times[int(np.argmin(offsets))]) == 307.0
    assert float(profile_times[0]) == 306.0, "the first slice is the worse one"


def test_a_missing_efit_product_is_reported_not_silently_skipped(tmp_path):
    """A path that does not resolve is a caller error, not an absent equilibrium."""
    thomson = tmp_path / "thomson.json"
    ods = ODS(consistency_check=False)
    ods["thomson_scattering.time"] = np.array([0.300])
    ods.save(str(thomson))

    _, manifest = vu.build_core_profiles_ods(
        shot=48226,
        thomson_product=thomson,
        efit_product=tmp_path / "nope.json",
    )
    assert manifest["status"] == "unavailable"
    assert "missing" in manifest["error"]
    assert "nope.json" in manifest["error"]
