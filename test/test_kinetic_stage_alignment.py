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


def test_a_product_whose_slices_were_all_rejected_is_not_kinetic():
    """`core_profiles` present does not mean `profiles_1d` is a list.

    When every profile time is out of tolerance the product still carries the
    IDS, and asking for the slice list there returned a float rather than
    raising -- so a parent-level guard crashed instead of answering.
    """
    ods = ODS(consistency_check=False)
    ods["core_profiles.time"] = np.array([0.306])
    assert "core_profiles" in ods
    assert vu._has_kinetic_slice(ods) is False


def test_membership_is_answered_where_omas_would_raise():
    """`in` walks the path, so a scalar parent raises instead of saying False.

    A reloaded product is where this bites: shot 39915's core_profiles has
    electron-only slices, and asking whether one carries `ion.0.temperature`
    raised `AttributeError: 'float' object has no attribute 'omas_data'`.
    """
    ods = ODS(consistency_check=False)
    ods["core_profiles.profiles_1d.0.electrons.temperature"] = np.linspace(9.0, 1.0, 3)
    # The parent resolves to an array, so the deep path is unanswerable by `in`.
    assert vu._path_present(ods, "core_profiles.profiles_1d.0.electrons.temperature.0.x") is False
    assert vu._path_present(ods, "core_profiles.profiles_1d.0.electrons.temperature") is True
    assert vu._has_kinetic_slice(ods) is False


def test_a_scalar_time_does_not_raise_out_of_an_optional_stage():
    """A single time saved as a float is legitimate; `len` on it is not.

    The same shape that crashed the lineage check, one function down: an
    optional stage must record a bad product, never raise past its caller.
    """
    ods = ODS(consistency_check=False)
    ods["core_profiles.time"] = 0.306
    assert vu._time_array_ms(ods, "core_profiles.time").tolist() == [306.0]
    assert vu._time_array_ms(ODS(consistency_check=False), "core_profiles.time") is None


def test_an_unreadable_efit_product_is_named_not_blamed_on_the_time_base(tmp_path):
    """48224's EFIT stage is `no_output`, so its product has no equilibrium.

    Falling through would report "no equilibrium" once per profile time and
    blame the time bases for a shot that was never reconstructed at all.
    """
    thomson = tmp_path / "thomson.json"
    t = ODS(consistency_check=False)
    t["thomson_scattering.time"] = np.array([0.300, 0.301])
    t.save(str(thomson))

    efit = tmp_path / "efit.json"
    e = ODS(consistency_check=False)
    e["dataset_description.data_entry.pulse"] = 48224
    e.save(str(efit))

    _, manifest = vu.build_core_profiles_ods(
        shot=48224, thomson_product=thomson, efit_product=efit
    )
    assert manifest["status"] == "unavailable"
    assert "no equilibrium.time" in manifest["error"]
    assert manifest.get("slices") is None, "it never reached the per-time loop"


def test_the_core_profile_psi_grid_is_the_equilibrium_s_own_weber(tmp_path):
    """#1292: the psi map built from the EFIT product is a per-radian g-file.

    ``from_equilibrium`` used to copy the ODS's weber flux into the g-file, and
    ``core_profiles`` takes its ``grid.psi`` from ``geq.to_omas()``, which
    multiplies by 2*pi -- so the stage published a psi grid 2*pi too large.
    """
    from vaft.omas.sample import sample_ods

    try:
        source = sample_ods(48224)
    except FileNotFoundError:
        pytest.skip("the 48224 sample is repository-only")
    product = tmp_path / "48224.json"
    source.save(str(product))

    out, manifest = vu.build_core_profiles_ods(
        shot=48224, thomson_product=product, ces_product=product, efit_product=product
    )
    assert manifest["status"] == "success", manifest.get("error")
    grid = np.asarray(out["core_profiles.profiles_1d.0.grid.psi"], dtype=float)
    ts = source["equilibrium.time_slice.0"]
    assert grid[0] == pytest.approx(float(ts["global_quantities.psi_axis"]), rel=1e-6)
    assert grid[-1] == pytest.approx(float(ts["global_quantities.psi_boundary"]), rel=1e-6)


def test_an_aligned_usable_slice_wins_over_an_earlier_aligned_failed_one():
    """#1331: Thomson and EFIT share a 1 ms grid on 39915, so every profile time
    is aligned; the first was 312 ms, a negative-pressure slice the kinetic fit
    never converged from, while 316 ms converges at scale 1.0."""
    profile = np.arange(308.0, 318.0)
    eq = np.arange(312.0, 329.0)
    usable = np.arange(315.0, 325.0)
    time_ms, why = vu.choose_profile_time(profile, eq, usable)
    assert time_ms == 315.0 and why == "aligned and usable"


def test_without_verdicts_or_usable_slices_the_best_aligned_time_is_kept():
    profile, eq = np.array([306.0, 307.0]), np.array([308.0, 309.0, 311.0])
    assert vu.choose_profile_time(profile, eq, None) == (307.0, "aligned (no per-slice verdicts)")
    assert vu.choose_profile_time(profile, eq, np.array([311.0])) == (307.0, "aligned (no aligned slice is usable)")


def test_usable_times_are_read_from_the_efit_collection_record():
    import json

    ods = ODS(consistency_check=False)
    ods["equilibrium.code.parameters"] = json.dumps({"efit_collection": {"slice_statuses": [
        {"time": 0.315, "overall_status": "usable"}, {"time": 0.312, "overall_status": "physical_failed"}]}})
    assert vu.usable_equilibrium_times_ms(ods).tolist() == [315.0]
    assert vu.usable_equilibrium_times_ms(ODS(consistency_check=False)) is None


def test_candidates_are_the_aligned_usable_times_nearest_first_then_earliest():
    profile = np.arange(308.0, 318.0)
    eq = np.arange(312.0, 329.0)
    usable = np.arange(315.0, 325.0)
    times, why = vu.profile_time_candidates(profile, eq, usable)
    assert why == "aligned and usable" and times == [315.0, 316.0, 317.0]
    assert vu.profile_time_candidates(profile, eq, None) == ([312.0], "aligned (no per-slice verdicts)")


def test_without_an_aligned_usable_slice_the_times_next_to_the_usable_window_come_first():
    """42985: Thomson at 322-331 ms, magnetic EFIT usable only from 335 ms."""
    profile = np.arange(322.0, 332.0)
    eq = np.arange(318.0, 340.0)
    usable = np.arange(335.0, 340.0)
    times, why = vu.profile_time_candidates(profile, eq, usable)
    assert why == "aligned, nearest the usable window" and times[:3] == [331.0, 330.0, 329.0]
