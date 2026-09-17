"""TGLF -> IMAS ``core_transport`` projection (issue #553 section 14).

The fixture under `test/data/gacode/tglf_vest_48224_profile/` is a real five-surface
TGLF run (SAT3, electromagnetic) on the packaged VEST 48224 kinetic state, trimmed to
the files the parser reads. Nothing here needs GACODE installed: the outputs are the
fixture, and the local inputs that carry the normalisation are rebuilt from the same
packaged sample, which is pure VAFT.

The claim under test is that nothing here is a copy. TGLF reports gyro-Bohm units and
runs on `r/a`; `core_transport` wants SI on `rho_tor_norm`. Both conversions are local
to the surface, both can be wrong by a factor that leaves every array the right shape,
and the momentum channel can additionally be wrong by a sign -- so each is asserted for
what it is rather than through an end-to-end smoke test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omas import ODS

from vaft.code.gacode.tglf import TGLFConfig, collect_tglf_outputs, prepare_tglf_input
from vaft.machine_mapping.turbulence import (
    ANOMALOUS_MODEL_INDEX,
    FLUX_MULTIPLIER,
    core_transport_from_tglf,
)
from vaft.ods_access import path_count

FIXTURE = Path(__file__).parent / "data" / "gacode" / "tglf_vest_48224_profile"
RADII = (0.2, 0.35, 0.5, 0.65, 0.8)

SAMPLE = None
try:  # pragma: no cover - depends on the repository-only sample being present
    from vaft.data.resources import data_path

    _candidate = Path(data_path("kineticEfit/ods_48224_300ms.json"))
    SAMPLE = _candidate if _candidate.exists() else None
except Exception:
    SAMPLE = None

requires_sample = pytest.mark.skipif(
    SAMPLE is None, reason="the packaged 48224 kinetic sample is a repository-only asset"
)


@pytest.fixture(scope="module")
def profile():
    from omas import load_omas_json

    from vaft.code.gacode.inputs import prepare_gacode_profile

    ods = load_omas_json(str(SAMPLE), consistency_check=False)
    return prepare_gacode_profile(ods, rho_max=0.95, z_eff=2.0, impurity="C")


@pytest.fixture(scope="module")
def surfaces(profile):
    """``(TGLFInput, TglfOutputs)`` per surface, from the fixture run."""
    return [
        (
            prepare_tglf_input(profile, rho, config=TGLFConfig(sat_rule=3, use_bper=True)),
            collect_tglf_outputs(FIXTURE / f"rho{rho}"),
        )
        for rho in RADII
    ]


@pytest.fixture()
def mapped(surfaces, profile):
    ods = ODS(consistency_check=False)
    report = core_transport_from_tglf(ods, surfaces, profile, time=0.3)
    return ods, report


# --------------------------------------------------------------------------
# the fixture is a real solved run
# --------------------------------------------------------------------------


def test_every_fixture_surface_solved(surfaces):
    assert [native.solved for _, native in surfaces] == [True] * len(RADII)
    assert {native.n_species for _, native in surfaces} == {3}


# --------------------------------------------------------------------------
# the model entry
# --------------------------------------------------------------------------


@requires_sample
def test_the_model_is_identified_as_turbulent_not_neoclassical(mapped):
    """Index 6 is `anomalous`, the data dictionary's own name for turbulent transport."""
    ods, report = mapped
    model = report["model"]
    assert ods[f"core_transport.model.{model}.identifier.index"] == ANOMALOUS_MODEL_INDEX
    assert ods[f"core_transport.model.{model}.identifier.name"] == "anomalous"


@requires_sample
def test_the_flux_multiplier_is_zero_because_the_energy_flux_is_already_total(mapped):
    """`flux_multiplier` adds the particle flux back into the energy flux.

    TGLF's energy weight is built from the full pressure moment, so the convective part
    is already in it and adding it again would double-count.
    """
    ods, report = mapped
    assert ods[f"core_transport.model.{report['model']}.flux_multiplier"] == FLUX_MULTIPLIER
    assert FLUX_MULTIPLIER == 0.0


@requires_sample
def test_a_second_call_reuses_the_model_rather_than_appending(surfaces, profile):
    ods = ODS(consistency_check=False)
    first = core_transport_from_tglf(ods, surfaces, profile, time=0.3)
    second = core_transport_from_tglf(ods, surfaces, profile, time=0.3)
    assert first["model"] == second["model"]
    assert path_count(ods, "core_transport.model") == 1


@requires_sample
def test_a_turbulent_model_does_not_overwrite_a_neoclassical_one(surfaces, profile):
    """Both mappers key on their own identifier, so an ODS can carry both."""
    ods = ODS(consistency_check=False)
    ods["core_transport.model.0.identifier.index"] = 5  # neoclassical
    report = core_transport_from_tglf(ods, surfaces, profile, time=0.3)
    assert report["model"] == 1
    assert ods["core_transport.model.0.identifier.index"] == 5
    assert ods["core_transport.model.1.identifier.index"] == ANOMALOUS_MODEL_INDEX


# --------------------------------------------------------------------------
# the two conversions that can be silently wrong
# --------------------------------------------------------------------------


@requires_sample
def test_the_grid_is_rho_tor_norm_and_not_the_r_over_a_tglf_ran_on(mapped, profile):
    """TGLF takes `r/a`; `core_transport` indexes fluxes by `rho_tor_norm`.

    They are not the same coordinate and not a small correction: on this state
    `r/a = 0.8` is `rho_tor_norm = 0.71`. A mapping that passed `r/a` straight through
    would produce an array of exactly the right shape at the wrong radii.
    """
    ods, report = mapped
    base = f"core_transport.model.{report['model']}.profiles_1d.0"
    grid = np.asarray(ods[f"{base}.grid_flux.rho_tor_norm"], dtype=float)

    assert grid.shape == (len(RADII),)
    assert np.all(np.diff(grid) > 0)
    assert np.all(grid < np.asarray(RADII)), "rho_tor_norm should sit inside r/a here"
    assert grid[-1] == pytest.approx(0.711, abs=0.01)

    # The bridge is the profile's own, so it must reproduce it exactly at its own points.
    rmin = np.asarray(profile.rmin, dtype=float)
    expected = np.interp(np.asarray(RADII), rmin / rmin[-1], np.asarray(profile.rho))
    assert grid == pytest.approx(expected)


@requires_sample
def test_the_fluxes_are_dimensionalised_by_the_surfaces_own_gyrobohm_unit(mapped, surfaces):
    """Each surface has its own `Q_GB`; a single global factor would be wrong.

    On this state `Q_GB` falls by an order of magnitude between `r/a = 0.5` and 0.8, so
    using one surface's unit everywhere shows up as a mis-shaped profile rather than a
    uniform offset.
    """
    ods, report = mapped
    base = f"core_transport.model.{report['model']}.profiles_1d.0"
    written = np.asarray(ods[f"{base}.electrons.energy.flux"], dtype=float)

    expected = np.array([
        native.gbflux["energy"][0] * local.normalisation.energy_flux
        for local, native in surfaces
    ])
    assert written == pytest.approx(expected)
    assert np.all(np.isfinite(written))

    units = np.array([local.normalisation.energy_flux for local, _ in surfaces])
    assert units.max() / units.min() > 5.0, (
        "the gyro-Bohm unit varies enough across these surfaces that a single factor "
        "would be visibly wrong; if it no longer does, this test proves less than it claims"
    )


@requires_sample
def test_the_particle_and_energy_units_differ_by_the_temperature(mapped, surfaces):
    """`Q_GB = Gamma_GB * k T_e`: the two channels must not share a factor."""
    local = surfaces[2][0]
    ratio = local.normalisation.energy_flux / local.normalisation.particle_flux
    assert ratio == pytest.approx(local.normalisation.electron_temperature)


@requires_sample
def test_the_ion_channels_carry_their_species_identity(mapped):
    """Electrons are TGLF species 0, so ion 0 of the IDS is TGLF species 1."""
    ods, report = mapped
    base = f"core_transport.model.{report['model']}.profiles_1d.0"
    assert ods[f"{base}.ion.0.label"] == "H+"
    assert ods[f"{base}.ion.0.z_ion"] == pytest.approx(1.0)
    # TGLF masses are deuterium-normalised; IMAS wants amu.
    assert ods[f"{base}.ion.0.element.0.a"] == pytest.approx(1.008, abs=0.02)
    assert ods[f"{base}.ion.1.label"] == "C6+"
    assert ods[f"{base}.ion.1.z_ion"] == pytest.approx(6.0)
    assert ods[f"{base}.ion.1.element.0.a"] == pytest.approx(12.011, abs=0.05)


@requires_sample
def test_the_electron_energy_flux_is_not_an_ion_one(mapped, surfaces):
    """A species-order slip leaves every array finite and the right shape."""
    ods, report = mapped
    base = f"core_transport.model.{report['model']}.profiles_1d.0"
    electron = np.asarray(ods[f"{base}.electrons.energy.flux"], dtype=float)
    ion = np.asarray(ods[f"{base}.ion.0.energy.flux"], dtype=float)
    assert electron[3] > 5.0 * ion[3], (
        "on this state the electron channel dominates by several times at mid-radius; "
        "if these were swapped the assertion would fail rather than pass quietly"
    )


# --------------------------------------------------------------------------
# what is deliberately not written
# --------------------------------------------------------------------------


@requires_sample
def test_the_momentum_flux_is_refused_when_the_run_was_given_no_rotation(mapped):
    """`prepare_gacode_profile` populates no `w0`, so VEXB_SHEAR reaches TGLF as zero.

    What comes back is the numerical residue of a plasma told not to rotate. Writing it
    as `momentum_tor.flux` would hand a consumer noise wearing the shape of a result.
    """
    ods, report = mapped
    base = f"core_transport.model.{report['model']}.profiles_1d.0"
    assert "momentum_tor.flux" not in report["written"]
    assert f"{base}.momentum_tor.flux" not in ods
    assert any("momentum_tor.flux" in reason for reason in report["skipped"])
    assert any("vexb_shear" in reason for reason in report["skipped"])


@requires_sample
def test_the_momentum_flux_is_the_negative_of_the_raw_column(surfaces, profile):
    """TGLF multiplies the toroidal stress by SIGN_IT and TGYRO by -SIGN_IT.

    The two cancel to a plain inversion whatever the current direction, so a mapping
    that applied `local.sign_it` once would be right only for a plasma whose current
    runs one way -- and on this state SIGN_IT is +1, which is exactly when that bug
    hides. Asserting the constant alone would restate its definition and reach none of
    the code, so this gives the surfaces a rotation and checks the written profile.
    """
    import dataclasses

    rotating = [
        (dataclasses.replace(local, vexb_shear=0.05, provenance={}), native)
        for local, native in surfaces
    ]
    ods = ODS(consistency_check=False)
    report = core_transport_from_tglf(ods, rotating, profile, time=0.3)
    base = f"core_transport.model.{report['model']}.profiles_1d.0"

    assert "momentum_tor.flux" in report["written"]
    written = np.asarray(ods[f"{base}.momentum_tor.flux"], dtype=float)

    raw = np.array([
        sum(
            native.gbflux["momentum"][index] * local.normalisation.momentum_flux
            for index in range(len(local.names))
        )
        for local, native in rotating
    ])
    assert written == pytest.approx(-raw)

    # The inversion has to be visible, or `approx(-raw)` would also pass on zeros.
    assert np.max(np.abs(raw)) > 0.0
    assert np.any(written * raw < 0.0), (
        "every surface's stress is zero, so this test cannot tell an inversion from an "
        "identity; the fixture no longer supports the claim"
    )
    # And it is not SIGN_IT applied once: that would be the identity here.
    assert all(local.sign_it == 1.0 for local, _ in rotating)
    assert written != pytest.approx(raw)


@requires_sample
def test_the_exchange_channel_is_named_as_having_no_flux_home(mapped):
    """It is a power density, W/m^3, and belongs in core_sources."""
    _, report = mapped
    assert any("exchange" in reason for reason in report["skipped"])


@requires_sample
def test_no_diffusivity_or_convection_split_is_invented(mapped):
    ods, report = mapped
    base = f"core_transport.model.{report['model']}.profiles_1d.0"
    assert f"{base}.grid_d.rho_tor_norm" not in ods
    assert f"{base}.grid_v.rho_tor_norm" not in ods
    assert any("grid_d" in reason for reason in report["skipped"])


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


@requires_sample
def test_a_surface_that_did_not_solve_is_dropped_and_named(surfaces, profile):
    import copy

    damaged = copy.deepcopy(surfaces)
    damaged[2][1].gbflux = None  # what an unsolved run looks like
    ods = ODS(consistency_check=False)
    report = core_transport_from_tglf(ods, damaged, profile, time=0.3)
    base = f"core_transport.model.{report['model']}.profiles_1d.0"

    assert any("did not solve" in reason for reason in report["skipped"])
    assert len(ods[f"{base}.grid_flux.rho_tor_norm"]) == len(RADII) - 1


@requires_sample
def test_an_input_without_a_normalisation_cannot_be_dimensionalised(surfaces, profile):
    """The gyro-Bohm unit is the whole conversion; without it there is nothing to write."""
    import dataclasses

    stripped = [(dataclasses.replace(local, normalisation=None), native)
                for local, native in surfaces]
    ods = ODS(consistency_check=False)
    report = core_transport_from_tglf(ods, stripped, profile, time=0.3)
    assert report["written"] == []
    assert any("normalisation" in reason for reason in report["skipped"])
    assert "core_transport.model.0.profiles_1d" not in ods


@requires_sample
def test_a_profile_with_no_coordinate_bridge_refuses_rather_than_assuming(surfaces, profile):
    import dataclasses

    ods = ODS(consistency_check=False)
    report = core_transport_from_tglf(
        ods, surfaces, dataclasses.replace(profile, rmin=None), time=0.3
    )
    assert report["written"] == []
    assert any("rho_tor_norm" in reason for reason in report["skipped"])


# --------------------------------------------------------------------------
# guards that must check the condition itself, not something next to it
# --------------------------------------------------------------------------


@requires_sample
def test_a_run_with_fewer_output_species_than_its_input_is_refused(surfaces, profile):
    """The count that is indexed is the output's, not the input's.

    `names` is what the input declared and `n_species` is what the run wrote into
    out.tglf.grid. An adiabatic-electron run makes them differ, and guarding only the
    first let the second raise IndexError out of a mapper that is meant to report.
    """
    import copy

    shrunk = copy.deepcopy(surfaces)
    native = shrunk[2][1]
    native.grid = {"n_species": 2, "n_xgrid": native.grid["n_xgrid"]}
    native.gbflux = {name: values[:2] for name, values in native.gbflux.items()}

    ods = ODS(consistency_check=False)
    report = core_transport_from_tglf(ods, shrunk, profile, time=0.3)  # must not raise
    assert report["written"] == []
    assert any("species count" in reason for reason in report["skipped"])


@requires_sample
def test_an_input_with_no_rotation_value_writes_no_momentum_even_unrecorded(
    surfaces, profile
):
    """The value decides, not the provenance record of it.

    An input built outside `prepare_tglf_input` can carry `vexb_shear=None` with an
    empty provenance -- `test_tglf_surrogate.py` builds exactly those. Reading only the
    record would conclude such a run was rotating.
    """
    import dataclasses

    unrecorded = [
        (dataclasses.replace(local, vexb_shear=None, provenance={}), native)
        for local, native in surfaces
    ]
    assert all(local.missing() == () for local, _ in unrecorded)

    ods = ODS(consistency_check=False)
    report = core_transport_from_tglf(ods, unrecorded, profile, time=0.3)
    assert "momentum_tor.flux" not in report["written"]


@requires_sample
def test_a_non_monotone_radial_grid_is_refused_rather_than_interpolated(surfaces, profile):
    """np.interp returns silently wrong values for an unsorted xp."""
    import dataclasses

    rmin = np.asarray(profile.rmin, dtype=float).copy()
    rmin[10], rmin[11] = rmin[11], rmin[10]
    ods = ODS(consistency_check=False)
    report = core_transport_from_tglf(
        ods, surfaces, dataclasses.replace(profile, rmin=rmin), time=0.3
    )
    assert report["written"] == []
    assert any("rho_tor_norm" in reason for reason in report["skipped"])


@requires_sample
def test_a_zero_toroidal_flux_has_no_sign_to_derive(profile):
    """SIGN_BT and SIGN_IT are +/-1 conventions; np.sign(0) is 0, which is neither."""
    import dataclasses

    from vaft.code.gacode.tglf import LocalConversionError

    with pytest.raises(LocalConversionError, match="torfluxa"):
        prepare_tglf_input(dataclasses.replace(profile, torfluxa=0.0), 0.5)


@requires_sample
def test_the_derived_signs_are_recorded_as_derived(surfaces):
    """They were hardcoded before, with nothing to say so."""
    local = surfaces[2][0]
    assert local.provenance["sign_bt"]["kind"] == "derived"
    assert local.provenance["sign_it"]["kind"] == "derived"
    # On VEST 48224 torfluxa < 0 and q > 0, which is what makes both +1.
    assert (local.sign_bt, local.sign_it) == (1.0, 1.0)


@requires_sample
@pytest.mark.parametrize(
    "flip_q, flip_flux, expected",
    [
        (False, False, (1.0, 1.0)),
        (True, False, (1.0, -1.0)),
        (True, True, (-1.0, 1.0)),
        (False, True, (-1.0, -1.0)),
    ],
)
def test_sign_it_follows_the_signed_q_as_gacodes_ipccw_does(
    profile, flip_q, flip_flux, expected
):
    """Cold review transport F1: taken from |q|, SIGN_IT could never differ from SIGN_BT.

    ``expro_locsim.f90:202-203``: ``btccw = -signb``, ``ipccw = -signq*signb``.
    """
    import dataclasses

    changed = dataclasses.replace(
        profile,
        q=-profile.q if flip_q else profile.q,
        torfluxa=-profile.torfluxa if flip_flux else profile.torfluxa,
    )
    local = prepare_tglf_input(changed, 0.5)
    assert (local.sign_bt, local.sign_it) == expected
    # Q_LOC stays |q|, as locpargen writes it.
    assert local.q_loc == pytest.approx(prepare_tglf_input(profile, 0.5).q_loc)
    assert local.q_loc > 0.0


@requires_sample
def test_a_q_that_changes_sign_has_no_current_direction(profile):
    import dataclasses

    from vaft.code.gacode.tglf import LocalConversionError

    q = np.array(profile.q, dtype=float)
    q[: q.size // 2] *= -1.0
    with pytest.raises(LocalConversionError, match="changes sign"):
        prepare_tglf_input(dataclasses.replace(profile, q=q), 0.5)
