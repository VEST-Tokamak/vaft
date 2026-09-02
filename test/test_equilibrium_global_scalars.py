"""The four global scalars a g-file does not carry, in the IMAS DD definitions (#238).

`beta_pol`, `beta_tor`, `beta_normal` and `li_3` were supplied by OMFIT's own
flux-surface solve and were empty once VAFT derived equilibria itself. What makes
them awkward is not the arithmetic: `beta_pol` has two definitions in circulation
that differ by 26%, and the DD gives `li_3` no formula at all.

These tests pin the definitions, the analytic anchors that do not depend on any
reference, and -- deliberately -- the size of the DD-versus-EFIT divergence, so a
later "fix" that quietly matches OMFIT fails here.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("omas")
pytest.importorskip("skimage")

from omas import load_omas_json
from scipy.constants import mu_0 as MU0

import vaft.omas.update as update
from vaft.compat import trapz_compat
from vaft.data.resources import data_path
from vaft.formula.equilibrium import (
    beta_normal_from_beta_tor,
    beta_poloidal_from_circumference,
    beta_poloidal_from_pressure_integral,
    beta_toroidal_from_p_B0,
    li_3_from_Bp2_volume_integral,
)


# --- analytic anchors, independent of any reference ---------------------------


def test_li_3_of_a_uniform_current_cylinder_is_one_half():
    """The textbook anchor. For a large-aspect-ratio circular plasma with uniform
    current density, `B_p(r) = mu0 I r / (2 pi a^2)` and

        int B_p^2 dV = int_0^a (mu0 I r / (2 pi a^2))^2 (2 pi R0)(2 pi r) dr
                     = mu0^2 I^2 R0 / 4

    so `li_3 = 2 (mu0^2 I^2 R0 / 4) / (mu0^2 I^2 R0) = 1/2` exactly, for any
    `I`, `R0` and `a`. Nothing about the reference shot enters this.
    """
    for ip, r0 in ((1.0e6, 1.7), (8.0e4, 0.4), (-3.5e5, 6.2)):
        bp2_volume = MU0**2 * ip**2 * r0 / 4.0
        assert li_3_from_Bp2_volume_integral(bp2_volume, ip, r0) == pytest.approx(0.5)


def test_the_betas_of_a_uniform_pressure_plasma_are_closed_form():
    """With `p` constant, `int p dV = p V` exactly, so each DD definition reduces
    to an expression with no integration error in it."""
    pressure, volume, r0, b0, ip, minor = 1.0e4, 2.5, 1.7, 2.3, 1.2e6, 0.6

    beta_tor = beta_toroidal_from_p_B0(pressure, b0)
    assert beta_tor == pytest.approx(2 * MU0 * pressure / b0**2)

    beta_pol = beta_poloidal_from_pressure_integral(pressure * volume, r0, ip)
    assert beta_pol == pytest.approx(4 * pressure * volume / (r0 * MU0 * ip**2))

    beta_normal = beta_normal_from_beta_tor(beta_tor, minor, b0, ip)
    assert beta_normal == pytest.approx(100 * beta_tor * minor * b0 / (ip / 1e6))

    # beta_normal is defined on magnitudes: reversing Ip must not flip its sign.
    assert beta_normal_from_beta_tor(beta_tor, minor, b0, -ip) == pytest.approx(beta_normal)
    assert beta_normal_from_beta_tor(beta_tor, minor, -b0, ip) == pytest.approx(beta_normal)


def test_the_two_beta_pol_conventions_differ_by_an_exact_geometric_factor():
    """Not a numerical disagreement. `beta_pol_DD / beta_pol_circ` is
    `2V / (R0 L_pol^2)` identically, for any inputs -- which is what makes them
    two definitions rather than two estimates (issue #318)."""
    for volume, r0, ip, length_pol, p_average in (
        (2.5, 1.7, 1.2e6, 5.0, 1.0e4),
        (1.0, 0.4, 8.0e4, 2.5, 3.0e1),
    ):
        dd = beta_poloidal_from_pressure_integral(p_average * volume, r0, ip)
        circumference = beta_poloidal_from_circumference(p_average, ip, length_pol)
        assert dd / circumference == pytest.approx(
            2.0 * volume / (r0 * length_pol**2), rel=1e-12
        )


# --- against the independent OMFIT reference ----------------------------------


@pytest.fixture(scope="module")
def reference():
    path = data_path("kineticEfit/ods_48224_300ms.json")
    if not path.exists():  # pragma: no cover - repository-only sample
        pytest.skip("kineticEfit reference ODS is not packaged in this build")
    return load_omas_json(str(path), consistency_check=False)


@pytest.fixture(scope="module")
def derived(reference):
    work = copy.deepcopy(reference)
    globals_ = work["equilibrium.time_slice.0.global_quantities"]
    for leaf in ("beta_pol", "beta_tor", "beta_normal", "li_3"):
        del globals_[leaf]
    update.update_equilibrium_global_quantities_beta_li(work)
    return work


def _stored(reference, leaf):
    return float(reference[f"equilibrium.time_slice.0.global_quantities.{leaf}"])


def _derived(derived, leaf):
    return float(derived[f"equilibrium.time_slice.0.global_quantities.{leaf}"])


def test_li_3_reproduces_the_reference(reference, derived):
    """The DD documents only "Internal inductance", so the ITER/Jackson third
    definition is written on the strength of this agreement, not on DD text."""
    assert _derived(derived, "li_3") == pytest.approx(_stored(reference, "li_3"), rel=0.01)


def test_the_efit_convention_reproduces_the_reference_beta_pol(reference, derived):
    """Proves the integrator: `int p dV`, `V` and `L_pol` are all right, because
    the convention the reference actually used comes back to 0.5%."""
    profiles = derived["equilibrium.time_slice.0.profiles_1d"]
    pressure = np.asarray(profiles["pressure"], float)
    volume = np.asarray(profiles["volume"], float)
    p_average = float(trapz_compat(pressure, x=volume)) / float(volume[-1])
    circumference = beta_poloidal_from_circumference(
        p_average, _derived(derived, "ip"), _derived(derived, "length_pol")
    )
    assert circumference == pytest.approx(_stored(reference, "beta_pol"), rel=0.005)


def test_the_dd_beta_pol_is_written_and_the_divergence_is_documented(reference, derived):
    """The DD leaf holds the DD definition even though it does not match OMFIT.
    This test exists to fail loudly if someone later "fixes" it by matching."""
    dd = _derived(derived, "beta_pol")
    stored = _stored(reference, "beta_pol")
    assert dd == pytest.approx(0.0227, rel=0.02)
    assert stored == pytest.approx(0.0287, rel=0.02)
    # 26% apart, and the gap is the geometric factor, not error.
    assert 0.75 < dd / stored < 0.82

    profiles = derived["equilibrium.time_slice.0.profiles_1d"]
    volume = float(np.asarray(profiles["volume"], float)[-1])
    r0 = float(derived["equilibrium.vacuum_toroidal_field.r0"])
    length_pol = _derived(derived, "length_pol")
    p_average = float(
        trapz_compat(np.asarray(profiles["pressure"], float), x=np.asarray(profiles["volume"], float))
    ) / volume
    circumference = beta_poloidal_from_circumference(
        p_average, _derived(derived, "ip"), length_pol
    )
    assert dd / circumference == pytest.approx(2.0 * volume / (r0 * length_pol**2), rel=1e-9)


def test_beta_tor_uses_the_dd_reference_field_and_says_so(reference, derived):
    """`B0` is `vacuum_toroidal_field.b0`, as the DD requires. That does not
    reproduce the reference, whose reference field is unidentified (#238); the
    gap is asserted rather than tuned away."""
    ratio = _derived(derived, "beta_tor") / _stored(reference, "beta_tor")
    assert 1.02 < ratio < 1.06, "beta_tor's known, documented divergence moved"
    # beta_normal inherits it, and nothing else.
    normal_ratio = _derived(derived, "beta_normal") / _stored(reference, "beta_normal")
    assert normal_ratio == pytest.approx(ratio, rel=0.05)


def test_length_pol_is_the_lcfs_perimeter(reference, derived):
    outline_r = np.asarray(reference["equilibrium.time_slice.0.boundary.outline.r"], float)
    outline_z = np.asarray(reference["equilibrium.time_slice.0.boundary.outline.z"], float)
    perimeter = float(
        np.sum(
            np.hypot(
                np.diff(np.r_[outline_r, outline_r[0]]),
                np.diff(np.r_[outline_z, outline_z[0]]),
            )
        )
    )
    assert _derived(derived, "length_pol") == pytest.approx(perimeter, rel=1e-6)


# --- the real sample and the summary ------------------------------------------


def test_the_packaged_sample_gets_physical_values_and_fills_the_summary():
    from vaft.database import _summary as summary_module
    from vaft.omas.sample import sample_ods

    try:
        ods = sample_ods()
    except Exception as exc:  # pragma: no cover - sample not packaged
        pytest.skip(f"39915 sample unavailable: {exc}")

    row = summary_module.extract_equilibrium_global(ods, 39915)[0]
    empty = {
        name
        for name, value in row.items()
        if value is None or (isinstance(value, float) and not np.isfinite(value))
    }
    assert not empty, f"summary columns still empty: {sorted(empty)}"

    # A low-beta VEST ohmic discharge: small betas, li of order one half.
    assert 0.0 < row["beta_pol"] < 1.0
    assert 0.0 < row["beta_tor"] < 0.1
    assert 0.0 < row["beta_normal"] < 5.0
    assert 0.1 < row["li_3"] < 3.0
    # The two conventions are both reported, and they are not the same number.
    assert row["beta_pol_circumference"] != row["beta_pol"]
    assert 0.0 < row["beta_pol_circumference"] < 1.0


def test_the_flux_surfaces_are_traced_once_not_twice():
    """Review finding: `int(B_pol^2 dV)` needs `bp_dl`, which is not a DD quantity
    and so is never written to the ODS -- the beta layer therefore repeated the
    whole trace the geometry updater had just done, at 0.95x its cost, while both
    docstrings claimed reuse. `vaft.database.summary` runs both over every shot in
    a range, so the doubling was the equilibrium cost of a database sweep.

    The surfaces are handed over now. What this pins is that they are handed over
    *and* that doing so changes no number.
    """
    from vaft.omas.sample import sample_ods

    try:
        shared = sample_ods()
    except Exception as exc:  # pragma: no cover - sample not packaged
        pytest.skip(f"39915 sample unavailable: {exc}")
    standalone = copy.deepcopy(shared)

    calls = []
    import vaft.process.equilibrium as process_equilibrium

    original = process_equilibrium.flux_surface_quantities

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    process_equilibrium.flux_surface_quantities = counted
    try:
        update.update_equilibrium_derived_profiles(shared)
        shared_traces = len(calls)
        calls.clear()
        update.update_equilibrium_global_quantities_beta_li(standalone)
        standalone_traces = len(calls)
    finally:
        process_equilibrium.flux_surface_quantities = original

    # Eight non-degenerate slices, so one trace each and no more, by either route.
    assert shared_traces == 8, f"entry point traced {shared_traces} times"
    assert standalone_traces == 8, f"standalone traced {standalone_traces} times"

    left = shared["equilibrium.time_slice.0.global_quantities"]
    right = standalone["equilibrium.time_slice.0.global_quantities"]
    for leaf in ("beta_pol", "beta_tor", "beta_normal", "li_3", "length_pol"):
        assert float(left[leaf]) == float(right[leaf]), leaf


# --- the reference major radius the DD scalars divide by ----------------------


def _minimal_ods(equilibrium_r0, equilibrium_b0, tf_r0=None, tf_b_r=None):
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods["equilibrium.vacuum_toroidal_field.r0"] = equilibrium_r0
    ods["equilibrium.vacuum_toroidal_field.b0"] = np.array([equilibrium_b0])
    if tf_r0 is not None:
        ods["tf.r0"] = tf_r0
    if tf_b_r is not None:
        ods["tf.b_field_tor_vacuum_r.data"] = np.full(8, tf_b_r)
    return ods


def test_a_stored_r0_that_disagrees_with_tf_is_rejected(caplog):
    """The VEST database stores an `equilibrium.vacuum_toroidal_field.r0` between
    0.19 and 0.35 on every shot sampled, while `tf.r0` is 0.4 -- and the two
    disagree about `B*R`, which is the physical invariant. `b0` alone matches
    `tf`'s field at 0.4, so `r0` is the corrupt half. `beta_pol` and `li_3` both
    divide by R_0, so trusting it inflates them by 1.15-2.1x.
    """
    from vaft.omas.update import resolve_reference_major_radius

    # Shot 39915 as the database actually holds it.
    ods = _minimal_ods(0.231317, 0.149799, tf_r0=0.4, tf_b_r=0.060704)
    with caplog.at_level("WARNING", logger="vaft.omas.update"):
        assert resolve_reference_major_radius(ods) == pytest.approx(0.4)
    assert "inconsistent with tf" in caplog.text
    assert "0.231" in caplog.text and "0.4" in caplog.text


def test_a_consistent_stored_r0_is_kept(caplog):
    """This detects one known corruption; it does not overrule a machine whose
    reference radius genuinely differs from its TF geometry."""
    from vaft.omas.update import resolve_reference_major_radius

    # b0*r0 == tf's B*R, so nothing is wrong and the equilibrium's own value stands.
    ods = _minimal_ods(0.6, 0.1, tf_r0=0.4, tf_b_r=0.06)
    with caplog.at_level("WARNING", logger="vaft.omas.update"):
        assert resolve_reference_major_radius(ods) == pytest.approx(0.6)
    assert "inconsistent" not in caplog.text


def test_without_tf_the_stored_r0_stands():
    """No cross-check available is not evidence of corruption."""
    from vaft.omas.update import resolve_reference_major_radius

    assert resolve_reference_major_radius(
        _minimal_ods(0.231317, 0.149799)
    ) == pytest.approx(0.231317)


def test_the_packaged_sample_is_unaffected_by_the_cross_check():
    """Its r0 is already 0.4, so the guard must be a no-op there -- a fix for bad
    data must not move good data."""
    from vaft.omas.sample import sample_ods
    from vaft.omas.update import resolve_reference_major_radius

    try:
        ods = sample_ods()
    except Exception as exc:  # pragma: no cover - sample not packaged
        pytest.skip(f"39915 sample unavailable: {exc}")

    stored = float(np.asarray(ods["equilibrium.vacuum_toroidal_field.r0"], float).ravel()[0])
    assert stored == pytest.approx(0.4)
    assert resolve_reference_major_radius(ods) == pytest.approx(stored)
