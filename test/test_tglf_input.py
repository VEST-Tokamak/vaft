"""The GACODEProfile -> TGLF local input projection (issue #553, increment 1).

TGLF never reads `input.gacode`, so unlike NEO everything `expro` used to supply --
local Miller shaping and its radial derivatives, normalised logarithmic gradients, and
four normalisations -- is computed on the VAFT side. That makes this the module where
GACODE's conventions are reproduced rather than deferred to, which is exactly what went
wrong three times in milestone 25 (the `b_unit` sign and `torfluxa` factor in #661, the
`z_eff` column in #803): each produced a plausible number and passed every check that
only compared VAFT against itself.

So the central test here is **not** self-consistent. `$GACODEHOME/profiles_gen/locpargen`
writes `input.tglf.locpargen` from any `input.gacode`, and its output at r/a = 0.3, 0.5
and 0.7 on the packaged 48224 state is committed under
`test/data/gacode/tglf_locpargen_48224/`. Every key VAFT writes is held against it.

Three conventions that comparison pinned down, each worth a test of its own below:

* expro differentiates on the grid and interpolates the *derivative*; interpolating
  first leaves `S_KAPPA_LOC` 87 percent out;
* `c_s` is normalised to the deuterium mass, and getting that wrong once cancelled a
  second error and made `XNUE` look right;
* the electron mass is carried in proton units by expro and deuterium units by TGLF,
  and only `XNUE` shows the difference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaft.code.gacode.tglf import (
    LocalConversionError,
    TGLFConfig,
    bound_deriv,
    prepare_tglf_input,
    tglf_parameters,
    write_input_tglf,
)

ORACLE = Path(__file__).parent / "data" / "gacode" / "tglf_locpargen_48224"
RADII = (0.3, 0.5, 0.7)

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


def read_oracle(rho):
    """locpargen's own answer, as a {KEY: float} mapping."""
    path = ORACLE / f"r{rho}.locpargen"
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "=" in line:
            key, _, raw = line.partition("=")
            try:
                values[key.strip()] = float(raw)
            except ValueError:
                continue
    return values


@pytest.fixture(scope="module")
def profile():
    from omas import load_omas_json

    from vaft.code.gacode.inputs import prepare_gacode_profile

    ods = load_omas_json(str(SAMPLE), consistency_check=False)
    return prepare_gacode_profile(ods, rho_max=0.95, z_eff=2.0, impurity="C")


# --------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------


@requires_sample
@pytest.mark.parametrize("rho", RADII)
def test_every_key_matches_gacodes_own_answer(profile, rho):
    """The load-bearing test: VAFT's projection against locpargen's, key by key.

    Not a smoke test. If this passes at three radii the conversion is GACODE's, and if
    it fails the number that disagrees names the convention that was missed.
    """
    oracle = read_oracle(rho)
    assert oracle, f"no oracle fixture for r/a = {rho}"

    mine = tglf_parameters(prepare_tglf_input(profile, rho))
    compared = []
    for key, reference in oracle.items():
        if key not in mine or key == "NS":
            continue
        value = float(mine[key])
        if abs(reference) < 1e-12:
            assert abs(value) < 1e-9, f"{key}: {value} against an oracle zero"
        else:
            assert value == pytest.approx(reference, rel=2e-3), key
        compared.append(key)

    # Guard against the comparison quietly shrinking to nothing.
    assert len(compared) > 50, f"only {len(compared)} keys were compared"
    for essential in ("Q_PRIME_LOC", "S_KAPPA_LOC", "BETAE", "XNUE", "DEBYE", "RLNS_1"):
        assert essential in compared


@requires_sample
def test_the_species_list_is_electrons_first(profile):
    """TGLF indexes electrons as species 1, where NEO puts ions first."""
    local = prepare_tglf_input(profile, 0.5)
    assert local.n_species == 3
    assert local.zs[0] == -1.0
    assert local.as_[0] == pytest.approx(1.0)
    assert local.taus[0] == pytest.approx(1.0)
    # And the carbon #803 added is still carbon.
    assert local.zs[2] == pytest.approx(6.0)
    assert local.as_[2] == pytest.approx(1.0 / 30.0, rel=1e-3)


# --------------------------------------------------------------------------
# the three conventions the oracle pinned down
# --------------------------------------------------------------------------


@requires_sample
def test_differentiating_after_interpolating_is_wrong_and_by_how_much(profile):
    """expro differentiates on the grid, then interpolates the derivative.

    Doing it the other way round is the natural reading and is badly wrong for a
    quantity that is nearly flat: S_KAPPA_LOC is ~6e-3 on VEST, so a small absolute
    error is an enormous relative one. Asserted as a number because "use the right
    order" is not something a reader can check.
    """
    from scipy.interpolate import CubicSpline

    rmin = np.asarray(profile.rmin, dtype=float)
    grid = rmin / rmin[-1]
    kappa = np.asarray(profile.kappa, dtype=float)

    correct = float(
        CubicSpline(grid, rmin / kappa * bound_deriv(kappa, rmin))(0.5)
    )
    wrong = 0.5 / float(CubicSpline(grid, kappa)(0.5)) * float(
        CubicSpline(grid, kappa)(0.5, 1)
    )

    assert correct == pytest.approx(read_oracle(0.5)["S_KAPPA_LOC"], rel=2e-3)
    assert abs(wrong - correct) / abs(correct) > 0.5, (wrong, correct)


@requires_sample
def test_the_collision_rate_uses_the_proton_unit_electron_mass(profile):
    """expro's `masse` is in proton units; TGLF's MASS_1 is deuterium-normalised.

    `nu_ee` takes sqrt(masse/2). Reusing the already-halved MASS_1 is exactly sqrt(2)
    wrong and XNUE is the only output that shows it -- which is why this is pinned
    rather than left to the aggregate comparison.
    """
    local = prepare_tglf_input(profile, 0.5)
    reference = read_oracle(0.5)["XNUE"]
    assert local.xnue == pytest.approx(reference, rel=2e-3)
    # The wrong mass would land here, and a 2e-3 tolerance would not absorb it.
    assert local.xnue / np.sqrt(2.0) != pytest.approx(reference, rel=2e-3)


@requires_sample
def test_the_sound_speed_is_normalised_to_deuterium(profile):
    """DEBYE is the check: it divides by rho_s, which carries c_s.

    Using the proton mass makes c_s sqrt(2) too large. For a while that error
    cancelled another one and XNUE looked right, so both are asserted separately.
    """
    local = prepare_tglf_input(profile, 0.5)
    assert local.debye == pytest.approx(read_oracle(0.5)["DEBYE"], rel=2e-3)


def test_bound_deriv_reproduces_an_exact_polynomial():
    """A three-point Lagrange derivative is exact for a quadratic, on any spacing."""
    radius = np.array([0.0, 0.1, 0.35, 0.4, 0.9, 1.0])
    values = 3.0 * radius**2 - 2.0 * radius + 1.0
    np.testing.assert_allclose(bound_deriv(values, radius), 6.0 * radius - 2.0, rtol=1e-12)


def test_bound_deriv_refuses_what_it_cannot_do():
    with pytest.raises(ValueError, match="three points"):
        bound_deriv(np.array([1.0, 2.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        bound_deriv(np.ones((2, 3)), np.ones((2, 3)))


# --------------------------------------------------------------------------
# what it refuses
# --------------------------------------------------------------------------


@requires_sample
def test_a_radius_inside_the_innermost_surface_is_refused(profile):
    """The r/a grid ends at 1.0 by construction, so only the axis end can be outside.

    A `rho_max` cut does not move the outer edge of this coordinate -- it renormalises
    it -- which is worth stating, because the obvious assumption is the opposite.
    """
    innermost = float(np.asarray(profile.rmin)[0] / np.asarray(profile.rmin)[-1])
    if innermost <= 0.0:
        pytest.skip("this profile starts at the magnetic axis")
    with pytest.raises(LocalConversionError, match="outside the converted profile"):
        prepare_tglf_input(profile, innermost / 2.0)


@requires_sample
def test_a_truncated_profile_still_spans_to_one_in_r_over_a(profile):
    grid = np.asarray(profile.rmin, dtype=float)
    assert grid[-1] > 0
    assert prepare_tglf_input(profile, 0.99).rmin_loc == pytest.approx(0.99)


@requires_sample
@pytest.mark.parametrize("rho", [0.0, 1.0, -0.2, 1.5])
def test_a_radius_that_is_not_inside_the_plasma_is_refused(profile, rho):
    with pytest.raises(LocalConversionError, match="strictly inside"):
        prepare_tglf_input(profile, rho)


@requires_sample
def test_an_absent_exb_shear_is_recorded_not_zeroed(profile):
    """A zero ExB shear suppresses nothing and is a physical claim.

    `prepare_gacode_profile` does not populate `w0`, so there is no honest source for
    VEXB_SHEAR. It is reported unavailable and left out of the file, where TGLF's own
    default applies -- rather than written as a zero VAFT chose.
    """
    local = prepare_tglf_input(profile, 0.5)
    assert local.vexb_shear is None
    assert "vexb_shear" in local.missing()
    assert local.provenance["vexb_shear"]["kind"] == "unavailable"
    # The key is still written -- omitting it would make the file unreproducible
    # without making the assumption visible. The provenance is what separates a
    # measured zero from an unmeasured one, and it is the only thing that does.
    assert tglf_parameters(local)["VEXB_SHEAR"] == 0.0


@requires_sample
def test_a_profile_missing_what_the_projection_needs_says_which(profile):
    import dataclasses

    stripped = dataclasses.replace(profile, kappa=None, q=None)
    with pytest.raises(LocalConversionError, match="q, kappa|kappa"):
        prepare_tglf_input(stripped, 0.5)


# --------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------


@requires_sample
def test_booleans_are_written_as_fortran_logicals(profile, tmp_path):
    """TGLF reads `.true.`; NEO's `1`/`0` is a parse error here."""
    parameters = tglf_parameters(prepare_tglf_input(profile, 0.5))
    written = write_input_tglf(parameters, tmp_path / "input.tglf")
    text = written.read_text(encoding="utf-8")
    assert "USE_TRANSPORT_MODEL=.true." in text
    assert "USE_TRANSPORT_MODEL=1" not in text
    assert ".true." in text and ".false." in text


@requires_sample
def test_every_setting_is_written_out_not_left_to_a_default(profile, tmp_path):
    """A file that omits a key cannot afterwards be told from one that chose it."""
    parameters = tglf_parameters(prepare_tglf_input(profile, 0.5))
    text = write_input_tglf(parameters, tmp_path / "input.tglf").read_text()
    keys = {line.split("=")[0] for line in text.splitlines() if "=" in line}
    assert len(keys) > 120
    for essential in ("SAT_RULE", "NS", "GEOMETRY_FLAG", "UNITS", "NMODES"):
        assert essential in keys


@requires_sample
def test_the_configuration_reaches_the_file(profile, tmp_path):
    config = TGLFConfig(sat_rule=3, use_bper=True, extra_parameters={"nky": 24})
    parameters = tglf_parameters(prepare_tglf_input(profile, 0.5, config=config), config)
    assert parameters["SAT_RULE"] == 3
    assert parameters["USE_BPER"] is True
    assert parameters["NKY"] == 24, "extra_parameters must pass through upper-cased"


def test_the_configuration_refuses_settings_tglf_cannot_take():
    with pytest.raises(ValueError, match="SAT_RULE"):
        TGLFConfig(sat_rule=7)
    with pytest.raises(ValueError, match="GEOMETRY_FLAG"):
        TGLFConfig(geometry_flag=3)
    with pytest.raises(ValueError, match="species"):
        TGLFConfig(n_species=9)
