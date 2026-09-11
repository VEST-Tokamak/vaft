"""Sauter and Redl neoclassical formulas, against NEO and against their limits.

The decisive test here is `test_sauter_reproduces_neo_reg18` and its Redl twin.
NEO carries its own implementations of both formulations
(`compute_Sauter` and `compute_Sauter_mod` in `neo/src/neo_theory.f90`) and
writes both to `out.neo.theory` on every run, so a stored run of the shipped
`reg18` regression case is an independent reference for the whole chain: the
coefficient fits, the collisionality convention and the current assembly.

The fixtures in `test/data/gacode/neo_reg18/` are a real NEO run of that case,
GACODE 6357db30, whose `out.neo.prec` reproduced the shipped reference value
exactly. Nothing here needs GACODE installed (issue #550).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaft.formula.neoclassical import (
    BootstrapCoefficients,
    coulomb_logarithm_electron_sauter,
    coulomb_logarithm_ion_sauter,
    electron_collisionality_sauter,
    ion_collisionality_sauter,
    redl_bootstrap_coefficients,
    redl_bootstrap_current,
    redl_neoclassical_conductivity,
    sauter_bootstrap_coefficients,
    sauter_bootstrap_current,
    sauter_neoclassical_conductivity,
    sauter_spitzer_conductivity,
    trapped_particle_fraction,
)

FIXTURES = Path(__file__).parent / "data" / "gacode"

#: Two stored NEO runs, deliberately in different regimes: reg18 is GACODE's
#: own conventional-aspect-ratio regression case with a carbon impurity, and
#: vest_48224 is VEST's packaged kinetic state, whose trapped fraction is half
#: again as large. A fit that matches in only one of them is not verified.
NEO_RUNS = {"reg18": FIXTURES / "neo_reg18", "vest_48224": FIXTURES / "neo_vest_48224"}


def _diagnostic_geo(directory: Path) -> dict[str, float]:
    """Read the named scalars from out.neo.diagnostic_geo's comment header."""
    values: dict[str, float] = {}
    for line in (directory / "out.neo.diagnostic_geo").read_text().splitlines():
        if not line.startswith("#") or "=" not in line:
            continue
        name, _, number = line[1:].partition("=")
        try:
            values[name.strip()] = float(number)
        except ValueError:
            continue
    return values


class _NeoRun:
    """One stored NEO run, in NEO's own normalised units.

    Reconstructing NEO's two collisionalities is the only non-obvious part, and
    it is done from NEO's own definitions in `neo/src/neo_theory.f90` rather
    than from the paper, because the point of the comparison is that VAFT's
    coefficient fits agree once both sides are handed the same numbers.
    `nu(is)` is the fifth per-species column of out.neo.equil, and the two
    `(4/3)/sqrt(pi)` and `(4/3)/sqrt(2 pi)` factors are NEO's.
    """

    def __init__(self, directory: Path) -> None:
        equilibrium = np.loadtxt(directory / "out.neo.equil")
        species = np.loadtxt(directory / "out.neo.species")
        self.theory = np.loadtxt(directory / "out.neo.theory")
        self.transport = np.loadtxt(directory / "out.neo.transport")

        geometry = _diagnostic_geo(directory)
        self.f_trap = geometry["f_trap"]
        self.i_over_psi_prime = geometry["I/psi'"]

        self.mass = species[0::2]
        self.charge = species[1::2]
        n_species = self.charge.size

        r_over_a, _dphidr, self.q, self.rho_star, self.rmaj = equilibrium[:5]
        self.density = equilibrium[7 + 0 :: 5]
        self.temperature = equilibrium[7 + 1 :: 5]
        self.dlnndr = equilibrium[7 + 2 :: 5]
        self.dlntdr = equilibrium[7 + 3 :: 5]
        collision_rate = equilibrium[7 + 4 :: 5]

        self.electron = int(np.argmin(self.charge))
        self.ions = [i for i in range(n_species) if i != self.electron]
        self.main_ion = self.ions[0]

        self.epsilon = r_over_a / self.rmaj
        self.z_eff = sum(
            self.density[i] * self.charge[i] ** 2 for i in self.ions
        ) / self.density[self.electron]

        electron_rate = (
            collision_rate[self.main_ion]
            * (4.0 / 3.0)
            / np.sqrt(np.pi)
            * np.sqrt(self.mass[self.main_ion] / self.mass[self.electron])
            * (self.temperature[self.main_ion] / self.temperature[self.electron]) ** 1.5
            / self.charge[self.main_ion] ** 2
        ) * (self.density[self.electron] / self.density[self.main_ion]) * self.z_eff / (
            self.charge[self.main_ion] ** 2
        )
        self.nu_e_star = (
            electron_rate
            * self.rmaj
            * abs(self.q)
            / (
                self.epsilon**1.5
                * np.sqrt(self.temperature[self.electron] / self.mass[self.electron])
            )
        )

        ion_rate = collision_rate[self.main_ion] * (4.0 / 3.0) / np.sqrt(2.0 * np.pi)
        # NEO's 2013 multi-species reading: scale by the summed ion density.
        self.nu_i_star = (
            ion_rate
            * self.rmaj
            * abs(self.q)
            / (
                self.epsilon**1.5
                * np.sqrt(self.temperature[self.main_ion] / self.mass[self.main_ion])
            )
            / self.density[self.main_ion]
            * sum(self.density[i] for i in self.ions)
        )

        self.pressure_electron = (
            self.density[self.electron] * self.temperature[self.electron]
        )
        self.pressure_ion = sum(
            self.density[i] * self.temperature[i] for i in self.ions
        )
        self.dp = sum(
            self.density[i] * self.temperature[i] * (self.dlntdr[i] + self.dlnndr[i])
            for i in range(n_species)
        )

    @property
    def neo_bootstrap_current(self) -> float:
        """NEO's own drift-kinetic <j_par B>, out.neo.transport column 2."""
        return float(self.transport[2])

    def current_arguments(self, **overrides) -> dict:
        """Keyword arguments for the two bootstrap-current entry points.

        NEO writes `+I/psi' * rho * p * (a/L_p)`, VAFT writes
        `-I_psi * dp/dpsi`; `a/L_p` is minus the radial logarithmic gradient, so
        the two agree with `I_psi = -(I/psi') * rho`. Both expressions are
        dimensionally homogeneous, so passing NEO's normalised quantities
        throughout is consistent.
        """
        arguments = dict(
            f_trap=self.f_trap,
            nu_e_star=self.nu_e_star,
            nu_i_star=self.nu_i_star,
            Z_eff=self.z_eff,
            I_psi=-self.i_over_psi_prime * self.rho_star,
            p_e=self.pressure_electron,
            p_i=self.pressure_ion,
            dp_dpsi=self.dp,
            dln_Te_dpsi=self.dlntdr[self.electron],
            dln_Ti_dpsi=self.dlntdr[self.main_ion],
        )
        arguments.update(overrides)
        return arguments


@pytest.fixture(scope="module")
def runs() -> dict[str, _NeoRun]:
    return {name: _NeoRun(path) for name, path in NEO_RUNS.items()}


@pytest.fixture(scope="module")
def reg18(runs) -> _NeoRun:
    return runs["reg18"]


@pytest.fixture(scope="module")
def vest(runs) -> _NeoRun:
    return runs["vest_48224"]


# --------------------------------------------------------------------------
# Verification against NEO
# --------------------------------------------------------------------------


def test_reg18_fixture_is_the_case_we_think_it_is(reg18):
    """Guard the fixture: three species, a carbon impurity, low collisionality."""
    assert reg18.charge.size == 3
    assert sorted(reg18.charge) == [-1.0, 1.0, 6.0]
    assert reg18.z_eff == pytest.approx(1.9109, abs=1e-4)
    assert reg18.epsilon == pytest.approx(0.17264, abs=1e-5)
    # Banana regime, which is where the trapped-particle physics is strongest
    # and where the two fits are most nearly equal.
    assert reg18.nu_e_star < 0.1
    assert reg18.nu_i_star < 0.1


@pytest.mark.parametrize("case", sorted(NEO_RUNS))
def test_sauter_reproduces_neo(runs, case):
    """VAFT's Sauter 1999 result equals NEO's, to NEO's own output precision.

    `out.neo.theory` column 10 is `SjparB`, NEO's `compute_Sauter`. NEO writes
    it in `e16.8`, so eight significant figures is all the file carries and
    agreement cannot be asserted tighter than that.

    Both regimes are checked because the two fits differ mainly through the
    trapped fraction, and reg18's 0.56 alone would not exercise the range VEST
    puts them in.
    """
    run = runs[case]
    current = sauter_bootstrap_current(**run.current_arguments())
    assert current == pytest.approx(run.theory[10], rel=1e-7)


@pytest.mark.parametrize("case", sorted(NEO_RUNS))
def test_redl_reproduces_neo(runs, case):
    """VAFT's Redl 2021 result equals NEO's `compute_Sauter_mod`.

    The last column of `out.neo.theory` is `jpar_Smod`.
    """
    run = runs[case]
    current = redl_bootstrap_current(**run.current_arguments())
    assert current == pytest.approx(run.theory[-1], rel=1e-7)


def test_the_two_fixtures_really_are_different_regimes(reg18, vest):
    """Guard the premise of parametrising over both runs."""
    assert reg18.f_trap == pytest.approx(0.563, abs=0.01)
    assert vest.f_trap == pytest.approx(0.732, abs=0.01)
    assert reg18.z_eff > 1.5 and vest.z_eff == pytest.approx(1.0, abs=1e-6)


def test_redl_is_closer_than_sauter_to_neo_on_the_vest_state(vest):
    """The reason both formulations exist, measured on VEST's own kinetic state.

    At f_trap = 0.73 the 1999 fit is outside the range it was built on and the
    2021 refit is not. Neither matches the drift-kinetic solve -- they are
    analytic approximations to it -- but Redl is several times closer, which is
    what makes it the defensible default for a spherical tokamak.

    This is a physics-model comparison, not a solver check: a Sauter-NEO gap of
    this size is a real property of the model, not a failure (issue #550).
    """
    arguments = vest.current_arguments()
    reference = vest.neo_bootstrap_current
    sauter_error = abs(sauter_bootstrap_current(**arguments) / reference - 1.0)
    redl_error = abs(redl_bootstrap_current(**arguments) / reference - 1.0)
    assert sauter_error > 0.05, "Sauter should visibly overshoot NEO here"
    assert redl_error < 0.03, "Redl should stay within a few percent"
    assert redl_error < 0.5 * sauter_error


def test_sauter_alpha_matches_neos_poloidal_flow_coefficient(reg18):
    """An independent check on alpha alone, not just on the assembled current.

    NEO writes `Sk = -alpha_S` as column 11, so this pins the ion-collisionality
    branch of the fit without the pressure gradients being able to hide an
    error in it.
    """
    coefficients = sauter_bootstrap_coefficients(
        reg18.f_trap, reg18.nu_e_star, reg18.nu_i_star, reg18.z_eff
    )
    assert coefficients.alpha == pytest.approx(-reg18.theory[11], rel=1e-7)


def test_the_two_models_agree_more_closely_at_conventional_aspect_ratio(reg18, vest):
    """The Sauter-Redl gap grows with the trapped fraction.

    Measured across the two stored runs rather than by moving one of them, so
    the comparison is between two real equilibria.
    """
    def gap(run):
        arguments = run.current_arguments()
        return abs(
            redl_bootstrap_current(**arguments)
            / sauter_bootstrap_current(**arguments)
            - 1.0
        )

    assert gap(reg18) < 0.02
    assert gap(vest) > 2.0 * gap(reg18)


def test_the_models_separate_at_spherical_tokamak_shape(reg18):
    """At VEST's trapped fraction the two fits no longer agree.

    The 1999 fit is an extrapolation there and the 2021 refit is not, so the
    growing gap is the documented reason `redl_*` exists (issue #550).
    """
    arguments = reg18.current_arguments()
    conventional = abs(
        redl_bootstrap_current(**arguments) / sauter_bootstrap_current(**arguments) - 1.0
    )
    arguments["f_trap"] = trapped_particle_fraction(0.6)
    spherical = abs(
        redl_bootstrap_current(**arguments) / sauter_bootstrap_current(**arguments) - 1.0
    )
    assert spherical > 3.0 * conventional


# --------------------------------------------------------------------------
# Trapped fraction
# --------------------------------------------------------------------------


def test_trapped_fraction_vanishes_on_axis():
    assert trapped_particle_fraction(0.0) == pytest.approx(0.0)


def test_trapped_fraction_rises_with_inverse_aspect_ratio():
    epsilon = np.linspace(0.0, 0.95, 40)
    assert np.all(np.diff(trapped_particle_fraction(epsilon)) > 0.0)


def test_trapped_fraction_at_vest_and_at_a_conventional_tokamak():
    """VEST traps far more of its distribution than a conventional device.

    At small inverse aspect ratio the formula should recover the textbook
    `sqrt(2 eps)` estimate; at VEST's it is far outside that expansion, which is
    the reason bootstrap current is a first-order concern for a spherical
    tokamak.
    """
    conventional = trapped_particle_fraction(0.1)
    vest = trapped_particle_fraction(0.6)
    assert conventional == pytest.approx(np.sqrt(2.0 * 0.1), rel=0.02)
    assert vest == pytest.approx(0.906, abs=0.005)
    assert vest > 2.0 * conventional


def test_trapped_fraction_is_within_a_percent_of_neos_shaped_value(reg18):
    """The circular approximation against a real shaped equilibrium.

    NEO computes f_trap by integrating over the field strength on the surface
    and writes it to out.neo.diagnostic_geo. reg18 is mildly shaped, so the
    circular formula should be close but not equal -- and the gap is the size
    of the error a caller accepts by using this instead of the equilibrium.
    """
    circular = trapped_particle_fraction(reg18.epsilon)
    assert circular == pytest.approx(reg18.f_trap, rel=0.01)
    assert circular != reg18.f_trap


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, np.nan])
def test_trapped_fraction_rejects_input_outside_its_domain(bad):
    with pytest.raises(ValueError):
        trapped_particle_fraction(bad)


# --------------------------------------------------------------------------
# Coulomb logarithms and collisionality
# --------------------------------------------------------------------------


def test_sauter_electron_coulomb_log_differs_from_the_nrl_one_by_its_constant():
    """The two conventions differ by exactly 31.3 - 30.9, and no more.

    Mixing them is the failure this pins: the collisionality fits here were
    built on the Sauter constant (issue #353).
    """
    from vaft.formula.equilibrium import coulomb_logarithm_from_n_T

    n_e, T_e = 5.0e19, 300.0
    difference = coulomb_logarithm_electron_sauter(n_e, T_e) - coulomb_logarithm_from_n_T(
        n_e, T_e
    )
    assert difference == pytest.approx(0.4, abs=1e-12)


def test_ion_coulomb_log_falls_with_the_cube_of_the_charge():
    single = coulomb_logarithm_ion_sauter(1.0e19, 100.0, 1.0)
    carbon = coulomb_logarithm_ion_sauter(1.0e19, 100.0, 6.0)
    assert single - carbon == pytest.approx(3.0 * np.log(6.0), rel=1e-12)


def test_electron_collisionality_scales_as_the_paper_says():
    base = dict(n_e=5.0e19, T_e=300.0, q=2.0, R=0.4, epsilon=0.1, Z_eff=2.0,
                ln_Lambda_e=15.0)
    reference = electron_collisionality_sauter(**base)
    assert electron_collisionality_sauter(**{**base, "n_e": 1.0e20}) == pytest.approx(
        2.0 * reference, rel=1e-12
    ), "linear in density"
    assert electron_collisionality_sauter(**{**base, "T_e": 600.0}) == pytest.approx(
        reference / 4.0, rel=1e-12
    ), "inverse square in temperature"
    assert electron_collisionality_sauter(**{**base, "epsilon": 0.4}) == pytest.approx(
        reference / 8.0, rel=1e-12
    ), "epsilon to the minus three halves"
    assert electron_collisionality_sauter(**{**base, "Z_eff": 4.0}) == pytest.approx(
        2.0 * reference, rel=1e-12
    ), "linear in Z_eff"
    assert electron_collisionality_sauter(**{**base, "q": -2.0}) == pytest.approx(
        reference, rel=1e-12
    ), "the magnitude of q is used, so the COCOS sign must not change nu_e*"


def test_ion_collisionality_scales_with_the_fourth_power_of_charge():
    base = dict(n_i=1.0e19, T_i=100.0, q=2.0, R=0.4, epsilon=0.3, Z=1.0,
                ln_Lambda_ii=15.0)
    reference = ion_collisionality_sauter(**base)
    assert ion_collisionality_sauter(**{**base, "Z": 2.0}) == pytest.approx(
        16.0 * reference, rel=1e-12
    )


def test_collisionality_defaults_to_the_sauter_coulomb_log():
    n_e, T_e = 5.0e19, 300.0
    explicit = electron_collisionality_sauter(
        n_e, T_e, 2.0, 0.4, 0.3, 2.0,
        ln_Lambda_e=coulomb_logarithm_electron_sauter(n_e, T_e),
    )
    implicit = electron_collisionality_sauter(n_e, T_e, 2.0, 0.4, 0.3, 2.0)
    assert implicit == pytest.approx(explicit, rel=1e-15)


# --------------------------------------------------------------------------
# Conductivity
# --------------------------------------------------------------------------


def test_neoclassical_conductivity_never_exceeds_spitzer():
    """Trapping can only remove current carriers, never add them."""
    spitzer = sauter_spitzer_conductivity(300.0, 2.0, 15.0)
    trapped = np.linspace(0.0, 0.95, 20)
    for collisionality in (0.0, 0.1, 1.0, 10.0):
        for model in (sauter_neoclassical_conductivity, redl_neoclassical_conductivity):
            values = model(spitzer, trapped, collisionality, 2.0)
            assert np.all(values <= spitzer * (1.0 + 1e-12))
            assert np.all(values > 0.0)


def test_conductivity_reduces_to_spitzer_with_no_trapped_particles():
    spitzer = sauter_spitzer_conductivity(300.0, 2.0, 15.0)
    assert sauter_neoclassical_conductivity(spitzer, 0.0, 0.5, 2.0) == pytest.approx(
        spitzer
    )
    assert redl_neoclassical_conductivity(spitzer, 0.0, 0.5, 2.0) == pytest.approx(
        spitzer
    )


def test_collisions_restore_the_conductivity_trapping_removed():
    """As nu_e* rises the neoclassical correction weakens towards Spitzer."""
    spitzer = sauter_spitzer_conductivity(300.0, 2.0, 15.0)
    collisionalities = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
    ratios = sauter_neoclassical_conductivity(spitzer, 0.6, collisionalities, 2.0) / spitzer
    assert np.all(np.diff(ratios) > 0.0)
    assert ratios[-1] > 0.9


def test_spitzer_conductivity_scales_with_temperature_to_the_three_halves():
    low = sauter_spitzer_conductivity(100.0, 1.0, 15.0)
    high = sauter_spitzer_conductivity(400.0, 1.0, 15.0)
    assert high / low == pytest.approx(8.0, rel=1e-12)


# --------------------------------------------------------------------------
# Bootstrap coefficients and their limits
# --------------------------------------------------------------------------


def test_bootstrap_coefficients_vanish_with_no_trapped_particles():
    """No trapped particles, no banana current: only alpha survives."""
    for model in (sauter_bootstrap_coefficients, redl_bootstrap_coefficients):
        coefficients = model(0.0, 0.5, 0.5, 2.0)
        assert coefficients.L31 == pytest.approx(0.0)
        assert coefficients.L32 == pytest.approx(0.0)
        assert coefficients.L34 == pytest.approx(0.0)


def test_sauter_alpha_reaches_its_banana_limit():
    """At nu_i* = 0 and f_t = 0, alpha is the paper's alpha_0 = -1.17."""
    coefficients = sauter_bootstrap_coefficients(0.0, 0.0, 0.0, 1.0)
    assert coefficients.alpha == pytest.approx(-1.17, rel=1e-12)


def test_redl_alpha_reaches_its_own_banana_limit():
    coefficients = redl_bootstrap_coefficients(0.0, 0.0, 0.0, 1.0)
    assert coefficients.alpha == pytest.approx(-0.62 / 0.53, rel=1e-12)


def test_alpha_is_negative_in_the_banana_regime():
    """The ion-temperature term opposes the density and electron terms."""
    for model in (sauter_bootstrap_coefficients, redl_bootstrap_coefficients):
        assert model(0.5, 0.01, 0.01, 1.5).alpha < 0.0


def test_collisions_suppress_the_bootstrap_coefficients():
    """L31 falls monotonically as the plasma leaves the banana regime."""
    collisionalities = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
    for model in (sauter_bootstrap_coefficients, redl_bootstrap_coefficients):
        l31 = np.asarray(model(0.5, collisionalities, collisionalities, 2.0).L31)
        assert np.all(np.diff(l31) < 0.0)


def test_l31_grows_with_the_trapped_fraction():
    trapped = np.linspace(0.0, 0.9, 20)
    for model in (sauter_bootstrap_coefficients, redl_bootstrap_coefficients):
        l31 = np.asarray(model(trapped, 0.05, 0.05, 1.5).L31)
        assert np.all(np.diff(l31) > 0.0)


def test_coefficients_are_a_named_tuple_in_a_fixed_order():
    coefficients = sauter_bootstrap_coefficients(0.5, 0.1, 0.1, 2.0)
    assert isinstance(coefficients, BootstrapCoefficients)
    assert tuple(coefficients) == (
        coefficients.L31,
        coefficients.L32,
        coefficients.L34,
        coefficients.alpha,
    )


def test_redl_sets_l34_equal_to_l31():
    """Documented: Redl does not refit L34, and the field exists for symmetry."""
    coefficients = redl_bootstrap_coefficients(0.5, 0.1, 0.1, 2.0)
    assert coefficients.L34 == coefficients.L31


# --------------------------------------------------------------------------
# Bootstrap current: sign, scaling and shape
# --------------------------------------------------------------------------


def _current_state(**overrides) -> dict:
    state = dict(
        f_trap=0.6, nu_e_star=0.05, nu_i_star=0.05, Z_eff=1.5,
        I_psi=0.2, p_e=800.0, p_i=400.0,
        dp_dpsi=-4.0e3, dln_Te_dpsi=-2.0, dln_Ti_dpsi=-2.0,
    )
    state.update(overrides)
    return state


def test_a_falling_pressure_profile_drives_a_positive_bootstrap_current():
    """With I > 0 and pressure falling outward, the L31 term is positive.

    A sign flip here is the signature of a COCOS or Wb-per-radian mistake in the
    caller, which is why the convention is asserted rather than left implicit.
    """
    assert sauter_bootstrap_current(**_current_state()) > 0.0
    assert redl_bootstrap_current(**_current_state()) > 0.0


def test_the_current_reverses_with_the_sign_of_the_flux_function():
    forward = sauter_bootstrap_current(**_current_state())
    reversed_field = sauter_bootstrap_current(**_current_state(I_psi=-0.2))
    assert reversed_field == pytest.approx(-forward, rel=1e-12)


def test_a_flat_plasma_drives_no_bootstrap_current():
    state = _current_state(dp_dpsi=0.0, dln_Te_dpsi=0.0, dln_Ti_dpsi=0.0)
    assert sauter_bootstrap_current(**state) == pytest.approx(0.0)
    assert redl_bootstrap_current(**state) == pytest.approx(0.0)


def test_the_current_is_linear_in_the_gradients():
    single = sauter_bootstrap_current(**_current_state())
    doubled = sauter_bootstrap_current(
        **_current_state(dp_dpsi=-8.0e3, dln_Te_dpsi=-4.0, dln_Ti_dpsi=-4.0)
    )
    assert doubled == pytest.approx(2.0 * single, rel=1e-12)


def test_the_ion_temperature_term_opposes_the_others():
    """alpha < 0 and L34 > 0, so an ion temperature gradient reduces the total."""
    without = sauter_bootstrap_current(**_current_state(dln_Ti_dpsi=0.0))
    with_gradient = sauter_bootstrap_current(**_current_state())
    assert with_gradient < without


# --------------------------------------------------------------------------
# Array/scalar behaviour and input validation
# --------------------------------------------------------------------------


def test_scalar_and_array_calls_agree_elementwise():
    trapped = np.array([0.2, 0.5, 0.7])
    vectorised = np.asarray(
        sauter_bootstrap_coefficients(trapped, 0.05, 0.05, 1.8).L31
    )
    elementwise = [
        sauter_bootstrap_coefficients(value, 0.05, 0.05, 1.8).L31 for value in trapped
    ]
    np.testing.assert_allclose(vectorised, elementwise, rtol=1e-15)


def test_scalar_input_returns_a_python_float():
    assert isinstance(trapped_particle_fraction(0.3), float)
    assert isinstance(sauter_bootstrap_coefficients(0.5, 0.1, 0.1, 2.0).L31, float)


def test_array_input_returns_an_array_of_the_same_shape():
    trapped = np.linspace(0.1, 0.8, 7)
    result = np.asarray(sauter_bootstrap_current(**_current_state(f_trap=trapped)))
    assert result.shape == trapped.shape


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (sauter_bootstrap_coefficients, {"f_trap": 1.5}),
        (sauter_bootstrap_coefficients, {"f_trap": -0.1}),
        (sauter_bootstrap_coefficients, {"nu_e_star": -1.0}),
        (sauter_bootstrap_coefficients, {"nu_i_star": -1.0}),
        (sauter_bootstrap_coefficients, {"Z_eff": 0.0}),
        (redl_bootstrap_coefficients, {"f_trap": np.nan}),
        (redl_bootstrap_coefficients, {"Z_eff": 0.5}),
    ],
)
def test_coefficients_reject_input_outside_their_domain(model, kwargs):
    arguments = {"f_trap": 0.5, "nu_e_star": 0.1, "nu_i_star": 0.1, "Z_eff": 2.0}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        model(**arguments)


def test_redl_refuses_a_charge_below_one():
    """sqrt(Z - 1) appears in the refit, so Z_eff < 1 is not merely unphysical."""
    with pytest.raises(ValueError, match="Z_eff must be >= 1"):
        redl_neoclassical_conductivity(1.0e6, 0.5, 0.1, 0.9)


@pytest.mark.parametrize(
    "function",
    [coulomb_logarithm_electron_sauter, electron_collisionality_sauter],
)
def test_non_positive_density_is_rejected(function):
    with pytest.raises(ValueError):
        if function is coulomb_logarithm_electron_sauter:
            function(0.0, 100.0)
        else:
            function(0.0, 100.0, 2.0, 0.4, 0.3, 2.0)


def test_non_finite_gradients_are_rejected():
    with pytest.raises(ValueError, match="dp_dpsi must be finite"):
        sauter_bootstrap_current(**_current_state(dp_dpsi=np.nan))


def test_the_spitzer_coefficient_agrees_with_the_packages_own_at_z_one():
    """An independent check on 1.9012e4, against a formula from a different source.

    `vaft.formula.equilibrium.spitzer_resistivity_from_T_e_Z_eff_ln_Lambda` is
    the NRL form with a linear Z dependence. At Z_eff = 1 the two charge
    treatments coincide, so the prefactors must agree to about a percent; a
    typo in either would be far larger than that.
    """
    from vaft.formula.equilibrium import spitzer_resistivity_from_T_e_Z_eff_ln_Lambda

    T_e, ln_Lambda = 1.0e3, 17.0
    mine = sauter_spitzer_conductivity(T_e, 1.0, ln_Lambda)
    theirs = 1.0 / spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(T_e, 1.0, ln_Lambda)
    assert mine == pytest.approx(theirs, rel=0.02)


def test_the_two_spitzer_conventions_diverge_at_higher_charge():
    """The documented reason not to mix them: N_Z is not a linear Z dependence.

    At Z_eff = 3 the two differ by tens of percent, which is why the
    neoclassical corrections here must be paired with this module's Spitzer
    reference rather than the NRL one.
    """
    from vaft.formula.equilibrium import spitzer_resistivity_from_T_e_Z_eff_ln_Lambda

    T_e, ln_Lambda, Z_eff = 1.0e3, 17.0, 3.0
    mine = sauter_spitzer_conductivity(T_e, Z_eff, ln_Lambda)
    theirs = 1.0 / spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(T_e, Z_eff, ln_Lambda)
    assert 1.15 < mine / theirs < 1.45
