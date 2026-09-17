"""The three virial identities, the three pairwise closures, and what closes.

Issue #546. Every assertion here is against the identities

    E1: 3*beta_p + li - mu_i        = S1 + S2
    E2:   beta_p + li + mu_i        = (RT/R0) * S2
    E3:   beta_p - (alpha-1)*li - mu_i = S3

rather than against a re-typed closed form. Re-typing a formula only proves the
code matches the typing, which is how the Lao beta_p sign survived: the pinned
test spelled out the same wrong coefficient the function did.
"""

import inspect
import math

import numpy as np
import pytest

from vaft.formula.equilibrium import (
    virial_bongard_from_S_alpha_mu,
    virial_bp_li_lihat_from_S123,
    virial_closure_denominators,
    virial_full_123_from_S_alpha_rt,
    virial_identity_residuals,
    virial_lao_from_S_alpha_mu_rt,
    virial_normalized_residual,
    virial_pair_12_from_S_mu_rt,
    virial_pair_13_from_S_alpha_mu,
    virial_pair_23_from_S_alpha_mu_rt,
    virial_residual_rms,
)

#: A well-conditioned point: every denominator (3a-2, a, a-1) is O(1).
S1, S2, S3, ALPHA, MU, RT = 0.9, 1.4, 0.5, 1.6, 0.12, 1.05

#: How each closure is called, given a dict of inputs.  One definition, used by
#: every test below, so a signature change cannot leave a stale copy behind.
SOLVERS = {
    "pair_12": lambda v: virial_pair_12_from_S_mu_rt(v["S1"], v["S2"], v["mu"], v["rt"]),
    "pair_13": lambda v: virial_pair_13_from_S_alpha_mu(
        v["S1"], v["S2"], v["S3"], v["alpha"], v["mu"]
    ),
    "pair_23": lambda v: virial_pair_23_from_S_alpha_mu_rt(
        v["S2"], v["S3"], v["alpha"], v["mu"], v["rt"]
    ),
}

#: The pairs, as (name, omitted identity index).  Index into the (e1, e2, e3)
#: triple that virial_identity_residuals returns.
PAIRS = (("pair_12", 2), ("pair_13", 1), ("pair_23", 0))

BASE = {"S1": S1, "S2": S2, "S3": S3, "alpha": ALPHA, "rt": RT, "mu": MU}


def _residuals(beta_p, li, mu_i=MU):
    return virial_identity_residuals(beta_p, li, mu_i, S1, S2, S3, ALPHA, RT)


# --- the closures solve the identities they claim to ------------------------


@pytest.mark.parametrize("name, omitted", PAIRS, ids=[p[0] for p in PAIRS])
def test_each_pair_satisfies_the_two_identities_it_uses(name, omitted):
    """The definition of a pairwise closure: the two kept identities close
    exactly, at machine precision, for any input."""
    beta_p, li = SOLVERS[name](BASE)
    residuals = _residuals(beta_p, li)
    for index, residual in enumerate(residuals):
        if index == omitted:
            continue
        assert residual == pytest.approx(0.0, abs=1e-12), (
            f"{name} does not satisfy the identity it is built from"
        )


@pytest.mark.parametrize("name, omitted", PAIRS, ids=[p[0] for p in PAIRS])
def test_the_omitted_identity_is_the_evidence(name, omitted):
    """Leave-one-identity-out: on inputs that are mutually consistent the
    omitted residual vanishes too, so any departure from zero is real
    inconsistency rather than the closure's own arithmetic."""
    consistent = _consistent_inputs()
    beta_p, li = SOLVERS[name](consistent)
    residuals = virial_identity_residuals(
        beta_p, li, consistent["mu"], consistent["S1"], consistent["S2"],
        consistent["S3"], consistent["alpha"], consistent["rt"],
    )
    assert residuals[omitted] == pytest.approx(0.0, abs=1e-12)
    assert beta_p == pytest.approx(consistent["beta_p"], abs=1e-12)
    assert li == pytest.approx(consistent["li"], abs=1e-12)


def _consistent_inputs():
    """Shafranov integrals manufactured from a chosen (beta_p, li, mu_i), so all
    three identities hold exactly and the three closures must agree."""
    beta_p, li, mu_i, alpha, rt = 0.85, 0.72, 0.19, 1.6, 1.05
    # E2 defines S2 given the triple; E1 then defines S1; E3 defines S3.
    s2 = (beta_p + li + mu_i) / rt
    s1 = 3.0 * beta_p + li - mu_i - s2
    s3 = beta_p - (alpha - 1.0) * li - mu_i
    return {
        "S1": s1, "S2": s2, "S3": s3, "alpha": alpha, "rt": rt,
        "mu": mu_i, "beta_p": beta_p, "li": li,
    }


def test_the_three_closures_agree_when_the_inputs_are_consistent():
    c = _consistent_inputs()
    b12, l12 = SOLVERS["pair_12"](c)
    b13, l13 = SOLVERS["pair_13"](c)
    b23, l23 = SOLVERS["pair_23"](c)
    for value in (b12, b13, b23):
        assert value == pytest.approx(c["beta_p"], abs=1e-12)
    for value in (l12, l13, l23):
        assert value == pytest.approx(c["li"], abs=1e-12)


def test_the_full_solve_recovers_all_three_unknowns():
    c = _consistent_inputs()
    beta_p, li, mu_i = virial_full_123_from_S_alpha_rt(
        c["S1"], c["S2"], c["S3"], c["alpha"], c["rt"]
    )
    assert beta_p == pytest.approx(c["beta_p"], abs=1e-12)
    assert li == pytest.approx(c["li"], abs=1e-12)
    assert mu_i == pytest.approx(c["mu"], abs=1e-12)
    residuals = virial_identity_residuals(
        beta_p, li, mu_i, c["S1"], c["S2"], c["S3"], c["alpha"], c["rt"]
    )
    assert residuals == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


# --- the relations between formulations, which the API must keep apart ------


def test_bongard_is_exactly_the_pair_13_closure():
    """Acceptance criterion of #546: the historical low-aspect-ratio closure is
    not merely close to the RT-free pair, it is the same algebra."""
    assert virial_bongard_from_S_alpha_mu(S1, S2, S3, ALPHA, MU) == (
        virial_pair_13_from_S_alpha_mu(S1, S2, S3, ALPHA, MU)
    )


def test_lao_beta_p_is_pair_12_but_lao_li_is_not():
    """The distinction #546 exists to make explicit. Lao's beta_p comes from
    E1+E2; his li takes beta_p - mu_i from the same pair and substitutes it into
    E3, so it is a three-relation result wearing a pairwise name."""
    beta_lao, li_lao = virial_lao_from_S_alpha_mu_rt(S1, S2, S3, ALPHA, MU, RT)
    beta_12, li_12 = virial_pair_12_from_S_mu_rt(S1, S2, MU, RT)
    assert beta_lao == pytest.approx(beta_12, abs=1e-12)
    assert li_lao != pytest.approx(li_12, abs=1e-9)


def test_the_full_solve_li_is_identically_the_lao_li():
    """Not a coincidence of these numbers: E1 and E2 fix beta_p - mu_i on their
    own, so E3 determines li the same way in both. Holds for any alpha != 1."""
    _, li_lao = virial_lao_from_S_alpha_mu_rt(S1, S2, S3, ALPHA, MU, RT)
    _, li_full, _ = virial_full_123_from_S_alpha_rt(S1, S2, S3, ALPHA, RT)
    assert li_full == pytest.approx(li_lao, abs=1e-12)
    for alpha in (0.4, 1.3, 2.7, -0.8):
        _, a = virial_lao_from_S_alpha_mu_rt(S1, S2, S3, alpha, MU, RT)
        _, b, _ = virial_full_123_from_S_alpha_rt(S1, S2, S3, alpha, RT)
        assert b == pytest.approx(a, abs=1e-12)


def test_the_full_solve_mu_i_is_independent_of_the_supplied_one():
    """The full solve is a different inverse problem: it takes no mu_i at all,
    which is what makes its mu_i comparable against a measured one."""
    beta_p, li, mu_i = virial_full_123_from_S_alpha_rt(S1, S2, S3, ALPHA, RT)
    pair_beta, pair_li = virial_pair_12_from_S_mu_rt(S1, S2, MU, RT)
    assert mu_i != pytest.approx(MU, abs=1e-9)
    assert beta_p != pytest.approx(pair_beta, abs=1e-9)
    assert li != pytest.approx(pair_li, abs=1e-9)


def test_the_deprecated_name_still_returns_the_same_numbers():
    assert virial_bp_li_lihat_from_S123(S1, S2, S3, ALPHA, RT) == (
        virial_full_123_from_S_alpha_rt(S1, S2, S3, ALPHA, RT)
    )


# --- sensitivity ------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("S1", {"pair_12", "pair_13"}),
        ("S2", {"pair_12", "pair_13", "pair_23"}),
        ("S3", {"pair_13", "pair_23"}),
        ("alpha", {"pair_13", "pair_23"}),
        ("rt", {"pair_12", "pair_23"}),
        ("mu", {"pair_12", "pair_13", "pair_23"}),
    ],
)
def test_each_closure_depends_on_exactly_the_inputs_its_identities_carry(name, expected):
    """The point of having three closures: they fail differently. pair_13 must
    not move when RT/R0 does, or comparing it against the others says nothing
    about RT sensitivity."""
    nudged = dict(BASE, **{name: BASE[name] + 0.05})

    def solve(v):
        return {key: solver(v) for key, solver in SOLVERS.items()}

    before, after = solve(BASE), solve(nudged)
    moved = {
        key for key in before
        if not np.allclose(before[key], after[key], rtol=0, atol=1e-12)
    }
    assert moved == expected


# --- singular and ill-conditioned limits ------------------------------------


@pytest.mark.parametrize(
    "label, call",
    [
        ("pair_13 at alpha = 2/3", lambda: virial_pair_13_from_S_alpha_mu(S1, S2, S3, 2.0 / 3.0, MU)),
        ("pair_23 at alpha = 0", lambda: virial_pair_23_from_S_alpha_mu_rt(S2, S3, 0.0, MU, RT)),
        ("full_123 at alpha = 1", lambda: virial_full_123_from_S_alpha_rt(S1, S2, S3, 1.0, RT)),
    ],
)
def test_a_singular_closure_is_indeterminate_not_a_large_number(label, call):
    """#546: near-singular cases must return indeterminate, not plausible-looking
    finite garbage that is then read as a physical failure."""
    values = call()
    assert all(math.isnan(v) for v in values), f"{label} returned {values}"


@pytest.mark.parametrize("delta", [1e-13, -1e-13, 0.0])
def test_the_singularity_is_caught_from_either_side(delta):
    assert all(math.isnan(v) for v in virial_pair_13_from_S_alpha_mu(
        S1, S2, S3, (2.0 + delta) / 3.0, MU
    ))


def test_a_closure_just_outside_the_singular_band_still_returns_numbers():
    """The guard must not swallow merely ill-conditioned cases: those are real
    results the conditioning evidence is there to qualify."""
    beta_p, li = virial_pair_13_from_S_alpha_mu(S1, S2, S3, 2.0 / 3.0 + 1e-6, MU)
    assert math.isfinite(beta_p) and math.isfinite(li)
    assert abs(beta_p) > 1e3, "expected the near-singular value to be large"


def test_pair_12_cannot_be_singular_because_it_never_sees_alpha():
    """It is the one closure built from the two relations that never mention
    alpha. That is structural, not a property of these numbers, so assert it on
    the signature: there is no alpha to make it indeterminate."""
    assert "alpha" not in inspect.signature(virial_pair_12_from_S_mu_rt).parameters
    beta_p, li = virial_pair_12_from_S_mu_rt(S1, S2, MU, RT)
    assert math.isfinite(beta_p) and math.isfinite(li)


def test_the_identities_stay_finite_where_the_inversions_fail():
    """Identity validity and inversion conditioning are different things: at
    alpha = 1 the full solve is singular while the residuals are ordinary
    numbers."""
    assert all(math.isnan(v) for v in virial_full_123_from_S_alpha_rt(S1, S2, S3, 1.0, RT))
    residuals = virial_identity_residuals(0.8, 0.7, 0.2, S1, S2, S3, 1.0, RT)
    assert all(math.isfinite(v) for v in residuals)


def test_the_denominators_name_each_closures_singular_limit():
    pair_13, pair_23, lao_li, full_123 = virial_closure_denominators(ALPHA)
    assert pair_13 == pytest.approx(3.0 * ALPHA - 2.0)
    assert pair_23 == pytest.approx(ALPHA)
    assert lao_li == pytest.approx(ALPHA - 1.0)
    assert full_123 == pytest.approx(4.0 * (ALPHA - 1.0))
    # Each vanishes exactly where its closure returns NaN.
    assert virial_closure_denominators(2.0 / 3.0)[0] == pytest.approx(0.0, abs=1e-15)
    assert virial_closure_denominators(0.0)[1] == 0.0
    assert virial_closure_denominators(1.0)[3] == 0.0


# --- residual normalization -------------------------------------------------


def test_the_normalized_residual_is_symmetric_and_floored():
    assert virial_normalized_residual(0.3, 10.0, 9.7) == pytest.approx(0.03)
    assert virial_normalized_residual(0.3, 9.7, 10.0) == pytest.approx(0.03)
    # The floor of 1 stops a residual looking huge when both sides pass zero.
    assert virial_normalized_residual(0.3, 0.01, -0.02) == pytest.approx(0.3)


def test_the_aggregate_is_an_rms_and_refuses_to_average_away_a_nan():
    assert virial_residual_rms(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert virial_residual_rms(0.0, 0.0, 3.0) == pytest.approx(math.sqrt(3.0))
    assert math.isnan(virial_residual_rms(0.0, 0.0, float("nan")))


def test_consistent_inputs_give_a_vanishing_aggregate():
    c = _consistent_inputs()
    beta_p, li, mu_i = virial_full_123_from_S_alpha_rt(
        c["S1"], c["S2"], c["S3"], c["alpha"], c["rt"]
    )
    e1, e2, e3 = virial_identity_residuals(
        beta_p, li, mu_i, c["S1"], c["S2"], c["S3"], c["alpha"], c["rt"]
    )
    lhs_rhs = (
        (3 * beta_p + li - mu_i, c["S1"] + c["S2"]),
        (beta_p + li + mu_i, c["rt"] * c["S2"]),
        (beta_p - (c["alpha"] - 1.0) * li - mu_i, c["S3"]),
    )
    normalized = [
        virial_normalized_residual(r, lhs, rhs)
        for r, (lhs, rhs) in zip((e1, e2, e3), lhs_rhs)
    ]
    assert virial_residual_rms(*normalized) == pytest.approx(0.0, abs=1e-12)

# --- what the review of #546 found, pinned so it cannot come back -----------


def test_a_normalized_residual_refuses_an_unknown_scale():
    """max() keeps its first argument against a NaN, so an unguarded
    max(1.0, |lhs|, |rhs|) would silently normalize by 1 when a side is
    unknown, and report a small residual derived from nothing."""
    assert math.isnan(virial_normalized_residual(0.3, 10.0, float("nan")))
    assert math.isnan(virial_normalized_residual(0.3, float("nan"), 10.0))


def test_the_full_solve_guards_the_determinant_it_documents():
    """The docstring and virial_closure_denominators both name 4*(alpha-1); the
    guard must be on that, not on a quarter of it."""
    eps = 0.4
    # |4*(alpha-1)| = 0.2, inside the band: rejected. Guarding |alpha-1|
    # instead would accept it, since 0.05 is well under eps.
    assert all(math.isnan(v) for v in virial_full_123_from_S_alpha_rt(
        S1, S2, S3, 1.05, RT, eps=eps
    ))
    # |4*(alpha-1)| = 1.2, outside it: still a number.
    assert all(math.isfinite(v) for v in virial_full_123_from_S_alpha_rt(
        S1, S2, S3, 1.3, RT, eps=eps
    ))


def test_the_historical_bongard_name_can_tune_the_same_guard():
    near = 2.0 / 3.0 + 1e-6
    assert all(math.isfinite(v) for v in virial_bongard_from_S_alpha_mu(S1, S2, S3, near, MU))
    assert all(math.isnan(v) for v in virial_bongard_from_S_alpha_mu(
        S1, S2, S3, near, MU, eps=1e-3
    ))
