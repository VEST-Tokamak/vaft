"""The toroidal phase audit: the kernel, the GPEC plumbing, and the verdict.

A complex toroidal harmonic does not say which reconstruction it belongs to,
``Re(C e^{-i n phi})`` or ``Re(C e^{+i n phi})``, and the two are different
fields.  Decision D-06 resolves that by measurement rather than assumption.

Every fixture here is synthetic: one rectangular coil loop per toroidal sector,
written out as a GPEC ``coil.in`` plus ``.dat`` pair and a ``gpec_cbrzphi``
file whose harmonics are computed by a Biot-Savart implementation written in
this file, independent of the one under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vaft.code.gpec import (
    audit_coil_field_phase,
    coil_filaments_from_coil_in,
    read_brzphi_harmonics,
    read_coil_control,
)
from vaft.process.perturbation import ToroidalPhaseAudit, toroidal_phase_audit
from vaft.validation.model import ValidationStatus
from vaft.validation.perturbation_phase import (
    GPEC_COIL_FIELD,
    PhaseAuditCriteria,
    phase_convention_verdict,
)

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
# The kernel
# --------------------------------------------------------------------------


def _reconstruct(stored: np.ndarray, phi: np.ndarray, n_tor: int) -> np.ndarray:
    """``Re(C e^{-i n phi})``, the field a stored harmonic stands for."""
    return np.real(stored[:, None, :] * np.exp(-1j * n_tor * phi)[None, :, None])


@pytest.fixture
def stored():
    rng = np.random.default_rng(20260918)
    return rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))


@pytest.mark.parametrize("n_tor", [1, 2, 3])
def test_a_field_built_from_the_stored_harmonic_prefers_it(stored, n_tor):
    phi = np.linspace(0.0, TWO_PI, 8 * n_tor, endpoint=False)
    audit = toroidal_phase_audit(stored, _reconstruct(stored, phi, n_tor), phi, n_tor=n_tor)
    assert audit.preferred == "stored"
    assert audit.stored_relative_norm < 1e-12
    assert audit.separation_ratio > 1e6
    assert max(audit.stored_component_relative_norm) < 1e-12


@pytest.mark.parametrize("n_tor", [1, 3])
def test_a_field_built_from_the_conjugate_prefers_the_conjugate(stored, n_tor):
    """A file in the other convention: the audit is what finds it."""
    phi = np.linspace(0.0, TWO_PI, 8 * n_tor, endpoint=False)
    sampled = _reconstruct(np.conjugate(stored), phi, n_tor)
    audit = toroidal_phase_audit(stored, sampled, phi, n_tor=n_tor)
    assert audit.preferred == "conjugate"
    assert audit.conjugate_relative_norm < 1e-12


def test_a_real_harmonic_has_no_winner_at_all(stored):
    """The one case where the question has no answer: a harmonic equal to its
    own conjugate reconstructs identically either way.  Naming a winner there
    would invent a result, so there is none to name."""
    real_only = np.real(stored).astype(complex)
    phi = np.linspace(0.0, TWO_PI, 16, endpoint=False)
    audit = toroidal_phase_audit(real_only, _reconstruct(real_only, phi, 1), phi, n_tor=1)
    assert audit.preferred is None
    assert audit.separation_ratio == pytest.approx(1.0)
    assert audit.stored_relative_norm == pytest.approx(audit.conjugate_relative_norm)


def test_a_field_with_no_content_at_this_mode_separates_nothing(stored):
    """The common failure the separation criterion exists for: project a field
    that is constant in phi onto n = 1 and both hypotheses land near one.  The
    absolute norms say why -- the projection returned essentially zero."""
    phi = np.linspace(0.0, TWO_PI, 16, endpoint=False)
    constant = np.tile(np.real(stored)[:, None, :], (1, phi.size, 1))
    audit = toroidal_phase_audit(stored, constant, phi, n_tor=1)
    assert audit.separation_ratio < 3.0
    assert min(audit.stored_relative_norm, audit.conjugate_relative_norm) > 0.5
    assert audit.stored_norm > 0.0
    assert audit.measured_norm < 1e-12 * audit.stored_norm


@pytest.mark.parametrize("phi_count", [3, 6])
def test_a_grid_that_cannot_resolve_the_mode_is_refused(stored, phi_count):
    """Nyquist for ``n`` itself. ``M <= 2|n|`` is refused rather than measured,
    because what an aliased projection returns does not look like a failure."""
    n_tor = 3
    aliased = np.linspace(0.0, TWO_PI, phi_count, endpoint=False)
    with pytest.raises(ValueError, match="cannot resolve n = 3"):
        toroidal_phase_audit(
            stored, _reconstruct(stored, aliased, n_tor), aliased, n_tor=n_tor
        )
    resolved = np.linspace(0.0, TWO_PI, 8 * n_tor, endpoint=False)
    good = toroidal_phase_audit(
        stored, _reconstruct(stored, resolved, n_tor), resolved, n_tor=n_tor
    )
    assert good.stored_relative_norm < 1e-12


def test_an_alias_from_a_second_harmonic_is_confidently_wrong(stored):
    """The limitation the docstring states, pinned because it is the one failure
    the verdict cannot catch: an alias is not "no content".  Sampled at four
    angles, a field carrying n = 1 alongside n = 3 projects onto n = 3 as
    exactly ``conj(C)`` -- a separation of 1e15 for the wrong answer.

    ``M > 2|n|`` refuses this particular case; nothing in the two numbers would
    have.  A field of unknown harmonic content needs an ``M`` chosen from that
    content, not from ``n``.
    """
    n_tor, extra = 3, 1
    contaminant = stored - np.conjugate(stored)
    four = np.linspace(0.0, TWO_PI, 4, endpoint=False)
    mixed_at = lambda phi: (
        _reconstruct(stored, phi, n_tor) + _reconstruct(contaminant, phi, extra)
    )
    with pytest.raises(ValueError, match="cannot resolve n = 3"):
        toroidal_phase_audit(stored, mixed_at(four), four, n_tor=n_tor)

    # Above Nyquist for n the alias is gone and the truth comes back.
    resolved = np.linspace(0.0, TWO_PI, 16, endpoint=False)
    audit = toroidal_phase_audit(stored, mixed_at(resolved), resolved, n_tor=n_tor)
    assert audit.preferred == "stored"
    assert audit.stored_relative_norm < 1e-12


def test_a_component_with_no_stored_coefficients_reports_no_relative_norm(stored):
    """Dividing by the smallest positive float turns an ordinary residual into
    1e307, which reads as a catastrophic error rather than as an undefined
    ratio."""
    blank = stored.copy()
    blank[:, 1] = 0.0
    phi = np.linspace(0.0, TWO_PI, 16, endpoint=False)
    audit = toroidal_phase_audit(blank, _reconstruct(stored, phi, 1), phi, n_tor=1)
    assert math.isnan(audit.stored_component_relative_norm[1])
    assert math.isnan(audit.conjugate_component_relative_norm[1])
    assert all(
        math.isfinite(value)
        for index, value in enumerate(audit.stored_component_relative_norm)
        if index != 1
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("target", ["stored", "sampled"])
def test_a_non_finite_input_is_refused_rather_than_propagated(stored, bad, target):
    """A NaN loses every comparison instead of failing one: it would come out
    as NaN against NaN with an infinite separation, which is indistinguishable
    from a decisive answer."""
    phi = np.linspace(0.0, TWO_PI, 16, endpoint=False)
    sampled = _reconstruct(stored, phi, 1)
    if target == "stored":
        stored = stored.copy()
        stored[0, 0] = bad
    else:
        sampled = sampled.copy()
        sampled[0, 0, 0] = bad
    with pytest.raises(ValueError, match=f"{target} holds 1 non-finite"):
        toroidal_phase_audit(stored, sampled, phi, n_tor=1)


@pytest.mark.parametrize(
    "phi",
    [
        np.linspace(0.0, math.pi, 16),          # half a period
        np.linspace(0.0, TWO_PI, 16),           # repeats the endpoint
        np.sort(np.random.default_rng(1).uniform(0.0, TWO_PI, 16)),  # not uniform
    ],
)
def test_a_phi_grid_that_is_not_one_equal_period_is_refused(stored, phi):
    sampled = _reconstruct(stored, phi, 1)
    with pytest.raises(ValueError, match="equally spaced over one full period"):
        toroidal_phase_audit(stored, sampled, phi, n_tor=1)


def test_shape_disagreements_are_refused(stored):
    phi = np.linspace(0.0, TWO_PI, 16, endpoint=False)
    sampled = _reconstruct(stored, phi, 1)
    with pytest.raises(ValueError, match="stored and phi_rad ask for"):
        toroidal_phase_audit(stored, sampled[:, :-1, :], phi, n_tor=1)
    with pytest.raises(ValueError, match=r"stored must be \(P, K\)"):
        toroidal_phase_audit(stored[:, 0], sampled, phi, n_tor=1)
    with pytest.raises(ValueError, match=r"sampled must be \(P, M, K\)"):
        toroidal_phase_audit(stored, sampled[0], phi, n_tor=1)


@pytest.mark.parametrize("n_tor", [0, 1.5])
def test_a_mode_number_that_is_not_a_non_zero_integer_is_refused(stored, n_tor):
    phi = np.linspace(0.0, TWO_PI, 16, endpoint=False)
    with pytest.raises(ValueError, match="non-zero integer"):
        toroidal_phase_audit(stored, _reconstruct(stored, phi, 1), phi, n_tor=n_tor)


def test_an_all_zero_harmonic_has_no_relative_norm(stored):
    phi = np.linspace(0.0, TWO_PI, 16, endpoint=False)
    zeros = np.zeros_like(stored)
    with pytest.raises(ValueError, match="identically zero"):
        toroidal_phase_audit(zeros, _reconstruct(zeros, phi, 1), phi, n_tor=1)


# --------------------------------------------------------------------------
# A synthetic GPEC run
# --------------------------------------------------------------------------

#: Toroidal sectors of the synthetic coil set, and an n = 1 current pattern.
SECTOR_ANGLES = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
SECTOR_CURRENTS = (1.0e3, 0.0, -1.0e3, 0.0)
#: The ``.dat`` winding multiplier.  Negative on purpose: GPEC's own
#: ``d3d_c.dat`` carries -4, and the sign is part of the driven current.
WINDING = -2.0
#: Where the synthetic field is evaluated, in (R, Z) metres.
PROBE_POINTS = ((1.30, 0.00), (1.35, 0.10), (1.25, -0.10), (1.40, 0.05))


def _sector_loop(phi0: float) -> np.ndarray:
    """A closed rectangular loop in the poloidal plane at toroidal angle ``phi0``."""
    corners = [(0.95, -0.25), (1.05, -0.25), (1.05, 0.25), (0.95, 0.25), (0.95, -0.25)]
    points = []
    for (r_a, z_a), (r_b, z_b) in zip(corners[:-1], corners[1:]):
        for step in np.linspace(0.0, 1.0, 12, endpoint=False):
            r = r_a + step * (r_b - r_a)
            z = z_a + step * (z_b - z_a)
            points.append((r * math.cos(phi0), r * math.sin(phi0), z))
    points.append(points[0])
    return np.asarray(points, dtype=float)


def _biot_savart(loops, currents, probes) -> np.ndarray:
    """Midpoint Biot-Savart, written here so the fixture does not use the
    implementation the audit uses."""
    field = np.zeros_like(probes)
    for loop, current in zip(loops, currents):
        mid = 0.5 * (loop[1:] + loop[:-1])
        segment = loop[1:] - loop[:-1]
        offset = probes[:, None, :] - mid[None, :, :]
        distance = np.linalg.norm(offset, axis=2)
        cross = np.cross(segment[None, :, :], offset)
        weight = 1.0e-7 * current / distance**3
        field += np.sum(cross * weight[:, :, None], axis=1)
    return field


#: How many toroidal angles the fixture projects over.  It has to be the
#: number the audit uses, because a discrete coil set has content at every
#: harmonic and a sampled projection aliases the ones above ``M/2`` into the
#: one being measured: at 64 against the audit's 48 the two agree only to
#: 5e-3, which is real disagreement about which aliases were folded in rather
#: than about the phase.
FIXTURE_PHI_SAMPLES = 48


def _reference_harmonics(n_tor: int, phi_samples: int = FIXTURE_PHI_SAMPLES) -> np.ndarray:
    """``C = 2 mean_phi(B e^{+i n phi})`` for the synthetic coil set, in
    ``(b_r, b_z, b_phi)`` order."""
    loops = [_sector_loop(angle) for angle in SECTOR_ANGLES]
    currents = [current * WINDING for current in SECTOR_CURRENTS]
    phi = np.linspace(0.0, TWO_PI, phi_samples, endpoint=False)
    harmonics = []
    for r, z in PROBE_POINTS:
        probes = np.column_stack((r * np.cos(phi), r * np.sin(phi), np.full(phi.size, z)))
        xyz = _biot_savart(loops, currents, probes)
        cylindrical = np.column_stack(
            (
                xyz[:, 0] * np.cos(phi) + xyz[:, 1] * np.sin(phi),
                xyz[:, 2],
                -xyz[:, 0] * np.sin(phi) + xyz[:, 1] * np.cos(phi),
            )
        )
        harmonics.append(2.0 * np.mean(cylindrical * np.exp(1j * n_tor * phi)[:, None], axis=0))
    return np.asarray(harmonics)


def _write_run(directory, harmonics, *, n_tor: int, machine: str = "synth") -> tuple:
    """Write a ``coil.in``, its ``.dat``, and a ``gpec_cbrzphi`` carrying
    ``harmonics``; return the two paths the audit takes."""
    data_dir = directory / "coil data"
    data_dir.mkdir()
    loops = [_sector_loop(angle) for angle in SECTOR_ANGLES]
    lines = [f"{len(loops)} 1 {loops[0].shape[0]} {WINDING:.6g}"]
    for loop in loops:
        lines.extend(f"{x:16.8e}{y:16.8e}{z:16.8e}" for x, y, z in loop)
    (data_dir / f"{machine}_synthset.dat").write_text("\n".join(lines) + "\n")

    currents = "\n".join(
        f"    coil_cur(1,{index})={value:g}"
        for index, value in enumerate(SECTOR_CURRENTS, start=1)
    )
    coil_in = directory / "coil.in"
    coil_in.write_text(
        "&COIL_CONTROL\n"
        # The quoted path holds slashes, which is not the namelist terminator.
        f'    data_dir="{data_dir}"   ! where the .dat files are\n'
        f'    machine="{machine}"\n'
        '    ip_direction="positive"\n'
        '    bt_direction="negative"\n'
        "    coil_num=1\n"
        '    coil_name(1)="synthset"\n'
        f"{currents}\n"
        "/\n"
        "&COIL_OUTPUT\n    gpec_interface=t\n/\n"
    )

    body = [
        " GPEC_CBRZPHI: External field by coils",
        " v0.0.0-synthetic",
        "",
        f"   n  =  {n_tor:4d}",
        f"   nr =  {len(PROBE_POINTS):4d}  nz =     1",
        "",
        "  l                r                z        real(b_r)        imag(b_r)"
        "        real(b_z)        imag(b_z)      real(b_phi)      imag(b_phi)",
    ]
    for (r, z), value in zip(PROBE_POINTS, harmonics):
        numbers = "".join(
            f"{part:17.8E}"
            for component in value
            for part in (component.real, component.imag)
        )
        body.append(f"  1{r:17.8E}{z:17.8E}{numbers}")
    brzphi = directory / f"gpec_cbrzphi_n{n_tor}.out"
    brzphi.write_text("\n".join(body) + "\n")
    return coil_in, brzphi


@pytest.fixture
def synthetic_run(tmp_path):
    return _write_run(tmp_path, _reference_harmonics(1), n_tor=1)


def test_the_synthetic_run_audits_as_stored(synthetic_run):
    coil_in, brzphi = synthetic_run
    result = audit_coil_field_phase(coil_in, brzphi, sample_points=len(PROBE_POINTS))
    assert result.n_tor == 1
    assert result.machine == "synth"
    assert result.coil_names == ("synthset",)
    assert result.audit.preferred == "stored"
    assert result.audit.stored_relative_norm < 1e-6
    assert result.audit.separation_ratio > 1e3


def test_a_conjugated_file_is_caught(tmp_path):
    coil_in, brzphi = _write_run(tmp_path, np.conjugate(_reference_harmonics(1)), n_tor=1)
    result = audit_coil_field_phase(coil_in, brzphi, sample_points=len(PROBE_POINTS))
    assert result.audit.preferred == "conjugate"
    assert result.audit.conjugate_relative_norm < 1e-6


def test_the_signed_winding_multiplier_is_applied(tmp_path):
    """``nw`` is signed and belongs to the driven current, not to a turn count.
    A file written as if it were 1 instead of -2 fails the audit outright."""
    coil_in, brzphi = _write_run(tmp_path, _reference_harmonics(1) / WINDING, n_tor=1)
    result = audit_coil_field_phase(coil_in, brzphi, sample_points=len(PROBE_POINTS))
    assert min(
        result.audit.stored_relative_norm, result.audit.conjugate_relative_norm
    ) > 1.0
    filaments = coil_filaments_from_coil_in(coil_in)
    assert filaments.currents_a.tolist() == [1.0e3 * WINDING, -1.0e3 * WINDING]


def test_a_zero_current_sector_is_dropped(synthetic_run):
    """Two of the four sectors carry no current and cost nothing."""
    coil_in, _ = synthetic_run
    filaments = coil_filaments_from_coil_in(coil_in)
    assert len(filaments.loops_xyz) == 2
    assert filaments.currents_a.size == 2


def test_the_data_dir_is_read_back_with_its_slashes(synthetic_run):
    """The namelist terminator is a line of its own; a quoted path full of
    slashes is not it."""
    coil_in, _ = synthetic_run
    control = read_coil_control(coil_in)
    assert control["machine"] == "synth"
    assert control["data_dir"].endswith("coil data")
    assert control["ip_direction"] == "positive"


def test_the_mode_number_comes_from_the_file_and_a_disagreement_raises(synthetic_run):
    coil_in, brzphi = synthetic_run
    assert read_brzphi_harmonics(brzphi).n_tor == 1
    with pytest.raises(ValueError, match="declares n = 1"):
        audit_coil_field_phase(coil_in, brzphi, n_tor=2)


def test_a_missing_geometry_file_names_it(synthetic_run):
    coil_in, brzphi = synthetic_run
    with pytest.raises(FileNotFoundError, match="synth_synthset.dat"):
        audit_coil_field_phase(coil_in, brzphi, coil_data_dir=coil_in.parent)


def test_a_multi_mode_file_is_refused_rather_than_audited_at_mode_123(tmp_path, synthetic_run):
    """GPEC's multi-mode writer prints ``n = 123`` for a file superposing modes
    1, 2 and 3 and declares the real list in a ``modes =`` note.  Reading the
    header as a mode number would audit a superposition at n = 123, which is
    not a mode and not what the columns hold.
    """
    coil_in, single = synthetic_run
    multi = tmp_path / "gpec_cbrzphi_n123.out"
    body = single.read_text().splitlines()
    body[0] = " GPEC_CBRZPHI: External field by coils (multi-mode harmonics)"
    body[1] = " modes = 1,2,3"
    body[3] = "   n  =   123"
    multi.write_text("\n".join(body) + "\n")

    harmonics = read_brzphi_harmonics(multi)
    assert harmonics.modes == (1, 2, 3)
    assert harmonics.n_tor is None
    assert read_brzphi_harmonics(single).modes == (1,)
    with pytest.raises(ValueError, match=r"superposes modes \(1, 2, 3\)"):
        audit_coil_field_phase(coil_in, multi)


def test_an_empty_region_is_refused_rather_than_averaged(synthetic_run):
    coil_in, brzphi = synthetic_run
    with pytest.raises(ValueError, match="no rows with l = 0"):
        read_brzphi_harmonics(brzphi, region=0)


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


def test_a_settled_audit_passes_and_names_the_convention(synthetic_run):
    coil_in, brzphi = synthetic_run
    result = audit_coil_field_phase(coil_in, brzphi, sample_points=len(PROBE_POINTS))
    verdict = phase_convention_verdict(result.audit, criteria=GPEC_COIL_FIELD)
    assert verdict.status is ValidationStatus.PASS
    assert verdict.convention == "stored"


@pytest.mark.parametrize(
    "stored_norm, conjugate_norm, separation",
    [
        (math.nan, math.nan, math.inf),   # what a non-finite sample used to give
        (1e-9, math.nan, math.inf),
        (1e-9, 5.0, math.nan),
    ],
)
def test_a_verdict_never_passes_on_a_number_that_is_not_one(
    stored_norm, conjugate_norm, separation
):
    """Every criterion is a condition to satisfy, not one to fail.  ``separation
    < 3`` and ``winner > 0.10`` are both False for a NaN, so written that way
    round the checks fall through to PASS -- which is how a poisoned audit came
    back as a settled convention."""
    audit = _audit_like(
        stored_relative_norm=stored_norm,
        conjugate_relative_norm=conjugate_norm,
        separation_ratio=separation,
    )
    verdict = phase_convention_verdict(audit, criteria=GPEC_COIL_FIELD)
    assert verdict.status is ValidationStatus.INDETERMINATE
    assert verdict.convention is None
    assert "no usable number" in verdict.reason


def test_an_exact_tie_is_indeterminate_rather_than_a_coin_toss():
    audit = _audit_like(
        stored_relative_norm=1.0, conjugate_relative_norm=1.0,
        separation_ratio=1.0, preferred=None,
    )
    verdict = phase_convention_verdict(audit, criteria=GPEC_COIL_FIELD)
    assert verdict.status is ValidationStatus.INDETERMINATE
    assert "no winner to name" in verdict.reason


def _audit_like(**overrides):
    """A ToroidalPhaseAudit with the fields a verdict reads, for the cases the
    kernel now refuses to produce."""
    fields = dict(
        n_tor=1, preferred="conjugate", stored_relative_norm=1.0,
        conjugate_relative_norm=1e-9, stored_component_relative_norm=(1.0,),
        conjugate_component_relative_norm=(1e-9,), separation_ratio=1e9,
        stored_norm=1.0, measured_norm=1.0, sample_count=4, phi_sample_count=16,
    )
    fields.update(overrides)
    return ToroidalPhaseAudit(**fields)


def test_an_unexcited_mode_is_indeterminate_not_a_failure(tmp_path, stored):
    """n = 2 on an n = 1 current pattern: nothing to measure, and saying so is
    not the same as saying the file is wrong."""
    phi = np.linspace(0.0, TWO_PI, 16, endpoint=False)
    # Not quite constant: a small n = 2 content, so the two hypotheses differ
    # and the separation rather than the tie is what decides.
    faint = np.real(stored)[:, None, :] + 1e-6 * _reconstruct(stored, phi, 2)
    audit = toroidal_phase_audit(stored, faint, phi, n_tor=2)
    verdict = phase_convention_verdict(audit, criteria=GPEC_COIL_FIELD)
    assert verdict.status is ValidationStatus.INDETERMINATE
    assert verdict.convention is None
    assert "only" in verdict.reason and "apart" in verdict.reason


def test_a_clear_winner_that_is_still_wrong_is_indeterminate(stored):
    """The second criterion, on its own: separated by far more than three and
    still not reproducing the field."""
    phi = np.linspace(0.0, TWO_PI, 16, endpoint=False)
    # A field 20% too strong: the stored hypothesis is still clearly the right
    # shape, and still misses the amplitude by twice the 0.10 bar.
    audit = toroidal_phase_audit(stored, 1.2 * _reconstruct(stored, phi, 1), phi, n_tor=1)
    assert audit.stored_relative_norm == pytest.approx(0.2)
    assert audit.separation_ratio > GPEC_COIL_FIELD.separation_ratio
    verdict = phase_convention_verdict(audit, criteria=GPEC_COIL_FIELD)
    assert verdict.status is ValidationStatus.INDETERMINATE
    assert "less bad of two bad answers" in verdict.reason


def test_the_criteria_are_the_legacy_branch_numbers():
    assert GPEC_COIL_FIELD.separation_ratio == 3.0
    assert GPEC_COIL_FIELD.relative_norm == 0.10


def test_the_criteria_are_a_required_argument(stored):
    """No default anywhere: a threshold that appears by itself is a threshold
    nobody chose."""
    phi = np.linspace(0.0, TWO_PI, 16, endpoint=False)
    audit = toroidal_phase_audit(stored, _reconstruct(stored, phi, 1), phi, n_tor=1)
    with pytest.raises(TypeError):
        phase_convention_verdict(audit)
    loose = PhaseAuditCriteria(separation_ratio=1.0, relative_norm=1.0, name="loose")
    assert phase_convention_verdict(audit, criteria=loose).status is ValidationStatus.PASS
