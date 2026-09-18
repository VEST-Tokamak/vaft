"""Fixed-boundary equilibrium scans over CHEASE inputs (issue #66).

The transformations are tested offline; the solve itself needs a CHEASE
binary and is skipped without one, as the other CHEASE integration tests are.
"""

import os

import numpy as np
import pytest

import vaft.data
from vaft.code import EquilibriumVariation, apply_equilibrium_variation, scan_chease
from vaft.data.eqdsk import read_geqdsk

SOURCE = "kineticEfit/g048224.00300"

needs_chease = pytest.mark.skipif(
    not (
        os.environ.get("CHEASEHOME")
        or os.environ.get("CHEASE_EXEC_DIR")
        or os.environ.get("CHEASE")
    ),
    reason="CHEASE scan integration test requires CHEASEHOME, CHEASE_EXEC_DIR or CHEASE",
)


@pytest.fixture(scope="module")
def geqdsk():
    return read_geqdsk(str(vaft.data.data_path(SOURCE)))


# ---------------------------------------------------------------------------
# What a variation may be
# ---------------------------------------------------------------------------

def test_a_variation_defaults_to_the_unperturbed_case():
    control = EquilibriumVariation("control")
    assert control.pressure_scale == 1.0
    assert control.current_peaking == 0.0
    assert not control.reshapes_boundary


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"current_peaking": -0.5}, "non-negative"),
        ({"pressure_scale": 0.0}, "positive"),
        ({"elongation_scale": -1.0}, "positive"),
    ],
)
def test_an_unphysical_variation_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EquilibriumVariation("bad", **kwargs)


def test_a_scan_needs_unique_labels():
    with pytest.raises(ValueError, match="unique"):
        scan_chease(
            str(vaft.data.data_path(SOURCE)),
            [EquilibriumVariation("same"), EquilibriumVariation("same")],
        )
    with pytest.raises(ValueError, match="at least one"):
        scan_chease(str(vaft.data.data_path(SOURCE)), [])


# ---------------------------------------------------------------------------
# What a variation does to the input, without running anything
# ---------------------------------------------------------------------------

def test_the_control_case_changes_nothing(geqdsk):
    modified, shape = apply_equilibrium_variation(geqdsk, EquilibriumVariation("control"))
    assert shape is None
    for key in ("PRES", "PPRIME", "FFPRIM", "RBBBS", "ZBBBS"):
        np.testing.assert_array_equal(
            np.asarray(modified[key], float), np.asarray(geqdsk[key], float)
        )


def test_pressure_scaling_moves_p_and_pprime_together(geqdsk):
    """p' is dp/dpsi and psi is untouched, so they must scale by the same factor."""
    modified, _ = apply_equilibrium_variation(
        geqdsk, EquilibriumVariation("beta", pressure_scale=1.5)
    )
    np.testing.assert_allclose(
        np.asarray(modified["PRES"], float), np.asarray(geqdsk["PRES"], float) * 1.5
    )
    np.testing.assert_allclose(
        np.asarray(modified["PPRIME"], float), np.asarray(geqdsk["PPRIME"], float) * 1.5
    )
    np.testing.assert_array_equal(
        np.asarray(modified["FFPRIM"], float), np.asarray(geqdsk["FFPRIM"], float)
    )


def test_current_peaking_redistributes_ffprime_at_fixed_integral(geqdsk):
    """A shape change, not an amplitude change.

    An amplitude change is the degree of freedom CHEASE normalizes away, which
    is why this preserves the integral and moves the weight inward instead.
    """
    modified, _ = apply_equilibrium_variation(
        geqdsk, EquilibriumVariation("li", current_peaking=1.0)
    )
    before = np.abs(np.asarray(geqdsk["FFPRIM"], float))
    after = np.abs(np.asarray(modified["FFPRIM"], float))
    psi_norm = np.linspace(0.0, 1.0, before.size)
    assert np.trapezoid(after, psi_norm) == pytest.approx(
        np.trapezoid(before, psi_norm), rel=1e-9
    )
    # The weight moved toward the axis: the inner half now carries more of it.
    half = before.size // 2
    assert after[:half].sum() / after.sum() > before[:half].sum() / before.sum()


def test_reshaping_uses_the_boundary_own_fitted_miller_parameters(geqdsk):
    """"10% more elongation" means 10% more than this discharge had."""
    from vaft.process.equilibrium import fit_miller_surface

    base = fit_miller_surface(
        (np.asarray(geqdsk["RBBBS"], float), np.asarray(geqdsk["ZBBBS"], float))
    ).surface
    modified, shape = apply_equilibrium_variation(
        geqdsk, EquilibriumVariation("shape", elongation_scale=1.10)
    )
    assert shape is not None
    _, _, kappa, _ = shape
    assert kappa == pytest.approx(base.kappa * 1.10, rel=1e-9)
    z_before = np.ptp(np.asarray(geqdsk["ZBBBS"], float))
    z_after = np.ptp(np.asarray(modified["ZBBBS"], float))
    assert z_after > z_before


def test_a_shape_variation_leaves_the_profiles_alone(geqdsk):
    modified, _ = apply_equilibrium_variation(
        geqdsk, EquilibriumVariation("shape", triangularity_shift=0.1)
    )
    for key in ("PRES", "PPRIME", "FFPRIM"):
        np.testing.assert_array_equal(
            np.asarray(modified[key], float), np.asarray(geqdsk[key], float)
        )


def test_the_source_is_not_mutated(geqdsk):
    before = np.asarray(geqdsk["PRES"], float).copy()
    apply_equilibrium_variation(geqdsk, EquilibriumVariation("x", pressure_scale=3.0))
    np.testing.assert_array_equal(np.asarray(geqdsk["PRES"], float), before)


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------

@needs_chease
def test_each_knob_moves_the_quantity_it_names(tmp_path):
    """The scan's whole claim, checked against a real solve.

    Peaking is the knob that moves li: scaling FF' uniformly does not, because
    CHEASE rescales the total current and a constant factor is exactly what it
    normalizes away.

    Elongation is read off CHEASE's own output, ``EQDSK_COCOS_02.OUT``, not the
    refined g-file: that file's RBBBS is restored from the input, so it reports
    the requested shape whether or not the solve used it (#887).
    """
    import vaft.omas
    from vaft.code import CHEASEConfig

    config = CHEASEConfig(
        nideal=6, nw=513, target_psin=0.993, relax=0.5, create_plot=False, timeout=900
    )
    cases = scan_chease(
        str(vaft.data.data_path(SOURCE)),
        [
            EquilibriumVariation("control"),
            EquilibriumVariation("beta_up", pressure_scale=1.5),
            EquilibriumVariation("li_up", current_peaking=1.0),
            EquilibriumVariation("elong_up", elongation_scale=1.10),
        ],
        config=config,
        workdir=tmp_path,
    )
    assert all(case.converged for case in cases), [c.error for c in cases]

    measured = {}
    for case in cases:
        ods = read_geqdsk(case.result.refined_geqdsk).to_omas()
        vaft.omas.update_equilibrium_derived_profiles(ods)
        node = ods["equilibrium.time_slice.0"]
        measured[case.variation.label] = (
            float(node["global_quantities.beta_normal"]),
            float(node["global_quantities.li_3"]),
            _elongation(_native_boundary(case.workdir)),
            np.nanmedian(
                vaft.omas.compute_grad_shafranov_residual(ods, time_slice=0).relative
            ),
        )

    control = measured["control"]
    assert measured["beta_up"][0] == pytest.approx(control[0] * 1.5, rel=0.05)
    assert measured["beta_up"][1] == pytest.approx(control[1], rel=0.01)
    assert measured["li_up"][1] > 1.5 * control[1]
    assert measured["elong_up"][2] == pytest.approx(control[2] * 1.10, rel=0.01)
    # A real shape change moves li too, but by a few percent (0.524 -> 0.499 on
    # 48224), not the factor peaking gives. The former "li unchanged to 1%"
    # held only because the solve never saw the new shape.
    assert abs(measured["elong_up"][1] / control[1] - 1.0) < 0.10
    # Every case is a genuine converged equilibrium, not just a file.
    for label, values in measured.items():
        assert values[3] < 0.15, (label, values[3])


@needs_chease
def test_a_scan_survives_a_case_the_solver_cannot_take(tmp_path):
    from vaft.code import CHEASEConfig

    config = CHEASEConfig(
        nideal=6, nw=513, target_psin=0.993, relax=0.5, create_plot=False, timeout=900
    )
    cases = scan_chease(
        str(vaft.data.data_path(SOURCE)),
        [
            EquilibriumVariation("control"),
            EquilibriumVariation("absurd", elongation_scale=25.0),
        ],
        config=config,
        workdir=tmp_path,
    )
    assert cases[0].converged
    assert len(cases) == 2
    assert cases[1].workdir.is_dir()


# ---------------------------------------------------------------------------
# Failure handling, pinned without the solver
# ---------------------------------------------------------------------------

def test_a_raising_case_is_recorded_and_the_scan_continues(monkeypatch, tmp_path):
    """``keep_going`` is the whole reason a scan is not a loop of solves."""
    import vaft.code.chease_scan as module

    calls = []

    def fake(geqdsk, config):
        calls.append(config.workdir)
        if len(calls) == 2:
            raise RuntimeError("mesh did not converge")
        return module.CHEASEResult(returncode=0, workdir=config.workdir)

    monkeypatch.setattr(module, "refine_equilibrium", fake)
    cases = scan_chease(
        str(vaft.data.data_path(SOURCE)),
        [EquilibriumVariation(name) for name in ("a", "b", "c")],
        workdir=tmp_path,
    )
    assert [case.variation.label for case in cases] == ["a", "b", "c"]
    assert cases[0].converged and cases[2].converged
    assert not cases[1].converged
    assert "mesh did not converge" in cases[1].error
    assert len(calls) == 3


def test_keep_going_false_lets_the_failure_out(monkeypatch, tmp_path):
    import vaft.code.chease_scan as module

    def fake(geqdsk, config):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "refine_equilibrium", fake)
    with pytest.raises(RuntimeError, match="boom"):
        scan_chease(
            str(vaft.data.data_path(SOURCE)),
            [EquilibriumVariation("only")],
            workdir=tmp_path,
            keep_going=False,
        )


def test_a_nonzero_exit_is_not_convergence(monkeypatch, tmp_path):
    """CHEASE subprocesses with check=False, so a failed solve returns a result.

    Its most likely failure -- a mesh that will not converge -- exits 1 and
    writes no refined g-file, so "did not raise" must not be read as success.
    """
    import vaft.code.chease_scan as module

    monkeypatch.setattr(
        module, "refine_equilibrium",
        lambda geqdsk, config: module.CHEASEResult(
            returncode=1, workdir=config.workdir, refined_geqdsk=None
        ),
    )
    cases = scan_chease(
        str(vaft.data.data_path(SOURCE)),
        [EquilibriumVariation("failed")],
        workdir=tmp_path,
    )
    assert not cases[0].converged
    assert "exited 1" in cases[0].error


# ---------------------------------------------------------------------------
# A shape knob must reach the boundary CHEASE solves (#887)
# ---------------------------------------------------------------------------

def _expeq_boundary(path):
    """The boundary an EXPEQ file hands CHEASE, in units of its R0.

    Line 4 is the point count and the next that many lines are ``R/R0 Z/R0``;
    see ``vaft.code.chease._write_expeq``.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[3])
    return np.array([[float(v) for v in line.split()] for line in lines[4 : 4 + count]])


def _elongation(rz):
    return float(np.ptp(rz[:, 1]) / np.ptp(rz[:, 0]))


def _native_boundary(workdir):
    """The LCFS CHEASE solved, from its own output before any restoration."""
    native = read_geqdsk(str(workdir / "EQDSK_COCOS_02.OUT"))
    return np.column_stack(
        [np.asarray(native["RBBBS"], float), np.asarray(native["ZBBBS"], float)]
    )


def test_a_shape_variation_reaches_the_expeq_boundary(monkeypatch, tmp_path):
    """RBBBS is not what CHEASE solves; EXPEQ is.

    With ``target_psin < 1`` the adapter traces the EXPEQ boundary from PSIRZ,
    which a shape variation does not touch. A scan that only rewrote RBBBS
    would hand CHEASE the unperturbed shape, and the refined g-file -- whose
    RBBBS is restored from the source -- would still report the requested
    elongation. So measure the file the solver reads.
    """
    import vaft.code.chease_scan as module
    from vaft.code import CHEASEConfig
    from vaft.code.chease import prepare_chease_inputs

    def materialize_only(geqdsk, config):
        prepare_chease_inputs(geqdsk, config)
        return module.CHEASEResult(returncode=1, workdir=config.workdir)

    monkeypatch.setattr(module, "refine_equilibrium", materialize_only)
    cases = scan_chease(
        str(vaft.data.data_path(SOURCE)),
        [
            EquilibriumVariation("control"),
            EquilibriumVariation("elong_up", elongation_scale=1.10),
        ],
        config=CHEASEConfig(target_psin=0.993, create_plot=False),
        workdir=tmp_path,
    )
    kappa = {
        case.variation.label: _elongation(_expeq_boundary(case.workdir / "EXPEQ"))
        for case in cases
    }
    assert kappa["elong_up"] == pytest.approx(1.10 * kappa["control"], rel=0.02)
