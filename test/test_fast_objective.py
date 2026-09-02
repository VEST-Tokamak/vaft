"""The two levers that make #308's objective affordable.

Measured on the packaged 39915 product with all 74 usable channels, one
objective evaluation cost ~7.2 s, of which ~97% was `compute_point_response_ods`
-- pure geometry, invariant under any resistance change, recomputed every time
-- and most of the rest was `np.linalg.eig` on the nonsymmetric `-M^-1 R`.

Lever 1 caches the geometry response across re-solves. Lever 2 decomposes the
symmetric-definite pencil with `eigh` instead, which the loader's reciprocity
fix (#347) made admissible. Neither may change a number beyond round-off.

On what the eigh tests do and do not cover: the *propagator* is checked entry
by entry against `eig` on a pencil spanning four decades of resistance -- the
case where a wrong similarity transform is fatal -- and agrees to 1e-13. The
*solve* is checked on the real 950-loop machine, where the two agree to
1.7e-12. There is deliberately no toy solve-level equivalence test: a random
SPD pencil with a wide R spread has max|I_p| / max|I| ~ 1e6 by construction
(tiny R makes R^-1 in the particular solution enormous), so the update
`I = I_p + P (I - I_p)` cancels catastrophically and amplifies round-off in
*either* decomposition by that factor. Measured: eig and eigh differ by ~7e-7
there, at every drive amplitude, while both propagators agree to 1e-15. The
real machine sits at max|I_p| / max|I| ~ 3 and shows none of it.
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from omas import ODS, load_omas_json

from vaft.omas.vacuum_magnetics import (
    VacuumMagneticsError,
    synthetic_vacuum_magnetics,
    vacuum_response,
)
from vaft.process.electromagnetics import solve_eddy_currents, wall_propagator
from vaft.validation.vacuum_benchmark import benchmark_wall_currents, plasma_free_interval


def _real_shot(shot: int) -> ODS:
    from vaft.data import resources
    import vaft.machine_mapping.em_coupling as em

    try:
        source = resources.data_path(f"samples/{shot}/source/pipeline-until-efit.json.gz")
    except Exception:  # pragma: no cover
        pytest.skip("packaged pipeline sample unavailable")
    if not Path(source).is_file():
        pytest.skip("packaged pipeline sample is repository-only")
    with gzip.open(source, "rt") as handle, tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False
    ) as plain:
        shutil.copyfileobj(handle, plain)
        plain_path = plain.name
    try:
        ods = load_omas_json(plain_path, consistency_check=False)
    finally:
        Path(plain_path).unlink(missing_ok=True)
    # Re-map em_coupling so the sample's stale asymmetric matrix is replaced
    # by the symmetrized one the loader now produces (#347).
    del ods["em_coupling"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        em.em_coupling(ods, shot=shot)
    return ods


@pytest.fixture(scope="module")
def shot_39915():
    return _real_shot(39915)


# --- lever 1: the geometry response is a per-shot constant ------------------


def test_cached_response_reproduces_the_uncached_forward_model_exactly(shot_39915):
    solved = benchmark_wall_currents(shot_39915)
    interval = plasma_free_interval(shot_39915)
    window = (interval.start, interval.end)
    kw = dict(per_family=None, window=window, validity_window=window)

    uncached = synthetic_vacuum_magnetics(solved, **kw)
    rows, response = vacuum_response(solved, **kw)
    cached = synthetic_vacuum_magnetics(solved, response=response, **kw)

    assert len(rows) == len(uncached) == len(cached)
    for a, b in zip(uncached, cached):
        np.testing.assert_array_equal(a.coil_eddy, b.coil_eddy)
        np.testing.assert_array_equal(a.coil, b.coil)
        np.testing.assert_array_equal(a.measured, b.measured)


def test_one_response_serves_every_resistance_the_fit_will_try(shot_39915):
    """The whole point: compute geometry once, re-solve the wall many times."""
    interval = plasma_free_interval(shot_39915)
    window = (interval.start, interval.end)
    kw = dict(per_family=None, window=window, validity_window=window)
    _, response = vacuum_response(benchmark_wall_currents(shot_39915), **kw)

    for scale in (0.5, 1.0, 1.7):
        solved = benchmark_wall_currents(shot_39915, resistance_scale=scale)
        cached = synthetic_vacuum_magnetics(solved, response=response, **kw)
        uncached = synthetic_vacuum_magnetics(solved, **kw)
        for a, b in zip(uncached, cached):
            np.testing.assert_array_equal(a.coil_eddy, b.coil_eddy)


def test_a_response_for_a_different_selection_is_refused(shot_39915):
    """Contracting geometry against the wrong channels would be silent garbage."""
    solved = benchmark_wall_currents(shot_39915)
    interval = plasma_free_interval(shot_39915)
    window = (interval.start, interval.end)
    _, response = vacuum_response(solved, per_family=2, window=window, validity_window=window)
    with pytest.raises(VacuumMagneticsError, match="must come from vacuum_response"):
        synthetic_vacuum_magnetics(
            solved, per_family=None, window=window, validity_window=window, response=response
        )


# --- lever 2: the symmetric-pencil propagator --------------------------------


def _spd_system(n: int = 12, seed: int = 308):
    """An SPD pencil with FOUR decades of resistance -- the genuinely hard case.

    A wrong similarity transform in the eigh back-transform is invisible at a
    narrow spread and fatal at a wide one, so the spread is deliberately wide.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(1.0, 2.0, size=(n, n))
    M = (base + base.T) / 2.0 + n * np.eye(n)
    R = np.diag(np.logspace(-4.0, 0.0, n))
    return R, M, rng


def test_eigh_propagator_matches_eig_to_round_off_on_a_wide_resistance_spread():
    """Four decades of R: the back-transform is checked entry by entry."""
    R, M, _ = _spd_system()
    eig = wall_propagator(R, M, 5.0e-5, method="eig")
    eigh = wall_propagator(R, M, 5.0e-5, method="eigh")
    assert np.max(np.abs(eig - eigh)) / np.max(np.abs(eig)) < 1e-13


def test_auto_falls_back_to_eig_when_the_matrix_is_not_symmetric():
    """A hand-built asymmetric M must take the general path, bit for bit."""
    R, M, rng = _spd_system()
    M[0, 1] *= 1.01
    time = np.linspace(0.0, 0.01, 120)
    drive = np.stack([np.sin(20.0 * time), -0.5 * time], axis=1)
    L = rng.uniform(-1.0, 1.0, size=(M.shape[0], 2)) * 1e-4
    auto = solve_eddy_currents(R, L, M, drive, time, method="auto")
    eig = solve_eddy_currents(R, L, M, drive, time, method="eig")
    np.testing.assert_array_equal(auto, eig)


def test_the_method_argument_is_validated():
    R, M, _ = _spd_system()
    with pytest.raises(ValueError, match="method must be"):
        wall_propagator(R, M, 5.0e-5, method="fast")


def test_eigh_and_eig_agree_on_the_real_machine_to_round_off(shot_39915):
    """950 loops, 2500 samples: the production regression the change must pass.

    Measured 1.7e-12 -- the accumulation floor for this size -- with no loop
    moving above 1e-6 of its own peak. A toy with an unphysically large
    particular solution can amplify round-off far above this; the real drive
    does not, which is why the criterion is set against the real machine.
    """
    from vaft.omas.process_wrapper import compute_impedance_matrices_ods

    R, L, M = compute_impedance_matrices_ods(shot_39915, [])
    assert np.array_equal(M, M.T), "fixture must carry the symmetrized matrix"
    pf = shot_39915["pf_active"]
    time = np.asarray(pf["time"], dtype=float)
    drive = np.array(
        [np.asarray(pf[f"coil.{i}.current.data"], dtype=float) for i in range(len(pf["coil"]))]
    ).T
    eig = solve_eddy_currents(R, L, M, drive, time, method="eig")
    auto = solve_eddy_currents(R, L, M, drive, time, method="auto")

    diff = np.abs(eig - auto)
    assert np.max(diff) / np.max(np.abs(eig)) < 1e-10
    peak = np.max(np.abs(eig), axis=0).clip(1e-30)
    assert not np.any(np.max(diff, axis=0) > 1e-6 * peak)
