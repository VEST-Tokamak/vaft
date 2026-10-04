"""Known phases, rank, masks and ambiguity for spectral components."""

from dataclasses import replace

import numpy as np
import pytest

from vaft.process.fluctuation import (
    FluctuationRecord, coherent_components, cross_spectral_matrix,
)


def _matrix():
    fs = 40_000
    time = np.arange(8_000) / fs
    rng = np.random.default_rng(1611)
    phases = (0.0, 0.45, -0.7)
    amplitudes = (1.0, 2.0, 0.5)
    records = [FluctuationRecord(name, time,
               scale * (np.sin(2 * np.pi * 2_000 * time + angle)
                        + 0.05 * rng.normal(size=time.size)))
               for name, scale, angle in zip(("ref", "b", "c"), amplitudes, phases)]
    return cross_spectral_matrix(records, nperseg=200)


def test_phase_sign_participation_and_scale_invariance():
    result = _matrix()
    components = coherent_components(result, reference="ref")
    peak = int(np.argmin(abs(result.frequency - 2_000)))
    assert components.component_defined[peak].all()
    assert np.nanmedian(components.coherent_fraction[peak]) > 0.98
    assert np.nanmedian(components.phase[peak, :, 1]) == pytest.approx(0.45, abs=0.05)
    assert np.nanmedian(components.phase[peak, :, 2]) == pytest.approx(-0.7, abs=0.05)
    np.testing.assert_allclose(components.participation[peak].sum(axis=-1), 1)
    np.testing.assert_allclose(components.eigenvalues[peak].sum(axis=-1), 3, atol=1e-9)
    np.testing.assert_allclose(result.raw_csd[peak, 0, 1, 1].real /
                               result.raw_csd[peak, 0, 0, 0].real, 4, rtol=0.05)
    np.testing.assert_allclose(components.participation[peak, :, 0],
                               components.participation[peak, :, 1], atol=0.02)


def test_missing_reference_and_single_diagnostic_do_not_claim_phase():
    result = _matrix()
    valid = result.valid.copy()
    matrix = result.matrix.copy()
    valid[0, :, 0] = False
    matrix[0, :, 0, :] = np.nan
    matrix[0, :, :, 0] = np.nan
    valid[-1, :, 1:] = False
    matrix[-1, :, 1:, :] = np.nan
    matrix[-1, :, :, 1:] = np.nan
    components = coherent_components(replace(result, valid=valid, matrix=matrix),
                                     reference="ref")
    assert components.component_defined[0].all()
    assert np.isnan(components.phase[0]).all()
    assert (components.n_diagnostics[-1] == 1).all()
    assert not components.component_defined[-1].any()
    assert np.isnan(components.coherent_fraction[-1]).all()


def test_degenerate_leading_pair_has_no_unique_participation():
    result = _matrix()
    matrix = result.matrix.copy()
    matrix[0, 0] = np.eye(3)
    components = coherent_components(replace(result, matrix=matrix), reference="ref")
    assert components.coherent_fraction[0, 0] == pytest.approx(1 / 3)
    assert not components.component_defined[0, 0]
    assert np.isnan(components.participation[0, 0]).all()
    assert np.isnan(components.phase[0, 0]).all()


def test_zero_loading_has_no_phase_and_insufficient_averages_have_no_fraction():
    result = _matrix()
    matrix = result.matrix.copy()
    matrix[0, 0] = np.array([[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]])
    components = coherent_components(replace(result, matrix=matrix), reference="ref")
    assert components.participation[0, 0, 2] == pytest.approx(0)
    assert np.isnan(components.phase[0, 0, 2])
    shared = result.shared_segments.copy()
    shared[0, 0] = 3  # Three records require at least four independent averages.
    insufficient = coherent_components(replace(result, shared_segments=shared),
                                       reference="ref")
    assert np.isnan(insufficient.coherent_fraction[0, 0])
    assert not insufficient.component_defined[0, 0]


def test_independent_component_reduces_dominant_fraction():
    fs = 40_000
    time = np.arange(16_000) / fs
    rng = np.random.default_rng(17)
    tone = np.sin(2 * np.pi * 2_000 * time)
    records = [
        FluctuationRecord("a", time, tone + 0.2 * rng.normal(size=time.size)),
        FluctuationRecord("b", time, np.sin(2 * np.pi * 2_000 * time + 0.4)
                          + 0.2 * rng.normal(size=time.size)),
        FluctuationRecord("noise", time, rng.normal(size=time.size)),
    ]
    matrix = cross_spectral_matrix(records, nperseg=200, segments_per_window=16)
    components = coherent_components(matrix, reference="a")
    peak = int(np.argmin(abs(matrix.frequency - 2_000)))
    assert 0.55 < np.nanmedian(components.coherent_fraction[peak]) < 0.9
    assert np.nanmedian(components.participation[peak, :, 2]) < 0.2


def test_requires_psd_normalization_and_known_reference():
    result = _matrix()
    with pytest.raises(ValueError, match="PSD normalization"):
        coherent_components(replace(result, normalization="none"), reference="ref")
    with pytest.raises(ValueError, match="reference"):
        coherent_components(result, reference="unknown")
