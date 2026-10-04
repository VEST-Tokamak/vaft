---
title: Cross-diagnostic fluctuation coherence
author: VEST team
date: 2026-10-05 00:00
category: guide
layout: post
permalink: /workflows/fluctuation-coherence/
guide:
  architecture: Scalar diagnostic selection, time alignment, spectral matrix, and coherent-component decomposition.
  prerequisites: Calibrated or explicitly labelled scalar signals with stored time axes and usable bandwidth metadata.
  expected: Time-resolved coherence, raw and normalized spectral matrices, component participation and relative phase.
related:
  notebooks: [fluctuation-diagnostics]
  api: [process, plot]
  data_sources: [sample-ods]
---

# Cross-diagnostic fluctuation coherence

The [fluctuation tutorial](https://github.com/VEST-Tokamak/vaft/blob/develop/tutorial/04_fluctuation_diagnostics_for_plasma_perturbations_and_transient_events.ipynb)
uses VEST shot 45531 to compare Mirnov, soft X-ray and H-alpha records. Its camera comparison uses
**shot 40600**: no 45531 camera frames or interferometer record are supplied by that archive.
Those missing diagnostics must not enter a 45531 matrix as zeros or as another shot's signals.

The processing functions work on plain scalar records. `vaft.omas.select_fluctuation_records`
extracts one explicitly named representative per diagnostic from an ODS or native IMAS entry;
`vaft.process.fluctuation` aligns and analyses the resulting records; `vaft.plot` draws the maps.
The selection API accepts Mirnov **field** only after calibration and integration of pickup
voltage, a named SXR chord and explicit brightness/power energy band, an interferometer line
integral, a camera ROI or a provenance-labelled temporal component, and a unique UV emission
identity. SXR and UV units must be supplied because calibrated IMAS values and legacy proxy
voltages/intensities differ. The selection source survives in every `FluctuationRecord`,
`CrossSpectralMatrix`, and `CoherentComponents` result. One representative per diagnostic
prevents a 52-chord array from receiving 52 times the weight of a single-channel diagnostic.

## Estimation and units

For two records, `cross_spectrogram` returns one-sided auto and cross spectral **densities**,
magnitude-squared coherence and the phase of **y relative to x**. Each output time window
averages at least two independent inner Fourier segments before dividing cross-power by the
auto PSDs. A single outer product would make coherence identically one and has no such
interpretation. The default averages four non-overlapping 256-sample inner segments; the outer
window advances by half its length. State the segment count and frequency resolution alongside
any coherence result.

For several records, the raw matrix has

$$S_{ij}(f,t)=\langle X_i(f,t)X_j^*(f,t)\rangle.$$

Its entry has units `(units_i × units_j)/Hz`. PSD normalization divides by
$\sqrt{S_{ii}S_{jj}}$, yielding a dimensionless Hermitian matrix with unit diagonal where
power is positive. Other explicit policies are `none`, whole-record `variance`, and `user` scales.
The raw physical-unit matrix remains in `raw_csd`. The default averaging count grows to at least
`N+1` inner segments for `N` selected diagnostics; fewer explicitly requested segments leave
component metrics undefined when the sample matrix is rank deficient by construction.

The dominant eigenvalue divided by the matrix trace is a **leading spectral fraction**, not a
probability or a significance level. Finite averaging raises it above `1/N` even for independent
signals. Compare a candidate with a null distribution using the same window, number of averages,
diagnostic count and selection procedure. A tied leading eigenvalue has no unique eigenvector;
participation and phase are then `NaN`. A missing diagnostic, out-of-band diagnostic, or zero
loading also has undefined participation or phase, never an invented zero response.

## Time, bandwidth and provenance

Records meet only on their common time interval. Rate reduction uses an anti-alias filter before
projection onto the requested grid. Declare physical usable bandwidths where known; otherwise
the result records `nyquist_only`, meaning the sampling ceiling is known but the sensor and
analogue passband are not. A requested band above a declared limit or conservative anti-alias
passband is rejected. Each record's source rate, target rate, operation, filter cutoff and
effective usable band are in `resampling`. The set of valid diagnostics can vary with frequency;
the `valid` mask and `n_diagnostics` record that partial overlap.

The 25 kHz grid in the 45531 tutorial examines only 2–8 kHz. It cannot establish that its H-alpha
record follows a 14 kHz branch: that frequency exceeds the grid's 12.5 kHz Nyquist. The Mirnov
record in the archive is pickup voltage (proportional to $dB/dt$), and its raw PSD is V²/Hz.
Without a calibrated integration it is not a magnetic-field PSD in T²/Hz. PSD normalization
supports dimensionless comparison but does not restore a physical field amplitude.

## What the result says

| Observation | Supported claim |
|---|---|
| Simultaneous onset | Activity shares a time; a common plasma evolution can cause it. |
| Same frequency | Records contain power in one band; separate sources can do that. |
| Pairwise coherence with stable phase | Two selected records maintain a spectral relation, subject to pickup and selection controls. |
| Dominant multi-diagnostic component | Selected records concentrate normalized spectral power in one direction, with measurable participation and relative phase. |
| Physical MHD mode identification | Needs spatial phase, mode-number and diagnostic transfer-function evidence beyond the spectral decomposition. |

Use the canonical maps `plot_cross_diagnostic_coherence_spectrogram`,
`plot_multi_diagnostic_coherent_spectrogram`, `plot_multi_diagnostic_coherent_fraction`,
`plot_multi_diagnostic_participation`, and `plot_multi_diagnostic_phase` to inspect these
quantities on their recorded time and frequency coordinates. The phase map is in degrees, the
process result in radians. The leading component is named a *dominant coherent spectral
component* until independent spatial evidence supports a physical mode identification.

## Small synthetic check

This demonstrates a known positive phase of y relative to x without relying on one discharge.
The independent noise makes the averaged estimate meaningful; two signals on the same frequency
alone would not establish coherent observation.

```python
import numpy as np
from vaft.process.fluctuation import (
    FluctuationRecord, coherent_components, cross_spectral_matrix,
)

fs = 40_000
t = np.arange(8_000) / fs
rng = np.random.default_rng(1611)
x = np.sin(2 * np.pi * 2_000 * t) + 0.1 * rng.normal(size=t.size)
y = np.sin(2 * np.pi * 2_000 * t + 0.4) + 0.1 * rng.normal(size=t.size)
records = [FluctuationRecord("reference", t, x, "V"),
           FluctuationRecord("second", t, y, "V")]
matrix = cross_spectral_matrix(records, nperseg=200, frequency_range=(1_000, 3_000))
component = coherent_components(matrix, reference="reference")
at_2khz = np.argmin(abs(component.frequency - 2_000))
print(np.nanmedian(component.phase[at_2khz, :, 1]))  # approximately +0.4 rad
```
