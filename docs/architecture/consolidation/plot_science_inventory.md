# Scientific computations inside `vaft.plot` (issue #2270)

Source-based inventory of numerical and physical transformations found in the
plotting layer (`vaft/plot/**`), with the owner each one should have.  Base:
`develop` at 7edf8e026.  Line numbers are those of that base.

Classification (issue #2270, Scope A):

* **reusable** -- a scientific transformation whose meaning does not depend on
  being drawn; belongs in `vaft.process` / `vaft.formula`.
* **view** -- view-specific transformation; stays in `vaft.plot.backend`.
* **render** -- rendering-specific; stays in the renderer.
* **delegation** -- already calls the canonical implementation.
* **unresolved** -- needs semantic/numerical verification before it moves.

Presence of NumPy/SciPy calls is not a criterion.

## Disposition in this pass (P1)

| Candidate | Was | Now | Equivalence |
| --- | --- | --- | --- |
| `recipes._rho_tor_of_psi_norm` | local resolver (stored rho_tor_norm if authentic, else `rho_tor_profile(q, psi)`) | thin wrapper over `vaft.process.profile.equilibrium_rho_tor_norm_table` | code moved verbatim; bit-identical |
| `recipes._build_core_profile_field` mapping | inline `interp(psi_N -> rho_tor) -> interp(rho_tor -> f)`, NaN beyond `psi_N > 1` | `vaft.process.profile.map_rho_tor_profile_to_rz`; the view keeps grid reading, orientation (`(R,Z)` -> `(Z,R)`), labels, overlays | bit-identical (same operations, same order) |
| `recipes._mapped_field_values` | inline sort + interp of a `psi_N` flux function | `vaft.process.equilibrium.psi_to_rz(..., fill_outside="edge")`, view masks `psi_N > 1` to NaN | bit-identical (`np.interp` clamps exactly as `psi_to_rz` clips) |
| `kinetic_overview._core_coordinate` | local `psi_norm`/`rho_pol_norm`/`rho_tor_norm` resolution of a core_profiles grid | `vaft.process.profile.core_profiles_grid_coordinate`; the view keeps time-matching the equilibrium slice | identical except one **correction**: a stored `sqrt(psi_N)` proxy is no longer drawn as `rho_tor_norm` (the fit is omitted, as for any coordinate the grid cannot supply). Detection compares with the grid's own `rho_pol_norm`, else its `psi` normalised by the matched slice's global flux, at 1e-6 (as `kinetic_state` does); a proxy whose `psi` ends differ from those globals is not caught |

`kinetic_state` uses `_core_coordinate` for `psi_norm` only and is unchanged.

## Inventory

| # | Candidate (develop line) | Call sites | Meaning | Canonical equivalent | Class | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `recipes._rho_tor_of_psi_norm` (4675) | core-profile 2-D maps, rational-surface overlays (`_resolve_rational_surfaces`) | slice `(psi_N, rho_tor_norm)` table, proxy refused | `process.profile._equilibrium_grid_at_index` / `_resolve_grid_rho` (private, `derive_radial_coordinates`) | reusable | **moved (P1)** |
| 2 | `recipes._build_core_profile_field` (4718) | `electron_temperature_field`, `electron_density_field` | profile(rho_tor) -> f(R,Z) | `process.equilibrium.psi_to_rz` is the `psi_N` counterpart | reusable + view | **split (P1)** |
| 3 | `recipes._mapped_field_values` (10971) | `equilibrium_*` 2-D field maps of 1-D flux functions | f(psi_N) -> f(R,Z) | `process.equilibrium.psi_to_rz` | reusable (duplicate) | **delegates (P1)** |
| 4 | `kinetic_overview._core_coordinate` (172) | `kinetic_overview`, `kinetic_state` | core grid in psi_N / rho_pol / rho_tor | none public | reusable | **moved (P1)** |
| 5 | `kinetic_overview._mapped_coordinate` (127) | `kinetic_overview` | Thomson/CX (R,Z) -> flux coordinate | `process.profile.equilibrium_mapping_points` | delegation | keep |
| 6 | `recipes._volume_weights_at` (6836) | `_impurity_inputs` -> impurity composition views | dV on a rho_tor grid from equilibrium V(rho_tor) | `omas.vest_upstream._volume_weights_on_grid` (same algorithm, private) | reusable (duplicate) | **moved (P2)**: `process.profile.volume_weights_on_grid` + `equilibrium_volume_weights`, used by the view, the composition stage and the kinetic-state inference |
| 7 | `recipes._mhd_linear_flux_surface_mesh` (14790) | `mhd_linear_geometry_island` | read/validate DCON R,z(psi_N, theta) mesh | none (writer `machine_mapping.gpec_ideal`) | view | keep |
| 8 | `recipes._surface_at` (14941) | island geometry | R,z at a target psi_N per theta column, NaN outside | none equivalent (`extract_flux_surface_contours` labels a different angle) | view | keep |
| 9 | `recipes._gpec_resonant_table` (~14350) | `mhd_linear_profile_resonant_flux`, `_island_width`, `_chirikov`, island geometry | Delta, Phi_res, phase, w_isl per rational surface | building blocks canonical (`process.perturbation.resonant_delta`, `formula.stability.*`); the pipeline only here | reusable | **moved (P2)**: `process.perturbation.resonant_table`; the view keeps the IDS/XML reading |
| 10 | island separatrix assembly (~15000) | `mhd_linear_geometry_island` | psi_r +- (w/2)|cos(xi/2)| | `formula.stability.helical_phase`, `process.equilibrium.pest_angle_from_jacobian_angle` | delegation (geometry local) | keep |
| 11 | `recipes._spectrogram_result` (11326) | `_build_spectrogram`, interferometer spectrogram | method dispatch (stft / hann_fft / cwt) | `process.fluctuation.compute_spectrogram`, `process.magnetics.mirnov_spectrogram`, `process.soft_x_rays.sxr_cwt_spectrogram` | delegation (parameter rewriting) | P3: fix pass-through, see B5 |
| 12 | `recipes._build_spectrogram` (11400) | `mirnov_spectrogram`, `soft_x_rays_spectrogram` | channel/time window, fs from median dt | `process.fluctuation._resolve_sample_rate` | view + delegation | P3 |
| 13 | `recipes._spectrogram_ridge` (~11460) | `_build_spectrogram` (`track=`) | ridge over a search band | `process.fluctuation.track_dominant_frequency` | delegation | P3: band bug, see B5 |
| 14 | `recipes._build_power_spectrum` (11661) | `*_spectrum` | Welch PSD + power-law fit | `process.fluctuation.compute_psd`, `fit_power_law_spectrum` | delegation | P3 |
| 15 | `recipes._default_display_band`, `_display_ceiling` | spectrograms | display ceiling from 1 % of map peak | -- | view | keep (must not feed analysis) |
| 16 | interferometer spectrogram (~2365) | `interferometer_spectrogram` | 200 Hz Butterworth high-pass, window from target df | `process.signal_processing.butterworth_highpass` | view with local analysis defaults | P3 |
| 17 | camera spectrogram / `_image_mhd_power` (~7900) | `camera_visible_spectrogram`, `camera_visible_image_mhd_power` | ROI sum, STFT, band centre = argmax | `process.camera_fluctuation.*`, `compute_spectrogram` | delegation + local peak pick | P3: centre pick to `process.camera_fluctuation` |
| 18 | `_build_mirnov_spatial_phase` + probe selection, `_toroidal_alias_period` (~2700) | `mirnov_spatial_phase` | probe set, preprocessing, wrapped-n fit, alias period 360/gcd | fit: `process.magnetics.toroidal_phase_fit_at_time` | reusable (selection/aliasing) | later: next to the fit in `process.magnetics` |
| 19 | `_coil_set_excitation` sector angle (~5150) | `coil_3d_*_current` | sector phi = vertex-weighted circular mean | `process.coils_non_axisymmetric.toroidal_mode_decomposition` (decomposition only) | unresolved | vertex vs length weighting to verify |
| 20 | `_build_power_balance` (~6340) and `plot/time.py` `time_power_balance` | `power_balance` | dW/dt = V_ind Ip, P_sync residual | `omas.formula_wrapper.compute_power_balance` | reusable (duplicated twice) | later: fold into `compute_power_balance` |
| 21 | `_build_vacuum_field` breakdown, `_vacuum_b_toroidal`, `_ecr_layers` (~4100-4320) | vacuum maps | |E_phi| |B_phi| / B_p, R0B0 nearest sample | `formula.startup.lloyd_figure_of_merit`, `omas.process_wrapper._vacuum_toroidal_product` | reusable (re-implemented) | later: call the canonical pair |
| 22 | `_build_impa_tf_profile` model (~2975) | `impa_profile_field` | B = mu0 24 I_TF / 2 pi R | `process.impa.toroidal_field` | reusable (duplicate, inline turn count) | later |
| 23 | local psi_N normalisations (~9580, `_psi_axis_boundary` ~10775, `_normalized_psi` ~13600, `_build_profile_gradient` ~16915) | profile abscissae, rational markers, 2-D maps, gradients | psi_N with differing fallback order | `formula.equilibrium.psi_normalised` | reusable (inconsistent copies) | later: one resolver returning `(psi_N, reference)` |
| 24 | `_equilibrium_coordinate_values` (~9555) | every `ProfileRecipe` | sqrt_phi_norm, rho_tor via q, r_minor | `data._derived.rho_tor_profile`, `formula.equilibrium.rho_tor_from_phi` | partial delegation | later: route through `equilibrium_rho_tor_norm_table` |
| 25 | `_build_profile_gradient` grid prep (~16830) | `*_profile_gradient` | grid normalisation and rho_tor verification | `process.profile_gradients.profile_gradient` (derivative only) | delegation + reusable preprocessing | later |
| 26 | `_rolling_median` (~9180) | `smooth=` on line recipes | display smoothing | `process.onset.median_smooth` | view | keep |
| 27 | `vaft/plot/soft_x_rays.py` spectrogram (328) | `vaft.plot.soft_x_rays.plot_soft_x_ray_spectrogram` | local `scipy.signal.spectrogram(mode="psd")` | `process.fluctuation.compute_spectrogram` | reusable (re-implemented) | P3: delegate or retire |
| 28 | `vaft/plot/mirnov.py` spectrogram path (120-290) | legacy `vaft.plot.mirnov.*` | gain division, preprocessing, resampling, fs fallback 250 kHz | `process.magnetics.*` | delegation with diverging policy | P3: align with the recipe path or retire |
| 29 | other plot-module physics: `plot/time.py` V_loop, `plot/profile.py` p from beta_tor, `kinetic_state` p_e and error propagation, `plot/history.py` regression metrics | respective plots | as named | `formula.equilibrium.loop_voltage_from_total_flux`, `electron_pressure`, `formula.statistics.*` | reusable (duplicates) | later |

Already delegating, no action: rational surfaces (`process.equilibrium.rational_surfaces`),
coherence (`process.fluctuation.cross_spectrum`), resistive Z_eff, Romero balance,
bootstrap, startup proxies, vacuum midplane, coil toroidal spectrum.  View or
render only: connection-length gridding, discharge-window axis limits, the
reference-slope anchor in `renderers/spectra.py`, phase-jump line breaking.

## B3 -- equilibrium volume weighting

`_volume_weights_at` and `omas.vest_upstream._volume_weights_on_grid` are the
same algorithm (V(rho_tor) interpolated to cell edges, differenced) with
different edge behaviour: on a degenerate V the view returns `None` (cylindrical
fallback in `process.impurity._radial_weights`), upstream returns zeros (0/0 in
`resolve_radial_composition`); upstream refuses `rho.size < 2`, the view does
not; both read equilibrium `rho_tor_norm` without the `sqrt(psi_N)` proxy
check.  No public `vaft.process` API produces dV on a profile grid
(`process.equilibrium.volume_average` is a 2-D cell average).  Disposition: one
public helper with proxy and monotonicity validation, used by both callers --
a separate PR, because unifying the edge policy changes upstream output.

**Done in P2.** `vaft.process.profile.volume_weights_on_grid` (array kernel)
and `equilibrium_volume_weights` (ODS adapter) replace both copies.
Bit-identical on an authentic radius (frozen copies; packaged 48224).
Corrections: a proxy radius is replaced by the `q`-derived one, the same
resolution `core_profiles` builds its grid with, or the slice is declined; a
constant `V` and an unusable grid are declined by every caller (upstream used
to return zeros, i.e. 0/0).  The derivation runs on a private copy of the
radius leaves, so the caller's ODS is never written.

## B4 -- MHD flux-surface geometry

`_mhd_linear_flux_surface_mesh` only reads and validates the mapped DCON mesh;
`_surface_at` interpolates radially per Jacobian-angle column and returns NaN
outside the mapped range, which is correct for clipped islands.  Neither has an
equivalent outside the plot layer, and the process straight-field-line
routines label a different angle, so they are not interchangeable.  The angle
conversion already delegates (`pest_angle_from_jacobian_angle`).  Both stay.
The reusable science in this area is `_gpec_resonant_table` (item 9), whose
per-surface derivation moved verbatim to
`vaft.process.perturbation.resonant_table` in P2.  The `m = round(n q)` sign
for negative q is unresolved.

## B5 -- spectral parameter semantics

No spectral algorithm is reimplemented in the recipe path; the
`vaft/plot/soft_x_rays.py` legacy spectrogram is the exception (item 27).
Parameter findings for the P3 PR:

1. Method-mismatched options pass the global option schema and are silently
   ignored (`nperseg` with `hann_fft`, `window_size` with `stft`,
   `max_frequency`/`frequency_range`/`track` on `*_spectrum`).
2. `track=True` without `frequency_range` bounds the ridge search by the
   **display** ceiling (`max_frequency` or the 1 %-of-peak heuristic), so the
   tracked ridge depends on a rendering choice.
3. `frequency_range` is a post-hoc crop of a full-band analysis for stft and
   hann_fft but the true analysis band for cwt; for coherence it is documented
   as display-only.
4. `nperseg` is clamped to the record length and `noverlap` to `[0, 0.99]`
   fraction; an odd `hann_fft` window is decremented -- the process layer
   would raise on each.
5. stft (density-scaled magnitude), hann_fft (2|X|/N amplitude) and cwt are
   all labelled "Magnitude".
6. An untimed signal gets `fs = 1.0` and is labelled Hz / s; the PSD path
   refuses the same input.
7. Camera spectrograms take `overlap=` (fraction) where the others take
   `noverlap=` (samples).

## Other defects found by the sweep

* `impa_profile_field` in voltage mode with no `time=` samples index 0 after
  subtracting `trace[0]`, so every plotted value is 0 and no model is drawn.
* `_build_power_balance` falls back to the slice *index* when a slice time is
  missing, then time-matches it as seconds; a missing P_Br counts as 0 in the
  "synchrotron" residual.
* The core-profile 2-D maps and `_mapped_field_values` contain cells by
  `psi_N <= 1` alone, so a private-flux region below an X-point is painted;
  `project_flux_function` uses the boundary outline as well.
* `omas.update.update_core_profiles_global_quantities_volume_average` reads
  equilibrium `rho_tor_norm` without the proxy check.
