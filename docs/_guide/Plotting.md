---
title: Plotting
author: VEST team
date: 2026-07-01 10:50
category: guide
layout: redirect
redirect_to: /workflows/experimental-interpretation/
---

`vaft.plot` draws OMAS-formatted VEST data: raw diagnostic traces, reconstructed
equilibria, fluctuation spectra and derived scalars. Everything is Matplotlib, and every
function below works the same in a notebook or a plain script.

The module is a star-import of ten submodules, so all of the names below are reachable
both as `vaft.plot.<name>` and through their submodule
(`vaft.plot.twodim.equilibrium_2d_profiles`, `vaft.plot.time.time_magnetics_ip`, ...).
Both spellings are used in the shipped notebooks.

Reference notebook:
[`plotting_sample_using_vaft_plot_module.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/plotting_sample_using_vaft_plot_module.ipynb).

## Sample data — runnable without a database

`vaft.omas` ships packaged sample shots, so every snippet on this page runs offline, with
no server and no `vaft.database` call:

```python
import vaft

ods = vaft.omas.sample_ods()   # one shot  (#39915)
odc = vaft.omas.sample_odc()   # three shots, as an ODS collection
```

`sample_ods()` returns a single ODS; `sample_odc()` returns an ODC of three shots. Pass the
ODC wherever an ODS is accepted and the shots are overlaid on shared axes. Internally, a
bare ODS is wrapped into a one-entry ODC by `vaft.omas.odc_or_ods_check`, so the time-trace
functions accept either. See
[Data structures]({{ site.baseurl }}/guide/Data_structures/) for the ODS/ODC model.

### The canonical snippet

```python
import matplotlib
matplotlib.use("Agg")        # drop this line in a notebook

import vaft

ods = vaft.omas.sample_ods()
odc = vaft.omas.sample_odc()

# time traces: single shot, then three shots overlaid
vaft.plot.time_magnetics_ip(ods, yunit='kA')
vaft.plot.time_magnetics_ip(odc, yunit='kA', xunit='ms', label='shot')

# PF coil currents, two selected coils, explicit legend
vaft.plot.time_pf_active_current(odc, indices=[0, 4], yunit='kA',
                                 label=['PF 5', 'PF 9-10'])

# 1D equilibrium profile: pass the ODC (see the warning below)
vaft.omas.update_equilibrium_profiles_1d_normalized_psi(ods)
vaft.plot.equilibrium_rho_tor_norm_pressure(odc)

# 2D equilibrium
vaft.plot.equilibrium_2d_profiles(ods, time_slice=2)
```

## Naming schemes

Several naming schemes coexist. The ones you need to recognise:

| Scheme | Example | Where |
|---|---|---|
| `time_<ids>_<quantity>` | `time_magnetics_ip(odc_or_ods, ...)` | time traces — **canonical** |
| `<ids>_time_<quantity>` | `magnetics_time_ip(odc_or_ods, ...)` | alias of the above, reversed word order |
| `equilibrium_<coordinate>_<quantity>` | `equilibrium_rho_tor_norm_pressure(odc, ...)` | 1D equilibrium profiles |
| `plot_<...>` | `plot_soft_x_ray_signal(ods, ...)`, `plot_scaling_fit(df, ...)` | soft X-rays, kinetic profiles, history |
| bare noun | `mirnov_signal(ods, ...)`, `vacuum_psi_contour(ods)` | Mirnov, 2D geometry |

For time traces, **`time_<ids>_<quantity>` is canonical** and new code should use it. The
reversed `<ids>_time_<quantity>` names are plain assignment aliases — the same function
objects, not wrappers — and the shipped notebooks mostly use that form, so both remain
valid:

```python
vaft.plot.time_magnetics_ip(ods)      # canonical
vaft.plot.magnetics_time_ip(ods)      # alias — identical object
```

Every function in the tables below with a `time_` prefix has the corresponding reversed
alias (`pf_active_time_current`, `equilibrium_time_q95`, `tf_time_coil_current`,
`barometry_time_pressure`, `spectrometer_uv_time_intensity`, ...). The soft X-ray functions
have `soft_x_rays_*` aliases in the same way.

## Shared keywords — and how they fail

The time-trace functions share four keywords. **Read this before trusting a figure:** an
invalid value for `xlim`, `label`, `xunit` or `yunit` does **not** raise. It either prints a
notice and falls back to the default, or is silently ignored while the axis label still
reflects what you asked for.

### `xlim`

| Value | Behaviour |
|---|---|
| `'plasma'` (default) | window from plasma-current on/off |
| `'coil'` | window spanning all PF coil-current on/off |
| `'none'` | no limit applied |
| `[t0, t1]` | a list of exactly two numbers, used literally |
| anything else | prints `Invalid xlim: ...`, then uses `'plasma'` |

If no signal is found to derive the window from, no limit is applied at all.

### `label`

| Value | Behaviour |
|---|---|
| `'shot'` / `'pulse'` | legend from `dataset_description.data_entry.pulse` |
| `'run'` | from `dataset_description.data_entry.run` |
| `'key'` | the ODC key |
| a list | used literally — **only if its length equals the number of ODSs** |
| anything else | prints `Invalid label: ...`, then uses `'key'` |

A list of the wrong length is not an error: it fails the length check and silently falls
back to `'key'`.

### `xunit` and `yunit`

`xunit` accepts `'s'` (default) and `'ms'`. **Any other value leaves the data in seconds but
still writes your string into the axis label** — you get a wrong label with no warning.

`yunit` behaves the same way, and its accepted values differ per function
(`'A'`/`'kA'`/`'MA'` for `time_magnetics_ip`, `'kA_T'`/`'MA_T'` for
`time_pf_active_current_turns`, `'Wb'`, `'V'`, `'T'`, `'J'`, `'Pa'`, `'a.u.'`, `''`
elsewhere). An unrecognised value means no unit conversion is applied, but the label still
says what you passed.

Neither keyword is validated anywhere. Stick to the values listed per function.

### `indices` — this one *does* raise

The sensor-selection keyword is the exception: an unusable value raises `ValueError`.

| Function family | Accepted `indices` |
|---|---|
| `time_pf_active_current`, `time_pf_active_current_turns` | `'used'` (default; coils carrying current), `'all'`, an int, a list of ints |
| `time_magnetics_flux_loop_flux`, `time_magnetics_flux_loop_voltage` | `'all'` (default), `'inboard'`, `'outboard'`, `'inboard_midplane'`, an int, a list of ints |
| `time_magnetics_b_field_pol_probe_field` | `'all'` (default), `'inboard'`, `'outboard'`, `'side'`, an int, a list of ints |

The probe groups are the ones described in
[Magnetics]({{ site.baseurl }}/guide/Magnetics/).

## Time traces

All of these take an ODS or an ODC as first argument, call `plt.show()` and return `None` —
they cannot be composed into an existing subplot grid.

### Magnetics

```python
vaft.plot.time_magnetics_ip(odc, yunit='kA', xunit='ms')
vaft.plot.time_magnetics_diamagnetic_flux(ods, yunit='Wb')
vaft.plot.time_magnetics_flux_loop_flux(ods, indices='inboard')
vaft.plot.time_magnetics_flux_loop_voltage(ods, indices='all')
vaft.plot.time_magnetics_b_field_pol_probe_field(ods, indices='inboard')
```

![Plasma current]({{ site.baseurl }}/assets/images/magnetics/plasma_current.png)

The flux-loop and probe functions lay their channels out in a subplot grid whose shape
depends on the group you selected:

![Inboard probe field]({{ site.baseurl }}/assets/images/magnetics/Inboard_B_z.png)

![Inboard flux loop]({{ site.baseurl }}/assets/images/magnetics/Inboard_flux_loop.png)

Loop voltage is computed as $-\mathrm{d}\psi/\mathrm{d}t$ from the flux-loop signal.

Two diamagnetic-flux functions exist and they are **not** the same:

- `time_magnetics_diamagnetic_flux(ods_or_odc, ...)` — the raw magnetics signal only.
- `time_diamagnetic_flux(ods_or_odc, ...)` — raw magnetics **plus** the equilibrium measured
  and equilibrium reconstructed values on one axis.

![Diamagnetic flux]({{ site.baseurl }}/assets/images/magnetics/diamagnetic_flux.png)

### PF coils

```python
vaft.plot.time_pf_active_current(odc, indices=[0, 4], yunit='kA',
                                 label=['PF 5', 'PF 9-10'])
vaft.plot.time_pf_active_current_turns(odc, yunit='kA_T')
```

`time_pf_active_current_turns` multiplies the coil current by its turn count; `yunit`
accepts `'kA_T'` and `'MA_T'` (anything else leaves the data in ampere-turns).

### Equilibrium scalars

Global quantities from the reconstructed equilibrium, all with the signature
`(odc_or_ods, label='shot', xunit='s', yunit=..., xlim='plasma')`:

| Function | Default `yunit` | Quantity |
|---|---|---|
| `time_equilibrium_plasma_current` | `'MA'` | plasma current |
| `time_equilibrium_li` | `''` | internal inductance $l_i$ |
| `time_equilibrium_beta_pol` | `''` | poloidal beta |
| `time_equilibrium_beta_tor` | `''` | toroidal beta |
| `time_equilibrium_beta_n` | `''` | normalized beta |
| `time_equilibrium_w_mhd` | `'J'` | MHD stored energy |
| `time_equilibrium_w_mag` | `'J'` | magnetic stored energy |
| `time_equilibrium_w_tot` | `'J'` | total stored energy |
| `time_equilibrium_q0` | `''` | $q$ on axis |
| `time_equilibrium_q95` | `''` | $q_{95}$ |
| `time_equilibrium_qa` | `''` | $q_a$ |
| `time_equilibrium_major_radius` | `'m'` | geometric axis $R$ |

```python
vaft.plot.time_equilibrium_q95(odc)
vaft.plot.time_equilibrium_beta_n(odc)
```

### Other diagnostics

```python
vaft.plot.time_tf_coil_current(ods, yunit='MA')
vaft.plot.time_tf_b_field_tor(ods, yunit='T')          # vacuum toroidal field
vaft.plot.time_tf_b_field_tor_vacuum_r(ods)            # B_tor * R
vaft.plot.time_barometry_pressure(ods, yunit='Pa')     # neutral pressure
vaft.plot.time_spectrometer_uv_intensity(ods, indices='all')
vaft.plot.time_impurity_effect(ods)                    # 3x2: Ip/Ha, flux/CIII, V_loop/OII
```

`time_impurity_effect` takes `impurity_lines=None` instead of a `yunit`.

### Single-ODS composite figures

These take one ODS (not an ODC) and build a multi-panel summary:

| Function | Signature |
|---|---|
| `time_energy` | `(ods, figsize=(4, 4))` — magnetic + thermal energy |
| `time_beta` | `(ods, figsize=(4, 4))` |
| `time_power_balance` | `(ods, figsize=(6, 6.5))` |
| `time_voltage_consumption` | `(ods, figsize=(4, 4))` |
| `time_virial_equilibrium_quantities` | `(ods, figsize=(8, 10))` |
| `plot_core_profiles_time_volume_averaged` | `(ods)` |
| `time_electromagnetics_current` | `(ods, label='shot', xunit='s', xlim='plasma', coil_indices='used', bpol_probes=..., flux_loops=..., onset=None, time_of_interest=None)` |

```python
vaft.plot.time_energy(ods)
vaft.plot.time_beta(ods)
vaft.plot.time_power_balance(ods, figsize=(8, 10))
```

`time_electromagnetics_current` draws a 2×3 electromagnetic overview. Its `bpol_probes` and
`flux_loops` defaults are **VEST-specific hard-coded channel indices**
(`inboard_bz` 4, `outboard_bz` 39, `fl_1` 2); pass your own dictionaries for a different
machine or probe set.

The quantities behind the energy, beta and power-balance panels are defined in
[Physics formulas]({{ site.baseurl }}/guide/Formula/).

## Time conventions

Different diagnostics define $t = 0$ differently, so overlaying shots is only meaningful once
they share a convention. `vaft.omas.change_time_convention` rebases the time base of an ODS
or ODC **in place** and records the active convention, so repeated calls are idempotent
rather than cumulative:

```python
vaft.plot.magnetics_time_ip(odc)                                # as acquired
vaft.omas.change_time_convention(odc, convention='breakdown')
vaft.plot.magnetics_time_ip(odc)                                # aligned on breakdown
```

Accepted conventions are `'daq'`, `'vloop'` (the default), `'ip'` and `'breakdown'`. Unlike
the plotting keywords, an unknown convention here raises `ValueError`.

## 1D equilibrium profiles

Twenty-four profile functions are generated for every (coordinate, quantity) pair, named
`equilibrium_<coordinate>_<quantity>`:

- **Coordinates** — `psi_norm`, `rho_tor_norm`, `r_major`, `r_minor`
- **Quantities** — `j_tor`, `q`, `pressure`, `pprime`, `f`, `ffprime`

All twenty-four share the signature
`(odc_or_ods, time_slices=None, labels_opt='shot', **plot_kwargs)`, and extra keyword
arguments are forwarded to `matplotlib.pyplot.plot`.

```python
vaft.omas.update_equilibrium_profiles_1d_normalized_psi(ods)   # populate psi_norm first

vaft.plot.equilibrium_rho_tor_norm_pressure(odc)
vaft.plot.equilibrium_rho_tor_norm_q(odc)
vaft.plot.equilibrium_r_major_j_tor(odc)
vaft.plot.equilibrium_psi_norm_ffprime(odc, linestyle='--')
```

> **Pass an ODC.** With an ODC of two or more shots, each shot's profile at its
> maximum-$I_p$ slice is drawn on shared axes. This is the path that works.
>
> **Do not pass `time_slices=` with a single ODS.** The time-slice key comparison is
> string-versus-integer, so every requested slice is rejected. You get a
> `Warning: Time slice key '0' not available in ODS. Available keys: [0, 1]. Skipping.`
> on stdout and **an empty figure — no exception**. The only single-ODS path that renders is
> a Jupyter session with `ipywidgets` installed and `time_slices=None`, which offers an
> interactive slice dropdown.

These functions also print progress lines to stdout on every call; that noise is expected.

`plot_onedim_profile_interactive(odc_or_ods, ods_group_name, quantity_name,
coordinate_name, time_slices=None, labels_opt='shot', **plot_kwargs)` is the engine behind
the generated names and can be called directly for a pair outside the generated set. It
carries exactly the same caveats.

`equilibrium_1d_radial(ods, time_slices=None)` produces radial-coordinate mapping figures,
useful as a consistency check on a reconstruction.

## 2D equilibrium, geometry and flux surfaces

```python
vaft.plot.equilibrium_2d_profiles(ods, time_slice=2)     # 2x3 grid of 2D profiles
vaft.plot.twodim_geometry_all(ods)                       # coils, passive structure, wall
vaft.plot.overlay_all(ods)                               # combined machine overlay
vaft.plot.vacuum_psi_contour(ods, time=None, cmap='viridis')
vaft.plot.overlay_all_with_vacuum_psi_contour(ods)
```

| Function | Signature |
|---|---|
| `equilibrium_2d_profiles` | `(ods, time_slice=None, figsize=(10, 6))` |
| `twodim_geometry_all` | `(ods)` |
| `overlay_all` | `(ods)` |
| `vacuum_psi_contour` | `(ods, time=None, cmap='viridis', fontsize=12, savepath=None)` |
| `overlay_all_with_vacuum_psi_contour` | `(ods, time=None, cmap='viridis', fontsize=12, savepath=None)` |
| `pf_passive_overlay` | `(ods, ax=None, colors=None, **kw)` |

Passing `savepath` writes the figure to disk. `pf_passive_overlay` is the one function here
that accepts an `ax`, so it can be layered onto a figure you are building yourself.

A reconstruction that models the plasma as filaments or as a grid of current
elements stores that representation in the `pf_plasma` IDS
(`vaft.omas.pf_plasma.set_plasma_elements` writes it, `plasma_elements` reads
it). `vaft.plot.pf_plasma_geometry_poloidal(ods, time=None)` draws the
elements coloured by their signed current at one instant, with the limiter
outline, so a filament fit and an element fit of the same slice read the same
way; the OMAS and IMAS adapters are `plot_pf_plasma_geometry_poloidal`.

## Mirnov coils

The Mirnov functions **return figures** and accept `ax=` and `show=`, so they compose into
your own subplot grids. Everything after the first argument is keyword-only.

| Function | Signature |
|---|---|
| `mirnov_signal` | `(ods, channels=None, *, probe_group='b_field_pol_probe', time_range=None, preprocess=False, gains=None, ax=None, show=True)` |
| `mirnov_spectrogram` | `(ods, channel=0, *, probe_group='b_field_pol_probe', time_range=None, preprocess=True, gain=None, sample_rate=None, window_size=500, time_resolution=1, max_frequency=None, cmap='hot_r', ax=None, show=True, return_result=False)` |
| `toroidal_mode_spectrum` | `(ods, channel_pair=(65, 67), *, probe_group='b_field_pol_probe', time_range=None, preprocess=True, gains=None, phase_geometry=np.pi/6, peak_threshold=0.1, sample_rate=None, axes=None, show=True, return_result=False)` |
| `toroidal_phase_mode_fit` | `(ods, center_time, *, channels=(64, 65, 66, 67), probe_group='b_field_pol_probe', time_range=None, frequencies=None, num_modes=2, candidate_n=tuple(range(0, 7)), window_size=500, preprocess=True, gains=None, sample_rate=None, peak_threshold=0.1, ax=None, show=True, save_path=None, return_result=False)` |

```python
import matplotlib.pyplot as plt
import vaft.plot as vplot

time_range = (0.304, 0.330)

fig, ax = vplot.mirnov_signal(ods, channels=[14, 37], time_range=time_range,
                              preprocess=False, show=False)
ax.set_title("Raw Mirnov voltage")

fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
vplot.mirnov_spectrogram(ods, channel=14, time_range=time_range,
                         max_frequency=80e3, ax=axes[0], show=False)
```

`mirnov_spectrogram` preprocesses by default (`preprocess=True`) and exposes `window_size`,
`time_resolution`, `max_frequency` and `cmap`. `toroidal_mode_spectrum` and
`toroidal_phase_mode_fit` infer the toroidal mode number from the phase difference between
channels; pass `return_result=True` to get the fit back alongside the figure.

The default channel numbers (65/67, 64–67) are VEST-specific — set `channels` /
`channel_pair` explicitly for other probe layouts.

## Soft X-rays

The cleanest API in the module: every function returns `(fig, ax)` (the overview returns
`(fig, axs)`) and takes `ax=` / `show=`.

| Function (+ alias) | Signature |
|---|---|
| `plot_soft_x_ray_los` (`soft_x_rays_los`) | `(ods, channels=None, arrays=None, ax=None, show_wall=True, show_channel_labels=False, title=None, show=True)` |
| `plot_soft_x_ray_signal` (`soft_x_rays_signal`, `soft_x_rays_time`) | `(ods, channels=None, arrays=None, time_range=None, baseline_range=None, scale=1.0, ylabel='Brightness proxy [a.u.]', ax=None, title=None, show=True)` |
| `plot_soft_x_ray_spectrogram` (`soft_x_rays_spectrogram`) | `(ods, channel=0, time_range=None, baseline_range=None, nperseg=1024, noverlap=None, max_frequency=90000.0, log_power=True, ax=None, title=None, show=True)` |
| `plot_soft_x_ray_pattern` (`soft_x_rays_pattern`) | `(ods, channels=None, arrays=None, time_range=None, baseline_range=None, scale=1.0, orientation='time_vertical', cmap='turbo', ax=None, title=None, show=True)` |
| `plot_soft_x_ray_overview` (`soft_x_rays_overview`) | `(ods, los_arrays=None, signal_channels=None, spectrogram_channel=0, pattern_channels=None, pattern_arrays=None, time_range=None, baseline_range=None, show=True)` |

```python
from vaft.plot import (
    plot_soft_x_ray_los,
    plot_soft_x_ray_signal,
    plot_soft_x_ray_spectrogram,
    plot_soft_x_ray_pattern,
    plot_soft_x_ray_overview,
)

time_range = (0.304, 0.328)
baseline_range = (0.304, 0.306)

plot_soft_x_ray_los(ods, arrays=["lowermid", "bottom"], show_channel_labels=True)
plot_soft_x_ray_signal(ods, channels=[0, 16], time_range=time_range,
                       baseline_range=baseline_range)
plot_soft_x_ray_spectrogram(ods, channel=16, time_range=time_range,
                            nperseg=512, noverlap=384, max_frequency=90_000)
plot_soft_x_ray_pattern(ods, channels=[16, 17, 18], time_range=time_range,
                        orientation="time_vertical")
plot_soft_x_ray_overview(ods, los_arrays=["lowermid", "bottom"],
                         signal_channels=[0, 16], spectrogram_channel=16,
                         time_range=time_range, baseline_range=baseline_range)
```

`arrays` accepts `'horizontal'`, `'vertical'`, `'lowermid'`, `'bottom'` and `'digitizer'`.
`baseline_range` selects the window subtracted as a baseline before plotting.

Unlike the time-trace keywords, `orientation` **is** validated: anything other than
`'time_vertical'` or `'time_horizontal'` raises `ValueError`.

## Kinetic profiles

Thomson scattering and charge exchange, all single-ODS:

| Function (+ alias) | Signature |
|---|---|
| `plot_thomson_radial_position` (`thomson_scattering_radial`) | `(ods, contour_quantity='psi_norm')` |
| `plot_thomson_time_series` (`thomson_scattering_time`) | `(ods)` |
| `plot_thomson_profiles` (`thomson_scattering_radial_profiles`) | `(ods, save_opt=0, file_name=None)` |
| `plot_electron_profile_with_thomson` | `(ods)` |
| `plot_TeNe_from_eq` | `(ods, save_opt=0, file_name=None, only_synthetic=True, synthetic_tag=None, plot_pressure=False, eq_time_index=0)` |
| `plot_electron_psi_profile` | `(ods, time_slice=None, figsize=(10, 6))` |
| `plot_electron_2d_profile` | `(ods, time_slice=None, figsize=(20, 8))` |
| `plot_electron_time_volume_averaged` | `(ods, figsize=(12, 6))` |
| `plot_equilibrium_and_core_profiles_pressure` | `(ods, figsize=(12, 6))` |
| `charge_exchange_radial` (`plot_ces_profile`) | `(ods, ion_index=0)` |
| `charge_exchange_time` | `(ods, ion_index=0)` |
| `charge_exchange_rho_profiles` | see the warning below |

```python
vaft.plot.plot_thomson_radial_position(ods)
vaft.plot.plot_thomson_time_series(ods)
vaft.plot.plot_thomson_profiles(ods, save_opt=1, file_name="thomson.png")
vaft.plot.plot_electron_2d_profile(ods, time_slice=2)
vaft.plot.plot_equilibrium_and_core_profiles_pressure(ods)
```

> **`charge_exchange_rho_profiles` writes to your ODS.** It is not a pure plotter: it fits
> the profiles and stores them back into `core_profiles.profiles_1d`, popping a colliding
> slice if one exists. This is by design (the results are meant to be reused), but call it on
> a copy if you need the input ODS left untouched.

`plot_pressure_profile_with_geqdsk(shot, time_ms, OMFITgeq, n_e_function, T_e_function,
geqdsk, save_opt=1)` also exists, but it needs OMFIT objects rather than an ODS alone.

## History and operational space

The `history` functions work on a **pandas DataFrame** of shot-level scalars, not on an ODS.
Build the DataFrame with the confinement pipeline (see
[Pipelines]({{ site.baseurl }}/guide/Pipelines/)), then:

```python
fig  = vaft.plot.plot_tauE_exp_vs_scaling_loglog(df, tauE_exp_col='tauE_s', figsize=(12, 8))
fig1 = vaft.plot.plot_scaling_fit(results, df, target_param='tauE_s')
fig2 = vaft.plot.plot_individual_parameter_effects(df, eng_params, target_param='tauE_s')
fig3 = vaft.plot.plot_correlation_heatmap(log_df, eng_params, target_param='tauE_s')
fig4 = vaft.plot.plot_regression_summary(results)

metrics_df = vaft.plot.compute_confinement_scaling_metrics(df, tauE_exp_col='tauE_s')
vaft.plot.plot_scaling_metrics_bars(metrics_df, figsize=(12, 4))

vaft.plot.plot_confinement_time_exp_vs_scaling(df, scaling_raw='IPB89', figsize=(10, 6))
vaft.plot.plot_confinement_time_exp_vs_scaling(df, scaling_raw='H98y2')
```

| Function | First argument |
|---|---|
| `plot_scaling_fit(results, df, target_param='tauE_s', figsize=(12, 5))` | fit results + DataFrame |
| `plot_individual_parameter_effects(df, eng_params, target_param='tauE_s', figsize=None, ncols=None, results=None)` | DataFrame |
| `plot_correlation_heatmap(log_df, eng_params, target_param='tauE_s', figsize=(10, 8))` | DataFrame |
| `plot_regression_summary(results, figsize=(10, 6))` | fit results |
| `plot_H_factor_vs_greenwald_fraction(df, tauE_exp_col='tauE_s', tauE_scaling_col=None, figsize=(8, 6))` | DataFrame |
| `plot_H_factor_distribution(df, tauE_exp_col='tauE_s', scaling_cols=None, figsize=(12, 5), bins=20)` | DataFrame |
| `plot_H_factor_vs_parameters(df, tauE_exp_col='tauE_s', scaling_cols=None, parameter_aliases=None, figsize=(14, 8))` | DataFrame |
| `plot_tauE_exp_vs_scaling_loglog(df, tauE_exp_col='tauE_s', scaling_cols=None, figsize=(12, 8))` | DataFrame |
| `plot_scaling_metrics_bars(metrics_df, metric_names=None, figsize=(12, 6), ...)` | metrics DataFrame |
| `plot_confinement_time_exp_vs_scaling(df, scaling_raw=None, figsize=(10, 6))` | DataFrame |
| `compute_confinement_scaling_metrics(df, tauE_exp_col='tauE_s', scaling_cols=None)` | DataFrame — **returns a DataFrame, not a figure** |

`plot_confinement_time_exp_vs_scaling` accepts `scaling_raw` as `'IPB'`/`'IPB89'`/`'ITER89P'`,
`'H98'`/`'H98y2'`, `'NSTX2006H'`, `'NSTX2006L'`, `'Kurskiev2022'`, a list of these, or `None`
for all four defaults.

> Careful with the name: `plot_confinement_time_exp_vs_scaling(df, ...)` — with the `plot_`
> prefix and a DataFrame — is the working function. The similarly named
> `confinement_time_exp_vs_scaling(ods_or_odc, ...)` is an empty stub that draws nothing.

Two history functions take an ODS/ODC instead of a DataFrame:

```python
vaft.plot.plot_bremsstrahlung_power_scaling_vs_fundamental_method(ods, Z_eff=2.0)
vaft.plot.plot_ohmic_power_flux_vs_dissipation_method(ods)
```

## Utilities

```python
vaft.plot.get_from_path(ods, 'equilibrium.time')          # None if the path is missing
vaft.plot.extract_labels_from_odc(odc, opt='shot')        # 'shot'/'pulse'/'run'/'key'
```

`time_equilibrium_analysis(ods, xunit='s', xlim='plasma')` builds a 3×2 equilibrium overview.
It is implemented but is not exercised by any test or notebook — check its output before
relying on it.

## Names that are exposed but do not work

`vaft/plot/__init__.py` star-imports every submodule, so the `vaft.plot` namespace contains
roughly 310 public names. Most of those are leaked imports (`np`, `plt`, `ODS`, ...) and
internal helpers. A few look like plotting entry points but produce nothing — avoid them:

| Name | What actually happens |
|---|---|
| `plot_onedim_profile` | calls an undefined helper; the resulting `NameError` is swallowed and an **empty figure** is drawn |
| `plot_equilibrium_pressure`, `plot_equilibrium_q`, `plot_core_profiles_ne`, `plot_core_profiles_te` | thin wrappers over `plot_onedim_profile` — same empty figures |
| `analysis_diagnostics` | raises `NameError` on undefined module globals |
| `analysis_electromagnetics` | empty body; returns `None`, draws nothing |
| `confinement_time_exp_vs_scaling` | empty body; use `plot_confinement_time_exp_vs_scaling(df, ...)` instead |

There is no top-view plotting: `vaft/plot/topview.py` contains only comments and defines no
functions.

For 1D equilibrium profiles use the generated `equilibrium_<coordinate>_<quantity>` names
described above — not the `plot_*` ones in this table.

## Notes

Return values are not uniform. `time.py` and most of `profile.py` call `plt.show()` and
return `None`; `mirnov.py`, `soft_x_rays.py` and `history.py` return figures. Only the Mirnov
and soft X-ray functions take `ax=` and `show=`, so only those can be embedded in a figure
you lay out yourself.

Because the data is plain OMAS, any recipe from the
[OMAS example gallery](https://gafusion.github.io/omas/auto_examples/index.html) also applies
to a VAFT ODS.

Source: [`vaft/plot`](https://github.com/VEST-Tokamak/vaft/blob/main/vaft/plot).
More worked examples: [Examples]({{ site.baseurl }}/guide/examples/).
