---
title: Equilibrium and kinetic profiles
author: VEST team
date: 2026-07-01 10:20
category: guide
layout: post
mermaid: true
permalink: /workflows/equilibrium-kinetic-profiles/
guide:
  architecture: Joins reconstructed equilibrium geometry to fitted kinetic diagnostics.
  prerequisites: GEQDSK equilibrium plus Thomson and charge-exchange measurements.
  expected: A core_profiles IDS and deterministic equilibrium-code input bundle.
related:
  notebooks: [kinetic-efit, chease-refinement]
  api: [process, code, plot]
  data_sources: [kinetic-48224, external-codes]
  outputs: [kinetic-profile, equilibrium-inputs]
---

Kinetic diagnostics on VEST — Thomson scattering (TS) and charge exchange spectroscopy (CES) — measure
$T_e$, $n_e$, $T_i$ and $V_{tor}$ at discrete $(R, Z)$ points. To become a `core_profiles` IDS these
point measurements must first be mapped onto a flux coordinate using an equilibrium, then fitted to a
smooth 1D profile in $\rho$.

`vaft.process` implements that two-stage pipeline, and `vaft.formula.fit_profile` provides the
underlying 1D fitting engine.

## Pipeline

```mermaid
flowchart LR
    A["TS / CES .mat"] -->|machine_mapping| B["ods: thomson_scattering<br/>charge_exchange"]
    E["GEQDSK / equilibrium"] --> C
    B --> C["equilibrium_mapping_*<br/>(R,Z) → ρ"]
    C --> D["profile_fitting_*<br/>ρ → Te, ne, Ti, Vtor"]
    D --> F["core_profiles<br/>profiles_1d"]
```

The two stages are deliberately separate: mapping depends on the equilibrium, fitting does not. You can
re-fit with a different model without recomputing the mapping.

## Loading kinetic diagnostics

Raw TS/CES `.mat` files are turned into IDS nodes by the machine mapping layer:

```python
import vaft
from omas import ODS

ods = ODS()

# thomson_scattering(ods, shotnumber, data_root=None, mat_file=None)
vaft.machine_mapping.thomson_scattering(ods, 46051, vaft.data.data_path("legacy/46051_NeTe.mat"))

# charge_exchange(ods, shotnumber, options="ces", data_root=None, mat_file=None)
vaft.machine_mapping.charge_exchange(ods, 47514, data_root=vaft.data.data_path("legacy/CES_47514.mat"))
```

`data_root` accepts either a directory or a path to a specific `*.mat` file. This populates
`thomson_scattering.channel[:].position.{r,z}`, `.t_e.data`, `.n_e.data` (plus `_error_upper`
siblings), and for CES `charge_exchange.channel[:].ion[:].{t_i,velocity_tor}.data`.

Sample files ship inside the package — see [Data structures]({{ site.baseurl }}/guide/Data_structures/)
for the ODS/IDS conventions, and use `vaft.data.data_path()` to resolve them:

| Sample | Packaged path |
|---|---|
| Thomson (shot 46051) | `legacy/46051_NeTe.mat` |
| Thomson (shot 39915) | `legacy/NeTe_Shot39915_v9_rev.mat` |
| CES (shot 47514) | `legacy/CES_47514.mat` |
| GEQDSK (shot 40330) | `efit/g040330.00320` |

## Stage 1 — equilibrium mapping

Each diagnostic channel sits at a fixed $(R, Z)$. The mapping functions interpolate $\psi(R,Z)$ from the
equilibrium and normalize it, returning one $\rho \in [0, 1]$ per channel:

$$\rho = \frac{\psi(R,Z) - \psi_{axis}}{\psi_{boundary} - \psi_{axis}}$$

```python
geq = vaft.data.read_geqdsk(vaft.data.data_path("efit/g040330.00320"))

mapped_rho = vaft.process.equilibrium_mapping_thomson_scattering(ods, geq)
mapped_rho_ces = vaft.process.equilibrium_mapping_charge_exchange(ods, geq)
```

Both accept a `vaft.data.GEQDSK`, an OMAS equilibrium ODS, or a legacy flux-surface mapping as `geq`.
Values are clipped to $[0, 1]$, so channels outside the last closed flux surface pile up at $\rho = 1$.
Building an equilibrium is covered in [Equilibrium]({{ site.baseurl }}/guide/Equilibrium/).

## Stage 2 — profile fitting

```python
n_e_fn, T_e_fn, coeffs_ne, coeffs_te, n_e_rho, T_e_rho = \
    vaft.process.profile_fitting_thomson_scattering(
        ods,
        time_ms=320.0,
        mapped_rho_position=mapped_rho,
        Te_order=3,
        Ne_order=3,
        uncertainty_option=1,       # weight the fit by per-channel error bars
        rho_points=100,
        fitting_function_te="polynomial",
        fitting_function_ne="polynomial",
    )
```

Note the return order: **density first, then temperature**. `n_e_fn` and `T_e_fn` are callables you can
evaluate on any $\rho \in [0,1]$; `n_e_rho` and `T_e_rho` are those functions already sampled on a
uniform grid of `rho_points`. `coeffs_*` is `None` for the `gp` and `linear` methods.

The CES counterpart mirrors it, with `ion_index` selecting the ion species:

```python
Vtor_fn, Ti_fn, coeffs_vtor, coeffs_ti, Vtor_rho, Ti_rho = \
    vaft.process.profile_fitting_charge_exchange(
        ods,
        time_ms=300.0,
        mapped_rho_position=mapped_rho_ces,
        Ti_order=3,
        Vtor_order=3,
        fitting_function_ti="polynomial",
        fitting_function_vtor="polynomial",
        ion_index=0,
    )
```

Here too the velocity function comes back before the temperature function.

### Fitting methods

`fitting_function_*` selects the model, and every variant is dispatched to `vaft.formula.fit_profile`:

| Value | Model |
|---|---|
| `polynomial` | $(1-\rho)\cdot P_n(\rho)$ — polynomial with an edge roll-off factor |
| `exponential` | $(1-\rho)\cdot \exp(P_n(\rho))$ — enforces positivity |
| `sqrt` / `sqrt_poly` | fits $P_n$ to $y^2$, returns $\sqrt{P_n}$ |
| `sqrt_exp` | same square-space trick with an exponential basis |
| `core_poly_edge_exp` | core polynomial blended into an edge exponential via a $\tanh$ transition |
| `gp` | Gaussian process regression (scikit-learn), with optional anchor points |
| `linear` | 1D interpolation through the data — no smoothing |

The `(1-\rho)` factor in the polynomial and exponential bases drives the fitted profile toward zero at
the boundary, which is usually what you want for $T_e$ and $n_e$ but is a real assumption — `gp` and
`core_poly_edge_exp` do not impose it.

`order` only affects the polynomial-family methods; it is ignored by `gp` and `linear`.

### Calling the fitting engine directly

For data that is not in an IDS, use the engine itself:

```python
y_eval, y_std_eval, fit_function, coeffs = vaft.formula.fit_profile(
    x, y, y_std,
    x_eval,
    order=3,
    uncertainty_option=1,
    fitting_function="gp",
    gp_anchor=(x_anchor, y_anchor, y_std_anchor),   # GP only
    n_restarts_optimizer=5,
)
```

It returns the profile on `x_eval`, its standard deviation (zeros for methods that carry no uncertainty
estimate), a callable, and the coefficients. `vaft.formula.make_fit_function(mode)` builds the bare
`polynomial` / `exponential` basis function if you want to fit it yourself. See
[Physics formulas]({{ site.baseurl }}/guide/Formula/) for the rest of the formula namespace.

## Writing `core_profiles`

`vaft.process.core_profiles` evaluates the fit callables and stores the result as a `profiles_1d` slice:

```python
ods = vaft.process.core_profiles(
    ods,
    time_ms=320.0,
    mapped_rho_position=mapped_rho,
    n_e_function=n_e_fn,
    T_e_function=T_e_fn,
    tol_ms=0.1,
)
```

It writes, for the new slice index `i`:

- `core_profiles.profiles_1d[i].time` — **seconds** (`time_ms` is divided by 1000)
- `core_profiles.profiles_1d[i].grid.rho_tor_norm` — 100 uniform points on $[0,1]$
- `.electrons.temperature` (eV) and `.electrons.density` / `.density_thermal` (m⁻³)
- `.electrons.temperature_fit` / `.density_fit` — the *measured* channel values on their mapped $\rho$,
  kept alongside the fit so you can overplot data against model
- `.ion[0]` labelled `H+`, populated with the electron profiles

That last point is an explicit simplification: **`core_profiles` copies $n_e$ and $T_e$ into the ion
channel**, it does not use CES data. Treat `ion[0]` as a placeholder unless you overwrite it yourself.

If a slice already exists within `tol_ms` of `time_ms` it is replaced rather than duplicated, so
re-fitting the same time in a loop is safe.

### Synthetic profiles from equilibrium pressure

When there is no Thomson data, profiles can be back-derived from the equilibrium pressure by assuming
$n_e$ and $T_e$ share a shape, with $P = 2 n_e T_e e$ and $g(\rho) = \sqrt{P(\rho)/P(0)}$:

```python
# Pin the on-axis temperature (eV) and let density follow
vaft.process.core_profiles_from_eq(ods, Te0_eV=100.0, eq_time_index=0)

# Or pin the density/temperature ratio (m^-3 per eV)
vaft.process.core_profiles_from_eq_ratio(ods, C_ne_over_Te=1.0e17, eq_time_index=0)
```

Both read `equilibrium.time_slice[i].profiles_1d.pressure` and write a `core_profiles` slice at the
matching equilibrium time. The factor of 2 absorbs the assumption $T_i = T_e$; there is no $Z_{eff}$ or
impurity modelling. These are synthetic profiles — useful to seed a code that needs *some* kinetic
input, not a measurement.

## Plotting

`vaft.plot` has dedicated views for each stage:

```python
vaft.plot.plot_thomson_radial_position(ods, contour_quantity="psi_norm")  # channels on flux surfaces
vaft.plot.plot_thomson_time_series(ods)                                   # raw Te/ne vs time
vaft.plot.plot_thomson_profiles(ods, save_opt=0, file_name=None)          # fitted profiles
vaft.plot.plot_electron_profile_with_thomson(ods)                         # fit vs measurement
vaft.plot.plot_electron_psi_profile(ods)
vaft.plot.plot_electron_time_volume_averaged(ods)
vaft.plot.plot_equilibrium_and_core_profiles_pressure(ods)                # consistency check
```

`plot_equilibrium_and_core_profiles_pressure` is the useful sanity check after a fit: if the pressure
implied by the kinetic profiles disagrees badly with the equilibrium pressure, the fit or the mapping
is wrong.

For CES, `vaft.plot.charge_exchange_rho_profiles` runs the mapping and fit internally and plots the
result in one call:

```python
vaft.plot.charge_exchange_rho_profiles(
    ods, eq=geq, time_ms=300.0, ion_index=0,
    fitting_function_ti="polynomial", Ti_order=3,
)
vaft.plot.charge_exchange_time(ods, ion_index=0)
```

`vaft.plot.plot_TeNe_from_eq(ods, ...)` plots the synthetic profiles produced by the
`core_profiles_from_eq*` helpers.

## Exporting

To hand fitted electron profiles to an external code:

```python
vaft.process.export_electron_profile_txt(
    n_e_fn, T_e_fn, coeffs_ne, coeffs_te,
    rho_points=100,
    filename="electron_profiles.txt",
)
```

The file is CSV with the header `psi_N, T_e [eV], n_e [m-3]`.

## Full example

```python
import numpy as np
import vaft
from omas import ODS

shot = 40330

geq = vaft.data.read_geqdsk(vaft.data.data_path("efit/g040330.00320"))
ods = geq.to_omas()
vaft.machine_mapping.dataset_description(
    ods, source=shot, options={"source_type": "shot"},
)
vaft.machine_mapping.thomson_scattering(
    ods, 46051, vaft.data.data_path("legacy/46051_NeTe.mat"),
)

mapped_rho = vaft.process.equilibrium_mapping_thomson_scattering(ods, geq)

for t_s in np.asarray(ods["thomson_scattering.time"], dtype=float):
    time_ms = float(t_s) * 1e3
    n_e_fn, T_e_fn, *_ = vaft.process.profile_fitting_thomson_scattering(
        ods, time_ms, mapped_rho,
        Te_order=2, Ne_order=2,
        fitting_function_te="polynomial",
        fitting_function_ne="polynomial",
    )
    ods = vaft.process.core_profiles(ods, time_ms, mapped_rho, n_e_fn, T_e_fn)

vaft.plot.plot_thomson_profiles(ods)
```

Note that `thomson_scattering.time` is in **seconds** while the fitting and `core_profiles` functions
take `time_ms` in **milliseconds** — the conversion above is not optional.

## Choosing a fitting method

VEST Thomson systems have few channels, so the fit is under-constrained and the method choice matters
more than it would on a large device:

- Start with `polynomial` at `order=2`. Higher orders oscillate between sparse channels.
- Use `gp` when you want an uncertainty band rather than a point estimate — it is the only method that
  returns a meaningful `y_std_eval`.
- Use `core_poly_edge_exp` when the edge gradient is the quantity of interest, since the polynomial
  bases force the profile to zero at $\rho = 1$ regardless of the data.
- `linear` is a diagnostic aid, not a physics result: it interpolates the channels exactly and will
  happily reproduce measurement noise.

Always keep `uncertainty_option=1` if error bars exist. With it disabled every channel is weighted
equally, including ones the diagnostic itself flagged as unreliable.

## References

- Notebook: [profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb)
- Source: [vaft/process/profile.py](https://github.com/VEST-Tokamak/vaft/blob/main/vaft/process/profile.py)
- Source: [vaft/plot/profile.py](https://github.com/VEST-Tokamak/vaft/blob/main/vaft/plot/profile.py)
- Source: [vaft/formula/utils.py](https://github.com/VEST-Tokamak/vaft/blob/main/vaft/formula/utils.py)
- Batch pipeline: [workflow/automatic_pipeline_2_corrective_data_update/update_thomson_scattering_and_core_profile.py](https://github.com/VEST-Tokamak/vaft/blob/main/workflow/automatic_pipeline_2_corrective_data_update/update_thomson_scattering_and_core_profile.py)
- Related: [Equilibrium]({{ site.baseurl }}/guide/Equilibrium/) ·
  [Data structures]({{ site.baseurl }}/guide/Data_structures/) ·
  [Physics formulas]({{ site.baseurl }}/guide/Formula/) ·
  [Examples]({{ site.baseurl }}/guide/examples/)
