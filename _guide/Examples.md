---
title: Examples
author: VEST team
date: 2026-07-01 11:10
category: guide
layout: post
permalink: /guide/examples/
---

# Notebook Examples

The VAFT repository ships **27 example notebooks** under [`notebooks/`](https://github.com/VEST-Tokamak/vaft/tree/main/notebooks). This page indexes all of them, groups them by theme, and shows verified code for the ones you are most likely to start from.

![VAFT overview]({{ site.baseurl }}/assets/images/IMG_3873.jpg)

## How to read this index

The notebooks are at two different levels of maturity, and it saves a lot of time to know which is which before you open one:

- **Runnable** — the notebook contains executable cells and a working data path. 16 notebooks.
- **Design shell** — the notebook is currently a structured markdown outline (objectives, expected inputs and outputs, planned sections) with **no code cells yet**. 11 notebooks. They are useful as specifications of where a pipeline stage is heading, but they will not execute anything.

Design shells are marked *(design shell)* below.

## Getting started and the database

| Notebook | Purpose |
| --- | --- |
| [`database_initialization_and_load.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/database_initialization_and_load.ipynb) | Install VAFT, verify the HSDS connection, list public shots, and load a shot into an ODS. |
| [`vest_experimental_data_list.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/vest_experimental_data_list.ipynb) | Tour of IMAS/OMAS concepts and of which VEST diagnostics are mapped into the database today. |
| [`vest_raw_signal_sql_database.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/vest_raw_signal_sql_database.ipynb) | Reach the legacy SQL DAQ store: field codes, slow vs fast channels, and raw waveform retrieval. |

Start here. This is the shortest path from a fresh install to a shot in memory.

```python
import vaft

# Is the HSDS backend reachable?
print(vaft.database.is_connect())

# Which shots are published?
shots = vaft.database.exist_shot("public")

# Load one shot as an OMAS ODS
ods = vaft.database.load(39915, directory="public")
```

`vaft.database.load` is the canonical entry point: it returns an OMAS ODS, and it also accepts an explicit `ids_name=` keyword when you want a native IMAS IDS instead. See [Database]({{ site.baseurl }}/guide/Database/) for the full surface.

If you have no network access, every example below can also be driven from the packaged sample data described in the next section.

## Data structures and IMAS

| Notebook | Purpose |
| --- | --- |
| [`read_and_convert_data_structure.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/read_and_convert_data_structure.ipynb) | Walk an equilibrium ODS key by key — `equilibrium.time_slice`, `profiles_1d`, and the rest of the hierarchy. |
| [`imas_omas_data_conversion.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/imas_omas_data_conversion.ipynb) | Bridge OMAS ODS objects and native IMAS AL5 HDF5 storage, in both directions. |

The packaged samples are the fastest way to get a realistic ODS without touching the network:

```python
import vaft

ods = vaft.omas.sample_ods()   # one shot
odc = vaft.omas.sample_odc()   # a collection of three shots

list(ods.keys())
print(ods["equilibrium.time"])
print(list(ods["equilibrium.time_slice.0"].keys()))
```

To reach a packaged file by name rather than through the sample helpers, use the resource accessor — it resolves paths inside the installed package:

```python
from vaft.data.resources import data_path

sample_path = data_path("omas/39915.json")
```

For the IMAS round trip:

```python
from vaft.imas import save_omas_imas, load_omas_imas

save_omas_imas(ods, user="test_user", machine="VEST", pulse=39915, run=0)
ods_back = load_omas_imas(user="test_user", machine="VEST", pulse=39915, run=0)
```

Related reading: [Data structures]({{ site.baseurl }}/guide/Data_structures/) and [Machine mapping]({{ site.baseurl }}/guide/Machine_mapping/).

## Diagnostics

| Notebook | Purpose |
| --- | --- |
| [`magnetic_diagnostics_processing.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/magnetic_diagnostics_processing.ipynb) | *(design shell)* Planned preprocessing, filtering, calibration, and sign conventions for `ip`, flux loops, and poloidal probes. |
| [`soft_x_ray_signal_analysis.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/soft_x_ray_signal_analysis.ipynb) | Map digitizer CSVs into a soft X-ray ODS, then plot lines of sight, signals, and spectrograms. |
| [`fluctuation_diagnostics_analysis.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/fluctuation_diagnostics_analysis.ipynb) | Mirnov coil fluctuation work: spectrograms, toroidal mode spectra, and mode-number fits. |
| [`fast_camera_video_analysis.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/fast_camera_video_analysis.ipynb) | *(design shell)* Planned camera image and video loading, plasma-behavior observation, and synchronization. |

![Probe example]({{ site.baseurl }}/assets/images/magnetics/Inboard_B_z.png)

The soft X-ray notebook is a good template for "raw file to ODS to plot" in one pass:

```python
from vaft.machine_mapping.soft_x_rays import soft_x_rays_from_digitizer_csv
from vaft.plot import plot_soft_x_ray_los, plot_soft_x_ray_signal, plot_soft_x_ray_spectrogram

ods = soft_x_rays_from_digitizer_csv(shot, daq_label, digitizer_file=plasma_file)

plot_soft_x_ray_los(ods, arrays=["lowermid", "bottom"])
plot_soft_x_ray_signal(ods)
plot_soft_x_ray_spectrogram(ods)
```

See [Magnetics]({{ site.baseurl }}/guide/Magnetics/) and [Processing]({{ site.baseurl }}/guide/Processing/).

## Equilibrium and stability

| Notebook | Purpose |
| --- | --- |
| [`equilibrium_refinement_using_chease.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/equilibrium_refinement_using_chease.ipynb) | Refine a GEQDSK with CHEASE: build `EXPEQ` and the namelist, resolve the binary, run, collect outputs. |
| [`forward_equilibrium_using_TES.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/forward_equilibrium_using_TES.ipynb) | Forward (Grad-Shafranov) equilibrium solve with TES, driven straight from an ODS. |
| [`electromagnetic_response_modeling_with_efund.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/electromagnetic_response_modeling_with_efund.ipynb) | *(design shell)* Planned EFUND response modeling over wall geometry, PF active and PF passive structures. |
| [`eddy_current_calculation_and_startup_analysis.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/eddy_current_calculation_and_startup_analysis.ipynb) | *(design shell)* Planned PF-passive eddy-current ODE solve and tokamak startup / null analysis. |
| [`magnetic_equilibrium_reconstruction_with_efit.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/magnetic_equilibrium_reconstruction_with_efit.ipynb) | *(design shell)* Planned EFIT reconstruction from magnetics, eddy currents, and PF coil information. |
| [`mhd_equilibrium_analysis.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/mhd_equilibrium_analysis.ipynb) | *(design shell)* Planned equilibrium loading, representative quantities, and coordinate transformations. |
| [`linear_ideal_stability_analysis_with_dcon.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/linear_ideal_stability_analysis_with_dcon.ipynb) | *(design shell)* Planned ideal MHD stability (delta-W) with DCON from the GPEC package. |
| [`linear_resistive_stability_analysis_with_rdcon.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/linear_resistive_stability_analysis_with_rdcon.ipynb) | *(design shell)* Planned resistive stability (Delta-prime) with RDCON. |
| [`perturbed_equilibrium_and_3d_response_with_gpec.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/perturbed_equilibrium_and_3d_response_with_gpec.ipynb) | *(design shell)* Planned perturbed equilibrium and non-axisymmetric 3D response with GPEC. |

CHEASE is the most complete code-coupling example in the repository. The `prepare_*` / `run_*` / `collect_*` triple is the pattern every code wrapper in `vaft.code` follows:

```python
from vaft.data import read_geqdsk
from vaft.data.resources import data_path
from vaft.code.chease import CHEASEConfig, prepare_chease_inputs, find_chease_executable

initial = read_geqdsk(data_path("efit/g039915.00319"))
ods = initial.to_omas()

config = CHEASEConfig(workdir=workdir)
inputs = prepare_chease_inputs(initial, config)   # writes EXPEQ + chease_namelist

executable = find_chease_executable(config)       # None if CHEASE is not installed
```

TES shows the same idea starting from a database shot rather than a file:

```python
import vaft
from vaft.code import tes

ods = vaft.database.load(39915)

cfg = tes.TESConfig(
    executable=RTES,
    workdir=WORKDIR,
    shot=39915,
    time=0.325,
    bt0=0.15,     # fix the toroidal field; omit to read it from the tf IDS
    eddy=True,    # treat pf_passive as eddy coils
)
inputs = tes.prepare_tes_inputs(ods, cfg)
```

Both binaries are optional: the notebooks degrade to input generation when the executable is absent. See [Equilibrium]({{ site.baseurl }}/guide/Equilibrium/) and [Stability]({{ site.baseurl }}/guide/Stability/).

## Profiles and transport

| Notebook | Purpose |
| --- | --- |
| [`profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb) | Map Thomson scattering onto equilibrium flux surfaces to fit core `Te` and `ne` profiles. |
| [`confinement_time_scaling.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/confinement_time_scaling.ipynb) | Single-shot workflow test, then a dataset-wide confinement-time scaling and regression study. |
| [`tokamak_power_balance.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/tokamak_power_balance.ipynb) | Ohmic input against radiated and conducted losses, with an Aurora-based impurity treatment. |

The profile notebook is the largest runnable example that stays inside pure VAFT:

```python
import vaft

ods = vaft.database.load(40330, directory="public")

vaft.plot.plot_thomson_radial_position(ods)
vaft.plot.plot_thomson_time_series(ods)
vaft.plot.plot_electron_profile_with_thomson(ods)
```

See [Profiles]({{ site.baseurl }}/guide/Profiles/) and [Formula]({{ site.baseurl }}/guide/Formula/).

## Analysis, V&V, and publication

| Notebook | Purpose |
| --- | --- |
| [`plotting_sample_using_vaft_plot_module.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/plotting_sample_using_vaft_plot_module.ipynb) | The plot module tour: naming conventions, ODS vs ODC input, and time-convention shifts. |
| [`publication_figures.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/publication_figures.ipynb) | Reproduce publication-quality composite figures at print DPI. |
| [`verification_and_validation.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/verification_and_validation.ipynb) | Cross-check volume-averaged parameters across shots and export a V&V spreadsheet. |
| [`multiple_tokamak_comparison.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/multiple_tokamak_comparison.ipynb) | *(design shell)* Planned cross-device comparison of geometry, equilibrium, and diagnostic signals. |

The plot module names functions as `{ids}_{coordinate}_{quantity}` — for example `time_magnetics_ip`. Any plot function accepts a single ODS, an ODC, or a list of ODS objects, which is what makes shot overlays trivial:

```python
import vaft

ods = vaft.omas.sample_ods()
odc = vaft.omas.sample_odc()

vaft.plot.time_magnetics_ip(ods)
vaft.plot.time_magnetics_ip(odc)          # overlays every shot in the collection

# Re-zero the time axis on breakdown, then replot
vaft.omas.change_time_convention(ods, convention="breakdown")
vaft.plot.time_magnetics_ip(ods)
```

![Magnetics example]({{ site.baseurl }}/assets/images/magnetics/plasma_current.png)

Composite figures are single calls:

```python
import vaft

ods = vaft.omas.sample_ods()
vaft.plot.overlay_all_with_vacuum_psi_contour(ods)
```

See [Plotting]({{ site.baseurl }}/guide/Plotting/) for the full catalogue.

## Operations and monitoring

| Notebook | Purpose |
| --- | --- |
| [`vest_daily_monitoring.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/vest_daily_monitoring.ipynb) | Day-to-day shot health check: load the day's shots into an ODC and compare key signals. |
| [`shot_characteristics_classification.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/shot_characteristics_classification.ipynb) | *(design shell)* Planned representative-signal extraction, shot classification, and summary sheets. |

These pair with the automated Snakemake stages described in [Pipelines]({{ site.baseurl }}/guide/Pipelines/).

## Known issues in the notebooks on `main`

Some notebooks on `main` have not caught up with two API changes. If you hit one of these, the fix is mechanical — and the working form is what this page uses throughout.

**1. `vaft.database.exist_ts_file()` does not exist.** It is called by `tokamak_power_balance.ipynb` and `verification_and_validation.ipynb` to discover processed shots. There is no replacement helper in the package; supply the shot list directly, as `profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb` now does:

```python
import pandas as pd

core_profile_shots = [40330]
df = pd.DataFrame({"Shot Number": core_profile_shots, "Status": ["core_profile"]})
```

**2. `vaft.omas.load_omas_json()` does not exist, and the packaged data moved.** Sample JSON files are no longer flat under `vaft/data/`; they live under `vaft/data/omas/`. Notebooks that still build a path like `os.path.join(os.path.dirname(vaft.__file__), "data", "39915.json")` — including `vest_daily_monitoring.ipynb` and parts of `read_and_convert_data_structure.ipynb` — will fail. Use the sample helpers or the resource accessor instead:

```python
import vaft
from vaft.data.resources import data_path

ods = vaft.omas.sample_ods()            # preferred
sample_path = data_path("omas/39915.json")   # or resolve the file explicitly
```

Note that `load_omas_json` *does* exist in the upstream `omas` package (`from omas import load_omas_json`); only the `vaft.omas` alias was removed.

## Recommended order

A newcomer should work through the runnable notebooks in this order:

1. [`database_initialization_and_load.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/database_initialization_and_load.ipynb) — install, connect, load your first shot.
2. [`read_and_convert_data_structure.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/read_and_convert_data_structure.ipynb) — learn the ODS hierarchy before you plot anything.
3. [`plotting_sample_using_vaft_plot_module.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/plotting_sample_using_vaft_plot_module.ipynb) — the plot naming convention pays for itself immediately.
4. [`vest_experimental_data_list.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/vest_experimental_data_list.ipynb) — find out which diagnostics are actually mapped.
5. [`profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb) — the first real physics workflow.
6. [`equilibrium_refinement_using_chease.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/equilibrium_refinement_using_chease.ipynb) — how VAFT wraps an external code.
7. [`confinement_time_scaling.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/confinement_time_scaling.ipynb) — scale up from one shot to the whole dataset.

Then branch by interest: [`imas_omas_data_conversion.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/imas_omas_data_conversion.ipynb) for interoperability, [`soft_x_ray_signal_analysis.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/soft_x_ray_signal_analysis.ipynb) and [`fluctuation_diagnostics_analysis.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/fluctuation_diagnostics_analysis.ipynb) for diagnostics, [`forward_equilibrium_using_TES.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/forward_equilibrium_using_TES.ipynb) for forward modeling, and [`publication_figures.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/publication_figures.ipynb) when it is time to write the paper.

If you have not installed VAFT yet, start at [Installation]({{ site.baseurl }}/guide/Installation/) and the [Quick start guide]({{ site.baseurl }}/guide/Quick_start_guide/). The complete function list is in the [API reference]({{ site.baseurl }}/guide/API_reference/).
