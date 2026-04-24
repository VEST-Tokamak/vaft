---
title: Examples
author: VEST team
date: 2026-04-24 16:20
category: guide
layout: post
permalink: /guide/examples/
---

# Notebook Examples

This page summarizes the example notebooks in the VAFT repository and highlights representative code paths that are already used in practice. The notebooks themselves live in [`notebooks/`](https://github.com/VEST-Tokamak/vaft/tree/main/notebooks).

![VAFT overview]({{ site.baseurl }}/assets/images/IMG_3873.jpg)

## Database initialization and load

Notebook: [`database_initialization_and_load.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/database_initialization_and_load.ipynb)

Use this notebook to understand the VEST database structure, verify the HSDS connection, list public shots, and load data into ODS or IDS objects.

```python
import vaft

connected = vaft.database.is_connect()
print(connected)

shots = vaft.database.exist_shot("public")
ods = vaft.database.load_ods(39915, directory="public")
```

## Plotting sample data with the VAFT plot module

Notebook: [`plotting_sample_using_vaft_plot_module.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/plotting_sample_using_vaft_plot_module.ipynb)

This notebook demonstrates the current VAFT plot API with both sample ODS data and time-convention changes.

```python
import vaft

ods = vaft.omas.sample_ods()
odc = vaft.omas.sample_odc()

vaft.plot.magnetics_time_ip(ods)
vaft.omas.change_time_convention(ods, convention="breakdown")
vaft.plot.magnetics_time_ip(ods)
```

![Magnetics example]({{ site.baseurl }}/assets/images/magnetics/plasma_current.png)

## Profile fitting using equilibrium and kinetic diagnostics

Notebook: [`profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb)

This notebook walks through Thomson scattering data loading, inspection of radial positions, and time-series plotting for equilibrium-linked profile analysis.

```python
import vaft

ods = vaft.database.load_ods(40330, directory="public")
vaft.plot.plot_thomson_radial_position(ods)
vaft.plot.plot_thomson_time_series(ods)
```

## IMAS-OMAS data conversion

Notebook: [`imas_omas_data_conversion.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/imas_omas_data_conversion.ipynb)

Use this notebook when you need to bridge ODS data and native IMAS AL5 HDF5 storage.

```python
import imas
from vaft.imas import save_omas_imas, load_omas_imas

factory = imas.IDSFactory()
equilibrium = factory.equilibrium()
equilibrium.time = [0.01]

# save_omas_imas / load_omas_imas examples are expanded in the notebook
```

## Confinement time scaling analysis

Notebook: [`confinement_time_scaling.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/confinement_time_scaling.ipynb)

This notebook demonstrates single-shot workflow testing and full-dataset statistical analysis for confinement scaling studies.

```python
import vaft

ods = vaft.database.load_ods(39915, directory="public")
# Continue with parameter extraction and scaling analysis as shown in the notebook
```

## Data structure walkthrough

Notebook: [`read_and_convert_data_structure.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/read_and_convert_data_structure.ipynb)

This notebook focuses on equilibrium data structure walkthroughs, especially EFIT / CHEASE-oriented ODS inspection and conversion paths.

## Experimental database browsing

Notebook: [`vest_experimental_data_list.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/vest_experimental_data_list.ipynb)

This notebook gives a broader introduction to IMAS and OMAS concepts and describes what VEST diagnostics are already mapped into the database.

![Probe example]({{ site.baseurl }}/assets/images/magnetics/Inboard_B_z.png)

## Additional notebooks

| Notebook | Focus |
| --- | --- |
| [`vest_daily_monitoring.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/vest_daily_monitoring.ipynb) | Daily monitoring and dashboard-style views |
| [`publication_figures.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/publication_figures.ipynb) | Reproducing publication-oriented figures |
| [`read_and_convert_data_structure.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/read_and_convert_data_structure.ipynb) | Equilibrium data structure walkthrough |
| [`database_initialization_and_load.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/database_initialization_and_load.ipynb) | Database connection and first-load workflow |

## Recommended order

1. Start with `database_initialization_and_load.ipynb`.
2. Move to `plotting_sample_using_vaft_plot_module.ipynb`.
3. Use `profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb` and `imas_omas_data_conversion.ipynb` for deeper workflows.
