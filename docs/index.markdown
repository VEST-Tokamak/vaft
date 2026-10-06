---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults
layout: home
title: VAFT — Versatile Analysis Framework for Tokamak
---

<section class="vaft-hero">
  <img class="vaft-hero-logo" src="{{ site.baseurl }}/assets/brand/vaft-wordmark.svg" alt="VAFT">
  <p class="vaft-hero-tagline">A machine-agnostic framework that takes tokamak data from raw VEST signals to verifiable IMAS datasets and reproducible physics analysis.</p>
  <p class="vaft-hero-commands"><code>vaft.setup()</code> <code>vaft help</code></p>
</section>

> **Integrate fusion science knowledge so it can be discovered, verified, compared, and studied.**

## Get started

<div class="vaft-start">
  <a class="vaft-start-card" href="{{ site.baseurl }}/workflows/start-here/"><strong>Install and set up</strong><span>Install VAFT, then run <code>vaft.setup()</code> or <code>vaft help</code>.</span></a>
  <a class="vaft-start-card" href="{{ site.baseurl }}/reference/notebooks/"><strong>Tutorials</strong><span>Runnable notebooks with verified outputs.</span></a>
  <a class="vaft-start-card" href="{{ site.baseurl }}/workflows/data-access-imas/"><strong>Guides</strong><span>Research workflows from data access to stability.</span></a>
  <a class="vaft-start-card" href="{{ site.baseurl }}/reference/api/"><strong>Catalogs and API</strong><span>The API, formula and process catalogs.</span></a>
  <a class="vaft-start-card" href="{{ site.baseurl }}/workflows/gui/"><strong>GUI</strong><span>Browse and plot data with <code>vaft gui</code>.</span></a>
</div>

## From the machine to the analysis

<ol class="vaft-pipeline">
  <li><strong>VEST</strong>Diagnostics and machine data from the tokamak.</li>
  <li><strong>IMAS ODS</strong>Standardized, per-shot OMAS data structures.</li>
  <li><strong>HSDS / FileDB</strong>The shared database, remote or on disk.</li>
  <li><strong>Analysis codes</strong>EFIT, CHEASE, GPEC, TokaMaker and VAFT's own models.</li>
</ol>

**VAFT** is a standardized, verifiable, and interoperable scientific framework for machine-agnostic
tokamak research. It integrates experimental data, reconstructed and simulated plasma states, and
analysis workflows so that fusion science knowledge can be discovered, verified, compared, and studied.
Its full end-to-end implementation on the
[VEST (Versatile Experiment Spherical Torus)](https://eng.snu.ac.kr/) tokamak at Seoul National
University serves as the reference implementation for modern, reproducible, and data-driven fusion
research.

It provides an [IMAS](https://imas.iter.org/)-compliant data interface built on the
[OMAS](https://gafusion.github.io/omas/) interface library and an
[HSDS](https://github.com/HDFGroup/hsds) remote HDF5 database. Read access to the VEST database is
open to all users; writing to the database is restricted to authorized accounts.

> Hong-Sik Yun, Sunjae Lee *et al* 2025 *Plasma Phys. Control. Fusion* **67** 115021 —
> "Developing an IMAS-compatible platform for the university-scale tokamak VEST and its application
> to operating characteristics analysis"
> ([doi:10.1088/1361-6587/ae1b6a](https://doi.org/10.1088/1361-6587/ae1b6a))

## What VAFT is

Four things, which together are what "framework" means here.

- **Standardized Data Interface** — standardized data representations, data processing,
  validation, visualization and community physics codes (EFIT, CHEASE, GPEC, TokaMaker, VFIT)
  integrated into one scientific workflow, interoperating rather than reimplemented.
- **Traceable & Reproducible Pipeline** — traceable and reproducible data products from machine design
  to reconstructed and simulated states, which is what makes a result verifiable against its
  provenance and assumptions. Versioning covers geometry, diagnostic mappings, calibration,
  conventions, processing logic and schema versions, not only source code.
- **FAIR Scientific Data Repository** — validated data preserved and shared in both native and standardized
  representations, following the FAIR principles, as the foundation for finding which experimental
  and modelling information exists for a shot.
- **Machine Knowledge Archive** — machine configuration and history, experimental history, decisions,
  documentation, tutorials and notebooks of the VEST tokamak since 2012, kept usable for long-term verification,
  comparison and study.

## Choose a path

<div class="entry-paths">
  <a class="entry-path-card" data-entry-path="workflows" href="{{ site.baseurl }}/workflows/start-here/">
    <strong>Research workflows</strong>
    <span>Move from data access through diagnostics, equilibrium, profiles, stability, and reproducible analysis outputs.</span>
    <span>Start here with a working result →</span>
  </a>
  <a class="entry-path-card" data-entry-path="reference" href="{{ site.baseurl }}/reference/api/">
    <strong>Library and data reference</strong>
    <span>Find the VAFT API, IMAS concepts, database sources, notebooks, citations, and support contacts.</span>
    <span>Open the VAFT API reference →</span>
  </a>
</div>

## Key features

| Capability | Description |
| --- | --- |
| **Remote database access** | Load per-shot OMAS ODS data from the VEST HSDS server with a single function call |
| **Machine mapping** | Convert native VEST diagnostic signals into standardized IMAS IDSs (magnetics, Thomson scattering, barometry, PF active, TF, spectrometer UV, charge exchange, etc.) |
| **Equilibrium & stability** | Interfaces for EFIT, CHEASE, GPEC (DCON/RDCON) — read/write code I/O in IDS format |
| **Physics formulas** | Equilibrium quantities (poloidal/toroidal flux, safety factor), stability metrics (beta limits, ballooning), confinement scaling laws (ITER89P, H98y2), Green's functions |
| **Signal processing** | Smoothing, baseline subtraction, noise reduction, electromagnetic field calculations, eddy current modeling |
| **Profile fitting** | Map kinetic diagnostics (Thomson scattering, CES) onto equilibrium flux surfaces; fit with GP, polynomial, or exponential models |
| **Visualization** | Time traces, 1D/2D profiles, flux surface contours, top-view, and operational-space maps |
| **IMAS interoperability** | Convert between OMAS ODS and IMAS-Python (AL5) data structures; export to NetCDF |

## Architecture

![The VEST data platform in five stages]({{ site.baseurl }}/assets/diagrams/vest_data_platform_overview.svg)

Experiment, experimental data processing, a per-shot database shared by reconstruction and simulation,
and access and analysis with VAFT. The reference view, with every section's content, is on
[Diagrams]({{ site.baseurl }}/reference/diagrams/#the-vest-data-platform).

### Available IMAS IDSs in the VEST database

**Experimental:**
`dataset_description` · `magnetics` · `tf` · `pf_active` · `barometry` · `spectrometer_uv` ·
`thomson_scattering` · `charge_exchange`

**Modelling:**
`wall` · `em_coupling` · `pf_passive` · `equilibrium` (EFIT/CHEASE) · `core_profiles` ·
`mhd_linear` (DCON/RDCON)

## Quick start

<!-- docs-snippet: skip needs-database (talks to a VEST database source) -->
```python
import vaft

ods = vaft.database.load(39915)     # OMAS ODS for one shot, from the public HSDS directory
ip = ods['magnetics.ip.0.data']     # IMAS-structured access
```

Install from source, add the HSDS client, then configure your connection:

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
python -m pip install -e .
```

The source install includes `h5pyd`. Start with the packaged offline sample, then configure HSDS only
when you need public VEST data. See [Start here]({{ site.baseurl }}/workflows/start-here/) for both paths.

## Documentation

| Guide | Contents |
| --- | --- |
| [Research workflows]({{ site.baseurl }}/workflows/start-here/) | Start with an offline result and continue through the VEST analysis chain |
| [Data access and IMAS]({{ site.baseurl }}/workflows/data-access-imas/) | Load, inspect, map and convert experimental data |
| [Equilibrium and kinetic profiles]({{ site.baseurl }}/workflows/equilibrium-kinetic-profiles/) | Reconstruct equilibria and fit diagnostic profiles |
| [Automated pipelines]({{ site.baseurl }}/workflows/automated-pipelines/) | Snakemake routine, corrective and summary workflows |
| [VAFT API]({{ site.baseurl }}/reference/api/) | Module-by-module library reference |
| [VEST diagnostics and data availability]({{ site.baseurl }}/reference/vest-diagnostics/) | Generated registry of data source, availability, lifecycle and VAFT mapping status |
| [Notebooks]({{ site.baseurl }}/reference/notebooks/) | Runnable notebooks, design shells and verified outputs |

## Resources

- **Source**: [github.com/VEST-Tokamak/vaft](https://github.com/VEST-Tokamak/vaft)
- **OMAS**: [gafusion.github.io/omas](https://gafusion.github.io/omas/) — Python API for IMAS data structures
- **OMFIT**: [omfit.io](https://omfit.io/) — integrated modeling and experimental data analysis framework
- **HSDS**: [github.com/HDFGroup/hsds](https://github.com/HDFGroup/hsds) — HDF5 REST-based data service
- **h5pyd**: [github.com/HDFGroup/h5pyd](https://github.com/HDFGroup/h5pyd) — Python client for HSDS
- **IMAS**: [IMAS-Data-Dictionary](https://github.com/iterorganization/IMAS-Data-Dictionary) — ITER Integrated Modelling & Analysis Suite

## Contact and support

All users may read the VEST datasets, while shared-database writes require authorization. For
maintenance, write access, technical-system routing, and issue reporting, use
[Contacts]({{ site.baseurl }}/reference/contacts/). For more about the laboratory and the VEST
device, visit [Nuplex](http://nuplex.snu.ac.kr).
