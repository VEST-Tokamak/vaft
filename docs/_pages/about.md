---
title: VEST and tokamak physics background
author: VEST team
date: 2026-07-01 12:00
category: pages
layout: post
permalink: /reference/vest-tokamak-physics/
guide:
  architecture: Scientific and platform context for the workflows and library reference.
  prerequisites: None.
  expected: A concise understanding of VEST, VAFT, IMAS/OMAS/HSDS, and how to cite the platform.
related:
  notebooks: [plotting-sample, confinement-scaling]
  api: [omas, process, plot]
  data_sources: [sample-ods, hsds-public]
  outputs: [first-result, confinement-scaling]
---

> **Integrate fusion science knowledge so it can be discovered, verified, compared, and studied.**

**VAFT** — the *Versatile Analytical Framework for Tokamak* — is a standardized, verifiable, and
interoperable scientific infrastructure for machine-agnostic tokamak research, developed by the
VEST team at **Seoul National University**. It integrates experimental data, reconstructed and
simulated plasma states, and analysis workflows so that fusion science knowledge can be discovered,
verified, compared, and studied. Its full end-to-end implementation on the VEST tokamak supports
routine experimental data processing, validation, modeling, physics analysis, and shared scientific
use across collaborating researchers and institutions, while serving as the reference
implementation for modern, reproducible, and data-driven fusion research.

Everything the framework exposes — remote shot loading, diagnostic mapping, physics formulas,
equilibrium and stability code interfaces, profile fitting and visualization — is organized around
standard IMAS data structures, so an analysis written against VEST data is, in principle, portable
to any IMAS-described machine.

![VEST]({{ site.baseurl }}/assets/images/IMG_3873.jpg)

## The VEST device

VEST (**Versatile Experiment Spherical Torus**) is a university-scale spherical torus operated at
Seoul National University. Its experimental campaigns produce the shot database that VAFT reads,
maps and post-processes: magnetics, TF and PF coil currents, barometry, UV spectrometry, Thomson
scattering and charge exchange, together with the modelled quantities derived from them
(wall and passive-structure models, equilibrium reconstructions, core profiles, linear MHD).

A per-shot walkthrough of the raw signals and their IMAS counterparts is in the
[Magnetics]({{ site.baseurl }}/guide/Magnetics/) and
[Machine mapping]({{ site.baseurl }}/guide/Machine_mapping/) guides.

## Foundations: IMAS, OMAS and HSDS

VAFT does not invent a data model. It composes three existing ones:

| Layer | Role in VAFT |
| --- | --- |
| [IMAS](https://github.com/iterorganization/IMAS-Data-Dictionary) | The ITER Integrated Modelling & Analysis Suite data dictionary — the schema every VEST quantity is mapped into (`magnetics`, `pf_active`, `equilibrium`, `core_profiles`, …). |
| [OMAS](https://gafusion.github.io/omas/) | The Python interface library. A VEST shot is handed to you as an OMAS `ODS`, so the whole OMAS/OMFIT ecosystem applies unchanged. |
| [HSDS](https://github.com/HDFGroup/hsds) | The HDF5 REST service hosting the VEST database. Shots live as per-IDS HDF5 images on the server and are fetched over HTTP. |

In practice that means one call gets you a fully IMAS-structured shot:

```python
import vaft

ods = vaft.database.load(39915)
time = ods['magnetics.time']
ip = ods['magnetics.ip.0.data']
```

See [Quick start guide]({{ site.baseurl }}/guide/Quick_start_guide/) to configure the connection,
and [Database]({{ site.baseurl }}/guide/Database/) for the storage model behind it.

## What the library contains

```text
vaft/
├── database/          # Remote database access (HSDS, raw SQL)
├── machine_mapping/   # Native VEST signals → IMAS IDS conversion
├── formula/           # Physics formulas (equilibrium, stability, Green's functions)
├── process/           # Signal processing, EM modelling, profile fitting
├── plot/              # Visualization (time, 1D, 2D, top-view, analysis)
├── omas/              # ODS utilities (shot metadata, sample data)
├── imas/              # IMAS-Python (AL5) interoperability
├── code/              # Code interfaces (EFIT, CHEASE, GPEC, Snakemake)
└── data/              # Sample data, geometry assets, calibration tables
```

Around the library sits the wider VEST data analysis platform: an automated **Snakemake** pipeline
carrying each shot from experiment through post-processing to simulation, and the IMAS database
itself. Both are described in [Pipelines]({{ site.baseurl }}/guide/Pipelines/); the per-module
surface is catalogued in the [API reference]({{ site.baseurl }}/guide/API_reference/).

## Citation

If VAFT or the VEST database contributes to your work, please cite:

> H.-S. Yun, S. Lee *et al*, "Developing an IMAS-compatible platform for the university-scale
> tokamak VEST and its application to operating characteristics analysis",
> *Plasma Physics and Controlled Fusion* **67** 115021 (2025).
> [doi:10.1088/1361-6587/ae1b6a](https://doi.org/10.1088/1361-6587/ae1b6a)

## Acknowledgements

The authors would like to thank O Meneghini at Proxima Fusion and J McClenaghan at General Atomics,
O Hoenen at ITER Organization for their technical advice. Some parts of the data processing were
performed using the code API in the [OMFIT](https://omfit.io/) integrated modeling framework.

This research was supported by the National Research Foundation of Korea (NRF) grant funded by the
Korean Government (MSIT) (RS-2023-00281276, RS-202400409564, and RS-2025-02304810).

## License

VAFT is distributed under the [Apache License 2.0](https://github.com/VEST-Tokamak/vaft/blob/main/LICENSE),
as declared by the package classifier (`License :: OSI Approved :: Apache Software License`) in
`pyproject.toml`.

## Getting in touch

Read access to the VEST database is open to everyone; write access, bug reports and collaboration
enquiries are covered on the [Contact]({{ site.baseurl }}/pages/contact/) page.

- Source: [github.com/VEST-Tokamak/vaft](https://github.com/VEST-Tokamak/vaft)
- Example notebooks: [Examples]({{ site.baseurl }}/guide/examples/)
