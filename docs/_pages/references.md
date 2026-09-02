---
title: References
author: VEST team
category: reference
layout: post
permalink: /reference/references/
guide:
  architecture: Primary citations, external standards, and documentation migration record.
  prerequisites: None.
  expected: Authoritative sources for scientific claims and a traceable legacy-page disposition.
related:
  notebooks: [database-initialization]
  api: [omas, imas, database]
  data_sources: [hsds-public]
---

VAFT connects an IMAS-compatible data platform, the OMAS Python interface, VEST experimental data,
and external equilibrium and stability codes. Use the original sources below for definitions and
scientific context; this site explains how VAFT applies them.

## VAFT platform paper

H.-S. Yun, S. Lee *et al.*, “Developing an IMAS-compatible platform for the university-scale tokamak
VEST and its application to operating characteristics analysis,” *Plasma Physics and Controlled
Fusion* **67** 115021 (2025), [doi:10.1088/1361-6587/ae1b6a](https://doi.org/10.1088/1361-6587/ae1b6a).

The documentation follows the paper’s platform architecture—Snakemake workflows, IMAS/HSDS storage,
the VAFT interface, and the EFIT → CHEASE → DCON/RDCON/GPEC analysis chain—but does not copy figures
from the publication.

## Standards and software

- [IMAS Data Dictionary](https://github.com/iterorganization/IMAS-Data-Dictionary)
- [OMAS](https://gafusion.github.io/omas/)
- [HDF5 Scalable Data Service](https://github.com/HDFGroup/hsds)
- [h5pyd](https://github.com/HDFGroup/h5pyd)
- [OMFIT](https://omfit.io/)
- [VAFT source and notebooks](https://github.com/VEST-Tokamak/vaft)

## Documentation migration inventory

Every page from the previous auto-generated guide is accounted for below. Compatibility routes use
permanent canonical destinations and a readable fallback link.

| Previous URL | Disposition | Canonical destination |
| --- | --- | --- |
{% for item in site.data.page_migrations -%}
| `{{ item.legacy_url }}` | {{ item.disposition }} | [{{ item.canonical_url }}]({{ site.baseurl }}{{ item.canonical_url }}) |
{% endfor %}
