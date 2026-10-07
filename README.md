<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/VEST-Tokamak/vaft/fc294b61bdb9e0d722f7eb1ea7c962ddda532959/docs/assets/brand/vaft-wordmark-dark-1024.png">
    <img src="https://raw.githubusercontent.com/VEST-Tokamak/vaft/fc294b61bdb9e0d722f7eb1ea7c962ddda532959/docs/assets/brand/vaft-wordmark-1024.png" alt="VAFT" width="480">
  </picture>
</p>

# VAFT — Versatile Analysis Framework for Tokamak

English | [한국어](README.ko.md) · [PyPI](https://pypi.org/project/vaft/) · [License](LICENSE)

> **Connecting nuclear fusion knowledge across disciplines for integrated tokamak research**

**VAFT is a scientific framework for organizing and analyzing tokamak data using IMAS structures.** It connects experimental data, reconstructed plasma states, simulation results, and analysis workflows through shared data structures and recorded processing history.

## What VAFT connects

VAFT maps device-specific diagnostics and machine data into the common data model defined by the [IMAS Data Dictionary](https://imas-data-dictionary.readthedocs.io/en/latest/). [OMAS](https://gafusion.github.io/omas/) provides a Python interface to those structures. VAFT uses the standardized records for processing and visualization and to exchange data with established physics codes such as EFIT and CHEASE. Native diagnostic files and solver outputs remain available alongside those records.

![Fusion research ecosystem connected by VAFT](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/fusion_research_ecosystem_presentation.svg)

The diagram places experiments, theory and modeling, and data-driven studies around measured, reconstructed, and simulated plasma states. VAFT lets researchers exchange and compare those states without replacing their specialized physics codes.

[Explore the diagram and its detailed version](https://vest-tokamak.github.io/vaft/develop/reference/diagrams/).

## Four enabling perspectives

![Four complementary capabilities of VAFT](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/vaft_four_pillars.svg)

These four capabilities work together rather than as successive stages. IMAS mappings provide a common interface; recorded configurations make processing traceable; and the repository and archive retain the data and machine context needed to interpret a result. For researchers, they address four practical needs:

- **Representation:** map diagnostic channels, device geometry, equilibria, and profiles to named IMAS structures while retaining source and flux conventions.
- **Research infrastructure:** keep native files and standardized shot records accessible, alongside notebooks and device configuration history.
- **Credibility:** record calibrations, mapping versions, solver settings, and quality checks so workflows are traceable and reproducible and results are verifiable.
- **Research practice and portability:** use the same data paths for reconstruction, modeling, and plotting; adapt those workflows to another device by supplying its mappings.

## How results are produced

![VAFT's managed scientific workflow](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/scientific_workflow.svg)

Machine descriptions and diagnostic measurements are registered and mapped to IMAS data structures. Diagnostic processing, equilibrium reconstruction, and simulation read and write those structures; the workflow records configurations and provenance and checks data quality before results are reused in analysis.

## Research with VAFT

Start with [offline sample data](tutorial/README.md), then [explore shots and diagnostics](https://vest-tokamak.github.io/vaft/workflows/data-access-imas/) or [reconstruct equilibria and fit profiles](https://vest-tokamak.github.io/vaft/workflows/equilibrium-kinetic-profiles/). The [research notebooks](notebooks/README.md) and [workflow guide](https://vest-tokamak.github.io/vaft/workflows/start-here/) show complete examples.

## VEST reference implementation

The [VEST tokamak](https://vest-tokamak.github.io/vaft/reference/vest-tokamak-physics/) at Seoul National University is VAFT's reference implementation: its diagnostics are mapped to IMAS and used in reconstruction, modeling, and analysis workflows. VAFT separates VEST-specific mappings from the data structures and tools those workflows share.

![VAFT's machine-agnostic architecture](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/machine_agnostic_architecture.svg)

A new device needs its own data access and diagnostic mappings; downstream tools can then work with the same IMAS structures. This is the intended path for extending VAFT beyond VEST, not a claim that every device integration is already implemented.

## Quick start

Install the published package, then inspect the packaged VEST sample without database credentials or external fusion codes:

```bash
pip install vaft
```

```python
import vaft

ods = vaft.omas.sample_ods()
print(sorted(ods.keys()))
```

For a plotted first result, follow [Start here](https://vest-tokamak.github.io/vaft/workflows/start-here/). For source installation or platform setup, see [install/README.md](install/README.md).

## Learn more

- [Documentation](https://vest-tokamak.github.io/vaft/) · [IMAS concepts](https://vest-tokamak.github.io/vaft/reference/imas-concepts/) · [Data access](https://vest-tokamak.github.io/vaft/reference/database-data-sources/) · [Equilibrium representations](https://vest-tokamak.github.io/vaft/develop/reference/equilibrium-representations/)
- [Tutorials](tutorial/README.md) · [Notebook catalog](notebooks/README.md) · [Contributing](CONTRIBUTING.md)
- [Citation and acknowledgements](https://vest-tokamak.github.io/vaft/reference/vest-tokamak-physics/) · [References](https://vest-tokamak.github.io/vaft/reference/references/) · [Third-party notices](THIRD_PARTY_NOTICES.md)
