<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/VEST-Tokamak/vaft/fc294b61bdb9e0d722f7eb1ea7c962ddda532959/docs/assets/brand/vaft-wordmark-dark-1024.png">
    <img src="https://raw.githubusercontent.com/VEST-Tokamak/vaft/fc294b61bdb9e0d722f7eb1ea7c962ddda532959/docs/assets/brand/vaft-wordmark-1024.png" alt="VAFT" width="480">
  </picture>
</p>

# VAFT — Versatile Analysis Framework for Tokamak

English | [한국어](README.ko.md) · [PyPI](https://pypi.org/project/vaft/) · [License](LICENSE)

> **Connecting nuclear fusion knowledge across disciplines for integrated tokamak research**

**VAFT is a standardized, verifiable, and interoperable scientific framework for machine-agnostic tokamak research.** It connects experimental data, reconstructed and simulated plasma states, and analysis workflows through shared data structures and traceable results.

## What VAFT connects

VAFT links machine-specific measurements, [IMAS](https://imas.iter.org/)/[OMAS](https://gafusion.github.io/omas/) representations, processing and visualization, and community physics codes. Standardized data complements the original scientific artifacts; it gives researchers a common way to compare and reuse them.

![Fusion research ecosystem connected by VAFT](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/fusion_research_ecosystem_presentation.svg)

Experimental work, theory and modeling, and data-driven methods contribute to shared scientific states that researchers can test, compare, and reuse. VAFT connects those states and activities; it does not replace the specialized physics codes behind them.

[Explore the diagram and its detailed version](https://vest-tokamak.github.io/vaft/reference/diagrams/).

## Four enabling perspectives

![Four complementary capabilities of VAFT](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/vaft_four_pillars.svg)

The four pillars are complementary capabilities, not successive stages: a standard interface and traceable pipeline make results comparable and reproducible, while the data repository and machine archive keep their evidence and context usable. Together they support four ways to think about the framework:

- **Representation:** map machine data and plasma states into interoperable IMAS structures while retaining their source and conventions.
- **Research infrastructure:** discover and share validated native and standardized data, notebooks, and machine knowledge.
- **Credibility:** record provenance, configuration, and checks so traceable, reproducible workflows produce verifiable results.
- **Research practice and portability:** connect experiments, reconstruction, modeling, and interpretation in workflows that can extend beyond one device.

## How results are produced

![VAFT's managed scientific workflow](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/scientific_workflow.svg)

Machine descriptions and measurements enter through ingestion and mapping; diagnostic processing, reconstruction, and interpretive simulation then read and write a shared IMAS scientific state. Configuration and provenance follow each product, while verification, validation, and quality assessment help turn it into analysis-ready data.

## Research with VAFT

Start with [offline sample data](tutorial/README.md), then [explore shots and diagnostics](https://vest-tokamak.github.io/vaft/workflows/data-access-imas/), [reconstruct equilibria and fit profiles](https://vest-tokamak.github.io/vaft/workflows/equilibrium-kinetic-profiles/), or [work through research notebooks](notebooks/README.md). The [workflows](https://vest-tokamak.github.io/vaft/workflows/start-here/) show how these steps fit together.

## VEST reference implementation

The [VEST tokamak](https://vest-tokamak.github.io/vaft/reference/vest-systems/) at Seoul National University is VAFT's end-to-end reference implementation, from device-specific data through standardized analysis products. VAFT's data model and workflow interfaces are designed for research beyond VEST.

![VAFT's machine-agnostic architecture](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/machine_agnostic_architecture.svg)

Device-specific access and mapping absorb differences before data reaches the common IMAS model and shared research framework. The figure shows the architecture intended for other devices and future studies; it does not claim that every device integration is already implemented.

## Quick start

Install the published package, then inspect the bundled sample without database credentials or external fusion codes:

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

- [Documentation](https://vest-tokamak.github.io/vaft/) · [Data access](https://vest-tokamak.github.io/vaft/reference/database-data-sources/) · [Equilibrium representations](https://vest-tokamak.github.io/vaft/reference/equilibrium-representations/)
- [Tutorials](tutorial/README.md) · [Notebook catalog](notebooks/README.md) · [Contributing](CONTRIBUTING.md)
- [Citation and acknowledgements](https://vest-tokamak.github.io/vaft/reference/vest-tokamak-physics/) · [References](https://vest-tokamak.github.io/vaft/reference/references/) · [Third-party notices](THIRD_PARTY_NOTICES.md)
