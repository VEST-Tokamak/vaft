---
layout: home
title: VAFT Documentation
---

<div class="hero-panel">
  <p class="eyebrow">VEST Tokamak Data Platform</p>
  <h1>Documentation for VAFT, the VEST analysis framework built on IMAS and OMAS.</h1>
  <p class="hero-lead">
    VAFT provides one place for database access, machine mapping, equilibrium workflows,
    profile fitting, and analysis notebooks used across the VEST data pipeline.
  </p>
  <div class="hero-actions">
    <a class="hero-button" href="{{ site.baseurl }}/guide/Quick_start_guide/">Open quick start</a>
    <a class="hero-button ghost" href="{{ site.baseurl }}/guide/Installation/">Installation</a>
    <a class="hero-button ghost" href="https://github.com/VEST-Tokamak/vaft">GitHub</a>
  </div>
</div>

<div class="feature-grid">
  <section class="feature-card">
    <p class="card-label">Database</p>
    <h2>Remote access to VEST HSDS shots</h2>
    <p>
      Load IMAS-backed ODS or native IDS objects from the public VEST database and
      move between remote and local storage with the same Python workflow.
    </p>
  </section>
  <section class="feature-card">
    <p class="card-label">Analysis</p>
    <h2>Reusable diagnostics and physics tools</h2>
    <p>
      VAFT bundles signal processing, equilibrium utilities, machine mapping, plotting,
      confinement analysis, and profile fitting in one package.
    </p>
  </section>
  <section class="feature-card">
    <p class="card-label">Interoperability</p>
    <h2>IMAS and OMAS in the same workflow</h2>
    <p>
      Work directly with IMAS IDS images when needed, or load the same shot as OMAS ODS
      for notebook-friendly inspection and plotting.
    </p>
  </section>
</div>

<div class="info-band">
  <div>
    <p class="card-label">Recommended path</p>
    <h2>Start with installation, then verify the database connection, then load a public shot.</h2>
  </div>
  <div class="info-actions">
    <a class="text-link" href="{{ site.baseurl }}/guide/Installation/">Install VAFT</a>
    <a class="text-link" href="{{ site.baseurl }}/guide/Quick_start_guide/">Run the quick start</a>
  </div>
</div>

## What You Can Do

<div class="capability-grid">
  <article class="capability-card">
    <h3>Load public shots</h3>
    <p>Use `vaft.database.load_ods` to fetch OMAS ODS data from the VEST HSDS database.</p>
  </article>
  <article class="capability-card">
    <h3>Inspect specific IDS data</h3>
    <p>Use `vaft.database.load(..., ids_name="equilibrium")` when you need native IMAS objects.</p>
  </article>
  <article class="capability-card">
    <h3>Compare diagnostics</h3>
    <p>Plot magnetics, equilibrium, and profile quantities in notebooks with standard Python tools.</p>
  </article>
  <article class="capability-card">
    <h3>Bridge to workflows</h3>
    <p>Connect notebook exploration to the Snakemake-based VEST processing pipeline.</p>
  </article>
</div>

## Core Example

```python
import vaft

if not vaft.database.is_connect():
    raise RuntimeError("HSDS connection is not ready")

ods = vaft.database.load_ods(39915, directory="public")
time = ods["magnetics.time"]
ip = ods["magnetics.ip.0.data"]
```

## References

- [Quick start]({{ site.baseurl }}/guide/Quick_start_guide/)
- [Installation]({{ site.baseurl }}/guide/Installation/)
- [Magnetics guide]({{ site.baseurl }}/guide/Magnetics/)
- [Equilibrium guide]({{ site.baseurl }}/guide/Equilibrium/)
- [VAFT repository](https://github.com/VEST-Tokamak/vaft)
