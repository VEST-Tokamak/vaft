---
layout: home
title: VAFT Documentation
---

<div class="hero-panel">
  <p class="eyebrow">VEST Tokamak Data Platform</p>
  <h1>VAFT is the working documentation hub for VEST data access, analysis, and equilibrium workflows.</h1>
  <p class="hero-lead">
    The site now focuses on the current `vaft` package API, public HSDS access, and the
    notebook-first analysis flow used in the repository today.
  </p>
  <div class="hero-actions">
    <a class="hero-button" href="{{ site.baseurl }}/guide/Quick_start_guide/">Open quick start</a>
    <a class="hero-button ghost" href="{{ site.baseurl }}/guide/Installation/">Installation</a>
    <a class="hero-button ghost" href="https://github.com/VEST-Tokamak/vaft">GitHub</a>
  </div>
</div>

<div class="stat-strip">
  <div class="stat-card">
    <span class="stat-value">HSDS</span>
    <span class="stat-label">remote VEST shot access</span>
  </div>
  <div class="stat-card">
    <span class="stat-value">IMAS + OMAS</span>
    <span class="stat-label">both object models supported</span>
  </div>
  <div class="stat-card">
    <span class="stat-value">Python 3.10-3.13</span>
    <span class="stat-label">current supported runtime</span>
  </div>
</div>

## Start Here

<div class="doc-grid">
  <a class="doc-card primary" href="{{ site.baseurl }}/guide/Quick_start_guide/">
    <span class="card-label">Guide 01</span>
    <h2>Quick start</h2>
    <p>Verify the HSDS connection, load a public shot, and inspect the returned ODS paths.</p>
    <span class="card-cta">Open guide</span>
  </a>
  <a class="doc-card" href="{{ site.baseurl }}/guide/Installation/">
    <span class="card-label">Guide 02</span>
    <h2>Installation</h2>
    <p>Install from PyPI or source and configure the HSDS client for remote database access.</p>
    <span class="card-cta">Install VAFT</span>
  </a>
  <a class="doc-card" href="{{ site.baseurl }}/guide/Magnetics/">
    <span class="card-label">Guide 03</span>
    <h2>Magnetics</h2>
    <p>Load magnetics data from an ODS shot and start from stable, path-based examples.</p>
    <span class="card-cta">See examples</span>
  </a>
  <a class="doc-card" href="{{ site.baseurl }}/guide/Equilibrium/">
    <span class="card-label">Guide 04</span>
    <h2>Equilibrium</h2>
    <p>Move between native IDS objects and notebook-friendly ODS access for equilibrium work.</p>
    <span class="card-cta">Open workflow</span>
  </a>
</div>

## Why VAFT

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

<div class="info-band accent-band">
  <div>
    <p class="card-label">Recommended path</p>
    <h2>Install the package, run `hsconfigure`, confirm `vaft.database.is_connect()`, then load a public shot.</h2>
  </div>
  <div class="info-actions">
    <a class="text-link" href="{{ site.baseurl }}/guide/Installation/">Install VAFT</a>
    <a class="text-link" href="{{ site.baseurl }}/guide/Quick_start_guide/">Run the quick start</a>
  </div>
</div>

## Core Example

<div class="capability-grid">
  <article class="capability-card code-card">
    <h3>Load a public shot</h3>
    <pre><code class="language-python">import vaft

if not vaft.database.is_connect():
    raise RuntimeError("HSDS connection is not ready")

ods = vaft.database.load_ods(39915, directory="public")
time = ods["magnetics.time"]
ip = ods["magnetics.ip.0.data"]</code></pre>
  </article>
  <article class="capability-card">
    <h3>Use the right object for the job</h3>
    <p><strong>`load_ods`</strong> is the default notebook entry point for analysis and plotting.</p>
    <p><strong>`load(..., ids_name=...)`</strong> is the path to use when you need a native IMAS IDS object.</p>
    <p><strong>`save_ods(..., env="local")`</strong> is the safe route for local export when remote write access is restricted.</p>
  </article>
</div>

## References

- [Quick start]({{ site.baseurl }}/guide/Quick_start_guide/)
- [Installation]({{ site.baseurl }}/guide/Installation/)
- [Magnetics guide]({{ site.baseurl }}/guide/Magnetics/)
- [Equilibrium guide]({{ site.baseurl }}/guide/Equilibrium/)
- [VAFT repository](https://github.com/VEST-Tokamak/vaft)
