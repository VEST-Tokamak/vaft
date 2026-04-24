---
layout: home
title: VAFT Documentation
---

# VAFT Documentation

VAFT is the VEST analysis framework for HSDS-backed database access, IMAS and OMAS interoperability, equilibrium workflows, and notebook-based tokamak data analysis.

<div class="home-note">
  Start with installation, run <code>hsconfigure</code>, confirm <code>vaft.database.is_connect()</code>,
  then load a public shot with <code>vaft.database.load_ods(...)</code>.
</div>

## Quick Links

- [Quick start]({{ site.baseurl }}/guide/Quick_start_guide/)
- [Installation]({{ site.baseurl }}/guide/Installation/)
- [Magnetics guide]({{ site.baseurl }}/guide/Magnetics/)
- [Equilibrium guide]({{ site.baseurl }}/guide/Equilibrium/)
- [VAFT repository](https://github.com/VEST-Tokamak/vaft)

## What VAFT Covers

- Remote access to public VEST HSDS shots
- ODS and native IDS workflows in the same repository
- Machine mapping, equilibrium utilities, plotting, and profile fitting
- Notebook exploration before automated pipeline execution

## Example

```python
import vaft

if not vaft.database.is_connect():
    raise RuntimeError("HSDS connection is not ready")

ods = vaft.database.load_ods(39915, directory="public")
time = ods["magnetics.time"]
ip = ods["magnetics.ip.0.data"]
```
