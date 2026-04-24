---
title: Equilibrium
author: VEST team
date: 2026-04-24 14:15
category: guide
layout: post
---

# Equilibrium Data

VAFT supports equilibrium workflows through both OMAS-friendly ODS access and native IMAS IDS loading.

## Load equilibrium as a native IDS

```python
import vaft

eq = vaft.database.load(
    shot=2,
    ids_name="equilibrium",
    directory="public",
    dd_version="3.41.0",
)
```

This returns an IMAS equilibrium object, which is useful when downstream code expects native IDS classes.

## Load the same shot as ODS

```python
ods = vaft.database.load_ods(2, directory="public")
```

This route is usually more convenient for interactive notebooks, path-based inspection, and quick plotting.

## Typical use cases

- inspect EFIT or CHEASE-derived equilibrium content stored in the VEST database
- map kinetic diagnostics onto flux coordinates
- prepare inputs for profile fitting and stability workflows
- bridge notebook analysis to the automated Snakemake pipeline

## Notes

Remote write access is restricted. If you generate or edit equilibrium data locally, use:

```python
vaft.database.save(eq, shot=2, env="local", dd_version="3.41.0")
```

or:

```python
vaft.database.save_ods(ods, shot=2, env="local")
```

<div class="note-card">
  <strong>Related notebooks:</strong> see
  `notebooks/database_initialization_and_load.ipynb` for database loading examples and
  `notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb`
  for equilibrium-linked profile workflows.
</div>
