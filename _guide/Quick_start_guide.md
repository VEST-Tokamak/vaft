---
title: Quick start guide
author: VEST team
date: 2026-04-24 14:15
category: guide
layout: post
---

# Quick Start

This guide replaces the older `vest` package examples with the current `vaft` workflow. The examples below are aligned with the repository as it exists today.

## 1. Check that HSDS is reachable

```python
import vaft

connected = vaft.database.is_connect()
print(connected)
```

`True` means the HSDS server is ready and your local `hsconfigure` settings are usable.

## 2. Load a public shot as ODS

For notebook exploration and analysis, the clearest entry point is `load_ods`.

```python
import vaft

ods = vaft.database.load_ods(39915, directory="public")
```

The returned object is an OMAS `ODS`, so you can inspect it with dotted paths:

```python
time = ods["magnetics.time"]
ip = ods["magnetics.ip.0.data"]
```

## 3. Discover what data exists

List available public shots:

```python
shots = vaft.database.exist_shot("public")
```

Load multiple shots at once:

```python
ods_list = vaft.database.load_ods([39915, 40330], directory="public")
```

## 4. Load a native IMAS IDS object

If you need a specific IDS object rather than ODS, use `vaft.database.load` with an explicit `ids_name`.

```python
eq = vaft.database.load(
    shot=2,
    ids_name="equilibrium",
    directory="public",
    dd_version="3.41.0",
)
```

That path is useful when working directly with IMAS-native objects or writing tooling around IDS files.

## 5. Save locally

Remote write access is restricted. For most users, the safe path is to save generated files locally.

Save IMAS image files locally:

```python
local_dir = vaft.database.save(eq, shot=2, env="local", dd_version="3.41.0")
```

Save an ODS-derived shot locally:

```python
local_dir = vaft.database.save_ods(ods, shot=39915, env="local")
```

## 6. What VAFT covers

VAFT sits on top of the VEST integrated data platform and is typically used for:

- loading HSDS-backed experimental and modeling data
- machine mapping into IMAS-compatible structures
- equilibrium and profile analysis
- signal processing and visualization
- notebook exploration before pipeline automation

<div class="note-card">
  <strong>Tip:</strong> use `load_ods` for fast inspection in notebooks, and use `load(..., ids_name=...)`
  only when you specifically need a native IMAS IDS object.
</div>

## Continue

- [Installation]({{ site.baseurl }}/guide/Installation/)
- [Magnetics guide]({{ site.baseurl }}/guide/Magnetics/)
- [Equilibrium guide]({{ site.baseurl }}/guide/Equilibrium/)
- [VAFT repository](https://github.com/VEST-Tokamak/vaft)
