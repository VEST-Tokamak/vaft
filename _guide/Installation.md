---
title: Start here
author: Sun jae Lee
date: 2026-07-01 09:00
category: guide
layout: post
permalink: /workflows/start-here/
guide:
  architecture: Entry point from installation to an offline ODS and optional public data.
  prerequisites: Python 3.10–3.13, Git, and a fresh virtual environment.
  expected: A local plasma-current plot, followed optionally by read-only shot 39915 metadata.
  status: Verified offline and against public HSDS.
related:
  notebooks: [database-initialization, plotting-sample]
  api: [omas, database, plot]
  data_sources: [sample-ods, hsds-public]
  outputs: [first-result, hsds-39915]
---

This path gets a new user from a clean Python environment to a visible result without requiring
database credentials or an external fusion code. Public VEST data access is the optional second step.

## 1. Install VAFT from source

VAFT supports Python 3.10–3.13. Use a virtual environment and install the repository source:

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
python -m pip install -e .
```

The declared dependencies include OMAS, IMAS-Python, plotting tools, Snakemake and `h5pyd`. NumPy 2
is the default numerical stack. The older PyPI package is retained for compatibility but is not the
recommended installation route.

### Development tooling

`dev` is the only optional-dependency group defined by the project:

```bash
python -m pip install -e ".[dev]"
```

## 2. Produce the first offline result

The packaged sample follows the same OMAS/IMAS paths as a VEST shot:

```python
import matplotlib.pyplot as plt
import vaft

ods = vaft.omas.sample_ods()
print(sorted(ods.keys()))

vaft.plot.magnetics_time_ip(ods)
plt.show()
```

Seeing the plasma-current trace completes the credential-free first workflow. Continue with
[experimental interpretation]({{ site.baseurl }}/workflows/experimental-interpretation/) or configure
the public database below.

## 3. Configure read-only public HSDS access

Run `hsconfigure` and enter the endpoint plus credentials supplied by the VEST team. Credentials stay
in the user configuration and must never be committed to a notebook or documentation asset.

```bash
>> hsconfigure
Enter new values or accept defaults in brackets with Enter.

Server endpoint []: http://147.46.36.244:5101
Username []: [assigned_username]
Password []: [assigned_password]
API Key [None]: 
Testing connection...
connection ok
Quit? (Y/N)Y
```

A successful read uses the public namespace and does not modify the database:

```python
import vaft

with vaft.database.open(39915, source="public", paths="equilibrium") as ods:
    print(ods["equilibrium.time"])
```

Use `vaft.database.open()` for lazy exploratory reads and `vaft.database.load()` when a workflow needs
a staged eager object. Remote saving is restricted to authorized operators; this workflow never calls
`vaft.database.save()`.

## 4. Optional external fusion codes

VAFT can prepare and collect inputs for EFIT, CHEASE, GPEC/DCON/RDCON and TES. Configure only the codes
you have installed:

```bash
export EFITHOME=/path/to/efit
export CHEASEHOME=/path/to/chease
export GPECHOME=/path/to/gpec
export TESHOME=/path/to/tes
```

Each executable belongs under its root’s `bin/` directory. The workflow guides degrade to deterministic
input preparation when a binary is absent.

## Expected outputs

- Offline: a plasma-current plot from the packaged sample ODS.
- HSDS: readable metadata or IDS paths for public shot 39915.
- External codes: an explicit readiness report before any solver is launched.
