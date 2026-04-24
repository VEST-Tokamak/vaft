---
title: Installation
author: VEST team
date: 2026-04-24 14:15
category: guide
layout: post
---

# Install VAFT

VAFT supports Python `3.10` through `3.13` and can be installed either from PyPI or from the repository source tree.

<div class="note-card">
  <strong>Recommendation:</strong> if you only want to use the public VEST database and the published package API,
  start with the PyPI install. Use the source install when you are developing VAFT itself.
</div>

## Option 1: Install from PyPI

```bash
python -m pip install vaft
python -m pip install "vaft[hsds]"
```

The extra `hsds` dependency installs `h5pyd`, which is required for remote HSDS access.

## Option 2: Install from source

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
python -m pip install -e .
python -m pip install -e ".[hsds]"
```

For development tooling:

```bash
python -m pip install -e ".[dev]"
```

## Configure the HSDS client

To connect to the VEST HSDS database, run:

```bash
hsconfigure
```

Use the server endpoint below and enter the credentials provided to you by the VEST team.

| Field | Value |
| --- | --- |
| Server endpoint | `http://147.46.36.244:5101` |
| Username | your assigned username |
| Password | your assigned password |
| API key | leave blank unless you were given one |

If you are testing public read access and do not have a personal account yet, the historically shared read-only credentials may still work depending on current server policy.

## Verify the connection

```python
import vaft

vaft.database.is_connect()
```

If the command returns `True`, the HSDS server is reachable and your local configuration is valid.

## Next step

After installation, continue with the [Quick start guide]({{ site.baseurl }}/guide/Quick_start_guide/) to load a public shot and inspect the returned data structure.
