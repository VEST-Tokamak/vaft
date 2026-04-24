# VAFT Documentation

VAFT is the VEST analysis framework for HSDS-backed database access, IMAS and OMAS interoperability, machine mapping, equilibrium workflows, and notebook-based tokamak data analysis.

## Installation

Install from PyPI:

```bash
python -m pip install vaft
python -m pip install "vaft[hsds]"
```

Install from source:

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
python -m pip install -e .
python -m pip install -e ".[hsds]"
```

## HSDS configuration

Run:

```bash
hsconfigure
```

Server endpoint:

```text
http://147.46.36.244:5101
```

Use the username and password assigned to you by the VEST team.

## Basic usage

```python
import vaft

if not vaft.database.is_connect():
    raise RuntimeError("HSDS connection is not ready")

ods = vaft.database.load_ods(39915, directory="public")
```

## Documentation

- Site: <https://vest-tokamak.github.io/vaft/>
- Quick start: <https://vest-tokamak.github.io/vaft/guide/Quick_start_guide/>
- Installation: <https://vest-tokamak.github.io/vaft/guide/Installation/>
- Repository: <https://github.com/VEST-Tokamak/vaft>
