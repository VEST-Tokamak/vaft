# CLAUDE.md — VAFT Development Guide

## Project Overview

**VAFT** (Versatile Analytical Framework for Tokamak) is a Python library for accessing and analyzing experimental data from the **VEST** (Versatile Experiment Spherical Torus) fusion tokamak at Seoul National University.

- **Version:** 0.1.0
- **Python:** >=3.10, <3.14
- **Data storage:** HDF5/HSDS (via h5py/h5pyd) with ODS data structures (via OMAS)
- **Homepage:** https://vest-tokamak.github.io/vaft/
- **Source:** https://github.com/VEST-Tokamak/vaft

## Build & Install

```bash
# Standard install
pip install vaft

# From source
pip install .

# Editable (development) install
pip install -e .

# With dev dependencies (pytest, black, flake8)
pip install -e ".[dev]"
```

Build system: **setuptools** configured via `pyproject.toml`. Version is read dynamically from `vaft/version.py`.

## Testing

```bash
# Run all tests
pytest test/

# Run a single test
pytest test/test_import.py
```

- Tests are in `test/` (flat directory, no conftest.py)
- No pytest configuration file — uses default pytest settings
- Some tests require a live HSDS server connection (`hsconfigure` with endpoint `http://147.46.36.244:5101`)
- Some tests require a MySQL database connection for raw DAQ data

## Code Style

- **Formatter:** `black` (default settings, no config file)
- **Linter:** `flake8` (default settings, no config file)
- No pre-commit hooks configured
- No CI/CD pipelines (no `.github/workflows/`)
- Type hints are used sparingly and inconsistently across the codebase

```bash
black vaft/
flake8 vaft/
```

## Project Structure

```
vaft/
├── vaft/                  # Main package
│   ├── process/           # Signal processing, equilibrium, magnetics, profiles, statistics
│   ├── formula/           # Physics formulas (equilibrium, stability, fittings, Green's functions, constants)
│   ├── plot/              # Visualization (1D, 2D, profiles, time series, top views, history)
│   ├── database/          # Data access layer
│   │   ├── ods.py         # HDF5/HSDS ODS database interface (h5pyd)
│   │   └── raw.py         # MySQL-based VEST raw DAQ signal interface
│   ├── omas/              # OMAS integration wrappers (formula_wrapper, process_wrapper)
│   ├── machine_mapping/   # VEST machine configuration and experiment metadata (vest.yaml)
│   ├── code/              # External code interfaces (EFIT active; CHEASE, GPEC are empty stubs)
│   ├── imas/              # IMAS data format support (from_omas conversion)
│   ├── data/              # Sample/test data files (.h5, .json, .mat, .csv, EFIT g-files)
│   └── version.py         # Version string (__version__)
├── test/                  # Test suite (pytest)
├── notebooks/             # Jupyter example notebooks
├── workflow/              # Automated data processing pipelines (Snakemake)
│   ├── automatic_pipeline_1_routine_data_processing/
│   ├── automatic_pipeline_2_corrective_data_update/
│   ├── automatic_pipeline_3_data_summary/
│   └── running_linear_stability/
├── docs/                  # Documentation source (Jekyll site)
├── pyproject.toml         # Build config, dependencies, tool settings
├── setup.py               # Legacy setup entrypoint (delegates to pyproject.toml)
└── requirements.txt       # Pinned dependencies (legacy, prefer pyproject.toml)
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `scipy` | Numerical computing |
| `matplotlib`, `seaborn` | Plotting |
| `pandas`, `xarray` | Data structures |
| `h5py`, `h5pyd` | HDF5 local and remote (HSDS) access |
| `omas` | IMAS/ODS data structure interface |
| `mysql-connector-python` | Raw DAQ database access |
| `numba` | JIT compilation for numerical routines |
| `scikit-learn` | Machine learning (profile fitting, regression) |
| `snakemake` | Workflow automation |
| `astropy` | Physical units and constants |

A vendored copy of OMAS is used (`vendor/omas/`, editable install via uv).

## Database Connectivity

VAFT connects to two external data sources:

1. **HSDS server** (HDF5 over HTTP) — for processed ODS data
   - Configure with `hsconfigure` command
   - Default endpoint: `http://147.46.36.244:5101`
   - Public reader credentials: username `reader`, password `test`

2. **MySQL database** — for raw DAQ signals
   - Config stored in `~/.vest/database_raw_info.yaml` (encrypted)
   - Encryption key in `~/.vest/encryption_key.key`
   - Uses connection pooling (pool size: 4, max retries: 3)

## Known Limitations & Code Issues

### SQL Injection Vulnerabilities (database/raw.py)
All SQL queries use f-string interpolation instead of parameterized queries. Shot numbers and field codes are inserted directly into query strings. This should be migrated to parameterized queries (`cursor.execute(query, params)`).

### Inconsistent Return Types (database/ods.py)
`exist_file()` declares return type `List[int]` but returns `True`, `False`, or `[]` depending on code path.

### Resource Leaks (database/raw.py)
Several functions (`date_from_shot`, `shots_from_date`, `last_shot`, `name`) acquire database connections without `try/finally` blocks, risking connection leaks on exceptions.

### Global Mutable State (database/raw.py)
`DB_POOL` is a module-level global modified by multiple functions. This creates thread-safety risks and makes testing difficult.

### Missing Import Bug (code/efit.py)
`gfile_to_omas()` calls `re.sub()` but never imports the `re` module. This will raise `NameError` at runtime when parsing gEQDSK filenames.

### Broad Exception Handling
Multiple locations catch `except Exception` or bare `except:` and silently continue (e.g., `process/profile.py`, `database/raw.py`, `code/efit.py`). This can mask real errors.

### Wildcard Imports
Most `__init__.py` files use `from .module import *`, causing namespace pollution and making it unclear which symbols are exported. No `__all__` definitions in submodules.

### Hardcoded VEST-Specific Constants
- `magnetics.py`: flux loop gain (11), vessel resistance (5.8e-4), baseline onset/offset times, smoothing windows
- `raw.py`: shot number ranges for DAQ table selection and trigger corrections are hardcoded with magic numbers

### Logging
No unified logging strategy. Mix of `print()` statements and `logging` module calls. Multiple places reset the root logger level with `logging.getLogger().setLevel(logging.WARNING)`.

### Missing Infrastructure
- No CI/CD pipeline
- No pre-commit hooks
- No conftest.py or pytest fixtures
- No `.flake8` or `pyproject.toml` tool config for linting
- No type checking (mypy/pyright) configuration
- Tests may require live server connections (not fully mockable)

### Stub Modules and Backup Files (code/)
- `chease.py` and `gpec.py` are empty placeholder files with no implementation
- `efit.sav` is a full duplicate backup of `efit.py` tracked in git (unnecessary)
- File I/O in `efit.py` uses manual `open()`/`close()` instead of context managers

### Commented-Out Code
- `magnetics.py`: ~200-line `toroidal_mode_analysis()` function is fully commented out
- `electromagnetics.py`: multiple commented debug print statements
- `efit.py`: ~50 commented-out code blocks (debug prints, alternative implementations, plotting)
- `test_import.py`: ODS import block commented out with traceback

### Numerical Edge Cases
- Division by zero not guarded in several equilibrium/profile calculations (e.g., `psi_boundary == psi_axis`)
- `np.errstate` used to suppress warnings rather than handling edge cases
- Matrix exponential in eddy current solver can overflow for large eigenvalues
- `efit.py`: normalizes data by dividing by min/max without zero checks
