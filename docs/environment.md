# Runtime environment variables

VAFT reads runtime settings from the process environment. Configure them after
installation in a shell startup file, a Conda activation script, an environment
module, or the equivalent mechanism used by your computing site.

VAFT uses two distinct naming conventions:

- external scientific-code installations use `{CODE}HOME` and point to a
  repository or installation root;
- paths and options owned by VAFT use descriptive `VAFT_*` names.

VAFT does not provide a separate runtime-configuration registry or reporting
API.

## External scientific-code roots

When a home variable is defined, the corresponding adapter uses the exact
paths below. A missing or non-executable file is an installation error; VAFT
does not silently fall back to another variable or `PATH`.

| Variable | Value type | Expected executables | Consuming adapter |
| --- | --- | --- | --- |
| `GPECHOME` | GPEC repository or installation root | `bin/dcon`, `bin/match`, `bin/rdcon`, `bin/stride`, `bin/gpec` | `vaft.code.gpec` |
| `CHEASEHOME` | CHEASE repository or installation root | `bin/chease` | `vaft.code.chease` |
| `EFITHOME` | EFIT repository or installation root | `bin/efit` | `vaft.code.efit`, `vaft.code.kineticEfit` |
| `TESHOME` | TES repository or installation root | `bin/rtes` | `vaft.code.tes` |

The expected installed layout is:

```text
$GPECHOME/
└── bin/{dcon,match,rdcon,stride,gpec}

$CHEASEHOME/
└── bin/chease

$EFITHOME/
└── bin/efit

$TESHOME/
└── bin/rtes
```

Compile or install each external project so its executable is present and has
execute permission at the documented location. External-code licensing and
installation remain the responsibility of the user and the upstream project.

### Backward-compatible executable variables

The following existing variables remain supported when the corresponding home
variable is unset. New installations should use `{CODE}HOME`.

| Variable | Value type | Behavior |
| --- | --- | --- |
| `EFIT` | EFIT executable or directory containing `efit` | Used only when `EFITHOME` is unset. |
| `CHEASE` | CHEASE executable or directory containing `chease` | Used only when `CHEASEHOME` is unset. |
| `CHEASE_EXEC_DIR` | Directory containing `chease`, or the executable path accepted by the existing adapter | Used after `CHEASE` when `CHEASEHOME` is unset. |
| `RTES` | Path to the `rtes` executable | Used only when `TESHOME` is unset. |

An explicitly supplied executable in an existing adapter configuration remains
supported. This is backward compatibility for those adapter APIs, not a global
configuration-precedence system.

## VAFT-owned settings

| Variable | Value type | Required | Purpose and default |
| --- | --- | --- | --- |
| `VAFT_FILEDB_DIR` | Directory path | For local FileDB workflows | Root of the canonical OMAS-first FileDB. |
| `VAFT_ADAS_DIR` | Directory path | No | OPEN-ADAS cache/data root. Defaults to the platform user-cache directory. |
| `VAFT_RAW_SAMPLE_PATH` | File path or path template | No | Offline raw-data dump; `{shot}` may be used in the path. |
| `VAFT_RAW_OFFLINE_ONLY` | Boolean option (`1`, `true`, `yes`, or `on`) | No | Disables live raw SQL access. Defaults to false. |

## Machine-specific overrides

These existing VEST integration variables are retained for compatibility.

| Variable | Value type | Purpose |
| --- | --- | --- |
| `VEST_SXR_GEOMETRY_TABLE` | File path | Explicit soft-X-ray geometry table. |
| `VEST_SXR_GEOMETRY_DIR` | Directory path | Directory searched for VEST soft-X-ray geometry data. |

## Externally owned environment

VAFT also respects settings owned by its dependencies or the operating system.
They are not renamed by VAFT.

| Variable | Owner | Purpose |
| --- | --- | --- |
| `IMAS_DD_VERSION_CONVERSION` | IMAS/OMAS integration | Data-dictionary version used for OMAS–IMAS conversion. |
| `IMAS_DD_CONVERSION` | IMAS/OMAS integration | Legacy fallback for `IMAS_DD_VERSION_CONVERSION`. |
| `IMAS_VERSION` | IMAS | Default IMAS version in compatibility APIs. |
| `OMAS_DEBUG_TOPIC` | OMAS | Enables dependency-owned OMAS debug topics such as `imas_code`. |
| `OMP_NUM_THREADS` | OpenMP/external code | EFIT thread count; the EFIT adapter defaults it to `1` when unset. |
| `USER`, `HOME` | Operating system | User identity and home-directory defaults. |
| `XDG_CACHE_HOME` | Freedesktop environment | Linux cache root used by OPEN-ADAS caching. |
| `LOCALAPPDATA` | Windows | Windows cache root used by OPEN-ADAS caching. |

HSDS credentials are configured with `hsconfigure` and the dependency-owned
`.hscfg` mechanism described in the main README; they are not VAFT environment
variables.

## Shell configuration examples

### Bash or Zsh

Add the relevant lines to `~/.bashrc` or `~/.zshrc`:

```bash
export GPECHOME=/opt/gpec
export CHEASEHOME=/opt/chease
export EFITHOME=/opt/efit
export TESHOME=/opt/tes

export VAFT_FILEDB_DIR=/data/VEST/FileDB
export VAFT_ADAS_DIR="$HOME/.cache/vaft/open_adas"
```

Start a new shell or source the file after editing it.

### Conda activation

Create `$CONDA_PREFIX/etc/conda/activate.d/vaft.sh` containing the same export
commands. If the environment is not active while creating the file, replace
`$CONDA_PREFIX` with that environment's absolute path.

```bash
export EFITHOME=/opt/efit
export CHEASEHOME=/opt/chease
export VAFT_FILEDB_DIR=/data/VEST/FileDB
```

### Environment module

A site modulefile can establish the roots without modifying user startup files:

```tcl
#%Module1.0
setenv GPECHOME /opt/gpec
setenv CHEASEHOME /opt/chease
setenv EFITHOME /opt/efit
setenv TESHOME /opt/tes
setenv VAFT_FILEDB_DIR /data/VEST/FileDB
```

After loading the environment, verify the compiled files directly, for example:

```bash
test -x "$EFITHOME/bin/efit"
test -x "$CHEASEHOME/bin/chease"
test -x "$GPECHOME/bin/gpec"
test -x "$TESHOME/bin/rtes"
```
