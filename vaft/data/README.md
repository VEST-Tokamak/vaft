# vaft/data data catalog

`vaft/data` contains runtime resources and repository-only samples. Data assets
are grouped one level deep only; Python package files stay at this directory
root. The GitHub repository contains every file listed below; the PyPI
distribution includes only the runtime geometry resources, GPEC templates,
`legacy/sql_table.txt`, and `omas/39915.json`. Clone the repository to access
the archived EFIT, IMAS, legacy diagnostic, digitizer, and additional OMAS
samples.

## Layout

| Directory | Files | Purpose |
| --- | --- | --- |
| `geometry/` | `Coil_info.mat`, `MD.yaml`, `VEST_DiscretizedCoilGeometry_Full_ver_1906.mat`, `VEST_DiscretizedCoilGeometry_Full_ver_2507.mat`, `VEST_MagneticsGeometry_Full_ver_2302.yaml`, `line_of_sight_endpoints.csv`, `table.yaml` | VEST magnetic, PF, and soft X-ray geometry metadata |
| `efit/` | `g039020.031180`, `g039915.00317`, `g039915.00319`, `g040330.00320`, `g040330.00321`, `g040330.00323`, `a039915.00319`, EFIT table files | GEQDSK/AEQDSK samples and EFIT reference tables |
| `omas/` | `39915.json`, `41524.json`, `41672.json`, `thomson_scattering.json` | OMAS/ODS sample and contract-test payloads |
| `imas/` | `vest_imas_3.40.1.nc` | IMAS-format sample container |
| `legacy/` | `41514.h5`, `46051_NeTe.mat`, `CES_47514.mat`, `IDS_47518.mat`, `NeTe_Shot39915_v9_rev.mat`, `digitizer_17592_45531.csv`, `digitizer_22577_45531.csv`, `shot_44740.json.gz`, `sql_table.txt` | Legacy diagnostic samples, raw SQL dump, and DB lookup table |
| `gpec/` | `*.in`, `vest_*.dat` | VEST GPEC-suite namelist templates and coil data |

## Access

Use `vaft.data.resources.data_path()` with explicit category paths:

```python
from vaft.data.resources import data_path

ods_path = data_path("omas/39915.json")
```

Repository-only samples such as `efit/g039915.00319` and
`legacy/46051_NeTe.mat` are available after cloning the repository, not after
`pip install vaft`.

Flat calls such as `data_path("39915.json")` are intentionally unsupported.
Deleted duplicate assets from older checkouts are not recreated here.
