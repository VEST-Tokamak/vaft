## How to get VEST pulse data?

### LOAD

Use `vaft.database.load(shot, source="public")` when a complete local IMAS staging set is needed.
This eager path invokes `hsget`; when `paths` is supplied it stages only
the selected top-level IDS and `dataset_description`, using a validated local
domain cache by default. Pass `cache="off"` to force a fresh download. Eager
loads prefer validated per-IDS h5image domains when present; use
`transport="canonical"` to force `hsget` or `transport="h5image"` to require
the derived path.

Use `vaft.database.open` for selection-based access without a temporary
staging directory:

```python
import vaft

with vaft.database.open(39915, source="public", paths="equilibrium") as ods:
    time = ods["equilibrium.time"]
    psi = ods["equilibrium.time_slice.0.profiles_2d.0.psi"]
```

`paths` may be a top-level IDS name, a detailed OMAS path, a list, or `None` to discover the available
domains. Only occurrence 0 is supported lazily; use the eager API for other
occurrences and native IDS objects.


### SAVE & DELETE

Save and delete is only available to the authorized users. If you want to store data in VEST server of get authentication to the file contact us. 
