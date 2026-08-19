# VEST `workflow/main`

This is the scaffold for the maintained OMAS-first VEST pipeline. All local
paths must come from `vaft.database.filedb.FileDB`; rules must not reconstruct paths or
refer to the legacy `/srv/vest.filedb/public/{shot}` layout.

Set `VAFT_FILEDB_DIR` to the deployment root and add shots to `config.yaml`.
The scientific rules will be added incrementally by issues #90 through #94.

The legacy server remains a read-only reference. Audit a mounted or copied
shot-first tree without modifying it:

```bash
python -m vaft.cli filedb audit /path/to/legacy --target-root /path/to/FileDB
```
