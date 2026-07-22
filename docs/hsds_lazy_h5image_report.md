# HSDS lazy access and per-IDS h5image report

This document records the VAFT database I/O architecture implemented for
[#51](https://github.com/VEST-Tokamak/vaft/issues/51), together with the two
shot 39915 benchmark passes used to select the default policy.

## Final architecture

Canonical IMAS images remain the only source of truth:

```text
/public/{shot}/master.h5
/public/{shot}/equilibrium.h5
/public/{shot}/magnetics.h5
...
```

Byte-exact, compressed per-IDS images are derived caches for eager loading:

```text
/public/{shot}/master.h5image.h5
/public/{shot}/equilibrium.h5image.h5
/public/{shot}/magnetics.h5image.h5
...
```

Each derived domain contains a 4 MiB chunked, gzip-level-1 `h5image` byte
stream and a JSON manifest. The manifest records the canonical URI and
revision, HDF5 backend and IMAS DD versions, OMAS and VAFT versions, byte size,
SHA-256, and creation time. Restoration streams the image into a local file
while validating its checksum; it does not copy the complete payload into
memory.

The access policy is intentionally tiered:

```text
specific signal        -> canonical direct HSDS lazy selection
one or more full IDS   -> local cache, then per-IDS h5image, then canonical hsget
complete native shot   -> per-IDS h5image -> native IDS mapping
complete ODS           -> per-IDS h5image -> native IDS -> fast OMAS conversion
```

The historical `/public/{shot}.omas.h5` remains readable as a compatibility
and benchmark baseline, but is no longer created by default.

## Public API and defaults

Remote database access uses a bare namespace:

```python
ods = vaft.database.load(
    39915,
    source="public",
    representation="omas",
    paths=["equilibrium", "magnetics"],
    transport="auto",
)
```

`transport="auto"` reuses a valid persistent local domain cache first, then a
valid per-IDS h5image, and finally falls back to canonical `hsget`.
`"canonical"` bypasses derived images and `"h5image"` is strict.

Direct lazy reads always use canonical HSDS selections:

```python
with vaft.database.open(
    39915, source="public", representation="omas", paths="equilibrium"
) as ods:
    psi = ods["equilibrium.time_slice.0.profiles_2d.0.psi"]

with vaft.database.open(
    39915, source="public", representation="imas", paths="equilibrium"
) as handle:
    psi = handle.get().time_slice[0].profiles_2d[0].psi
```

Remote saves keep the canonical write authoritative:

```python
vaft.database.save(data, 39915, target="public", derived_cache="auto")
```

`derived_cache="auto"` creates per-IDS images. Explicit values are `"none"`,
`"imas-images"`, `"omas"`, and `"both"`. A derived-cache failure emits a
warning but does not roll back a successful canonical save. Native IDS writes
reject the OMAS cache modes.

Local files are handled separately by `vaft.omas.load/save` and
`vaft.imas.load/save`. These importers detect OMAS JSON/HDF5, IMAS HDF5 image
sets, IMAS NetCDF, and GEQDSK without a format argument.

## IMAS to OMAS conversion

For IMAS-Python AL5, VAFT now performs exactly one `DBEntry.get()` per stored
IDS and traverses only populated nodes with `iter_nonempty_()`. Scalar, string,
array, nested AOS, and uncertainty companion nodes are written directly with
`ODS.setraw()`. Staging passes the actual stored IDS list, avoiding empty gets
across the entire data dictionary. AL4 retains the compatibility converter.

The final full-shot conversion p50 is 2.44 s and selective equilibrium
conversion p50 is 0.34 s, below the respective 10 s and 1.5 s targets.

## Benchmark methodology

- Endpoint fingerprint: `fc67d46d5de5c680`
- Shot: `39915`
- Python 3.12.13, h5pyd 0.20.0, h5py 3.16.0, OMAS 0.94.2
- Values were checked for shape, dtype, and numerical parity.
- Cold eager runs used `cache="off"`.
- Lazy runs verified that no `hsget` subprocess was used.
- Final runs used independent temporary staging per method and removed it
  immediately afterward.

### Benchmark 1: rollout smoke

The first pass used one measured iteration to validate the derived domains,
transport fallback, and parity before enabling the policy.

| Method | Canonical total | Per-IDS h5image total |
|---|---:|---:|
| Full ODS | 37.15 s | 7.44 s |
| Full native IDS mapping | 35.68 s | 6.07 s |
| Selective equilibrium ODS | 18.91 s | 2.34 s |
| Selective equilibrium native | 18.91 s | 2.31 s |
| Selective magnetics ODS | 4.00 s | 1.75 s |
| Selective magnetics native | 2.64 s | 0.82 s |

Direct OMAS lazy access took 0.512 s for the equilibrium 2D signal and 0.096
s for the magnetics 1D signal. The smoke pass read the historical monolithic
OMAS payload into memory; the final pass below uses the production 4 MiB
streaming restoration path.

### Benchmark 2: final repeated benchmark

The final pass excluded one warm-up and reports five measured iterations.

| Method | Download/staging p50 | Conversion p50 | Total p50 | Total p95 |
|---|---:|---:|---:|---:|
| Full ODS, canonical | 30.37 s | 2.44 s | 32.80 s | 39.96 s |
| Full ODS, per-IDS h5image | 4.77 s | 2.15 s | **6.92 s** | 8.11 s |
| Full native, canonical | 30.37 s | 0.98 s | 31.34 s | 38.52 s |
| Full native, per-IDS h5image | 4.77 s | 0.99 s | **5.81 s** | 6.96 s |
| Equilibrium ODS, canonical | 18.49 s | 0.34 s | 18.83 s | 20.28 s |
| Equilibrium ODS, per-IDS h5image | 2.37 s | 0.33 s | **2.74 s** | 2.85 s |
| Equilibrium native, canonical | 17.25 s | 0.23 s | 17.48 s | 20.34 s |
| Equilibrium native, per-IDS h5image | 2.24 s | 0.29 s | **2.70 s** | 2.76 s |
| Magnetics ODS, canonical | 2.65 s | 1.47 s | 4.33 s | 5.42 s |
| Magnetics ODS, per-IDS h5image | 0.38 s | 1.30 s | **1.63 s** | 2.16 s |
| Magnetics native, canonical | 2.35 s | 0.41 s | 2.77 s | 2.88 s |
| Magnetics native, per-IDS h5image | 0.32 s | 0.43 s | **0.75 s** | 0.83 s |
| Historical full OMAS h5image | 5.93 s | 4.90 s | 11.30 s | 11.73 s |

Direct lazy p50 remained fastest for individual signals:

| Representation | Equilibrium 2D | Magnetics 1D |
|---|---:|---:|
| OMAS lazy | 0.423 s | 0.098 s |
| Native IMAS lazy | 0.500 s | 0.098 s |

All canonical, per-IDS, native, OMAS, and lazy representative values matched.
The per-IDS representation used 81,453,505 allocated bytes versus 105,028,824
bytes for the canonical HSDS representation, a 22.4% reduction. Keeping both
representations increases total storage by about 77.6% over canonical-only
storage.

## Decision

The per-IDS representation exceeded every activation threshold:

- cold full and selective staging improved by more than 20%;
- allocated representation size improved by more than 15%;
- full ODS was faster, rather than up to 10% slower, than the historical full
  OMAS cache.

Therefore eager `transport="auto"` prefers valid per-IDS images, while direct
lazy access stays canonical. New full OMAS cache generation is opt-in. A
future server deployment should generate derived images asynchronously and
add stale-cache garbage collection, but those operational changes are not
required for the client-side policy implemented here.
