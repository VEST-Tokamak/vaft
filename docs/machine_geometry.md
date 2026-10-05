# Shared machine geometry

`vaft.plot.machine_geometry.machine_geometry_registry(data, manifest=manifest)`
extracts immutable source records through the existing OMAS/native IMAS path
accessor. Optional `families` selects registered families. An unknown family
raises; missing geometry in a registered family produces no record.

This first stage registers Thomson and CES measurement sites, Langmuir sites,
SXR and interferometer LOS, and stored non-axisymmetric conductor segments.
CES sites do not define a CES LOS. Dynamic CES coordinates are accepted as static
only when all stored samples agree. Unknown toroidal position remains unknown.
A partially populated reflection point causes the LOS to be omitted.

Each record retains cylindrical source coordinates (m, rad), a semantic kind,
IDS source path and JSON provenance. Pass the cross-shot fixture manifest to
retain its source shots, artifact checksums, source times, processing histories,
geometry reference and nonphysical-discharge flag. Without a manifest, source
shots/eras remain unknown; stored IDS comments and code parameters are retained.

`project_machine_geometry(record, view)` returns existing `GeometryLayer` or
`Geometry3DLayer` models for `rz`, `top`, `3d` or `camera`. It returns `None` for
views requiring an unknown phi. Cartesian axes follow `X=R cos(phi)` and
`Y=R sin(phi)`. Straight LOS legs are sampled in Cartesian space before
projection; a reflected LOS retains every stored corner. Coil endpoints are
retained, with gaps between discontinuous elements. No loop closure is invented.

Camera projection requires the existing calibrated `CameraProjection` instance
via `projection=`. The adapter converts metres to centimetres, calls its
`project` method and inserts NaN at invalid samples so the renderer cannot join
across those samples. Sampling resolution is a presentation control.

The representation also supports explicit trajectories and unit Cartesian
launch directions. `axis_length` only controls the displayed axis extent.
Thomson chord and EC/NBI source mapping, canonical plot integration, and the
fixture four-view atlas follow in separate PRs for issue #1610.
