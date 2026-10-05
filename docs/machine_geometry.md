# Shared machine geometry

`vaft.plot.machine_geometry.machine_geometry_registry(data, manifest=manifest)`
extracts immutable source records through the existing OMAS/native IMAS path
accessor. Optional `families` selects registered families. An unknown family
raises; missing geometry in a registered family produces no record.

The registry includes Thomson and CES measurement sites, Langmuir sites,
SXR and interferometer LOS, and stored non-axisymmetric conductor segments.
CES sites do not define a CES LOS. Dynamic CES coordinates are accepted as static
only when all stored samples agree. Unknown toroidal position remains unknown.
A partially populated reflection point causes the LOS to be omitted.

Each record retains cylindrical source coordinates (m, rad), a semantic kind,
IDS source path and JSON provenance. Pass the cross-shot fixture manifest to
retain its source shots, artifact checksums, source times, processing histories,
geometry reference and nonphysical-discharge flag. Without a manifest, source
shots/eras remain unknown; stored IDS comments and code parameters are retained.
If an IDS has several source artifacts, the channel identifier selects the
matching frequency when unambiguous. Otherwise the record marks its source
ambiguous and retains all candidates.

`project_machine_geometry(record, view)` returns existing `GeometryLayer` or
`Geometry3DLayer` models for `rz`, `top`, `3d` or `camera`. It returns `None` for
views requiring an unknown phi. Cartesian axes follow `X=R cos(phi)` and
`Y=R sin(phi)`. Straight LOS legs are sampled in Cartesian space before
projection; a reflected LOS retains every stored corner. Coil endpoints are
retained, with gaps between discontinuous elements. No loop closure is invented.
When phi is unknown, the R-Z view marks stored vertices as points and does not
connect them into an unsupported projected chord.

Camera projection requires the existing calibrated `CameraProjection` instance
via `projection=`. The adapter converts metres to centimetres, calls its
`project` method and inserts NaN at invalid samples so the renderer cannot join
across those samples. Sampling resolution is a presentation control.

The representation also supports explicit trajectories and unit Cartesian
launch directions. `axis_length` only controls the displayed axis extent.
The fixture manifest adds the mapper-derived Thomson 8MM10→1MM10 chord and
five scattering sites. Those sites lie on the derived chord; the positions
remain distinct from direct density and temperature measurements. It also
carries the 39915-era provisional CAD EC origin/steering and the model-derived
NBI source/beam axis. The NBI source and aperture heights are checked against
`mdescr_VEST_190307.dat`; the axis is not an as-built hardware claim. The
fixture includes one full MID coil sector using every conductor element from
the packaged GPEC geometry. The MID reference model is labelled shot 48226;
its projection on the fixture's 39915 machine does not imply simultaneous
operation. The camera reference is a separate 39915 calibration asset pair;
no camera frame is copied. CES LOS and gas injection remain absent.

The geometry additions are model/mapper records in the manifest when DD 3.41
has no appropriate source leaf. Every such record cites its source key,
processing and source hashes. NBI's `source_model_key: 0` is a configuration
key, not a physical VEST shot. Canonical plot integration and the four-view
atlas follow in the next PR for issue #1610.
