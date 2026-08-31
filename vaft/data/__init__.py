"""File-format and data-structure compatibility helpers.

The public ``vaft.data`` namespace is for portable file/data representations
such as EFIT GEQDSK. Packaged sample files remain available through
``vaft.data.resources``.
"""

from importlib import import_module

__all__ = [
    "BoundaryRepresentation",
    "Contour",
    "DerivationProvenance",
    "DerivedValue",
    "EquilibriumConvention",
    "EquilibriumData",
    "GEQDSK",
    "Gap",
    "GlobalEquilibriumDescriptors",
    "KEQDSK",
    "MEQDSK",
    "MillerFitResult",
    "MillerSequenceResult",
    "MillerSurface",
    "data_path",
    "available_samples",
    "from_imas",
    "from_equilibrium",
    "from_omas",
    "read_geqdsk",
    "read_aeqdsk",
    "read_keqdsk",
    "read_meqdsk",
    "resources",
    "sample_camera_visible_frame_paths",
    "sample_geqdsk",
    "sample",
    "sample_manifest",
    "open_adas",
    "to_imas",
    "to_omas",
    "write_geqdsk",
    "VFITResult",
    "SolovevConstraint",
    "SolovevEquilibrium",
    "StrikePoint",
    "Topology",
    "ValidationIssue",
    "ValidationReport",
    "XPoint",
    "read_vfit",
]

_EXPORT_MAP = {
    "BoundaryRepresentation": (".equilibrium", "BoundaryRepresentation"),
    "Contour": (".equilibrium", "Contour"),
    "DerivationProvenance": (".equilibrium", "DerivationProvenance"),
    "DerivedValue": (".equilibrium", "DerivedValue"),
    "EquilibriumConvention": (".equilibrium", "EquilibriumConvention"),
    "EquilibriumData": (".equilibrium", "EquilibriumData"),
    "GEQDSK": (".eqdsk", "GEQDSK"),
    "Gap": (".equilibrium", "Gap"),
    "GlobalEquilibriumDescriptors": (".equilibrium", "GlobalEquilibriumDescriptors"),
    "KEQDSK": (".keqdsk", "KEQDSK"),
    "MEQDSK": (".meqdsk", "MEQDSK"),
    "MillerFitResult": (".equilibrium", "MillerFitResult"),
    "MillerSequenceResult": (".equilibrium", "MillerSequenceResult"),
    "MillerSurface": (".equilibrium", "MillerSurface"),
    "from_imas": (".eqdsk", "from_imas"),
    "from_equilibrium": (".eqdsk", "from_equilibrium"),
    "from_omas": (".eqdsk", "from_omas"),
    "read_geqdsk": (".eqdsk", "read_geqdsk"),
    "read_aeqdsk": (".aeqdsk", "read_aeqdsk"),
    "read_keqdsk": (".keqdsk", "read_keqdsk"),
    "read_meqdsk": (".meqdsk", "read_meqdsk"),
    "to_imas": (".eqdsk", "to_imas"),
    "to_omas": (".eqdsk", "to_omas"),
    "write_geqdsk": (".eqdsk", "write_geqdsk"),
    "data_path": (".resources", "data_path"),
    "available_samples": (".resources", "available_samples"),
    "sample_camera_visible_frame_paths": (".resources", "sample_camera_visible_frame_paths"),
    "sample_geqdsk": (".resources", "sample_geqdsk"),
    "sample": (".resources", "sample"),
    "sample_manifest": (".resources", "sample_manifest"),
    "VFITResult": (".vfit", "VFITResult"),
    "SolovevConstraint": (".equilibrium", "SolovevConstraint"),
    "SolovevEquilibrium": (".equilibrium", "SolovevEquilibrium"),
    "StrikePoint": (".equilibrium", "StrikePoint"),
    "Topology": (".equilibrium", "Topology"),
    "ValidationIssue": (".equilibrium", "ValidationIssue"),
    "ValidationReport": (".equilibrium", "ValidationReport"),
    "XPoint": (".equilibrium", "XPoint"),
    "read_vfit": (".vfit", "read_vfit"),
}


def __getattr__(name: str):
    if name in {"resources", "open_adas"}:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
