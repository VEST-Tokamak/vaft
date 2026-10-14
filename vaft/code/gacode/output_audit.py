"""Derivability/retention view over existing native dataclasses and mapping audits.

This inventory enumerates fields from the actual native containers; new parser
fields appear as review_required rather than silently missing from the audit.
"""

from dataclasses import fields

from .cgyro.outputs import CgyroOutputs
from .tglf.outputs import TglfOutputs

_PRIMARY = {
    "TGLF": {
        "gbflux": (
            "out.tglf.gbflux",
            "species,quantity",
            "gyro-Bohm",
            "core_transport.model[].profiles_1d",
            "local SI flux units and r/a -> rho_tor_norm",
        ),
        "eigenvalue_spectrum": (
            "out.tglf.eigenvalue_spectrum",
            "ky,mode",
            "c_s/a",
            "gyrokinetics_local.linear",
            "GKDB rates and ion-negative omega",
        ),
        "ky_spectrum": (
            "out.tglf.ky_spectrum",
            "ky",
            "ky*rho_s",
            "gyrokinetics_local",
            "GKDB perpendicular wavevector",
        ),
        "sum_flux_spectrum": (
            "out.tglf.sum_flux_spectrum",
            "species,field,ky,quantity",
            "weighted gyro-Bohm contribution",
            "gyrokinetics_local.non_linear",
            "sum over ky, not density; GKDB flux rescaling",
        ),
    },
    "CGYRO": {
        "grid": (
            "out.cgyro.grids",
            "p,theta,energy,xi,ky",
            "native grid",
            "gyrokinetics_local",
            "audited subset only; retain full native grid",
        ),
        "equilibrium": (
            "out.cgyro.equilibrium",
            "species,geometry",
            "native normalization",
            "gyrokinetics_local",
            "audited subset only; retain geometry/calibration",
        ),
        "frequency": (
            "bin.cgyro.freq",
            "ky,time",
            "c_s/a",
            "gyrokinetics_local.linear",
            "native ion-direction sign -> ion-negative, GKDB rate",
        ),
        "growth_rate": (
            "bin.cgyro.freq",
            "ky,time",
            "c_s/a",
            "gyrokinetics_local.linear",
            "GKDB rate; convergence required for interpretation",
        ),
        "ballooning": (
            "bin.cgyro.phib/aparb/bparb",
            "extended theta",
            "complex native field",
            "gyrokinetics_local.linear",
            "phi only audited; retain unsupported fields",
        ),
        "flux": (
            "bin.cgyro.ky_flux",
            "species,moment,field,ky,time",
            "gyro-Bohm contribution",
            "gyrokinetics_local.non_linear; core_transport",
            "explicit selected-window mean, GKDB/SI scales",
        ),
    },
}
_NATIVE = {
    "TGLF": {
        "field_spectrum",
        "intensity_spectrum",
        "density_spectrum",
        "temperature_spectrum",
        "nete_crossphase_spectrum",
        "nsts_crossphase_spectrum",
        "ql_flux_spectrum",
        "ql_intensity_spectrum",
        "width_spectrum",
        "spectral_shift_spectrum",
        "ave_p0_spectrum",
        "saturation_parameters",
        "spectral_species",
    },
    "CGYRO": {"time", "time_error", "record_index", "ion_direction"},
}
_TGLF_AXES = {
    "field_spectrum": "ky,mode,(vector,phi,Apar,Bpar)",
    "intensity_spectrum": "species,ky,mode,(density,temperature,parallel_velocity,parallel_energy)",
    "density_spectrum": "ky,species; sqrt(mode-summed intensity)",
    "temperature_spectrum": "ky,species; sqrt(mode-summed intensity)",
    "nete_crossphase_spectrum": "ky,mode",
    "nsts_crossphase_spectrum": "species,ky,mode",
    "ql_flux_spectrum": "species,field,mode,ky,(particle,energy,toroidal_stress,parallel_stress,exchange)",
    "ql_intensity_spectrum": "species,ky,mode,moment",
    "width_spectrum": "ky",
    "spectral_shift_spectrum": "ky",
    "ave_p0_spectrum": "ky",
    "saturation_parameters": "named scalar parameters",
    "spectral_species": "1-based native species indices",
}

_META = {
    "directory",
    "grid",
    "precision",
    "version",
    "errors",
    "files",
    "info",
    "exit_message",
}


def output_audit():
    """List every parsed output, class, interpretation, retention and plot route.

    Storage A/B/C is orthogonal to transformation status and qualification.
    No output is declared derivable merely because an approximation exists.
    """
    records = []
    for solver, container in [("TGLF", TglfOutputs), ("CGYRO", CgyroOutputs)]:
        for f in fields(container):
            name = f.name
            if name in _PRIMARY[solver]:
                source, axes, unit, destination, conversion = _PRIMARY[solver][name]
                cls = "A"
                reason = "faithful audited subset; preserve native source for lossless reconstruction"
            elif solver == "CGYRO" and name == "record_index":
                source = "out.cgyro.time"
                axes = "raw print-record indices"
                unit = "index"
                destination = None
                conversion = "first-strictly-increasing restart selection"
                cls = "B"
                reason = "reproducible from retained raw time table; preserve selection provenance"
            elif name in _NATIVE[solver]:
                source = (
                    (
                        "out.tglf."
                        + {
                            "ql_flux_spectrum": "QL_flux_spectrum",
                            "ql_intensity_spectrum": "QL_intensity_spectrum",
                            "saturation_parameters": "scalar_saturation_parameters",
                        }.get(name, name)
                    )
                    if solver == "TGLF"
                    else "out.cgyro.time / info"
                )
                axes = _TGLF_AXES.get(name, "raw print record or scalar provenance")
                unit = (
                    "rad"
                    if "crossphase" in name
                    else "native CGYRO a/c_s"
                    if name == "time"
                    else "native solver normalization (see container definition)"
                )
                if name == "spectral_species":
                    source = "species index headers in out.tglf.intensity_spectrum / QL spectra"
                destination = None
                conversion = "none"
                cls = "C"
                reason = "not reproducible from standardized averaged fluxes/eigenvalues; retain source"
            elif name in _META:
                source = "native metadata"
                axes = "metadata"
                unit = "not applicable"
                destination = "code metadata or sidecar"
                conversion = "provenance"
                cls = "C"
                reason = "retain settings, precision, revision and source evidence"
            else:
                source = name
                axes = "review_required"
                unit = "review_required"
                destination = None
                conversion = "unsupported"
                cls = "C"
                reason = "review_required: new parser field; retain losslessly"
            records.append(
                {
                    "solver": solver,
                    "quantity": name,
                    "storage_class": cls,
                    "native_source": source,
                    "axes": axes,
                    "native_units": unit,
                    "imas_destination": destination,
                    "transformation": conversion,
                    "retention": reason,
                    "derivation_dependencies": [
                        "native source",
                        "effective input",
                        "normalization",
                        "axis/sign conventions",
                    ],
                    "qualification": [
                        "missing data separate from zero",
                        "model assumptions",
                        "convergence or selected-window statistics where relevant",
                    ],
                    "owner": f"vaft.code.gacode.{solver.lower()}",
                    "plot_route": "registered gyrokinetics/turbulent_transport for audited quantities; vaft.plot.gyrokinetics for native-only diagnostics",
                    "evidence": "existing mapping audit and native fixture tests; see Gyrokinetic_results.md",
                }
            )
    records.append(
        {
            "solver": "CGYRO",
            "quantity": "complex_field_history",
            "storage_class": "C",
            "native_source": "bin.cgyro.kxky_phi/apar/bpar",
            "axes": "radial slot,saved theta,ky,raw time",
            "native_units": "complex native CGYRO field",
            "imas_destination": None,
            "transformation": "restart-aligned memory map",
            "retention": "irreducible native history; averages cannot reconstruct phases",
            "derivation_dependencies": [
                "grid",
                "precision",
                "restart-aware time",
                "Fourier convention",
            ],
            "qualification": [
                "saved-theta coverage",
                "missing field",
                "partial records",
            ],
            "owner": "vaft.code.gacode.cgyro",
            "plot_route": "plot_gyrokinetics_field_potential",
            "evidence": "b493397 upstream array equality and FFT oracle on real 39915 fixture",
        }
    )
    for name, dependencies in [
        ("zonal_fraction", ["phi", "ky", "spectral weights"]),
        ("zonal_shear_proxy", ["phi", "kx", "ky"]),
        (
            "window_statistics",
            ["native flux history", "simulation time", "window", "gap/block policy"],
        ),
        (
            "spatial_field",
            [
                "native complex field",
                "p/ky",
                "box length",
                "saved theta",
                "Fourier sign and half-sum",
            ],
        ),
    ]:
        records.append(
            {
                "solver": "shared",
                "quantity": name,
                "storage_class": "B",
                "native_source": None,
                "axes": "method dependent",
                "native_units": "explicit in process result",
                "imas_destination": None,
                "transformation": "derived",
                "retention": "retain dependencies and method, not a fabricated IDS field",
                "derivation_dependencies": dependencies,
                "qualification": [
                    "source applicability",
                    "sampling",
                    "method assumptions",
                ],
                "owner": "vaft.process.gyrokinetics",
                "plot_route": "prepared native/derived plot",
                "evidence": "analytical tests; spatial field also native upstream FFT oracle",
            }
        )
    return records


def mapping_transformations():
    """Expose the existing leaf-level mapping evidence without replacing its registry."""
    from vaft.machine_mapping.gyrokinetics import MAPPING_AUDIT, TGLF_MAPPING_AUDIT

    return {"CGYRO": dict(MAPPING_AUDIT), "TGLF": dict(TGLF_MAPPING_AUDIT)}
