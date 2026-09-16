"""Synthetic ODS factories for the canonical computed plots sample 39915 cannot build.

Every factory takes the packaged 39915 sample ODS (deep-copied before any edit) and
returns an ODS that `vaft.plot.backend.recipes.build_model(name, normalize_entries(ods))`
accepts.  Factories that do not need the sample ignore it.  OPTIONS lists the
build_model keyword options a name needs on top of the ODS.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import numpy as np
from omas import ODS

# ---------------------------------------------------------------------------
# nbi_profile_* -- adapted from test/test_nbi_registered_plots.py (fixture `ods`)
# ---------------------------------------------------------------------------
_ZONES = 4


def make_nbi(_sample: ODS) -> ODS:
    from vaft.code.nubeam.outputs import (
        NUBEAMFluxSurfaceAverages,
        NUBEAMOutputs,
        NUBEAMRadialGrid,
    )
    from vaft.machine_mapping.core_sources import core_sources_from_nubeam

    outputs = NUBEAMOutputs(
        workdir=Path("/nonexistent/nubeam-run"),  # only stored, never opened
        runid="TESTRUN",
        profiles={
            "pbe": np.array([100.0, 200.0, 300.0, 400.0]),
            "pbi": np.array([10.0, 20.0, 30.0, 40.0]),
            "curbeam": np.array([1.0, 2.0, 3.0, 4.0]),
        },
        grid=NUBEAMRadialGrid(
            rho=np.linspace(0.0, 1.0, _ZONES + 1),
            volume=np.linspace(0.0, 8.0, _ZONES + 1),
            area=np.linspace(0.0, 2.0, _ZONES + 1),
        ),
        flux_surface=NUBEAMFluxSurfaceAverages(
            f=np.full(_ZONES + 1, 1.0),
            gm1=np.full(_ZONES + 1, 2.0 * np.pi),
            gm5=np.full(_ZONES + 1, 4.0),
        ),
    )
    out = ODS()
    core_sources_from_nubeam(out, outputs, b0=2.0)
    return out


# ---------------------------------------------------------------------------
# interferometer_spectrogram -- no test builds a channel (test_spectrogram_methods.py
# only checks the option is honoured); shape from _build_interferometer_spectrogram.
# ---------------------------------------------------------------------------
def make_interferometer(_sample: ODS) -> ODS:
    ods = ODS(consistency_check=False)
    fs = 1.0e6
    t = np.arange(8192) / fs + 0.30
    trend = 1.0e19 * np.clip((t - 0.30) / 0.004, 0.0, 1.0)
    wiggle = 5.0e16 * np.sin(2 * np.pi * 20.0e3 * t)
    ods["interferometer.channel.0.n_e_line.data"] = trend + wiggle
    ods["interferometer.channel.0.n_e_line.time"] = t
    return ods


# ---------------------------------------------------------------------------
# passive_structure_time_current / magnetics_overview_* -- adapted from
# test/test_vacuum_field_map.py (`solved` fixture) and the tutorial session 02
# spine (test/test_tutorial_session_02.py): the sample with its eddy currents solved.
# ---------------------------------------------------------------------------
_SOLVED: ODS | None = None


def make_eddy_solved(sample: ODS) -> ODS:
    global _SOLVED
    if _SOLVED is None:
        import vaft.omas

        ods = copy.deepcopy(sample)
        vaft.omas.compute_eddy_currents(ods, [], [])
        _SOLVED = ods
    return _SOLVED


# ---------------------------------------------------------------------------
# impa_time_field / soft_x_rays_geometry_lines_of_sight / chease_overview_profile_validity
# already build from the raw packaged sample (verified); identity factory.
# (test_layout_contract.py, test_plot_machine_overview.py, test_chease_validation_plots.py)
# ---------------------------------------------------------------------------
def make_identity(sample: ODS) -> ODS:
    return sample


# ---------------------------------------------------------------------------
# coil_3d_* -- adapted from test/test_machine_mapping_coils_non_axisymmetric.py
# (`geometry_ods` fixture)
# ---------------------------------------------------------------------------
def make_coils_3d(_sample: ODS) -> ODS:
    from vaft.machine_mapping.coils_non_axisymmetric import coils_non_axisymmetric

    ods = ODS()
    coils_non_axisymmetric(ods)
    return ods


# ---------------------------------------------------------------------------
# pf_plasma_geometry_poloidal -- adapted from test/test_pf_plasma.py (_four_filaments)
# ---------------------------------------------------------------------------
def make_pf_plasma(_sample: ODS) -> ODS:
    from vaft.omas.pf_plasma import set_plasma_elements

    ods = ODS()
    time = np.array([0.30, 0.31, 0.32])
    r = np.array([0.40, 0.40, 0.40, 0.55])
    z = np.array([0.00, 0.15, -0.15, 0.00])
    currents = np.outer([0.5, 0.15, 0.15, 0.2], [0.0, 8.0e4, 6.0e4])
    set_plasma_elements(
        ods, r, z, width=0.01, height=0.01, currents=currents, time=time,
        code_name="VFIT",
        code_parameters="<parameters><source>filament</source></parameters>",
        comment="four filaments",
    )
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = np.array([0.1, 0.8, 0.8, 0.1])
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = np.array([-1.0, -1.0, 1.0, 1.0])
    return ods


# ---------------------------------------------------------------------------
# core_profiles on the sample equilibrium -- shape after the packaged 48224 kinetic
# sample (vaft/data/kineticEfit/ods_48224_300ms.json, used by
# test/test_neoclassical_comparison.py); leaves are synthetic parabolas on the
# equilibrium's own rho_tor_norm grid, at the equilibrium slice time.
# ---------------------------------------------------------------------------
def _with_core_profiles(sample: ODS, *, zeff: bool) -> ODS:
    ods = copy.deepcopy(sample)
    eq = "equilibrium.time_slice.0"
    rho = np.asarray(ods[f"{eq}.profiles_1d.rho_tor_norm"], dtype=float)
    shape = np.clip(1.0 - rho**2, 0.0, None)
    cp = "core_profiles.profiles_1d.0"
    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    ods["core_profiles.time"] = np.array([float(ods[f"{eq}.time"])])
    ods[f"{cp}.time"] = float(ods[f"{eq}.time"])
    ods[f"{cp}.grid.rho_tor_norm"] = rho
    ods[f"{cp}.grid.psi"] = np.asarray(ods[f"{eq}.profiles_1d.psi"], dtype=float)
    te = 20.0 + 180.0 * shape
    ne = 1.0e18 + 2.0e19 * shape
    ods[f"{cp}.electrons.temperature"] = te
    ods[f"{cp}.electrons.density"] = ne
    ods[f"{cp}.electrons.density_thermal"] = ne
    ods[f"{cp}.ion.0.label"] = "H"
    ods[f"{cp}.ion.0.z_ion"] = 1.0
    ods[f"{cp}.ion.0.element.0.a"] = 1.0
    ods[f"{cp}.ion.0.element.0.z_n"] = 1.0
    ods[f"{cp}.ion.0.element.0.atoms_n"] = 1
    ods[f"{cp}.ion.0.temperature"] = 0.8 * te
    ods[f"{cp}.ion.0.density"] = ne
    ods[f"{cp}.ion.0.density_thermal"] = ne
    ods[f"{cp}.pressure_thermal"] = 1.602e-19 * (ne * te + ne * 0.8 * te)
    if zeff:
        ods[f"{cp}.zeff"] = np.full(rho.size, 1.5)
    return ods


def make_core_profiles(sample: ODS) -> ODS:
    return _with_core_profiles(sample, zeff=False)


def make_power_balance(sample: ODS) -> ODS:
    """summary_time_power_balance -- shape after the packaged 48224 kinetic sample
    (test/test_omas_plot_adapters.py only asserts 39915 does NOT offer it).

    compute_power_balance integrates magnetic energy over every equilibrium slice,
    so the sample's trailing Ip=0 vacuum slice (psi_axis == psi_boundary, empty mask)
    is dropped; each kept slice gets profiles_1d.volume (the volume updater fills
    global_quantities.volume from its last point) and a core_profiles slice.
    """
    ods = copy.deepcopy(sample)
    n = len(ods["equilibrium.time_slice"])
    keep = [i for i in range(n) if float(ods[f"equilibrium.time_slice.{i}.global_quantities.ip"]) != 0.0]
    for i in reversed([i for i in range(n) if i not in keep]):
        del ods[f"equilibrium.time_slice.{i}"]
    ods["equilibrium.time"] = np.asarray(ods["equilibrium.time"], dtype=float)[keep]
    for leaf in ("equilibrium.vacuum_toroidal_field.b0",):
        if leaf in ods and np.asarray(ods[leaf]).size == n:
            ods[leaf] = np.asarray(ods[leaf], dtype=float)[keep]
    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    ods["core_profiles.time"] = np.asarray(ods["equilibrium.time"], dtype=float)
    for k in range(len(keep)):
        eq = f"equilibrium.time_slice.{k}"
        rho = np.asarray(ods[f"{eq}.profiles_1d.rho_tor_norm"], dtype=float)
        # plasma volume from the boundary outline: shoelace area x 2 pi R_centroid
        br = np.asarray(ods[f"{eq}.boundary.outline.r"], dtype=float)
        bz = np.asarray(ods[f"{eq}.boundary.outline.z"], dtype=float)
        area = 0.5 * abs(np.dot(br, np.roll(bz, -1)) - np.dot(bz, np.roll(br, -1)))
        ods[f"{eq}.profiles_1d.volume"] = 2.0 * np.pi * float(br.mean()) * area * rho**2
        shape = np.clip(1.0 - rho**2, 0.0, None)
        cp = f"core_profiles.profiles_1d.{k}"
        te = 20.0 + 180.0 * shape
        ne = 1.0e18 + 2.0e19 * shape
        ods[f"{cp}.time"] = float(ods[f"{eq}.time"])
        ods[f"{cp}.grid.rho_tor_norm"] = rho
        ods[f"{cp}.grid.psi"] = np.asarray(ods[f"{eq}.profiles_1d.psi"], dtype=float)
        ods[f"{cp}.electrons.temperature"] = te
        ods[f"{cp}.electrons.density"] = ne
        ods[f"{cp}.electrons.density_thermal"] = ne
        ods[f"{cp}.ion.0.label"] = "H"
        ods[f"{cp}.ion.0.z_ion"] = 1.0
        ods[f"{cp}.ion.0.element.0.a"] = 1.0
        ods[f"{cp}.ion.0.element.0.z_n"] = 1.0
        ods[f"{cp}.ion.0.element.0.atoms_n"] = 1
        ods[f"{cp}.ion.0.temperature"] = 0.8 * te
        ods[f"{cp}.ion.0.density"] = ne
        ods[f"{cp}.ion.0.density_thermal"] = ne
        ods[f"{cp}.pressure_thermal"] = 1.602e-19 * (ne * te + ne * 0.8 * te)
    return ods


def make_neoclassical(sample: ODS) -> ODS:
    return _with_core_profiles(sample, zeff=True)


# ---------------------------------------------------------------------------
# camera_visible_* -- adapted from test/test_camera_visible_image_api.py (_with_frames):
# 39915 has a packaged pose, so frames added at the equilibrium slice time project.
# ---------------------------------------------------------------------------
_ROWS, _COLS = 1024, 1280


def make_camera(sample: ODS) -> ODS:
    ods = copy.deepcopy(sample)
    times = [float(ods[f"equilibrium.time_slice.{i}.time"]) for i in range(len(ods["equilibrium.time_slice"]))]
    ods["camera_visible.channel.0.name"] = "Fast Camera"
    ods["camera_visible.channel.0.detector.0.lines_n"] = _ROWS
    ods["camera_visible.channel.0.detector.0.columns_n"] = _COLS
    for i, t in enumerate(times[:3]):
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.image_raw"] = np.full((_ROWS, _COLS), 100.0 + i)
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.time"] = t
    return ods


# ---------------------------------------------------------------------------
# ntms_time_delta_prime -- adapted from test/test_ntms_delta_prime_plot.py (_ods)
# ---------------------------------------------------------------------------
def make_ntms(_sample: ODS) -> ODS:
    surfaces_by_time = [[(1, 2, 0.5), (1, 3, -1.5)], [(1, 2, 0.8), (1, 3, -1.2)]]
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 39915
    ods["ntms.ids_properties.homogeneous_time"] = 1
    ods["ntms.time"] = np.array([0.316, 0.317])
    for index, surfaces in enumerate(surfaces_by_time):
        for position, (n_tor, m_pol, value) in enumerate(surfaces):
            entry = ods["ntms"]["time_slice"][index]["mode"][position]
            entry["n_tor"] = int(n_tor)
            entry["m_pol"] = int(m_pol)
            entry["deltaw"][0]["name"] = "classical"
            entry["deltaw"][0]["value"] = float(value)
    return ods


# ---------------------------------------------------------------------------
# mhd_linear_* -- adapted from test/test_mhd_linear_validation_plots.py
# (_mhd_linear_ods + _add_eigenfunction)
# ---------------------------------------------------------------------------
def _add_eigenfunction(ods, base, *, n_tor, harmonics=(-3.0, -2.0, -1.0), n_psi=16):
    psi_n = np.linspace(0.05, 0.98, n_psi)
    m = np.asarray(harmonics, dtype=float)
    envelope = np.exp(-((psi_n[:, None] - np.linspace(0.3, 0.8, m.size)[None, :]) ** 2) / 0.01)
    amplitude = envelope * np.linspace(1.0, 3.0, m.size)[None, :]
    ods[f"{base}.plasma.grid_type.index"] = -1
    ods[f"{base}.plasma.grid_type.name"] = "inverse_psi_hamada_fourier"
    ods[f"{base}.plasma.grid.dim1"] = psi_n
    ods[f"{base}.plasma.grid.dim2"] = m
    ods[f"{base}.plasma.displacement_perpendicular.real"] = amplitude
    ods[f"{base}.plasma.displacement_perpendicular.imaginary"] = np.zeros_like(amplitude)
    q = 1.0 + 7.0 * psi_n**2
    singular_factor = m[None, :] - n_tor * q[:, None]
    ods[f"{base}.plasma.b_field_perturbed.coordinate1.real"] = np.zeros_like(amplitude)
    ods[f"{base}.plasma.b_field_perturbed.coordinate1.imaginary"] = singular_factor * amplitude
    ods[f"{base}.m_pol_dominant"] = float(m[-1])


def make_mhd_linear(_sample: ODS) -> ODS:
    energies = {1: [-0.4, -0.5, -0.6], 2: [0.2, 0.15, 0.1]}
    times = (0.316, 0.317, 0.318)
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 41234
    ods["mhd_linear.ids_properties.homogeneous_time"] = 1
    ods["mhd_linear.time"] = list(times)
    ods["mhd_linear.code.parameters"] = '<parameters><eigenfunction_grid radial_stride="4"/></parameters>'
    for slice_index in range(len(times)):
        for position, n_tor in enumerate(sorted(energies, reverse=True)):
            base = f"mhd_linear.time_slice.{slice_index}.toroidal_mode.{position}"
            ods[f"{base}.n_tor"] = n_tor
            ods[f"{base}.energy_perturbed"] = energies[n_tor][slice_index]
            _add_eigenfunction(ods, base, n_tor=n_tor)
    return ods


# ---------------------------------------------------------------------------
# chease_overview_refinement_summary -- adapted from test/test_chease_validation_plots.py
# (_chease_ods)
# ---------------------------------------------------------------------------
_COMPARISON_METRICS = {
    "0": {
        "q_rms_rel": 0.01, "pressure_rms_rel": 0.02, "pprime_rms_rel": 0.03,
        "ffprim_rms_rel": 0.04, "psi_axis_abs_diff": 1e-4, "psi_boundary_abs_diff": 2e-4,
        "boundary_r_rms": 1e-3, "boundary_z_rms": 2e-3, "boundary_rz_rms": 3e-3,
        "boundary_points": 64.0, "current_abs_diff": 10.0, "current_rel_diff": 0.001,
    },
    "1": {
        "q_rms_rel": 0.05, "pressure_rms_rel": 0.06, "pprime_rms_rel": 0.07,
        "ffprim_rms_rel": 0.08, "psi_axis_abs_diff": 3e-4, "psi_boundary_abs_diff": 4e-4,
        "boundary_r_rms": 4e-3, "boundary_z_rms": 5e-3, "boundary_rz_rms": 6e-3,
        "boundary_points": 64.0, "current_abs_diff": 20.0, "current_rel_diff": 0.002,
    },
}
_RECORDS_SUMMARY = [
    {"input": "g041234.00300", "status": "completed"},
    {"input": "g041234.00310", "status": "completed"},
]


def make_chease(_sample: ODS) -> ODS:
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 41234
    ods["equilibrium.code.name"] = "chease"
    ods["equilibrium.code.library.0.name"] = "chease"
    ods["equilibrium.code.parameters"] = json.dumps(
        {"comparison_metrics": _COMPARISON_METRICS, "records_summary": _RECORDS_SUMMARY}
    )
    times = np.array([0.300, 0.310])
    ods["equilibrium.time"] = times
    for index, (q0, q95) in enumerate(((1.05, 3.2), (1.10, 3.5))):
        root = f"equilibrium.time_slice.{index}"
        ods[f"{root}.time"] = float(times[index])
        ods[f"{root}.profiles_1d.q"] = np.linspace(q0, q95, 33)
        ods[f"{root}.profiles_1d.pressure"] = np.linspace(2.0e4, 0.0, 33)
        ods[f"{root}.global_quantities.q_axis"] = float(q0)
        ods[f"{root}.global_quantities.q_95"] = float(q95)
    return ods


# ---------------------------------------------------------------------------
SYNTHETIC: dict[str, Callable[[ODS], ODS]] = {
    "nbi_profile_electron_heating": make_nbi,
    "nbi_profile_ion_heating": make_nbi,
    "nbi_profile_current_drive": make_nbi,
    "interferometer_spectrogram": make_interferometer,
    "passive_structure_time_current": make_eddy_solved,
    "impa_time_field": make_identity,
    "soft_x_rays_geometry_lines_of_sight": make_identity,
    "coil_3d_geometry3d": make_coils_3d,
    "coil_3d_geometry_topview": make_coils_3d,
    "pf_plasma_geometry_poloidal": make_pf_plasma,
    "neoclassical_profile_bootstrap_current": make_neoclassical,
    "electron_temperature_field": make_core_profiles,
    "electron_density_field": make_core_profiles,
    "summary_time_power_balance": make_power_balance,
    "camera_visible_image": make_camera,
    "camera_visible_image_frame": make_camera,
    "camera_visible_image_efit_overlay": make_camera,
    "camera_visible_image_field_line": make_camera,
    "camera_visible_animation_frames": make_camera,
    "ntms_time_delta_prime": make_ntms,
    "mhd_linear_time_energy_perturbed": make_mhd_linear,
    "mhd_linear_profile_displacement": make_mhd_linear,
    "mhd_linear_profile_b_field_perturbed": make_mhd_linear,
    "magnetics_overview_vacuum": make_eddy_solved,
    "magnetics_overview_plasma_residual": make_eddy_solved,
    "chease_overview_refinement_summary": make_chease,
    "chease_overview_profile_validity": make_identity,
}

#: build_model options a name needs beyond the ODS (factories cannot pass options).
OPTIONS: dict[str, dict] = {
    "camera_visible_image_field_line": {"field_line_start": (0.4, 0.0)},
}

#: Names no factory could make build, with the exact error.
UNSUPPORTED: dict[str, str] = {}


# ---------------------------------------------------------------------------
# camera_visible_image_fluctuation / _mhd_power / camera_visible_spectrogram --
# adapted from test/test_camera_fluctuation_plots.py (fixture `camera_ods`):
# 200 frames at 50 kfps with a 6 kHz oscillation on the right half.
# ---------------------------------------------------------------------------


def make_camera_fluctuation(_sample: ODS) -> ODS:
    from vaft.machine_mapping.camera_visible import (
        vfit_camera_visible_dynamic,
        vfit_camera_visible_static,
    )

    rows, cols, n_frames, frame_rate = 8, 12, 200, 50_000.0
    time = np.arange(n_frames) / frame_rate + 0.3
    background = 80.0 + 20.0 * np.sin(2 * np.pi * 300.0 * time)
    frames = np.broadcast_to(background[:, None, None], (n_frames, rows, cols)).copy()
    frames[:, :, cols // 2:] += (15.0 * np.sin(2 * np.pi * 6_000.0 * time))[:, None, None]
    out = ODS()
    out["dataset_description.data_entry.pulse"] = 39915
    vfit_camera_visible_static(out, lines_n=rows, columns_n=cols, exposure_time_s=1.91e-5)
    vfit_camera_visible_dynamic(out, images=list(frames.astype(int)), times_s=list(time))
    return out


for _name in ("camera_visible_image_fluctuation", "camera_visible_image_mhd_power", "camera_visible_spectrogram"):
    SYNTHETIC[_name] = make_camera_fluctuation


# ---------------------------------------------------------------------------
# mhd_linear_profile_resonant_flux / _island_width -- adapted from
# test/test_gpec_resonant_plots.py (fixture `mapped`): an ODS built the way a
# real ideal-GPEC run reaches one, from the packaged netCDF fixtures.
# ---------------------------------------------------------------------------


def make_gpec_resonant(_sample: ODS) -> ODS:
    import tempfile

    from gpec_nc_fixtures import write_control_nc, write_cylindrical_nc, write_profile_nc
    from vaft.machine_mapping.gpec_ideal import gpec_ideal

    out = ODS(consistency_check=False)
    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir)
        write_control_nc(path, n=1)
        write_cylindrical_nc(path, n=1)
        write_profile_nc(path, n=1, rational_q=(2.0, 3.0))
        gpec_ideal(out, str(path), {"modes": [1]})
    return out


for _name in ("mhd_linear_profile_resonant_flux", "mhd_linear_profile_island_width"):
    SYNTHETIC[_name] = make_gpec_resonant
