import copy
import gzip
import json

import numpy as np
import pytest
from scipy import ndimage, signal

from vaft.database import raw as raw_db
from vaft.database._local import load_ods
from omas import ODS

from vaft.omas import save as save_ods
from vaft.omas.vest_upstream import (
    _canonical_diagnostics_time,
    _validate_diagnostics_time_coordinates,
    archive_raw_source,
    build_diagnostics_ods,
    build_eddy_ods,
    build_mhd_linear_ods,
    build_static_ods,
    machine_era_for_shot,
    write_stage_product,
)
from vaft.machine_mapping.pf_active import vfit_pf
from vaft.process.magnetics import VestMagneticsProcessingConfig, vest_equilibrium_magnetics_signals


@pytest.mark.parametrize(
    ("shot", "expected"),
    [
        (43016, "vest-pre-43017-pf1906"),
        (43017, "vest-43017-45957-pf1906"),
        (45957, "vest-43017-45957-pf1906"),
        (45958, "vest-45958-45966-pf2507"),
        (45966, "vest-45958-45966-pf2507"),
        (45967, "vest-45967-plus-pf2507"),
    ],
)
def test_machine_era_boundaries_are_explicit(shot, expected):
    assert machine_era_for_shot(shot).name == expected


def test_default_diagnostics_grid_is_half_open_and_25_khz():
    time = _canonical_diagnostics_time(0.26, 0.36, 4e-5)

    assert time.size == 2_500
    assert time[0] == pytest.approx(0.26)
    assert time[-1] < 0.36
    np.testing.assert_allclose(np.diff(time), 4e-5)


def test_native_mirnov_coordinate_marks_magnetics_heterogeneous():
    processed_time = _canonical_diagnostics_time(0.26, 0.36, 4e-5)
    native_time = np.arange(0.26, 0.36, 4e-6)
    ods = ODS(consistency_check=False)
    ods["magnetics.time"] = processed_time
    ods["magnetics.b_field_pol_probe.0.voltage.time"] = native_time
    ods["magnetics.b_field_pol_probe.0.voltage.data"] = np.ones(native_time.size)

    metadata = _validate_diagnostics_time_coordinates(
        ods, processed_time, tstart=0.26, tend=0.36, dt=4e-5
    )

    assert ods["magnetics.ids_properties.homogeneous_time"] == 0
    assert metadata["native_mirnov"][0]["sampling_rate"] == pytest.approx(250_000.0)


def test_static_product_is_not_a_reference_shot_container():
    ods, manifest = build_static_ods("vest-45958-45966-pf2507")

    assert set(ods.keys()) == {
        "wall",
        "pf_active",
        "pf_passive",
        "em_coupling",
        "magnetics",
        "tf",
    }
    assert "dataset_description" not in ods
    assert "pf_active.time" not in ods
    assert "pf_passive.time" not in ods
    assert "pf_passive.loop.0.current" not in ods
    assert np.shape(ods["em_coupling.mutual_passive_passive"]) == (950, 950)
    assert manifest["channel_status"]["pf_active"]["disabled_channels"] == [
        "PF2",
        "PF3",
        "PF4",
        "PF7",
        "PF8",
    ]


def test_static_product_marks_time_independent_ids_as_such():
    """No IDS in the static product ever gets a `.time` node.

    Per the IMAS DD, `homogeneous_time` must be 2 when only constant/static
    nodes are filled. `pf_active`/`magnetics`/`tf` default to 1 in their
    shared static builders (correct once the per-shot diagnostics stage adds
    `.time`), and `pf_passive` inherits 1 from the packaged asset -- both
    need an explicit override here since this product never adds `.time`.
    `wall` and `em_coupling` have no dynamic counterpart at all, so they are
    fixed at the source instead.
    """
    ods, _ = build_static_ods("vest-45958-45966-pf2507")

    for ids_name in ("wall", "em_coupling", "pf_active", "pf_passive", "magnetics", "tf"):
        assert ods[f"{ids_name}.ids_properties.homogeneous_time"] == 2
        assert f"{ids_name}.time" not in ods


def test_static_wall_completeness():
    """The wall's limiter description is self-describing without its manifest.

    `description_2d.type` is a distinct node from `limiter.type` and was
    never populated; the outline is a genuinely closed polygon but the DD's
    `closed` flag was never set to say so.
    """
    ods, _ = build_static_ods("vest-45958-45966-pf2507")

    description = ods["wall.description_2d.0"]
    assert description["type.index"] == 1
    assert "vessel" not in description

    n_units = len(description["limiter.unit"])
    assert n_units >= 1
    for unit_index in range(n_units):
        unit = description[f"limiter.unit.{unit_index}"]
        r = unit["outline.r"]
        z = unit["outline.z"]
        assert unit["closed"] == int(r[0] == r[-1] and z[0] == z[-1])
        assert unit["name"]


def _write_raw_dump(path, shot, fields, pulse_datetime=None):
    payload = {
        "shot": shot,
        "fields": {
            str(field): {"data": values, "type": "slow"}
            for field, values in fields.items()
        },
    }
    if pulse_datetime is not None:
        payload["pulse_datetime"] = pulse_datetime
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_unavailable_diagnostic_does_not_corrupt_valid_sibling(tmp_path):
    shot = 43017
    raw = tmp_path / "raw.json.gz"
    _write_raw_dump(raw, shot, {12: np.linspace(1.0, 2.0, 200).tolist()})
    static_path = tmp_path / "static.json.gz"
    static_manifest = tmp_path / "static-manifest.json"
    static, manifest = build_static_ods(machine_era_for_shot(shot).name)
    write_stage_product(
        static,
        manifest,
        output=static_path,
        metadata=static_manifest,
    )

    ods, diagnostics_manifest = build_diagnostics_ods(
        shot=shot,
        raw_source=raw,
        static_ods=static_path,
        tstart=0.0,
        tend=0.005,
        dt=4e-5,
    )

    assert diagnostics_manifest["status"] == "partial"
    assert diagnostics_manifest["channel_status"]["barometry"]["status"] == "success"
    for name in ("pf_active", "spectrometer_uv", "tf", "magnetics"):
        assert diagnostics_manifest["channel_status"][name]["status"] == "unavailable"
    # IMPA extends magnetics, so it cannot be mapped when magnetics is missing;
    # it must say so rather than fail the build.
    assert diagnostics_manifest["channel_status"]["impa"]["status"] == "unavailable"
    assert diagnostics_manifest["channel_status"]["impa"]["missing_channels"] == [
        114, 115, 116, 117, 118, 119, 120, 121
    ]
    assert {216, 217, 218}.issubset(
        diagnostics_manifest["channel_status"]["magnetics"]["missing_channels"]
    )
    assert {
        "magnetics:field-216",
        "magnetics:field-217",
        "magnetics:field-218",
    }.issubset(diagnostics_manifest["quality_summary"]["missing"])
    assert "barometry" in ods
    assert "tf" not in ods
    assert np.all(np.asarray(ods["barometry.gauge.0.pressure.data"]) > 0)


def test_diagnostics_ods_carries_the_raw_dumps_pulse_datetime(tmp_path):
    """A raw dump carrying pulse_datetime (issue #126) must reach dataset_description.

    dump_all_raw_signals_for_shot() writes pulse_datetime from the SQL `shot`
    table when it runs against live SQL; build_diagnostics_ods() must read it
    back out of the archived dump and set dataset_description.pulse_time_begin,
    with no live SQL call of its own.
    """
    shot = 43017
    raw = tmp_path / "raw.json.gz"
    _write_raw_dump(
        raw,
        shot,
        {13: np.linspace(1.0, 2.0, 200).tolist()},
        pulse_datetime="2026-05-01T08:30:00",
    )
    static_path = tmp_path / "static.json.gz"
    static, manifest = build_static_ods(machine_era_for_shot(shot).name)
    write_stage_product(
        static, manifest, output=static_path, metadata=tmp_path / "static-manifest.json"
    )

    ods, _ = build_diagnostics_ods(
        shot=shot,
        raw_source=raw,
        static_ods=static_path,
        tstart=0.0,
        tend=0.005,
        dt=4e-5,
    )

    assert ods["dataset_description.pulse_time_begin"] == "2026-05-01T08:30:00"


def test_diagnostics_ods_leaves_pulse_time_unset_for_a_dump_without_one(tmp_path):
    """Older dumps written before pulse_datetime existed must not break the stage."""
    shot = 43017
    raw = tmp_path / "raw.json.gz"
    _write_raw_dump(raw, shot, {13: np.linspace(1.0, 2.0, 200).tolist()})
    static_path = tmp_path / "static.json.gz"
    static, manifest = build_static_ods(machine_era_for_shot(shot).name)
    write_stage_product(
        static, manifest, output=static_path, metadata=tmp_path / "static-manifest.json"
    )

    ods, _ = build_diagnostics_ods(
        shot=shot,
        raw_source=raw,
        static_ods=static_path,
        tstart=0.0,
        tend=0.005,
        dt=4e-5,
    )

    assert "dataset_description.pulse_time_begin" not in ods


def test_barometry_time_is_heterogeneous_not_homogeneous(tmp_path):
    """barometry stores time per-gauge, so homogeneous_time must be 0, not 1.

    Per the DD, homogeneous_time=1 means the IDS's dynamic quantities share a
    single time array at the IDS root (`<ids>.time`); barometry never writes
    one, only `gauge.0.pressure.time`. homogeneous_time=1 without a root
    `.time` claims a time base that does not exist -- the correct value for
    "time values are stored in the various time fields at lower levels" is 0.
    """
    shot = 43017
    raw = tmp_path / "raw.json.gz"
    _write_raw_dump(raw, shot, {12: np.linspace(1.0, 2.0, 200).tolist()})
    static_path = tmp_path / "static.json.gz"
    static, manifest = build_static_ods(machine_era_for_shot(shot).name)
    write_stage_product(
        static, manifest, output=static_path, metadata=tmp_path / "static-manifest.json"
    )

    ods, diagnostics_manifest = build_diagnostics_ods(
        shot=shot,
        raw_source=raw,
        static_ods=static_path,
        tstart=0.0,
        tend=0.005,
        dt=4e-5,
    )
    assert diagnostics_manifest["channel_status"]["barometry"]["status"] == "success"

    assert ods["barometry.ids_properties.homogeneous_time"] == 0
    assert "barometry.time" not in ods
    assert "barometry.gauge.0.pressure.time" in ods

    output_path = tmp_path / "diagnostics.json"
    metadata_path = tmp_path / "diagnostics-manifest.json"
    write_stage_product(ods, diagnostics_manifest, output=output_path, metadata=metadata_path)

    from omas import load_omas_json

    reloaded = load_omas_json(str(output_path), consistency_check=True)
    assert reloaded["barometry.ids_properties.homogeneous_time"] == 0


def test_eddy_ods_carries_no_stage_specific_impedance_cache_keys(tmp_path):
    """pf_passive comes from the curated static ODS, which never holds a cache.

    A previous stage-specific workaround stripped `pf_passive.{R,L,M}_mat` from
    the eddy product after the fact. Nothing writes those keys any more: the
    solver keeps the impedance matrices as locals, and the versioned static
    ODS carries only `pf_passive.{ids_properties,loop}`. This asserts the
    invariant holds without any stripping step, including a consistency-checked
    reload (those keys are not valid IMAS locations).
    """
    shot = 43017
    static_path = tmp_path / "static.json"
    static_manifest = tmp_path / "static-manifest.json"
    static, manifest = build_static_ods(machine_era_for_shot(shot).name)
    write_stage_product(static, manifest, output=static_path, metadata=static_manifest)

    diagnostics = ODS(consistency_check=False)
    time = np.linspace(0.0, 0.01, 50)
    # pf_active must carry real per-coil currents: build_eddy_ods does not
    # backfill it from the static ODS the way build_diagnostics_ods does.
    diagnostics["pf_active"] = copy.deepcopy(static["pf_active"])
    diagnostics["pf_active.time"] = time
    n_coils = len(diagnostics["pf_active.coil"])
    for coil_index in range(n_coils):
        diagnostics[f"pf_active.coil.{coil_index}.current.time"] = time
        diagnostics[f"pf_active.coil.{coil_index}.current.data"] = np.zeros_like(time)
    diagnostics["magnetics.ip.0.time"] = time
    diagnostics["magnetics.ip.0.data"] = 1e4 * np.sin(np.linspace(0.0, 1.0, 50))
    diagnostics_path = tmp_path / "diagnostics.json"
    save_ods(diagnostics, diagnostics_path)

    ods, eddy_manifest = build_eddy_ods(
        shot=shot,
        diagnostics_ods=diagnostics_path,
        static_ods=static_path,
        filament_r=[0.35, 0.35, 0.35],
        filament_z=[0.25, 0.0, -0.25],
        filament_fraction=[1 / 3, 1 / 3, 1 / 3],
        dt_sub=5e-5,
    )

    assert "pf_passive.R_mat" not in ods
    assert "pf_passive.L_mat" not in ods
    assert "pf_passive.M_mat" not in ods

    output_path = tmp_path / "eddy.json"
    metadata_path = tmp_path / "eddy-manifest.json"
    write_stage_product(ods, eddy_manifest, output=output_path, metadata=metadata_path)

    # A stray cache key is not a valid IMAS location, so a consistency-checked
    # reload is the strongest check that none survived.
    from omas import load_omas_json

    reloaded = load_omas_json(str(output_path), consistency_check=True)
    assert "pf_passive.R_mat" not in reloaded
    assert "pf_passive.L_mat" not in reloaded
    assert "pf_passive.M_mat" not in reloaded


def test_eddy_ods_updates_homogeneous_time_for_ids_that_gain_a_time_node(tmp_path):
    """pf_passive goes from independent (2) to homogeneous (1) once eddy adds `.time`.

    pf_passive is copied from the static product, which is independent
    (homogeneous_time=2, no `.time`). compute_eddy_currents() then adds
    `pf_passive.time`, so the flag must flip to 1 to match -- copying the
    static value forward unchanged would leave it claiming "no time" while a
    time array is actually present. wall/em_coupling never gain `.time`, so
    they must stay at 2.
    """
    shot = 43017
    static_path = tmp_path / "static.json"
    static_manifest = tmp_path / "static-manifest.json"
    static, manifest = build_static_ods(machine_era_for_shot(shot).name)
    write_stage_product(static, manifest, output=static_path, metadata=static_manifest)
    assert static["pf_passive.ids_properties.homogeneous_time"] == 2

    diagnostics = ODS(consistency_check=False)
    time = np.linspace(0.0, 0.01, 50)
    diagnostics["pf_active"] = copy.deepcopy(static["pf_active"])
    diagnostics["pf_active.time"] = time
    n_coils = len(diagnostics["pf_active.coil"])
    for coil_index in range(n_coils):
        diagnostics[f"pf_active.coil.{coil_index}.current.time"] = time
        diagnostics[f"pf_active.coil.{coil_index}.current.data"] = np.zeros_like(time)
    diagnostics["magnetics.ip.0.time"] = time
    diagnostics["magnetics.ip.0.data"] = 1e4 * np.sin(np.linspace(0.0, 1.0, 50))
    diagnostics_path = tmp_path / "diagnostics.json"
    save_ods(diagnostics, diagnostics_path)

    ods, _ = build_eddy_ods(
        shot=shot,
        diagnostics_ods=diagnostics_path,
        static_ods=static_path,
        filament_r=[0.35, 0.35, 0.35],
        filament_z=[0.25, 0.0, -0.25],
        filament_fraction=[1 / 3, 1 / 3, 1 / 3],
        dt_sub=5e-5,
    )

    assert "pf_passive.time" in ods
    assert ods["pf_passive.ids_properties.homogeneous_time"] == 1
    assert "wall.time" not in ods
    assert ods["wall.ids_properties.homogeneous_time"] == 2
    assert "em_coupling.time" not in ods
    assert ods["em_coupling.ids_properties.homogeneous_time"] == 2


def test_explicit_raw_archive_is_copied_with_machine_readable_provenance(tmp_path):
    source = tmp_path / "source.json.gz"
    output = tmp_path / "raw" / "output" / "vest_daq_raw.json.gz"
    _write_raw_dump(source, 39915, {13: [1.0, 2.0]})

    manifest = archive_raw_source(shot=39915, source=source, output=output)

    assert output.read_bytes() == source.read_bytes()
    assert manifest["source"] == {"kind": "archive", "name": source.name}
    assert manifest["output"]["sha256"]


def test_plain_raw_archive_is_normalized_to_the_gzip_workflow_product(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"shot": 39915, "fields": {"13": {"data": [1.0]}}}),
        encoding="utf-8",
    )
    output = tmp_path / "vest_daq_raw.json.gz"

    archive_raw_source(shot=39915, source=source, output=output)

    with gzip.open(output, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["shot"] == 39915
    assert set(payload["fields"]) == {"13"}


def test_live_raw_export_honors_a_plain_json_output(tmp_path, monkeypatch):
    output = tmp_path / "vest_daq_raw.json"

    def fake_dump(shot, path):
        assert shot == 39915
        assert path.endswith(".json.gz")
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump({"shot": shot, "fields": {"13": {"data": [1.0]}}}, handle)
        return True

    monkeypatch.setattr(raw_db, "dump_all_raw_signals_for_shot", fake_dump)

    manifest = archive_raw_source(shot=39915, output=output)

    assert json.loads(output.read_text(encoding="utf-8"))["shot"] == 39915
    assert manifest["source"] == {"kind": "vest-sql", "name": None}
    assert sorted(path.name for path in tmp_path.iterdir()) == [output.name]


def test_explicit_raw_archive_rejects_a_different_shot(tmp_path):
    source = tmp_path / "source.json.gz"
    _write_raw_dump(source, 39916, {13: [1.0, 2.0]})

    with pytest.raises(ValueError, match="shot mismatch"):
        archive_raw_source(
            shot=39915,
            source=source,
            output=tmp_path / "vest_daq_raw.json.gz",
        )


def test_stage_writer_produces_reloadable_gzip_ods(tmp_path):
    ods, manifest = build_static_ods("vest-pre-43017-pf1906")
    output = tmp_path / "static.json.gz"
    metadata = tmp_path / "manifest.json"
    write_stage_product(ods, manifest, output=output, metadata=metadata)

    reloaded, _ = load_ods(output)
    assert len(reloaded["pf_passive.loop"]) == 950
    recorded = json.loads(metadata.read_text(encoding="utf-8"))
    assert recorded["output"]["name"] == "static.json.gz"
    assert recorded["output"]["sha256"]


def test_stage_writer_is_byte_deterministic(tmp_path):
    ods, manifest = build_static_ods("vest-pre-43017-pf1906")
    outputs = []
    for index in range(2):
        output = tmp_path / str(index) / "static.json.gz"
        write_stage_product(
            ods,
            manifest,
            output=output,
            metadata=tmp_path / str(index) / "manifest.json",
        )
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]


def _write_gpec_output(
    workdir, module, mode, *, time_label="00319", mlow=-8, mhigh=16, **variables
):
    import numpy as np
    import xarray as xr

    run_dir = workdir / time_label / module / f"nn={mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{module}_output_n{mode}.nc"
    mpert = mhigh - mlow + 1
    attrs = {"mlow": mlow, "mhigh": mhigh, "mpert": mpert, "mband": 0, "n": mode}
    if module == "dcon":
        ds = xr.Dataset(
            {"W_t_eigenvalue": (("mode", "i"), [[variables["w_t"], 0.0]])},
            coords={"i": [0, 1], "mode": [1]},
            attrs=attrs,
        )
    else:
        m_values = variables.get("m_values", [mlow + 4])
        msing = len(m_values)
        delta_prime = np.zeros((msing, msing, 2), dtype=float)
        delta_prime[0, 0, 0] = variables["delta_prime"]
        ds = xr.Dataset(
            {
                "Delta_prime": (("r", "r_prime", "i"), delta_prime),
                "r": (("r",), m_values),
                "psi_n_rational": (("r",), [0.1 * (i + 1) for i in range(msing)]),
                "q_rational": (("r",), [float(m) / mode for m in m_values]),
            },
            coords={"i": [0, 1]},
            attrs=attrs,
        )
    ds.to_netcdf(run_dir / filename)
    return run_dir


def test_build_mhd_linear_ods_reuses_the_adapters_directory_layout_and_round_trips(tmp_path):
    workdir = tmp_path / "gpec"
    _write_gpec_output(workdir, "dcon", 1, w_t=-0.42)
    _write_gpec_output(workdir, "rdcon", 1, delta_prime=1.23)
    # STRIDE mode 1 deliberately left unrun -- exercises the "missing" cell.

    ods, manifest = build_mhd_linear_ods(
        shot=39915,
        time_values=["00319"],
        workdir=workdir,
        modules=("dcon", "rdcon", "stride"),
        modes=(1,),
    )

    assert ods["mhd_linear.ids_properties.homogeneous_time"] == 1
    assert ods["mhd_linear.time"] == pytest.approx([0.319])
    modes = ods["mhd_linear"]["time_slice"][0]["toroidal_mode"]
    assert len(modes) == 2
    assert {mode["n_tor"] for mode in modes.values()} == {1}

    assert manifest["status"] == "success"
    assert manifest["modules_modes"]["t=00319/dcon/n=1"]["status"] == "success"
    assert manifest["modules_modes"]["t=00319/rdcon/n=1"]["status"] == "success"
    assert manifest["modules_modes"]["t=00319/stride/n=1"]["status"] == "missing"
    assert manifest["input"]  # at least one .nc file hashed

    output = tmp_path / "mhd_linear.json"
    metadata = tmp_path / "mhd_linear_manifest.json"
    write_stage_product(ods, manifest, output=output, metadata=metadata)
    reloaded, _ = load_ods(output)
    assert len(reloaded["mhd_linear"]["time_slice"][0]["toroidal_mode"]) == 2


def test_build_mhd_linear_ods_covers_multiple_refined_time_slices(tmp_path):
    workdir = tmp_path / "gpec"
    _write_gpec_output(workdir, "dcon", 1, time_label="00300", w_t=-0.1)
    _write_gpec_output(workdir, "dcon", 1, time_label="00320", w_t=-0.2)

    ods, manifest = build_mhd_linear_ods(
        shot=39915,
        time_values=["00300", "00320"],
        workdir=workdir,
        modules=("dcon",),
        modes=(1,),
    )

    assert ods["mhd_linear.time"] == pytest.approx([0.300, 0.320])
    assert len(ods["mhd_linear"]["time_slice"]) == 2
    assert ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["energy_perturbed"] == pytest.approx(-0.1)
    assert ods["mhd_linear"]["time_slice"][1]["toroidal_mode"][0]["energy_perturbed"] == pytest.approx(-0.2)
    assert manifest["modules_modes"]["t=00300/dcon/n=1"]["status"] == "success"
    assert manifest["modules_modes"]["t=00320/dcon/n=1"]["status"] == "success"


def test_build_mhd_linear_ods_reads_separate_canonical_module_work_trees(tmp_path):
    dcon_workdir = tmp_path / "gpec" / "dcon" / "39915" / "n=1" / "work"
    rdcon_workdir = tmp_path / "gpec" / "rdcon" / "39915" / "n=1" / "work"
    _write_gpec_output(dcon_workdir, "dcon", 1, w_t=-0.42)
    _write_gpec_output(rdcon_workdir, "rdcon", 1, delta_prime=1.23)

    ods, manifest = build_mhd_linear_ods(
        shot=39915,
        time_values=["00319"],
        module_workdirs={("dcon", 1): dcon_workdir, ("rdcon", 1): rdcon_workdir},
        modules=("dcon", "rdcon"),
        modes=(1,),
    )

    assert len(ods["mhd_linear"]["time_slice"][0]["toroidal_mode"]) == 2
    assert manifest["modules_modes"]["t=00319/dcon/n=1"]["status"] == "success"
    assert manifest["modules_modes"]["t=00319/rdcon/n=1"]["status"] == "success"


def test_build_mhd_linear_ods_does_not_construct_a_gpec_case_inputs(tmp_path, monkeypatch):
    """`build_mhd_linear_ods` used to fabricate a `GPECCaseInputs` with a fake
    `geqdsk=Path("unused")` just to call `module_dir`. It must resolve run
    directories directly from `workdir`/`time_ms` instead."""
    import vaft.code.gpec as gpec_pkg

    workdir = tmp_path / "gpec"
    _write_gpec_output(workdir, "dcon", 1, w_t=-0.42)

    def _fail_if_constructed(*args, **kwargs):
        raise AssertionError("build_mhd_linear_ods must not construct GPECCaseInputs")

    monkeypatch.setattr(gpec_pkg, "GPECCaseInputs", _fail_if_constructed)

    ods, manifest = build_mhd_linear_ods(
        shot=39915,
        time_values=["00319"],
        workdir=workdir,
        modules=("dcon",),
        modes=(1,),
    )

    assert manifest["modules_modes"]["t=00319/dcon/n=1"]["status"] == "success"


def test_pf_current_processing_matches_the_reference_filter(tmp_path):
    shot = 39915
    source = tmp_path / "raw.json.gz"
    samples = np.sin(np.linspace(0.0, 20.0, 1200)) + np.linspace(0.0, 0.2, 1200)
    _write_raw_dump(
        source,
        shot,
        {field: samples.tolist() for field in (5, 59, 62, 65)},
    )

    _time, currents = vfit_pf(shot, raw_source=source)
    centered = samples - np.mean(samples)
    taps = signal.firwin(251, 2500.0, pass_zero="lowpass", fs=25_000.0)
    expected = ndimage.uniform_filter1d(
        signal.filtfilt(taps, 1, centered), size=10
    ) * -5e4
    np.testing.assert_allclose(currents[0], expected)


def test_missing_md_channel_keeps_later_channel_in_its_ordered_slot():
    time = np.arange(1200, dtype=float) * 4e-6
    waveform = np.sin(np.linspace(0.0, 10.0, time.size))
    channels = [
        {"field_code": 1, "kind": "b_field_pol_probe", "calibration": 1.0},
        {"field_code": 2, "kind": "b_field_pol_probe", "calibration": 1.0},
    ]

    _target_time, _flux, probes = vest_equilibrium_magnetics_signals(
        39915,
        channels,
        lambda _shot, field: None if field == 1 else (time, waveform),
        config=VestMagneticsProcessingConfig(),
        allow_missing=True,
    )

    assert len(probes) == 2
    assert probes[0].size == 0
    assert probes[1].size > 0
