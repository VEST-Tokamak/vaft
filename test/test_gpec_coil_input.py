"""GPEC coil-input generation from the canonical VEST 3D coil configuration.

The reference for numerical equivalence is the working shot-48226 @ 300 ms
ideal-GPEC run: its ``coil.in`` is committed under
``test/data/gpec_reference_48226/`` and its coil geometry is byte-identical
to the packaged ``vaft/data/gpec/vest_MID.dat`` (the bundle named it
``vest_12inch_20turn.dat``; VAFT canonicalizes it as the MID set).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from vaft.code import gpec
from vaft.code.gpec import CoilInputSpec, emit_coil_dat, stage_coil_data, write_coil_in
from vaft.code.gpec._runtime import package_vest_dir
from vaft.data.resources import data_path
from vaft.machine_mapping.coils_non_axisymmetric_geometry import (
    GPEC_COIL_DAT_HEADER,
    coil_set_from_dat,
    coil_set_from_xyz_loops,
    load_vest_3d_coil_config,
    parse_gpec_coil_dat,
)
from vaft.machine_mapping.conventions import VEST_GPEC_COIL_DIRECTIONS

REFERENCE_DIR = Path(__file__).parent / "data" / "gpec_reference_48226"
REFERENCE_CURRENTS = (200.0, 200.0, 0.0, -200.0, -200.0, 0.0)

GFILE_TEXT = "  EFITD   01/01/2024   #  48226  300ms        3  65  65\n 1.0 2.0 3.0\n"


def _parse_coil_control(text: str) -> tuple[dict, dict[int, str], dict[int, list[float]]]:
    """Tolerant ``&COIL_CONTROL`` reader: scalars, coil names, sector currents."""
    body = text.split("&COIL_CONTROL", 1)[1].split("\n/", 1)[0]
    scalars: dict = {}
    names: dict[int, str] = {}
    currents: dict[int, list[float]] = {}
    for line in body.splitlines():
        line = line.split("!", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        name_match = re.fullmatch(r"coil_name\((\d+)\)", key)
        cur_match = re.fullmatch(r"coil_cur\((\d+),(\d+)\)", key)
        if name_match:
            names[int(name_match.group(1))] = value.strip('"').strip("'")
        elif cur_match:
            set_index = int(cur_match.group(1))
            values = [float(token) for token in value.replace(",", " ").split()]
            currents.setdefault(set_index, []).extend(values)
        else:
            scalars[key] = value.strip('"').strip("'")
    return scalars, names, currents


@pytest.fixture()
def generated(tmp_path):
    out = tmp_path / "coil.in"
    write_coil_in(
        package_vest_dir() / "coil.in",
        out,
        data_dir=tmp_path / "coil",
        specs=[CoilInputSpec("MID", REFERENCE_CURRENTS)],
        machine="vest",
    )
    return out


def test_generated_coil_in_matches_reference_semantics(generated):
    scalars, names, currents = _parse_coil_control(generated.read_text(encoding="utf-8"))
    ref_scalars, ref_names, ref_currents = _parse_coil_control(
        (REFERENCE_DIR / "coil.in").read_text(encoding="utf-8")
    )

    for key in ("machine", "ip_direction", "bt_direction", "ceq_type"):
        assert scalars[key] == ref_scalars[key], key
    assert int(scalars["coil_num"]) == int(ref_scalars["coil_num"]) == 1
    assert currents[1] == pytest.approx(ref_currents[1])
    # Names differ by design: the bundle's "12inch_20turn" is VAFT's
    # canonical "MID"; both resolve 6-sector sets with identical geometry.
    assert names[1] == "MID"
    assert ref_names[1] == "12inch_20turn"


def test_generated_geometry_equals_reference_geometry():
    header = parse_gpec_coil_dat(data_path("gpec/vest_MID.dat"))
    assert header[:4] == (6, 1, 100, 20.0)


def test_stage_coil_data_is_byte_identical(tmp_path):
    config = load_vest_3d_coil_config(coil_sets=["MID"])
    (staged,) = stage_coil_data([config["MID"]], tmp_path, machine="vest")
    assert staged.name == "vest_MID.dat"
    assert staged.read_bytes() == Path(data_path("gpec/vest_MID.dat")).read_bytes()


def test_emit_coil_dat_reconstructs_numerically(tmp_path):
    config = load_vest_3d_coil_config()
    for name, coil_set in config.coil_sets.items():
        emitted = emit_coil_dat(coil_set, tmp_path / f"vest_{name}.dat")
        ncoil, nsec, npts, nw, points = parse_gpec_coil_dat(emitted)
        _, _, _, ref_nw, ref_points = parse_gpec_coil_dat(coil_set.dat_path)
        assert (ncoil, nsec, npts) == (6, 1, ref_points.shape[1])
        assert nw == ref_nw
        np.testing.assert_allclose(points, ref_points, atol=1e-6)


def test_write_coil_in_validates_sector_count(tmp_path):
    with pytest.raises(ValueError, match="sectors"):
        write_coil_in(
            package_vest_dir() / "coil.in",
            tmp_path / "coil.in",
            data_dir=tmp_path,
            machine="vest",
            specs=[CoilInputSpec("MID", (1.0, 2.0))],
        )


def test_write_coil_in_requires_specs(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        write_coil_in(
            package_vest_dir() / "coil.in",
            tmp_path / "coil.in",
            data_dir=tmp_path,
            machine="vest",
            specs=[],
        )


def test_write_coil_in_multiple_sets(tmp_path):
    out = tmp_path / "coil.in"
    write_coil_in(
        package_vest_dir() / "coil.in",
        out,
        data_dir=tmp_path,
        machine="vest",
        specs=[
            CoilInputSpec("UP", (1.0,) * 6),
            CoilInputSpec("LOW", (-1.0,) * 6),
        ],
    )
    scalars, names, currents = _parse_coil_control(out.read_text(encoding="utf-8"))
    assert int(scalars["coil_num"]) == 2
    assert names == {1: "UP", 2: "LOW"}
    assert currents[2] == pytest.approx([-1.0] * 6)


@pytest.fixture()
def case(tmp_path):
    geqdsk = tmp_path / "g048226.00300"
    geqdsk.write_text(GFILE_TEXT, encoding="utf-8")
    return gpec.GPECCaseInputs(
        shot=48226,
        time_ms=300,
        geqdsk=geqdsk,
        workdir=tmp_path / "run",
    )


@pytest.fixture()
def no_gpec_env(monkeypatch):
    monkeypatch.delenv(gpec.GPEC_HOME_ENV, raising=False)


def test_prepare_with_coil_specs_generates_inputs(no_gpec_env, case):
    config = gpec.GPECSuiteConfig(
        modules=("gpec",),
        modes=(1,),
        gpec=gpec.IdealGPECOptions(
            coil_specs=(CoilInputSpec("MID", REFERENCE_CURRENTS),)
        ),
    )
    result = gpec.prepare_gpec_suite_case(case, config)
    assert result.ok

    run_dir = case.workdir / "00300" / "gpec" / "nn=1"
    staged = run_dir / "coil" / "vest_MID.dat"
    assert staged.read_bytes() == Path(data_path("gpec/vest_MID.dat")).read_bytes()

    scalars, names, currents = _parse_coil_control(
        (run_dir / "coil.in").read_text(encoding="utf-8")
    )
    assert names == {1: "MID"}
    assert currents[1] == pytest.approx(list(REFERENCE_CURRENTS))
    assert scalars["data_dir"] == str((run_dir / "coil").resolve())


def test_explicit_coil_in_wins_over_coil_specs(no_gpec_env, case, tmp_path):
    override = tmp_path / "override_coil.in"
    override.write_text("&coil /\n", encoding="utf-8")
    case.coil_in = override
    config = gpec.GPECSuiteConfig(
        modules=("gpec",),
        modes=(1,),
        gpec=gpec.IdealGPECOptions(
            coil_specs=(CoilInputSpec("MID", REFERENCE_CURRENTS),)
        ),
    )
    result = gpec.prepare_gpec_suite_case(case, config)
    assert result.ok
    run_dir = case.workdir / "00300" / "gpec" / "nn=1"
    assert (run_dir / "coil.in").read_text(encoding="utf-8") == override.read_text(encoding="utf-8")
    assert not (run_dir / "coil").exists()


def test_default_prepare_is_unchanged_without_coil_specs(no_gpec_env, case):
    result = gpec.prepare_gpec_suite_case(
        case, gpec.GPECSuiteConfig(modules=("gpec",), modes=(1,))
    )
    assert result.ok
    run_dir = case.workdir / "00300" / "gpec" / "nn=1"
    text = (run_dir / "coil.in").read_text(encoding="utf-8")
    assert "coil_num=3" in text
    assert 'coil_name(1)="UP"' in text
    assert not (run_dir / "coil").exists()


def test_namelist_strings_are_not_json_escaped(tmp_path):
    r"""A Fortran namelist has no escape sequences, so a path must survive verbatim.

    ``json.dumps`` would render ``C:\gpec\coil`` as ``"C:\gpec\coil"``; GPEC
    reads that literally and resolves a directory that does not exist. Invisible
    on POSIX, where a path carries no backslashes at all.
    """
    from vaft.code.gpec._runtime import write_template

    template = tmp_path / "coil.in"
    template.write_text(
        '&COIL_CONTROL\n data_dir=""\n machine="y"\n/\n', encoding="utf-8"
    )
    target = tmp_path / "out.in"

    write_template(
        template,
        target,
        {"data_dir": Path(r"C:\vaft data\39915\gpec\nn=1\coil"), "machine": "vest"},
    )
    rendered = target.read_text(encoding="utf-8")

    assert r'"C:\vaft data\39915\gpec\nn=1\coil"' in rendered
    assert "\\\\" not in rendered, "namelist values must not be backslash-escaped"
    assert '"vest"' in rendered


def test_namelist_quoting_keeps_the_repository_delimiter():
    """The packaged template and the 48226 reference both use double quotes."""
    from vaft.code.gpec._runtime import _quote_namelist_string

    assert _quote_namelist_string("/srv/vest/coil") == '"/srv/vest/coil"'
    assert _quote_namelist_string(r"C:\vaft\coil") == r'"C:\vaft\coil"'
    # The delimiter is the only in-string metacharacter Fortran recognises.
    assert _quote_namelist_string('say "hi"') == '"say ""hi"""'


# ---------------------------------------------------------------------------
# A second machine: geometry supplied by the caller, direction words explicit
# ---------------------------------------------------------------------------

SYNTHETIC_SETS = {"rmpu": 4, "rmpl": 8, "efcc": 4, "pf_n1": 16}


def _tiny_loop(phi_deg: float, radius: float = 1.4, half_height: float = 0.1) -> np.ndarray:
    """A closed five-point rectangle centred at toroidal angle ``phi_deg``."""
    phi = np.deg2rad(phi_deg)
    r_in, r_out = radius - 0.05, radius + 0.05
    corners = [(r_in, -half_height), (r_out, -half_height), (r_out, half_height), (r_in, half_height), (r_in, -half_height)]
    return np.array([[r * np.cos(phi), r * np.sin(phi), z] for r, z in corners])


@pytest.fixture(scope="module")
def synthetic_machine():
    config = {}
    for name, count in SYNTHETIC_SETS.items():
        loops = [_tiny_loop(360.0 * k / count) for k in range(count)]
        config[name] = coil_set_from_xyz_loops(name, loops, turns=2.0, description=f"synthetic {name}")
    return config


def test_coil_set_from_xyz_loops_validates_and_records_sector_angles(synthetic_machine):
    config = synthetic_machine
    assert config["rmpl"].sector_angles_deg == pytest.approx(tuple(45.0 * k for k in range(8)))
    assert config["rmpl"].dat_path is None
    with pytest.raises(ValueError, match="turns must be positive"):
        coil_set_from_xyz_loops("x", [_tiny_loop(0.0)], turns=0.0)
    with pytest.raises(ValueError, match="not closed"):
        coil_set_from_xyz_loops("x", [_tiny_loop(0.0)[:-1]], turns=1.0)
    with pytest.raises(ValueError, match="one point count per set"):
        coil_set_from_xyz_loops("x", [_tiny_loop(0.0), _tiny_loop(90.0)[:3]], turns=1.0)


def test_in_memory_sets_round_trip_through_dat_with_the_canonical_header(tmp_path, synthetic_machine):
    config = synthetic_machine
    staged = stage_coil_data([config["rmpl"], config["pf_n1"]], tmp_path / "coil", machine="synth")
    assert [p.name for p in staged] == ["synth_rmpl.dat", "synth_pf_n1.dat"]
    header = staged[0].read_text(encoding="utf-8").splitlines()[0]
    assert GPEC_COIL_DAT_HEADER == ("ncoil", "nsec", "npts", "nw")
    assert header.split() == ["8", "1", "5", "2"]  # ncoil nsec npts nw, space separated, not the legacy swap
    ncoil, nsec, npts, nw, points = parse_gpec_coil_dat(staged[0])
    assert (ncoil, nsec, npts, nw) == (8, 1, 5, 2.0)
    for k, filament in enumerate(config["rmpl"].filaments):
        assert np.allclose(points[k], filament.points_xyz, atol=1e-6)
    reread = coil_set_from_dat(staged[0], "rmpl")
    assert len(reread.filaments) == 8 and reread.turns == 2.0 and reread.dat_path == staged[0]


def test_second_machine_coil_in_carries_explicit_directions_and_all_sets(tmp_path, synthetic_machine):
    config = synthetic_machine
    specs = [
        CoilInputSpec("rmpu", tuple(float(k) for k in range(4))),
        CoilInputSpec("rmpl", tuple(float(k) for k in range(8))),
        CoilInputSpec("efcc", (1.0, -1.0, 1.0, -1.0)),
        CoilInputSpec("pf_n1", tuple([0.5] * 16)),
    ]
    out = tmp_path / "coil.in"
    write_coil_in(
        package_vest_dir() / "coil.in",
        out,
        data_dir=tmp_path / "coil",
        specs=specs,
        machine="synth",
        coil_config=config,
        ip_direction="positive",
        bt_direction="negative",
    )
    scalars, names, currents = _parse_coil_control(out.read_text(encoding="utf-8"))
    assert scalars["machine"] == "synth"
    assert int(scalars["coil_num"]) == 4
    assert scalars["ip_direction"] == "positive" and scalars["bt_direction"] == "negative"
    assert names == {1: "rmpu", 2: "rmpl", 3: "efcc", 4: "pf_n1"}
    assert currents[2] == pytest.approx([float(k) for k in range(8)])
    assert [spec.name for spec in gpec.read_coil_in(out)] == ["rmpu", "rmpl", "efcc", "pf_n1"]


def test_second_machine_refuses_to_inherit_geometry_or_directions(tmp_path, synthetic_machine):
    config = synthetic_machine
    common = dict(data_dir=tmp_path / "coil", specs=[CoilInputSpec("rmpu", (1.0, 0.0, -1.0, 0.0))])
    with pytest.raises(ValueError, match="coil_config .* is required"):
        write_coil_in(package_vest_dir() / "coil.in", tmp_path / "a.in", machine="synth", **common)
    with pytest.raises(ValueError, match="ip_direction and bt_direction must be given"):
        write_coil_in(package_vest_dir() / "coil.in", tmp_path / "b.in", machine="synth", coil_config=config, **common)
    with pytest.raises(ValueError, match="must be one of"):
        write_coil_in(
            package_vest_dir() / "coil.in", tmp_path / "c.in", machine="synth", coil_config=config,
            ip_direction="ccw", bt_direction="negative", **common,
        )


def test_vest_direction_words_match_the_reference_run(generated):
    reference, _, _ = _parse_coil_control((REFERENCE_DIR / "coil.in").read_text(encoding="utf-8"))
    scalars, _, _ = _parse_coil_control(generated.read_text(encoding="utf-8"))
    assert scalars["machine"] == "vest"
    for key in ("ip_direction", "bt_direction"):
        assert scalars[key] == reference[key] == VEST_GPEC_COIL_DIRECTIONS[key]


def test_prepare_for_a_second_machine_stages_generated_dat_files(no_gpec_env, case, synthetic_machine):
    config = synthetic_machine
    options = gpec.IdealGPECOptions(
        coil_specs=(CoilInputSpec("rmpl", tuple(float(k) for k in range(8))),),
        machine="synth",
        coil_config=config,
        ip_direction="positive",
        bt_direction="negative",
    )
    result = gpec.prepare_gpec_suite_case(
        case, gpec.GPECSuiteConfig(modules=("gpec",), modes=(1,), gpec=options)
    )
    assert result.ok
    run_dir = case.workdir / "00300" / "gpec" / "nn=1"
    assert (run_dir / "coil" / "synth_rmpl.dat").exists()
    scalars, names, _ = _parse_coil_control((run_dir / "coil.in").read_text(encoding="utf-8"))
    assert scalars["machine"] == "synth" and names == {1: "rmpl"}


def test_second_machine_options_are_refused_at_construction():
    with pytest.raises(ValueError, match="coil_specs, coil_config, ip_direction, bt_direction must be given"):
        gpec.IdealGPECOptions(machine="synth")
    with pytest.raises(ValueError, match="ip_direction, bt_direction must be given"):
        gpec.IdealGPECOptions(machine="synth", coil_specs=(), coil_config={})


def test_resolve_accepts_the_vest_config_object_and_rejects_misuse(synthetic_machine):
    from vaft.code.gpec import resolve_coil_inputs

    vest = load_vest_3d_coil_config(coil_sets=["MID"])
    config, ip, bt = resolve_coil_inputs("vest", vest, None, None, ["MID"])
    assert config["MID"] is vest["MID"] and (ip, bt) == ("positive", "negative")
    with pytest.raises(ValueError, match="at least one coil set"):
        resolve_coil_inputs("vest", None, None, None, [])
    with pytest.raises(ValueError, match="more than once"):
        resolve_coil_inputs("vest", None, None, None, ["MID", "MID"])
    with pytest.raises(ValueError, match="plain word"):
        resolve_coil_inputs("../x", synthetic_machine, "positive", "negative", ["rmpl"])
    with pytest.raises(ValueError, match="key must equal the set name"):
        resolve_coil_inputs("synth", {"RMPL": synthetic_machine["rmpl"]}, "positive", "negative", ["RMPL"])
    with pytest.raises(ValueError, match="must be one of"):
        resolve_coil_inputs("vest", None, "", None, ["MID"])


def test_explicitly_empty_coil_specs_do_not_fall_back_to_the_template(no_gpec_env, case):
    config = gpec.GPECSuiteConfig(
        modules=("gpec",), modes=(1,), gpec=gpec.IdealGPECOptions(coil_specs=())
    )
    with pytest.raises(ValueError, match="at least one"):
        gpec.prepare_gpec_suite_case(case, config)


def test_missing_backing_file_is_an_error_not_a_silent_reemit(tmp_path, synthetic_machine):
    from dataclasses import replace

    ghost = replace(synthetic_machine["rmpu"], dat_path=tmp_path / "gone.dat")
    with pytest.raises(FileNotFoundError):
        stage_coil_data([ghost], tmp_path / "coil", machine="synth")
