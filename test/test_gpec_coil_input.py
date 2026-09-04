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
from vaft.machine_mapping.coil_geometry_3d import (
    load_vest_3d_coil_config,
    parse_gpec_coil_dat,
)

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
    (staged,) = stage_coil_data([config["MID"]], tmp_path)
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
            specs=[CoilInputSpec("MID", (1.0, 2.0))],
        )


def test_write_coil_in_requires_specs(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        write_coil_in(
            package_vest_dir() / "coil.in",
            tmp_path / "coil.in",
            data_dir=tmp_path,
            specs=[],
        )


def test_write_coil_in_multiple_sets(tmp_path):
    out = tmp_path / "coil.in"
    write_coil_in(
        package_vest_dir() / "coil.in",
        out,
        data_dir=tmp_path,
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
