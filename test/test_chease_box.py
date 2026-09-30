"""CHEASE's EQDSK output box as two sizes, NRBOX and NZBOX (#459).

Upstream CHEASE reads the two independently; the adapter used to tie them to
one ``nw``. The output box only interpolates the solution, so a rectangular
box must leave the global descriptors where they were.
"""

from __future__ import annotations

import os

import pytest

from vaft.code.chease import CHEASEConfig, _namelist_lines

PARAMS = {"CSSPEC": 0.0, "ASPCT": 1.0, "R0EXP": 1.0, "B0EXP": 1.0, "SIGNB0XP": 1, "SIGNIPXP": 1, "CURRT": 1.0, "QSPEC": 1.0}
needs_chease = pytest.mark.skipif(
    not (os.environ.get("CHEASEHOME") or os.environ.get("CHEASE_EXEC_DIR") or os.environ.get("CHEASE")),
    reason="CHEASE integration test requires CHEASEHOME, CHEASE_EXEC_DIR or CHEASE",
)


def _box(config):
    return [line.strip() for line in _namelist_lines(config, PARAMS) if "BOX=" in line]


def test_square_box_is_the_default():
    assert _box(CHEASEConfig(nw=257)) == ["NRBOX=257,", "NZBOX=257,"]
    assert CHEASEConfig(nw=257).resolved_nh == 257


def test_rectangular_box_is_written_as_asked():
    assert _box(CHEASEConfig(nw=513, nh=257)) == ["NRBOX=513,", "NZBOX=257,"]


@pytest.mark.parametrize("kwargs", [dict(nh=1), dict(nw=1), dict(nh=2301), dict(nw=4000)])
def test_box_sizes_are_validated(kwargs):
    # CHEASE silently clamps either side to NPBPS = 2300 (psibox.f90).
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        CHEASEConfig(**kwargs)


@needs_chease
def test_rectangular_box_leaves_the_descriptors_alone(tmp_path):
    from vaft.code.chease import refine_equilibrium
    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

    values = {}
    for nw, nh in ((513, None), (513, 257)):
        workdir = tmp_path/f"{nw}_{nh}"
        result = refine_equilibrium(sample_geqdsk(), CHEASEConfig(workdir=workdir, nw=nw, nh=nh, create_plot=False))
        assert result.ok
        native = read_geqdsk(workdir/"EQDSK_COCOS_02.OUT")
        assert (int(native["NW"]), int(native["NH"])) == (nw, nh or nw)
        d = derive_global_descriptors(as_equilibrium(native)).values
        values[nh] = {k: float(d[k].value) for k in ("ip", "q0", "q95", "li_virial", "volume", "magnetic_axis_r")}
    for name, value in values[None].items():
        assert values[257][name] == pytest.approx(value, rel=1e-3), name


def test_manifest_keeps_output_grid_an_integer(tmp_path):
    import json

    from vaft.code.chease import prepare_chease_inputs
    from vaft.data.resources import sample_geqdsk

    prepare_chease_inputs(sample_geqdsk(), CHEASEConfig(workdir=tmp_path, nw=129, nh=65, create_plot=False))
    manifest = json.loads((tmp_path/"chease_input_manifest.json").read_text())
    assert type(manifest["output_grid"]) is int and type(manifest["output_grid_z"]) is int
    assert (manifest["output_grid"], manifest["output_grid_z"]) == (129, 65)
