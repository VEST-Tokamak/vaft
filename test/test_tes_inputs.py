"""TES input deck: limited default, limiter sampling, coil model (issue #1469).

The deck is built from the packaged 39915 sample with the canonical VEST
limiter, exactly like the forward notebook; no ``rtes`` binary is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.code.tes import TESConfig, prepare_tes_inputs
from vaft.code.tes.inputs import (
    _clip_limiter_to_grid,
    _coils_from_ods,
    _element_pf_rows,
    _grid_box,
    _legacy_pf_rows,
)
from vaft.machine_mapping.wall import wall as canonical_wall
from vaft.omas.sample import sample_ods


@pytest.fixture(scope="module")
def ods():
    data = sample_ods(39915)
    canonical_wall(data)
    return data


def _deck(ods, tmp_path, config=None, **overrides):
    base = dict(workdir=tmp_path, shot=39915, time=0.325, bt0=0.15,
                constraint_source="magnetics", betap=0.05)
    base.update(overrides)
    config = config or TESConfig(**base)
    inputs = prepare_tes_inputs(ods, config)
    return config, inputs.cinput.read_text().splitlines()


def _value(lines, key):
    (line,) = [ln for ln in lines if ln.split() and ln.split()[0] == key]
    return line.split()[1:]


def _block(lines, key, count):
    """The ``count`` numbers following a bare ``key`` line (LIMR / LIMZ)."""
    start = lines.index(key) + 1
    values: list[float] = []
    for line in lines[start:]:
        values.extend(float(v) for v in line.split())
        if len(values) >= count:
            return np.asarray(values[:count])
    raise AssertionError(f"{key} block is short")


def _coil_rows(lines):
    start = next(i for i, ln in enumerate(lines) if ln.startswith("NCOIL"))
    n = int(lines[start].split()[1])
    return np.array([[float(v) for v in lines[start + 1 + k].split()] for k in range(n)])


def test_default_deck_has_no_shape_control(ods, tmp_path):
    _, lines = _deck(ods, tmp_path)

    assert _value(lines, "FIX_SHAPE") == ["0"]
    assert _value(lines, "NXPT") == ["0"]
    assert _value(lines, "NISO") == ["0"]
    assert _value(lines, "NCGRP_FOR_SHAPE") == ["0"]


def test_default_limiter_resolves_the_inboard_midplane(ods, tmp_path):
    # TES tests the boundary flux only at listed limiter points. The canonical
    # center-stack face is one edge from Z = -0.575 to +0.575 m, so without
    # resampling an inboard-midplane contact cannot be seen.
    config, lines = _deck(ods, tmp_path)
    n = int(_value(lines, "NLIM")[0])
    r, z = _block(lines, "LIMR", n), _block(lines, "LIMZ", n)

    assert np.min(np.hypot(r - 0.105, z)) < 0.01
    r0, r1, z0, z1 = _grid_box(config)
    edges = np.hypot(np.diff(np.r_[r, r[0]]), np.diff(np.r_[z, z[0]]))
    z_next = np.r_[z[1:], z[0]]
    on_box = (np.isclose(z, z0, atol=1e-5) & np.isclose(z_next, z0, atol=1e-5)) | (
        np.isclose(z, z1, atol=1e-5) & np.isclose(z_next, z1, atol=1e-5))
    assert edges[~on_box].max() <= config.limiter_spacing + 1e-6   # wall edges only
    assert r.min() >= r0 - 1e-9 and r.max() <= r1 + 1e-9
    assert z.min() >= z0 - 1e-9 and z.max() <= z1 + 1e-9


def test_clipping_follows_the_grid_box_instead_of_cutting_chords():
    config = TESConfig()
    r0, r1, z0, z1 = _grid_box(config)
    # a square wall taller than the grid: its top and bottom edges leave it
    r = np.array([0.2, 0.6, 0.6, 0.2])
    z = np.array([-1.0, -1.0, 1.0, 1.0])

    rc, zc = _clip_limiter_to_grid(r, z, config)

    corners = {(round(a, 9), round(b, 9)) for a, b in zip(rc, zc)}
    assert corners == {(0.2, round(z0, 9)), (0.6, round(z0, 9)), (0.6, round(z1, 9)), (0.2, round(z1, 9))}


def test_clipping_rejects_a_wall_outside_the_grid():
    with pytest.raises(ValueError, match="fewer than 3 points"):
        _clip_limiter_to_grid(np.array([2.0, 3.0, 3.0]), np.array([0.0, 0.0, 1.0]), TESConfig())


def test_element_coil_model_keeps_every_winding_as_a_filament(ods, tmp_path):
    _, lines = _deck(ods, tmp_path)
    rows = _coil_rows(lines)
    pf = ods["pf_active"]
    n_elements = sum(len(pf[f"coil.{c}.element"]) for c in range(len(pf["coil"])))

    # eddy is off; only in-grid elements sharing a TES node are folded together
    assert 0 < n_elements - len(rows) < 50
    assert np.all(rows[:, 4] == 1)                      # turns folded into the current
    assert len(np.unique(rows[:, 6])) == len(rows)      # one group per row
    # PF1 ampere-turns: 632 turns at the measured current
    pf1_current = np.interp(0.325, pf["time"], pf["coil.0.current.data"])
    pf1 = rows[np.isclose(rows[:, 0], 0.053)]
    assert pf1[:, 5].sum() * 1000.0 == pytest.approx(632.0 * pf1_current, rel=1e-6)


def test_shape_control_requires_the_legacy_coil_model(ods, tmp_path):
    with pytest.raises(ValueError, match="coil_model='legacy'"):
        _deck(ods, tmp_path, fix_shape=1)


def test_legacy_preset_restores_the_double_null_deck(ods, tmp_path):
    config = TESConfig.legacy_double_null(
        workdir=tmp_path, shot=39915, time=0.325, bt0=0.15,
        constraint_source="magnetics", betap=0.05,
    )
    _, lines = _deck(ods, tmp_path, config=config)
    rows = _coil_rows(lines)

    assert _value(lines, "FIX_SHAPE") == ["1"]
    assert _value(lines, "NXPT") == ["2"]
    assert [float(v) for v in _value(lines, "ISOR")] == [0.64, 0.11, 0.3, 0.3]
    assert _value(lines, "NCGRP_FOR_SHAPE") == ["18"]
    assert len(rows) == 36
    assert sorted(set(rows[:, 6].astype(int))) == list(range(1, 19))
    # clipped vertices only: the 1.15 m center-stack face stays one edge
    n = int(_value(lines, "NLIM")[0])
    r, z = _block(lines, "LIMR", n), _block(lines, "LIMZ", n)
    assert np.hypot(np.diff(np.r_[r, r[0]]), np.diff(np.r_[z, z[0]])).max() > 1.1


def _with_passive_currents(ods, amps):
    data = ods.copy()
    data["pf_passive.time"] = np.array([0.3, 0.35])
    for i in range(len(data["pf_passive.loop"])):
        data[f"pf_passive.loop.{i}.current"] = np.array([amps, amps])
    return data


def _tes_node(r, z, config):
    """TES's own node choice for an in-grid row (update_jphi.cpp), 0-based."""
    dr = (config.rmax - config.rmin) / (config.nr - 1)
    dz = (config.zmax - config.zmin) / (config.nz - 1)
    ll, jj = int((r - config.rmin) / dr), int((z - config.zmin) / dz)
    ll += r - (config.rmin + ll * dr) > dr / 2
    jj += z - (config.zmin + jj * dz) > dz / 2
    return ll, jj


@pytest.mark.parametrize("coil_model", ["elements", "legacy"])
def test_in_grid_rows_occupy_distinct_nodes_and_keep_their_current(ods, coil_model):
    # TES ASSIGNS an in-grid row's current density to one grid node, so two
    # rows on one node used to keep only the last (274 of 535 passive loops).
    data = _with_passive_currents(ods, 10.0)
    config = TESConfig(eddy=True, coil_model=coil_model)
    rows = np.array(_coils_from_ods(data, config, 0.325))
    inside = (
        (rows[:, 0] > config.rmin) & (rows[:, 0] < config.rmax)
        & (rows[:, 1] > config.zmin) & (rows[:, 1] < config.zmax)
    )
    nodes = [_tes_node(r, z, config) for r, z in rows[inside, :2]]

    assert len(nodes) == len(set(nodes))
    # TES indexes its group table by id in an array sized NCOIL
    assert sorted(set(rows[:, 6].astype(int))) == list(range(1, len(set(rows[:, 6])) + 1))
    assert rows[:, 6].max() <= len(rows)
    dr = (config.rmax - config.rmin) / (config.nr - 1)
    dz = (config.zmax - config.zmin) / (config.nz - 1)
    assert np.allclose(rows[inside, 2], dr) and np.allclose(rows[inside, 3], dz)
    # total ampere-turns are conserved: every passive loop carries 10 A
    n_loops = len(data["pf_passive.loop"])
    pf_rows = (_element_pf_rows if coil_model == "elements" else _legacy_pf_rows)(data, 0.325)
    pf_total = sum(row[4] * row[5] for row in pf_rows)
    assert (rows[:, 4] * rows[:, 5]).sum() == pytest.approx(pf_total + n_loops * 10.0 / 1000.0)


def test_clip_edges_are_not_resampled(ods, tmp_path):
    # The box edges the clip adds run across the open chamber neck; they are
    # the cut, not wall, and must not become dozens of limiter candidates.
    config, lines = _deck(ods, tmp_path)
    n = int(_value(lines, "NLIM")[0])
    r, z = _block(lines, "LIMR", n), _block(lines, "LIMZ", n)
    _, _, z0, z1 = _grid_box(config)

    assert np.sum(np.isclose(z, z1, atol=1e-5)) == 2
    assert np.sum(np.isclose(z, z0, atol=1e-5)) == 2


def test_clipping_a_concave_wall_that_leaves_and_reenters_the_box():
    config = TESConfig()
    r0, r1, z0, z1 = _grid_box(config)
    # a U-shaped wall whose two arms stick out above the grid
    r = np.array([0.2, 0.6, 0.6, 0.5, 0.5, 0.3, 0.3, 0.2])
    z = np.array([-0.5, -0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

    rc, zc = _clip_limiter_to_grid(r, z, config)

    assert np.all(rc >= r0 - 1e-12) and np.all(rc <= r1 + 1e-12)
    assert np.all(zc >= z0 - 1e-12) and np.all(zc <= z1 + 1e-12)
    # the notch between the arms survives: (0.3..0.5, 0.0) is still wall
    assert {(0.5, 0.0), (0.3, 0.0)} <= set(zip(np.round(rc, 9), np.round(zc, 9)))


def test_passive_currents_are_written_in_kA(ods):
    data = ods.copy()
    data["pf_passive.time"] = np.array([0.3, 0.35])
    for i in range(len(data["pf_passive.loop"])):
        data[f"pf_passive.loop.{i}.current"] = np.array([0.0, 0.0])
    data["pf_passive.loop.0.current"] = np.array([1000.0, 1000.0])    # [A]

    with_eddy = np.array(_coils_from_ods(data, TESConfig(eddy=True), 0.325))
    without = np.array(_coils_from_ods(data, TESConfig(eddy=False), 0.325))

    eddy_kA = (with_eddy[:, 4] * with_eddy[:, 5]).sum() - (without[:, 4] * without[:, 5]).sum()
    assert eddy_kA == pytest.approx(1.0)                               # 1 kA, not 1e-3 kA


def test_missing_reference_betap_is_a_clear_error(ods, tmp_path):
    with pytest.raises(ValueError, match="TESConfig.betap"):
        _deck(ods, tmp_path, constraint_source="equilibrium", betap=None)
