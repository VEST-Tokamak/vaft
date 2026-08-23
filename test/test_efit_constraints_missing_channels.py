"""Missing-diagnostic-channel handling in the EFIT constraints/k-file stage.

`vfit_equilibrium_form_constraints` and `generate_kfile` read
`magnetics.b_field_pol_probe`/`magnetics.flux_loop` under the assumption
that every AoS index has data. OMAS array-of-structures grow contiguously
from index 0, so a real shot with even one missing channel (a normal DAQ
dropout, not a bug) used to crash the constraints stage outright, or -- if
naively patched to just skip the missing index -- would have silently
shrunk the channel count and shifted every later channel's identity against
EFIT's fixed-geometry tables and the legacy `broken`-index numbering.

These tests build one fully-populated diagnostics/eddy ODS from a synthetic
raw dump (the same technique as test_vest_upstream.py), then delete specific
`field.data`/`flux.data` entries to simulate real missing channels at
chosen positions, and verify the constraints and k-file stages handle every
case: missing-first, missing-middle, and multiple missing channels, for
both `bpol_probe` and `flux_loop`.
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from omas import load_omas_json

from vaft.code.efit import generate_constraints_ods, generate_kfile
from vaft.code.efit.config import EFITConstraintConfig, EFITProfileConfig, EFITScientificConfig
from vaft.machine_mapping.magnetics import TOROIDAL_MIRNOV_REFERENCE_CHANNELS, vest_md_channel_definitions
from vaft.omas.vest_upstream import (
    build_diagnostics_ods,
    build_eddy_ods,
    build_static_ods,
    machine_era_for_shot,
    write_stage_product,
)


SHOT = 43016
SLOW_DT = 4e-5
DEFAULT_UNCERTAINTY = [1e-4, 1e-4, 5e-2, 3e-2, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2]
DEFAULT_WEIGHTING = [1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01]


def _all_magnetics_field_codes() -> list[int]:
    defs = vest_md_channel_definitions()
    codes = {int(d["field_code"]) for d in defs}
    codes |= {int(c["field_code"]) for c in TOROIDAL_MIRNOV_REFERENCE_CHANNELS}
    return sorted(codes)


def _write_raw_dump(path: Path, shot: int, n_samples: int) -> None:
    """A raw archive covering every field the diagnostics stage needs.

    Real sampling: the loader recomputes time as `arange(n)*SLOW_DT`
    regardless of any time array supplied here, so `n_samples` must be large
    enough for the dump to cover the diagnostics window (0.26-0.36 s).
    """
    codes = _all_magnetics_field_codes() + [1, 5, 59, 62, 65, 102, 257]
    t = np.arange(n_samples) * SLOW_DT
    fields = {}
    for code in sorted(set(codes)):
        if code == 102:
            # A plasma-current-like waveform so the Ip>20kA constraint
            # time-selection window is non-empty.
            data = 60000 * np.clip(np.sin((t - 0.26) / 0.1 * np.pi), 0, None)
        else:
            data = np.sin(t * 10) * 0.01 + 0.02
        fields[str(code)] = {"data": data.tolist(), "type": "slow"}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"shot": shot, "fields": fields}, handle)


@pytest.fixture(scope="module")
def full_eddy_ods(tmp_path_factory):
    """One fully-populated eddy ODS: no channel is naturally missing.

    Built once per test module (the raw->static->diagnostics->eddy chain is
    expensive); each test takes a deep copy so it can delete channels
    without affecting other tests.
    """
    tmp = tmp_path_factory.mktemp("constraints_fixture")
    raw_path = tmp / "raw.json.gz"
    _write_raw_dump(raw_path, SHOT, n_samples=10_000)

    era = machine_era_for_shot(SHOT)
    static, static_manifest = build_static_ods(era.name)
    static_path = tmp / "static.json"
    write_stage_product(static, static_manifest, output=static_path, metadata=tmp / "static-manifest.json")

    diag_ods, diag_manifest = build_diagnostics_ods(
        shot=SHOT, raw_source=raw_path, static_ods=static_path,
        tstart=0.26, tend=0.36, dt=4e-5, run=1,
    )
    diag_path = tmp / "diagnostics.json"
    write_stage_product(diag_ods, diag_manifest, output=diag_path, metadata=tmp / "diagnostics-manifest.json")

    eddy_ods, _ = build_eddy_ods(
        shot=SHOT, diagnostics_ods=diag_path, static_ods=static_path,
        filament_r=[0.35, 0.35, 0.35], filament_z=[0.25, 0.0, -0.25],
        filament_fraction=[1 / 3, 1 / 3, 1 / 3], dt_sub=5e-5,
    )

    # A handful of toroidal-mirnov reference channels never get data from
    # this synthetic dump's field codes; fill them so the fixture has zero
    # naturally-missing channels and every test's missing set is exact.
    time_axis = eddy_ods["magnetics.time"]
    for i in range(len(eddy_ods["magnetics.b_field_pol_probe"])):
        if "field.data" not in eddy_ods[f"magnetics.b_field_pol_probe.{i}"]:
            eddy_ods[f"magnetics.b_field_pol_probe.{i}.field.data"] = np.full_like(time_axis, 1e-3)
    for i in range(len(eddy_ods["magnetics.flux_loop"])):
        if "flux.data" not in eddy_ods[f"magnetics.flux_loop.{i}"]:
            eddy_ods[f"magnetics.flux_loop.{i}.flux.data"] = np.full_like(time_axis, 1e-3)

    n_probes = len(eddy_ods["magnetics.b_field_pol_probe"])
    n_flux = len(eddy_ods["magnetics.flux_loop"])
    assert all(f"magnetics.b_field_pol_probe.{i}.field.data" in eddy_ods for i in range(n_probes))
    assert all(f"magnetics.flux_loop.{i}.flux.data" in eddy_ods for i in range(n_flux))
    return eddy_ods


def _extract_array(text: str, name: str, next_name: str) -> list[float]:
    start = text.index(f"{name}=") + len(f"{name}=")
    end = text.index(f"{next_name}=")
    chunk = text[start:end]
    return [float(x) for x in chunk.replace("\n", " ").split(",") if x.strip()]


def _build_kfile_config(nbcoil: int) -> EFITScientificConfig:
    # No real mhdin.dat table exists in this test, so `nfsum` falls back to
    # the full 26-segment count; supply a matching-shape (trivial) coil
    # constraint matrix so generate_kfile's own unrelated shape guard
    # doesn't block a test that isn't exercising PF-coil constraints.
    return EFITScientificConfig(
        profile=EFITProfileConfig(kppcur=2, kffcur=2),
        constraints=EFITConstraintConfig(
            coil_constraint_matrix=tuple((1.0,) for _ in range(nbcoil)),
            coil_constraint_targets=(0.0,),
        ),
    )


@pytest.mark.parametrize(
    ("missing_probes", "missing_flux", "case_id"),
    [
        ((0,), (), "bpol-missing-first"),
        ((33,), (), "bpol-missing-middle"),
        ((0, 5, 10, 30, 67), (), "bpol-missing-multiple"),
        ((), (0,), "flux-missing-first"),
        ((), (5,), "flux-missing-middle"),
        ((), (0, 3, 10), "flux-missing-multiple"),
        ((0, 67), (0, 10), "both-missing"),
    ],
)
def test_constraints_and_kfile_survive_missing_channels(
    full_eddy_ods, tmp_path, missing_probes, missing_flux, case_id
):
    ods = copy.deepcopy(full_eddy_ods)
    n_probes = len(ods["magnetics.b_field_pol_probe"])
    n_flux = len(ods["magnetics.flux_loop"])

    probe_identifiers = {
        i: ods[f"magnetics.b_field_pol_probe.{i}.identifier"] for i in missing_probes
    }
    flux_identifiers = {i: ods[f"magnetics.flux_loop.{i}.identifier"] for i in missing_flux}
    for i in missing_probes:
        del ods[f"magnetics.b_field_pol_probe.{i}.field.data"]
    for i in missing_flux:
        del ods[f"magnetics.flux_loop.{i}.flux.data"]

    times = np.array([0.30, 0.301])
    ods["equilibrium.time"] = times

    # (1) generate_constraints_ods must not crash, and must preserve the
    # full, contiguous channel count regardless of which are missing.
    generate_constraints_ods(
        ods, SHOT, str(tmp_path), "", times,
        DEFAULT_UNCERTAINTY, DEFAULT_WEIGHTING,
        broken=[], fit=0, FFCUR=2, PPCUR=2,
    )

    EQ = ods["equilibrium"]
    assert len(EQ["time_slice.0.constraints.bpol_probe"]) == n_probes
    assert len(EQ["time_slice.0.constraints.flux_loop"]) == n_flux

    for i in missing_probes:
        for t_idx in (0, 1):
            entry = EQ[f"time_slice.{t_idx}.constraints.bpol_probe.{i}"]
            assert entry["measured"] == 0.0
            assert entry["weight"] == 0.0
            assert entry["source"] == probe_identifiers[i]
    for i in missing_flux:
        for t_idx in (0, 1):
            entry = EQ[f"time_slice.{t_idx}.constraints.flux_loop.{i}"]
            assert entry["measured"] == 0.0
            assert entry["weight"] == 0.0
            assert entry["source"] == flux_identifiers[i]

    # Valid channels keep their identity: nonzero weight, unchanged index,
    # and a finite (not accidentally zeroed) measured value.
    for i in range(n_probes):
        if i in missing_probes:
            continue
        entry = EQ[f"time_slice.0.constraints.bpol_probe.{i}"]
        assert entry["weight"] != 0.0
        assert np.isfinite(entry["measured"])
        assert entry["source"] == ods[f"magnetics.b_field_pol_probe.{i}.identifier"]
    for i in range(n_flux):
        if i in missing_flux:
            continue
        entry = EQ[f"time_slice.0.constraints.flux_loop.{i}"]
        assert entry["weight"] != 0.0
        assert np.isfinite(entry["measured"])

    # (2) OMAS must accept the result through a full consistency-checked
    # save/reload round trip.
    constraints_path = tmp_path / f"{case_id}-constraints.json"
    from omas import save_omas_json

    save_omas_json(ods, str(constraints_path))
    reloaded = load_omas_json(str(constraints_path), consistency_check=True)
    assert len(reloaded["equilibrium.time_slice.0.constraints.bpol_probe"]) == n_probes
    assert len(reloaded["equilibrium.time_slice.0.constraints.flux_loop"]) == n_flux
    for i in missing_probes:
        assert reloaded[f"equilibrium.time_slice.0.constraints.bpol_probe.{i}.weight"] == 0.0
    for i in missing_flux:
        assert reloaded[f"equilibrium.time_slice.0.constraints.flux_loop.{i}.weight"] == 0.0

    # (3) The generated k-file itself: FWTMP2/FWTSI must be 0 exactly at
    # the missing positions and 1 everywhere else, with unchanged indices;
    # EXPMP2/COILS must stay finite (never NaN) at those positions too.
    kfile_dir = tmp_path / f"{case_id}-kfile"
    config = _build_kfile_config(nbcoil=len(EQ["time_slice.0.constraints.pf_current"]))
    generate_kfile(ods, SHOT, save_dir=str(kfile_dir), config=config)
    kfiles = sorted((kfile_dir / "kfile").glob("*"))
    assert kfiles, "generate_kfile produced no k-file"
    text = kfiles[0].read_text()

    expmp2 = _extract_array(text, "EXPMP2", "BITMPI")
    fwtmp2 = _extract_array(text, "FWTMP2", "PLASMA")
    coils = _extract_array(text, "COILS", "PSIBIT")
    fwtsi = _extract_array(text, "FWTSI", "FWTMP2")

    assert len(expmp2) == n_probes
    assert len(fwtmp2) == n_probes
    assert len(coils) == n_flux
    assert len(fwtsi) == n_flux

    for i in range(n_probes):
        assert np.isfinite(expmp2[i])
        assert fwtmp2[i] == (0 if i in missing_probes else 1)
    for i in range(n_flux):
        assert np.isfinite(coils[i])
        assert fwtsi[i] == (0 if i in missing_flux else 1)
