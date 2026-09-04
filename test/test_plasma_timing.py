"""The plasma window from an ODS with its provenance (issue #409, PR-ii).

The packaged shots pin the numbers the raw-database corpus study produced;
the synthetic products walk the source hierarchy -- slow H-alpha, fast
H-alpha, plasma current -- through every state the owner named: a channel
absent, railed, flat, mislabelled or invalidated, a shot with light but no
current, current but no light, neither, and the two disagreeing either way.
"""
from __future__ import annotations

import ast
import gzip
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from omas import ODS, load_omas_json

import vaft.omas.plasma_timing as module
from vaft.machine_mapping.utils import (
    VestConfigurationError,
    resolve_diagnostics_time_policies,
    resolve_plasma_timing_policy,
)
from vaft.omas.plasma_timing import (
    AGREEMENT_CONSISTENT,
    AGREEMENT_HALPHA_LEADS_IP_LARGE,
    AGREEMENT_HALPHA_ONLY,
    AGREEMENT_IP_BEFORE_HALPHA,
    AGREEMENT_IP_ONLY,
    AGREEMENT_NONE,
    SOURCE_H_FAST,
    SOURCE_H_PRIMARY,
    SOURCE_IP,
    PlasmaTimingError,
    analysis_span,
    halpha_sources,
    halpha_usability,
    plasma_timing,
)
from vaft.process.onset import active_window

DT = 4e-5
RNG = np.random.default_rng(409)

# Corpus numbers (ms) from the 205-shot study on issue #409.
PACKAGED = {
    39915: dict(onset=0.3065, offset=0.3308),
    41672: dict(onset=0.3125, offset=0.3517),
    41524: dict(onset=0.3146, offset=(0.336, 0.337)),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pipeline_ods(shot: int) -> ODS:
    from vaft.data import resources

    try:
        source = resources.data_path(f"samples/{shot}/source/pipeline-until-efit.json.gz")
    except Exception:  # pragma: no cover
        pytest.skip("packaged pipeline sample unavailable")
    if not Path(source).is_file():
        pytest.skip("packaged pipeline sample is repository-only")
    with gzip.open(source, "rt") as handle, tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False
    ) as plain:
        shutil.copyfileobj(handle, plain)
        plain_path = plain.name
    try:
        return load_omas_json(plain_path, consistency_check=False)
    finally:
        Path(plain_path).unlink(missing_ok=True)


def grid(t0: float = 0.26, t1: float = 0.36) -> np.ndarray:
    return np.arange(t0, t1, DT)


def light(t, *, onset=0.306, offset=0.331, amplitude=1.0, noise=3e-3, rise=2e-3):
    """H-alpha-like emission: fast rise, plateau, sharp end, white noise."""
    y = np.zeros_like(t)
    on = (t >= onset) & (t <= offset)
    y[on] = amplitude * np.clip((t[on] - onset) / rise, 0.0, 1.0)
    return y + noise * RNG.standard_normal(t.size)


def current(t, *, onset=0.3068, offset=0.3306, peak=60e3, noise=150.0, rise=8e-3):
    """Plasma current: ramp to the peak, slow decay, quench at ``offset``."""
    y = np.zeros_like(t)
    on = (t >= onset) & (t <= offset)
    ramp = np.clip((t[on] - onset) / rise, 0.0, 1.0)
    y[on] = peak * ramp * (1.0 - 0.3 * np.clip((t[on] - onset - rise) / (offset - onset), 0.0, 1.0))
    return y + noise * RNG.standard_normal(t.size)


def pickup_only(t, *, noise=150.0):
    """Coil-firing pickup: 1 ms bipolar spikes of 2 kA on a quiet Rogowski."""
    y = noise * RNG.standard_normal(t.size)
    for t_fire in (0.281, 0.293, 0.307):
        m = (t >= t_fire) & (t < t_fire + 1e-3)
        y[m] += 2e3 * np.sin(2 * np.pi * (t[m] - t_fire) / 1e-3)
    return y


def synthetic_ods(
    *,
    slow=None,
    fast=None,
    ip=None,
    t=None,
    slow_label="H-alpha_6563",
    fast_label="H-alpha_6563",
    validity=None,
    validity_timed=None,
) -> ODS:
    t = grid() if t is None else t
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 41672
    ods["spectrometer_uv.ids_properties.homogeneous_time"] = 1
    ods["spectrometer_uv.time"] = t
    for channel in range(3):  # OMAS arrays of structures cannot skip an index
        ods[f"spectrometer_uv.channel.{channel}.name"] = f"channel {channel}"
    for channel, line, label, data in ((0, 0, slow_label, slow), (2, 0, fast_label, fast)):
        if data is None:
            continue
        base = f"spectrometer_uv.channel.{channel}.processed_line.{line}"
        ods[f"{base}.label"] = label
        ods[f"{base}.wavelength_central"] = 656.3e-9
        ods[f"{base}.intensity.data"] = np.asarray(data, dtype=float)
    if validity is not None:
        ods["spectrometer_uv.channel.0.processed_line.0.intensity.validity"] = validity
    if validity_timed is not None:
        ods["spectrometer_uv.channel.0.processed_line.0.intensity.validity_timed"] = validity_timed
    if ip is not None:
        ods["magnetics.ids_properties.homogeneous_time"] = 1
        ods["magnetics.time"] = t
        ods["magnetics.ip.0.data"] = np.asarray(ip, dtype=float)
    return ods


# ---------------------------------------------------------------------------
# Packaged shots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shot", sorted(PACKAGED))
def test_packaged_shots_time_the_plasma_from_the_slow_h_alpha_line(shot):
    ods = _pipeline_ods(shot)
    expected = PACKAGED[shot]

    timing = plasma_timing(ods)

    assert timing.found
    assert timing.onset_source == SOURCE_H_PRIMARY
    assert timing.offset_source == SOURCE_H_PRIMARY
    assert timing.agreement == AGREEMENT_CONSISTENT
    assert timing.onset == pytest.approx(expected["onset"], abs=5e-4)
    if isinstance(expected["offset"], tuple):
        lo, hi = expected["offset"]
        assert lo <= timing.offset <= hi
    else:
        assert timing.offset == pytest.approx(expected["offset"], abs=5e-4)
    assert timing.ip is not None and timing.ip.found
    assert abs(timing.onset_delta_s) <= 2.5e-3
    assert abs(timing.offset_delta_s) <= 2.5e-3
    assert "offset_at_record_end" not in timing.flags
    json.dumps(timing.record())


def test_packaged_usability_records_the_fast_channel_resampling_and_era_shift():
    ods = _pipeline_ods(41672)
    primary, fast = halpha_sources()

    slow = halpha_usability(ods, primary)
    resampled = halpha_usability(ods, fast)

    assert slow.usable and slow.notes["resampled"] is False
    assert resampled.usable and resampled.notes["resampled"] is True
    assert resampled.notes["native_rate_hz"] == 250e3
    assert resampled.notes["time_shift_s"] == 0.26  # shot >= 41660
    assert slow.metrics["peak_over_sigma"] > 100


def test_reading_a_product_materialises_nothing():
    """A bare ``ods[path]`` creates the path; every read here must not."""
    ods = _pipeline_ods(39915)
    del ods["spectrometer_uv.channel.2"]
    before = set(ods.flat())

    timing = plasma_timing(ods)
    for source in halpha_sources():
        halpha_usability(ods, source)

    assert set(ods.flat()) == before
    assert timing.candidates[1].reason == "present"
    assert timing.onset_source == SOURCE_H_PRIMARY


# ---------------------------------------------------------------------------
# The hierarchy on synthetic products
# ---------------------------------------------------------------------------


def test_the_span_is_the_configured_plasma_analysis_window():
    span = analysis_span()
    window = resolve_diagnostics_time_policies().windows["plasma_analysis"]

    assert (span.tstart, span.tend) == (window.tstart, window.tend) == (0.28, 0.36)
    assert span.reference_start == pytest.approx(0.26)


def test_a_normal_synthetic_shot_is_consistent_and_timed_from_the_light():
    t = grid()
    ods = synthetic_ods(slow=light(t), fast=light(t, noise=5e-3), ip=current(t))

    timing = plasma_timing(ods)

    assert timing.agreement == AGREEMENT_CONSISTENT
    assert timing.onset_source == SOURCE_H_PRIMARY
    assert timing.onset == pytest.approx(0.306, abs=3e-4)
    assert timing.offset == pytest.approx(0.331, abs=5e-4)
    assert timing.fallback_reason is None
    assert timing.window == (timing.onset, timing.offset)


def test_slow_channel_absent_falls_back_to_the_validated_fast_channel():
    t = grid()
    ods = synthetic_ods(fast=light(t), ip=current(t))

    timing = plasma_timing(ods)

    assert timing.onset_source == SOURCE_H_FAST
    assert "optical_fallback_fast" in timing.flags
    assert timing.candidates[0].reason == "present"
    assert timing.candidates[0].notes["absent"] is True
    assert "h_alpha_primary: present" in timing.fallback_reason
    assert timing.agreement == AGREEMENT_CONSISTENT


def test_a_railed_fast_channel_is_not_used_and_the_current_answers():
    t = grid()
    railed = np.clip(light(t, amplitude=8.0), None, 5.0)  # 5 % of samples at the rail
    ods = synthetic_ods(fast=railed, ip=current(t))

    timing = plasma_timing(ods)

    assert timing.candidates[1].reason == "not_railed"
    assert timing.candidates[1].metrics["railed_fraction"] > 0.01
    assert timing.onset_source == SOURCE_IP
    assert timing.agreement == AGREEMENT_IP_ONLY
    assert timing.onset == pytest.approx(0.3068, abs=1e-3)


def test_no_h_alpha_at_all_is_a_normal_state_answered_by_the_current():
    t = grid()
    ods = synthetic_ods(ip=current(t))

    timing = plasma_timing(ods)

    assert timing.found
    assert timing.onset_source == timing.offset_source == SOURCE_IP
    assert timing.agreement == AGREEMENT_IP_ONLY
    assert timing.optical is None
    assert "h_alpha_primary: present" in timing.fallback_reason
    assert "h_alpha_fast: present" in timing.fallback_reason
    assert "halpha_dark_with_ip_pulse" not in timing.flags


def test_a_flat_quantized_baseline_is_unusable():
    t = grid()
    quantized = np.round(light(t, noise=1e-4), 3)  # the 2026-08 records: 1e-3 steps, MAD 0
    ods = synthetic_ods(slow=quantized, ip=current(t))

    usability = halpha_usability(ods, halpha_sources()[0])

    assert not usability.usable
    assert usability.reason == "baseline_live"
    assert usability.metrics["baseline_mad"] < 2e-4
    assert plasma_timing(ods).onset_source == SOURCE_IP


def test_a_wrong_label_is_skipped():
    t = grid()
    ods = synthetic_ods(slow=light(t), slow_label="OI_7770", ip=current(t))

    usability = halpha_usability(ods, halpha_sources()[0])

    assert usability.reason == "label"
    assert usability.notes["stored_label"] == "OI_7770"
    assert plasma_timing(ods).onset_source == SOURCE_IP


def test_an_invalidated_node_is_skipped_by_scalar_and_by_timed_validity():
    t = grid()
    scalar = synthetic_ods(slow=light(t), ip=current(t), validity=-2)
    assert halpha_usability(scalar, halpha_sources()[0]).reason == "validity"

    timed = np.full(t.size, 0)
    timed[t >= 0.28] = -2  # the whole search stretch invalid, the reference fine
    by_time = synthetic_ods(slow=light(t), ip=current(t), validity_timed=timed)
    usability = halpha_usability(by_time, halpha_sources()[0])
    assert usability.reason == "validity"
    assert usability.metrics["valid_fraction"] < 0.9

    suspect = synthetic_ods(slow=light(t), ip=current(t), validity=-1)
    assert halpha_usability(suspect, halpha_sources()[0]).usable


def test_pickup_alone_is_no_plasma_and_no_window_is_assumed():
    t = grid()
    ods = synthetic_ods(slow=light(t, amplitude=0.0), ip=pickup_only(t))

    timing = plasma_timing(ods)

    assert not timing.found
    assert timing.window is None
    assert timing.agreement == AGREEMENT_NONE
    assert "no_plasma_timing" in timing.flags
    assert "h_alpha_primary: usable but no light" in timing.fallback_reason
    assert "ip_principal:" in timing.fallback_reason
    assert timing.candidates[0].usable
    json.dumps(timing.record())


def test_a_dark_usable_slow_channel_is_not_overruled_by_the_fast_one():
    t = grid()
    ods = synthetic_ods(slow=light(t, amplitude=0.0), fast=light(t), ip=current(t))

    timing = plasma_timing(ods)

    assert timing.onset_source == SOURCE_IP
    assert timing.agreement == AGREEMENT_IP_ONLY
    assert "halpha_dark_with_ip_pulse" in timing.flags
    assert timing.optical_source.role == SOURCE_H_PRIMARY
    assert not timing.optical.found


def test_current_leading_the_light_is_flagged_and_the_light_still_wins():
    t = grid()
    ods = synthetic_ods(slow=light(t, onset=0.310), ip=current(t, onset=0.3065))

    timing = plasma_timing(ods)

    assert timing.agreement == AGREEMENT_IP_BEFORE_HALPHA
    assert "ip_before_halpha" in timing.flags
    assert timing.onset == pytest.approx(0.310, abs=3e-4)
    assert timing.onset_delta_s < -1e-3


def test_light_leading_the_current_by_a_lot_is_flagged():
    t = grid()
    ods = synthetic_ods(slow=light(t, onset=0.290, offset=0.331), ip=current(t, onset=0.306))

    timing = plasma_timing(ods)

    assert timing.agreement == AGREEMENT_HALPHA_LEADS_IP_LARGE
    assert "halpha_leads_ip_large" in timing.flags
    assert timing.onset == pytest.approx(0.290, abs=3e-4)


def test_offset_disagreement_is_flagged_and_the_offset_stays_with_the_light():
    t = grid()
    ods = synthetic_ods(slow=light(t, offset=0.325), ip=current(t, offset=0.3306))

    timing = plasma_timing(ods)

    assert "offset_disagreement" in timing.flags
    assert timing.offset == pytest.approx(0.325, abs=5e-4)
    assert timing.offset_delta_s > 3e-3
    assert timing.agreement == AGREEMENT_CONSISTENT


def test_pre_ionization_light_starts_the_window_and_is_a_second_segment():
    t = grid()
    early = light(t, onset=0.290, offset=0.297, amplitude=0.4, noise=0.0)
    main = light(t, onset=0.306, offset=0.331)
    ods = synthetic_ods(slow=early + main, ip=current(t, onset=0.3068))

    timing = plasma_timing(ods)

    assert timing.onset == pytest.approx(0.290, abs=3e-4)
    assert "multiple_segments" in timing.flags
    assert timing.agreement == AGREEMENT_HALPHA_LEADS_IP_LARGE
    assert len(timing.optical.segments) == 2


def test_light_without_current_is_halpha_only():
    t = grid()
    ods = synthetic_ods(slow=light(t), ip=150.0 * RNG.standard_normal(t.size))

    timing = plasma_timing(ods)

    assert timing.found
    assert timing.agreement == AGREEMENT_HALPHA_ONLY
    assert "ip_no_pulse" in timing.flags
    assert timing.onset_source == SOURCE_H_PRIMARY
    assert not timing.ip.found


def test_a_product_starting_inside_the_range_is_still_timed_and_flagged():
    t = grid(0.28, 0.36)
    ods = synthetic_ods(slow=light(t), ip=current(t), t=t)

    timing = plasma_timing(ods)

    assert timing.found
    assert "reference_inside_search" in timing.flags
    assert timing.onset == pytest.approx(0.306, abs=5e-4)


def test_an_invalidated_plasma_current_is_refused_as_source_and_cross_check():
    """Review finding: the current used to be promoted regardless of its validity."""
    t = grid()
    ods = synthetic_ods(ip=current(t))
    ods["magnetics.ip.0.validity"] = -2

    timing = plasma_timing(ods)

    assert not timing.found
    assert timing.ip is None
    assert timing.ip_checks["validity"] is False
    assert "ip_unusable" in timing.flags
    assert "ip_principal: validity" in timing.fallback_reason
    assert timing.agreement == AGREEMENT_NONE

    lit = synthetic_ods(slow=light(t), ip=current(t))
    lit["magnetics.ip.0.validity"] = -2
    with_light = plasma_timing(lit)
    assert with_light.found and with_light.source == SOURCE_H_PRIMARY
    assert with_light.agreement == AGREEMENT_HALPHA_ONLY
    assert "ip_unusable" in with_light.flags and with_light.onset_delta_s is None


def test_an_ip_only_product_starting_inside_the_range_is_flagged_too():
    """Review finding: the flag only travelled with an optical candidate."""
    t = grid(0.28, 0.36)
    ods = synthetic_ods(ip=current(t), t=t)

    timing = plasma_timing(ods)

    assert timing.source == SOURCE_IP
    assert "reference_inside_search" in timing.flags


def test_missing_plasma_current_is_the_one_error():
    t = grid()
    ods = synthetic_ods(slow=light(t))

    with pytest.raises(PlasmaTimingError, match="magnetics.ip.0"):
        plasma_timing(ods)


# ---------------------------------------------------------------------------
# Policy wiring and layering
# ---------------------------------------------------------------------------


def test_sources_are_ordered_by_digitizer_rate_not_by_position(monkeypatch):
    reversed_signals = list(reversed(module.SIGNALS))
    monkeypatch.setattr(module, "SIGNALS", reversed_signals)

    primary, fast = halpha_sources()

    assert (primary.channel, primary.role) == (0, SOURCE_H_PRIMARY)
    assert (fast.channel, fast.role) == (2, SOURCE_H_FAST)
    assert primary.label == fast.label == "H-alpha_6563"


def test_the_policy_rules_are_active_window_arguments_and_reach_the_detector():
    policy = resolve_plasma_timing_policy()
    t = grid()
    ods = synthetic_ods(slow=light(t), ip=current(t))

    baseline = plasma_timing(ods, policy=policy)
    import dataclasses

    retuned = dataclasses.replace(policy, h_alpha={**policy.h_alpha, "fraction": 0.5})
    high = plasma_timing(ods, policy=retuned)

    assert high.optical.onset.evidence["threshold"] > baseline.optical.onset.evidence["threshold"]
    assert high.optical.onset.evidence["fraction"] == 0.5
    for rule in (policy.h_alpha, policy.ip):
        active_window(t, light(t), **{k: v for k, v in rule.items()})


def test_a_misspelled_rule_or_unknown_window_is_a_configuration_error(tmp_path):
    import yaml

    from vaft.machine_mapping.utils import _resolve_info_file_path, load_yaml

    document = load_yaml(_resolve_info_file_path(None))

    bad_rule = dict(document)
    bad_rule["plasma_timing"] = {**document["plasma_timing"], "ip": {**document["plasma_timing"]["ip"], "cutoff_khz": 2.0}}
    path = tmp_path / "bad_rule.yaml"
    path.write_text(yaml.safe_dump(bad_rule))
    with pytest.raises(VestConfigurationError, match="cutoff_khz"):
        resolve_plasma_timing_policy(info_file=str(path))

    bad_window = dict(document)
    bad_window["plasma_timing"] = {**document["plasma_timing"], "window": "plasma_analisys"}
    path = tmp_path / "bad_window.yaml"
    path.write_text(yaml.safe_dump(bad_window))
    with pytest.raises(VestConfigurationError, match="'window' must name"):
        resolve_plasma_timing_policy(info_file=str(path))


def test_every_configured_channel_has_a_digitizer_rate(monkeypatch):
    """Review finding: a channel without a cadence used to sort last silently."""
    from vaft.machine_mapping import spectrometer_uv

    for _, channel, _, _, _ in spectrometer_uv.SIGNALS:
        assert channel in spectrometer_uv.CHANNEL_CADENCE_HZ

    extended = list(module.SIGNALS) + [(999, 7, 0, "H-alpha_6563", 656.3e-9)]
    monkeypatch.setattr(module, "SIGNALS", extended)
    with pytest.raises(PlasmaTimingError, match="channel 7"):
        halpha_sources()


def test_rule_validation_refuses_call_site_keys_booleans_and_bad_fractions(tmp_path):
    import yaml

    from vaft.machine_mapping.utils import _resolve_info_file_path, load_yaml

    document = load_yaml(_resolve_info_file_path(None))
    block = document["plasma_timing"]

    def resolve_with(**changes):
        doc = dict(document)
        doc["plasma_timing"] = {**block, **changes}
        path = tmp_path / f"{len(list(tmp_path.iterdir()))}.yaml"
        path.write_text(yaml.safe_dump(doc))
        return resolve_plasma_timing_policy(info_file=str(path))

    with pytest.raises(VestConfigurationError, match="'fs'"):
        resolve_with(ip={**block["ip"], "fs": 25000.0})
    with pytest.raises(VestConfigurationError, match="'reference_mask'"):
        resolve_with(h_alpha={**block["h_alpha"], "reference_mask": 0})
    with pytest.raises(VestConfigurationError, match="hold_s.*not a boolean"):
        resolve_with(h_alpha={**block["h_alpha"], "hold_s": True})
    with pytest.raises(VestConfigurationError, match="principal_only.*true or false"):
        resolve_with(ip={**block["ip"], "principal_only": 1})
    with pytest.raises(VestConfigurationError, match="prefilter_samples.*integer"):
        resolve_with(h_alpha={**block["h_alpha"], "prefilter_samples": 2.5})
    with pytest.raises(VestConfigurationError, match="min_valid_fraction.*<= 1"):
        resolve_with(usability={**block["usability"], "min_valid_fraction": 90})
    with pytest.raises(VestConfigurationError, match="rail_level.*positive"):
        resolve_with(usability={**block["usability"], "rail_level": 0})
    # a legitimate retune still resolves, and the policy is cached per file
    policy = resolve_with(ip={**block["ip"], "cutoff_hz": None})
    assert policy.ip["cutoff_hz"] is None
    assert policy.h_alpha["prefilter_samples"] == 5


def test_importing_the_timing_module_leaves_the_root_logger_alone():
    """Review finding: the import chain reaches vaft.database.raw (through the
    machine_mapping package), which used to call logging.basicConfig at import
    and hand every consumer an INFO-level stderr handler."""
    import subprocess
    import sys

    code = (
        "import logging\n"
        "root = logging.getLogger()\n"
        "before = (root.level, len(root.handlers))\n"
        "import vaft.omas.plasma_timing\n"
        "print((root.level, len(root.handlers)) == before)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True", out.stdout + out.stderr


def test_the_omas_layer_reads_only_and_the_process_layer_knows_no_diagnostic():
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    validation = [m for m in imported if m.startswith("vaft.validation")]
    assert validation == ["vaft.validation.imas"], validation
    assert not [m for m in imported if m.startswith(("vaft.plot", "vaft.database"))]
    # no ODS write: no subscript assignment on an ODS argument, no set_path
    writes = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == "ods"
    ]
    assert writes == []
    assert "set_path" not in Path(module.__file__).read_text(encoding="utf-8")

    import vaft.process.onset as onset

    text = Path(onset.__file__).read_text(encoding="utf-8")
    for literal in ("spectrometer_uv", "vest.yaml", "magnetics.ip", "machine_mapping", "vaft.omas import"):
        assert literal not in text, literal
