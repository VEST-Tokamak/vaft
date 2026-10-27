"""Which raw DAQ fields each diagnostic registry entry is made from.

:func:`raw_sources` answers, for a registry id and a shot, the question the
mappers answer implicitly when they load: which archive fields, under which
calibration, feed that diagnostic. Each resolver below is built from the same
shot resolver its mapper calls (``resolve_vest_diagnostic``,
``coil_field_code_by_index``, ``magnetics_wiring_for_shot``, ...), so wiring
eras, calibration revisions and channel exclusions come out the way the
mapper applies them. ``test/test_raw_sources_mapper_agreement.py`` runs the
real mappers on a recording stub of :func:`vaft.database.raw.vest_load` and
requires the fields they load to equal what this module reports.

The join from registry id to resolver is stated once, in the registry record:
``source.raw_fields: "<module>:<function>"`` in ``vest.yaml``.

Reported are the ``vest.yaml`` era values. A ``processing_config`` passed to
one run (a workflow override) is not reflected: it never changes which
field is read, but its ``calibration_mode`` could change how a magnetics
factor is applied. What a specific product used is in its ``code.parameters``.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field as dataclass_field
from functools import lru_cache
from typing import Any

from .registry import load_diagnostic_registry

__all__ = ["RawField", "raw_field_name", "raw_sources"]


@dataclass(frozen=True)
class RawField:
    """One archive field a diagnostic is made from, resolved for one shot.

    ``label`` is VAFT's name for the signal (the mapper's channel label);
    ``daq_name`` is the legacy offline name table's entry for the field, which
    can be stale or empty -- the live DB (:func:`vaft.database.raw.name`) is
    the authority for it. ``raw_unit`` is the unit of the stored DAQ value.
    ``role`` says what the mapper uses the field for (``signal`` for the
    measurement itself; ``fl10_reference``, ``plasma_window``,
    ``plasma_current``, ``tf_current``, ``pf_current`` for inputs it only
    corrects or gates with). ``channel`` is the IDS-side node the field feeds,
    where there is one. ``known_fault`` is the recorded reason when
    ``vest.yaml`` marks the channel broken for this shot: it is still a
    source, the mapper still reads it, and its validity says the rest.
    """

    field: int
    label: str
    daq_name: str
    raw_unit: str
    daq_type: str | None
    role: str
    channel: str | None
    calibration: Mapping[str, Any] = dataclass_field(default_factory=dict)
    config: str | None = None
    known_fault: str | None = None


@lru_cache(maxsize=1)
def _daq_names() -> dict[int, str]:
    from vaft.database.raw import SQL_TABLE_PATH

    with open(SQL_TABLE_PATH, "r", encoding="utf-8") as handle:
        table = json.load(handle)
    names: dict[int, list[str]] = {}
    for name, field in table.items():
        names.setdefault(int(field), []).append(str(name))
    # Field 25 carries two legacy names; joined rather than one picked.
    return {field: " / ".join(entries) for field, entries in names.items()}


def raw_field_name(field: int) -> str:
    """The legacy offline name of a DAQ field, or ``""`` when the table has none.

    Offline counterpart of :func:`vaft.database.raw.name`, which needs a live
    DB pool. The table is the shipped ``sql_table.txt`` snapshot and is not
    kept in step with field reuse, so it can name a field's former signal.
    """
    return _daq_names().get(int(field), "")


def _daq_type(value: Any) -> str | None:
    if value is None:
        return None
    # Some vest.yaml blocks record the DAQ as 0/1 rather than slow/fast.
    return {0: "slow", 1: "fast"}.get(value, str(value)) if not isinstance(value, str) else value


def _numeric(value: Any) -> Any:
    """Calibration numbers as numbers: PyYAML reads ``-3.0e4`` (no exponent
    sign) as a string, which the mappers ``float()`` and a consumer should not
    have to."""
    if isinstance(value, Mapping):
        return {key: _numeric(item) for key, item in value.items()}
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _record(
    field: int,
    label: str,
    role: str,
    *,
    channel: str | None = None,
    raw_unit: str = "V",
    daq_type: Any = None,
    calibration: Mapping[str, Any] | None = None,
    config: str | None = None,
    known_fault: str | None = None,
) -> RawField:
    return RawField(
        field=int(field),
        label=str(label),
        daq_name=raw_field_name(field),
        raw_unit=str(raw_unit or "V"),
        daq_type=_daq_type(daq_type),
        role=role,
        channel=channel,
        calibration=_numeric(dict(calibration or {})),
        config=config,
        known_fault=known_fault,
    )


def _single(shot: int, key: str, label: str, channel: str | None, role: str = "signal") -> RawField:
    from .utils import resolve_vest_diagnostic

    config = resolve_vest_diagnostic(int(shot), key)
    source = config["source"]
    return _record(
        source["field"],
        label,
        role,
        channel=channel,
        raw_unit=source.get("raw_unit", "V"),
        daq_type=source.get("daq_type"),
        calibration=config.get("calibration"),
        config=f"0.diagnostics.{key}",
    )


# ---------------------------------------------------------------------------
# Resolvers, one per registry id. Named in vest.yaml `source.raw_fields`.
# ---------------------------------------------------------------------------


def dataset_description_raw_fields(shot: int) -> tuple[RawField, ...]:
    """None: the entry is shot metadata (date, remark), not a waveform."""
    del shot
    return ()


def tf_raw_fields(shot: int) -> tuple[RawField, ...]:
    return (_single(shot, "tf", "TF coil current", "tf.coil[0].current"),)


def barometry_raw_fields(shot: int) -> tuple[RawField, ...]:
    return (_single(shot, "barometry_main", "Main chamber pressure gauge", "barometry.gauge[0].pressure"),)


def ech_raw_fields(shot: int) -> tuple[RawField, ...]:
    from .ec_launchers import FORWARD_DIAGNOSTIC, REFLECTED_DIAGNOSTIC

    return (
        _single(shot, FORWARD_DIAGNOSTIC, "6 kW ECH forward power detector", "ec_launchers.beam[0].power_launched"),
        _single(shot, REFLECTED_DIAGNOSTIC, "6 kW ECH reflected power detector", "ec_launchers.beam[0].power_launched"),
    )


def pf_active_raw_fields(shot: int) -> tuple[RawField, ...]:
    from .pf_active import _coil_gain_by_index, coil_field_code_by_index

    codes = coil_field_code_by_index(int(shot))
    gains = _coil_gain_by_index(int(shot))
    names = _pf_circuit_names()
    records = []
    for index in sorted(codes):
        records.append(
            _record(
                codes[index],
                f"{names.get(index, f'PF circuit {index}')} current",
                "signal",
                channel=f"pf_active.coil[{index}].current",
                calibration={"type": "linear", "operation": "multiply", "factor": gains.get(index, 0.0)},
                config="0.diagnostics.pf_active.processing.coil_field_codes",
            )
        )
    return tuple(records)


@lru_cache(maxsize=1)
def _pf_circuit_names() -> dict[int, str]:
    from .utils import load_yaml, _resolve_info_file_path

    content = load_yaml(_resolve_info_file_path(None))
    block = (content.get(0) or content.get("0") or {}).get("pf_active") or {}
    return {int(index): str(entry.get("name")) for index, entry in (block.get("channel") or {}).items()}


def _plasma_current_fields(shot: int, role: str) -> tuple[RawField, ...]:
    """The Rogowski field and, where compensation is on, the FL10 reference."""
    from .magnetics import _plasma_processing_for_shot

    config, reference, _baseline, _sign = _plasma_processing_for_shot(int(shot))
    source = config["source"]
    records = [
        _record(
            source["field"],
            "Plasma-current Rogowski coil",
            role,
            channel="magnetics.rogowski_coil[0].current",
            raw_unit=source.get("raw_unit", "V"),
            daq_type=source.get("daq_type"),
            calibration=config.get("calibration"),
            config="0.diagnostics.plasma_current",
        )
    ]
    mode = reference.get("mode", "subtract")
    if mode != "disabled":
        # The two compensation modes read their FL10 field and gain from
        # different keys, as vfit_plasma_current / _apply_fl10_windowed_compensation do.
        if mode == "subtract_fl10_windowed":
            fl10 = reference["fl10"]
            field = fl10["field"]
            calibration = {
                **{key: value for key, value in fl10.items() if key != "field"},
                "effective_resistance_ohm": reference["effective_resistance_ohm"],
            }
            config_path = "0.diagnostics.plasma_current.processing.reference.fl10"
        else:
            field = reference["field"]
            calibration = {
                key: reference[key] for key in ("flux_gain", "effective_resistance_ohm") if key in reference
            }
            config_path = "0.diagnostics.plasma_current.processing.reference"
        records.append(
            _record(
                field,
                "Flux loop #10 voltage (limiter-current compensation)",
                "fl10_reference" if role == "signal" else role,
                calibration={"mode": mode, **calibration},
                config=config_path,
            )
        )
    return tuple(records)


def _diamagnetic_record(shot: int) -> RawField | None:
    from .magnetics import _diamagnetic_config, diamagnetic_field_for_shot

    field = diamagnetic_field_for_shot(int(shot))
    if field is None:
        return None
    config = _diamagnetic_config(int(shot))
    source = config.get("source") or {}
    processing = config.get("processing") or {}
    # No `calibration` block: the conversion is the shunt and the TF circuit
    # (vest_diamagnetic_flux_detailed).
    calibration = {
        key: processing[key] for key in ("rogowski_shunt", "tf_circuit") if key in processing
    }
    return _record(
        field,
        "Diamagnetic hi-sensitivity TF-current Rogowski coil",
        "signal",
        channel="magnetics.rogowski_coil[1].current",
        raw_unit=source.get("raw_unit", "V"),
        daq_type=source.get("daq_type"),
        calibration=calibration,
        config="0.diagnostics.diamagnetic_flux",
    )


def rogowski_coil_raw_fields(shot: int) -> tuple[RawField, ...]:
    plasma = _plasma_current_fields(shot, "signal")[0]
    diamagnetic = _diamagnetic_record(shot)
    return (plasma,) if diamagnetic is None else (plasma, diamagnetic)


def ip_raw_fields(shot: int) -> tuple[RawField, ...]:
    return tuple(
        record if record.role != "signal" else _replace(record, channel="magnetics.ip[0].data")
        for record in _plasma_current_fields(shot, "signal")
    )


def diamagnetic_flux_raw_fields(shot: int) -> tuple[RawField, ...]:
    """The TF Rogowski, the H-alpha that gates the plasma window, and Ip's inputs.

    Empty when ``vest.yaml`` records no usable sensor for the shot: the mapper
    then writes no diamagnetic flux and reads nothing for it.
    """
    from .magnetics import HALPHA_RAW_FIELD

    sensor = _diamagnetic_record(shot)
    if sensor is None:
        return ()
    return (
        _replace(sensor, channel="magnetics.diamagnetic_flux[0].data"),
        _record(HALPHA_RAW_FIELD, "H-alpha 656 nm (slow PMT)", "plasma_window"),
        *(_replace(record, role="plasma_current", channel=None) for record in _plasma_current_fields(shot, "signal")),
    )


def _wiring_records(shot: int, kind: str, path: str) -> list[RawField]:
    from .magnetics import equilibrium_magnetics_processing_config, known_magnetics_faults
    from .magnetics_registry import resolve_equilibrium_magnetics_sensors

    operation = equilibrium_magnetics_processing_config(int(shot)).calibration_mode
    faults = known_magnetics_faults(int(shot))
    records = []
    for sensor in resolve_equilibrium_magnetics_sensors(int(shot)):
        if sensor.kind != kind:
            continue
        # `name` is the DAQ label of the field the sensor reads for this shot.
        records.append(
            _record(
                sensor.field_code,
                sensor.name,
                "signal",
                channel=f"{path}[{sensor.kind_index}]",
                daq_type="slow",
                calibration={"operation": operation, "factor": float(sensor.calibration)},
                config="0.diagnostics.equilibrium_magnetics.processing.wiring",
                known_fault=faults.get((kind, sensor.kind_index)),
            )
        )
    return records


def flux_loop_raw_fields(shot: int) -> tuple[RawField, ...]:
    return tuple(_wiring_records(shot, "flux_loop", "magnetics.flux_loop"))


def b_field_pol_probe_raw_fields(shot: int) -> tuple[RawField, ...]:
    """Equilibrium probes, then the toroidal phase references, then the
    fluctuation Mirnovs -- the order the mapper appends them in.

    The Mirnov voltages are stored raw; their amplifier gain is a divisor a
    reader applies (``"applied": "downstream"``), not one the ODS carries.
    """
    from .magnetics import (
        FLUCTUATION_MIRNOV_FIRST_SHOT,
        _load_fluctuation_mirnov_channels,
        toroidal_mirnov_reference_channels,
    )

    records = _wiring_records(shot, "b_field_pol_probe", "magnetics.b_field_pol_probe")
    index = len(records)
    for channel in toroidal_mirnov_reference_channels(int(shot)):
        records.append(
            _record(
                channel["field_code"],
                channel["name"],
                "signal",
                channel=f"magnetics.b_field_pol_probe[{index}].voltage",
                daq_type="slow",
                calibration={"operation": "divide", "factor": float(channel["gain"]), "applied": "downstream"},
                config="0.diagnostics.toroidal_mirnov_reference",
            )
        )
        index += 1
    if int(shot) >= FLUCTUATION_MIRNOV_FIRST_SHOT:
        for channel in _load_fluctuation_mirnov_channels(int(shot)):
            records.append(
                _record(
                    channel["field"],
                    channel["identifier"],
                    "signal",
                    channel=f"magnetics.b_field_pol_probe[{index}].voltage",
                    raw_unit=channel.get("unit", "V"),
                    daq_type="fast",
                    calibration={"operation": "divide", "factor": float(channel["gain"]), "applied": "downstream"},
                    config="0.diagnostics.fluctuation_mirnov",
                )
            )
            index += 1
    return tuple(records)


def halo_currents_raw_fields(shot: int) -> tuple[RawField, ...]:
    """The outboard limiter current monitors, mapped to ``magnetics.shunt``."""
    from .magnetics import LIMITER_SHUNT_CHANNELS, LIMITER_SHUNT_RESISTANCE

    del shot
    return tuple(
        _record(
            channel["field_code"],
            channel["name"],
            "signal",
            channel=f"magnetics.shunt[{index}].voltage",
            calibration={"transfer_ohm": LIMITER_SHUNT_RESISTANCE},
            config="0.halo_current.limiter_monitor",
        )
        for index, channel in enumerate(LIMITER_SHUNT_CHANNELS)
    )


def internal_probe_raw_fields(shot: int) -> tuple[RawField, ...]:
    """IMPA Hall and Bz channels, then the TF, Ip and PF context it compensates with."""
    from .impa import _bz_specs, _channel_specs, _probe_node, _resolve_gain, resolve_impa_config

    config = resolve_impa_config(int(shot))
    calibration = config.get("calibration") or {}
    gain = _resolve_gain(calibration, int(shot))
    hall_node = _probe_node(str((config.get("tf_compensation") or {}).get("orientation", "toroidal")))
    records = [
        _record(
            spec["field"],
            spec.get("label", f"IMPA {spec['index']}"),
            "signal",
            channel=f"{hall_node} (IMPA {spec['index']})",
            daq_type=spec.get("daq_type"),
            calibration={"gain": gain},
            config="impa.channels",
        )
        for spec in _channel_specs(config, int(shot))
    ]
    records += [
        _record(
            spec["field"],
            spec.get("label", f"IMPA Bz {spec['index']}"),
            "signal",
            channel=f"magnetics.b_field_pol_probe (IMPA Bz {spec['index']})",
            daq_type=spec.get("daq_type"),
            calibration={"gain": calibration.get("bz_gain")},
            config="impa.bz_channels",
        )
        for spec in _bz_specs(config, int(shot))
    ]
    context = [
        *(_replace(record, role="tf_current", channel=None) for record in tf_raw_fields(shot)),
        *(_replace(record, role="plasma_current", channel=None) for record in _plasma_current_fields(shot, "signal")),
        *(_replace(record, role="pf_current", channel=None) for record in pf_active_raw_fields(shot)),
    ]
    return tuple(records + context)


def spectrometer_uv_raw_fields(shot: int) -> tuple[RawField, ...]:
    from .spectrometer_uv import CHANNEL_CADENCE_HZ, SIGNALS

    del shot
    return tuple(
        _record(
            field,
            label,
            "signal",
            channel=f"spectrometer_uv.channel[{channel}].processed_line[{line}].intensity",
            daq_type="fast" if CHANNEL_CADENCE_HZ.get(channel, 0.0) > 25e3 else "slow",
            # The mapper stores the negated raw voltage (spectrometer_uv).
            calibration={"type": "linear", "operation": "multiply", "factor": -1.0},
            config="vaft.machine_mapping.spectrometer_uv.SIGNALS",
        )
        for field, channel, line, label, _wavelength in SIGNALS
    )


def langmuir_probes_raw_fields(shot: int) -> tuple[RawField, ...]:
    """Voltage and current of each assembly installed for the shot.

    The channel names the assembly's identifier, not a slot: the mapper fills
    ``embedded[]`` contiguously and skips an installed assembly whose voltage
    field is empty for the shot, which only the waveform can tell.
    """
    from .langmuir_probes import ASSEMBLIES, resolve_langmuir_probe_config

    records = []
    for assembly in ASSEMBLIES:
        config = resolve_langmuir_probe_config(assembly["key"], int(shot))
        first_shot = config.get("first_shot")
        if first_shot is not None and int(shot) < int(first_shot):
            continue
        source = config["source"]
        calibration = config.get("calibration") or {}
        for quantity, key, cal in (
            ("voltage", "voltage_field", {"voltage_gain": calibration.get("voltage_gain")}),
            ("current", "current_field", {"current_divisor": calibration.get("current_divisor")}),
        ):
            records.append(
                _record(
                    source[key],
                    f"{assembly['name']} {quantity}",
                    "signal",
                    channel=f"langmuir_probes.embedded (identifier langmuir_probes:{assembly['key']})",
                    raw_unit=source.get("raw_unit", "V"),
                    daq_type=source.get("daq_type"),
                    calibration=cal,
                    config=f"0.langmuir_probes.{assembly['key']}",
                )
            )
    return tuple(records)


def _replace(record: RawField, **changes: Any) -> RawField:
    from dataclasses import replace

    return replace(record, **changes)


# ---------------------------------------------------------------------------


def _resolver(spec: str) -> Callable[[int], tuple[RawField, ...]]:
    module_name, _, function_name = spec.partition(":")
    if not module_name or not function_name:
        raise ValueError(f"raw_fields must be '<module>:<function>', got {spec!r}")
    return getattr(importlib.import_module(module_name), function_name)


def raw_sources(diagnostic: str, shot: int) -> tuple[RawField, ...]:
    """The raw DAQ fields registry entry ``diagnostic`` is made from for ``shot``.

    ``diagnostic`` is a :func:`~vaft.machine_mapping.registry.load_diagnostic_registry`
    id (``magnetics.ip``, ``pf_active``, ...). Raises ``KeyError`` for an
    unknown id and ``ValueError`` for one whose registry record states no
    ``source.raw_fields`` (file-based, reference or model sources). An empty
    tuple means the entry reads no field for this shot -- a sensor not yet
    installed, or shot metadata only.
    """
    records = _registry_sources()
    if diagnostic not in records:
        raise KeyError(f"unknown diagnostic registry id {diagnostic!r}")
    source_type, spec = records[diagnostic]
    if not spec:
        raise ValueError(
            f"diagnostic {diagnostic!r} states no source.raw_fields: its source is "
            f"{source_type!r}, not raw DAQ fields"
        )
    return tuple(_resolver(spec)(int(shot)))


@lru_cache(maxsize=1)
def _registry_sources() -> dict[str, tuple[str, str | None]]:
    """``{id: (source type, raw_fields)}``, parsed once: vest.yaml takes ~0.25 s."""
    return {
        identifier: (str(record["source"].get("type")), record["source"].get("raw_fields"))
        for identifier, record in load_diagnostic_registry().items()
    }
