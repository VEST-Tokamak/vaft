"""VEST hard X-ray mapping for the OMAS ``hard_x_rays`` IDS (prototype).

The VEST HXR detector (Lim, Jo et al., Fusion Eng. Des. 170 (2021) 112522) is
four BC-408 scintillator films of 10/100/200/300 um on SiPM arrays, read in
current mode on digitizer 17592 channels 61-64 -- the same digitizer, record
and TTL trigger as the 17592 soft X-ray channels 1-40.  The acquisition script
(``sample_v4.py``) did not keep those traces.  It averaged them into 1 ms bins,
L2-normalised each bin, ran a neural-network unfolding model
(``deconvolution1.model``) and wrote the result as
``digitizer_hxr_Eflux_{daq}_{shot}.csv``: one row per bin, one column per
nominal energy.  That file is what survives for almost every shot, so it is
what this mapping reads.

What the file is, and is not:

* **Relative intensity.** The detector has no absolute calibration; values are
  arbitrary units, not photons s^-1 m^-2 sr^-1 as the IDS ``radiance`` unit
  states.  ``ids_properties.comment`` says so.
* **Five of six energies.** The model returns 40/60/80/110/140/210 keV, but
  the writer's ``[0:-1]`` slice dropped the 210 keV column from every file.
* **Nominal energies, not bands.** The labels are the film thicknesses'
  sensitivity peaks, not calibrated bin edges, so ``energies`` carries the
  label and ``lower_bound``/``upper_bound`` are left unset rather than invented.
* **Signed.** The unfolding rescales by the signed channel sum, and on most
  archived shots some bins are negative.  Those bins are marked suspect (-1)
  in ``radiance.validity_timed``, not removed; ``radiance.validity`` is the
  worst bin, as ``vaft.validation.validity.aggregate_validity`` defines it.
* **One record per file, usually.** The writer appended, so a re-run adds a
  second 40-row block to the same file; such files are rejected because which
  block belongs to the shot is not recorded.

Machine time is the HXR trigger from the ShotLog-derived trigger settings.
When a shot has no HXR entry the SXR entry is used instead: HXR is read from
the same digitizer record as SXR, so the two share one trigger.  With neither,
the axis stays trigger-relative and a warning is raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np

from .soft_x_rays import _load_trigger_settings, resolve_sxr_trigger_settings
from .utils import resolve_data_root, set_path

HXR_DAQ_LABEL = "17592"
HXR_FLUX_ENERGIES_KEV: tuple[float, ...] = (40.0, 60.0, 80.0, 110.0, 140.0)
"""Energy labels of the archived Eflux columns, in file order."""
HXR_MODEL_ENERGIES_KEV: tuple[float, ...] = (*HXR_FLUX_ENERGIES_KEV, 210.0)
"""Energy labels the unfolding model returns; the last never reached a file."""
HXR_BIN_WIDTH_S = 1e-3
HXR_BINS_PER_RECORD = 40
HXR_RAW_CHANNELS = 4
HXR_SCINTILLATOR_THICKNESS_M: tuple[float, ...] = (10e-6, 100e-6, 200e-6, 300e-6)
"""BC-408 film thickness of digitizer channels 61-64, in order (paper table)."""
HXR_UNFOLDING_MODEL = {
    "name": "deconvolution1.model",
    "saved_model.pb": "bf92f47fe23e4b29391ed132976b128cd2fc7b66d4d228131b15456eaba11cf5",
    "variables.data-00000-of-00001": "faac44f848ed1250b96dbfebf49582848772001e9b9646514724a75ac1bbad50",
    "variables.index": "ed84a56c167f36a494d570b6e838ea9d8ac52f71dbafa916403076a62fbc8d0f",
}
"""SHA-256 of the one unfolding model found with the archived Eflux files."""

_COMMENT = (
    "VEST hard X-ray unfolded energy flux (sample_v4 Eflux CSV); relative intensity in "
    "arbitrary units, not absolute radiance. Energies are nominal labels; the 210 keV "
    "output was dropped by the legacy writer. Negative bins are suspect (-1) in validity_timed."
)


@dataclass(frozen=True)
class HXRTimeAlignment:
    """Resolved relation between the trigger-relative HXR axis and the shot clock."""

    offset_seconds: float
    source: str
    detail: str


def resolve_hxr_time_alignment(
    shot: int,
    *,
    trigger_settings_path: str | Path | None = None,
) -> HXRTimeAlignment:
    """Resolve the machine-time offset of an HXR record.

    An ``HXR`` trigger entry is authoritative.  Without one the ``SXR`` entry
    is used, because the HXR channels are part of the same digitizer-17592
    record as the SXR channels.  With neither, time stays trigger-relative.
    """
    path = resolve_sxr_trigger_settings(trigger_settings_path)
    entry: Any = None
    if path is not None:
        settings = _load_trigger_settings(str(path))
        entry = settings.get(int(shot), settings.get(str(int(shot))))
    if isinstance(entry, Mapping):
        for label, source in (("HXR", "hxr_trigger"), ("SXR", "sxr_cotrigger")):
            item = entry.get(label)
            if isinstance(item, Mapping) and item.get("start_time_ms") is not None:
                start_ms = float(item["start_time_ms"])
                return HXRTimeAlignment(start_ms * 1e-3, source, f"{label} start_time_ms={start_ms:g}")
    detail = f"Shot {shot} has no usable HXR or SXR trigger setting; using trigger-relative time."
    warnings.warn(detail, RuntimeWarning, stacklevel=2)
    return HXRTimeAlignment(0.0, "trigger_relative", detail)


def hxr_file_candidates(
    shot: int,
    variant: str = "Eflux",
    data_root: str | Path | None = None,
    daq_label: str = HXR_DAQ_LABEL,
) -> list[Path]:
    """Candidate locations of one archived HXR CSV.

    ``{root}/{shot}/`` is the ``legacy/hard_x_rays/{shot}/`` layout
    ``vaft.database.filedb.FileDB.legacy`` resolves; the flat candidate covers
    a hand-assembled directory.
    """
    root = resolve_data_root(data_root)
    filename = f"digitizer_hxr_{variant}_{daq_label}_{int(shot)}.csv"
    return [root / str(int(shot)) / filename, root / filename, root / "hard_x_rays" / str(int(shot)) / filename]


def _resolve_hxr_file(shot: int, variant: str, data_root: str | Path | None, explicit: str | Path | None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"HXR CSV file not found: {path}")
        return path
    candidates = hxr_file_candidates(shot, variant, data_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find an HXR {variant} CSV for shot {shot}. Searched: {searched}")


def load_hxr_flux_csv(path: str | Path) -> np.ndarray:
    """Read one Eflux CSV as a ``(bins, energies)`` array.

    Rejects anything but one 40 x 5 record: an appended file holds several
    records and nothing says which belongs to the shot.
    """
    data = np.loadtxt(Path(path), delimiter=",", ndmin=2)
    expected = (HXR_BINS_PER_RECORD, len(HXR_FLUX_ENERGIES_KEV))
    if data.shape != expected:
        records = data.shape[0] / HXR_BINS_PER_RECORD
        raise ValueError(
            f"{path}: HXR Eflux shape {data.shape}, expected {expected}. "
            + (
                f"It holds {records:g} appended records; the legacy writer opened the file in "
                "append mode, so which record belongs to the shot is unknown."
                if data.shape[1] == expected[1] and data.shape[0] % HXR_BINS_PER_RECORD == 0
                else "It is not a legacy Eflux record."
            )
        )
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: HXR Eflux contains non-finite values.")
    return data


def load_hxr_raw_csv(path: str | Path, *, sample_rate: float = 125e6 / 128.0) -> dict[str, np.ndarray]:
    """Read a legacy HXR raw CSV (one row per scintillator channel).

    Only shots 40140 (four channels, decimated) and 40142 (one channel, full
    rate -- pass ``sample_rate=125e6``) kept raw traces.  Values are the
    acquisition script's baseline-subtracted, polarity-flipped relative volts.
    Returns ``time`` (s, trigger-relative) and ``data`` shaped
    ``(samples, channels)``.
    """
    rows = np.loadtxt(Path(path), delimiter=",", ndmin=2)
    if rows.shape[0] > HXR_RAW_CHANNELS:
        raise ValueError(f"{path}: {rows.shape[0]} rows; an HXR raw CSV has at most {HXR_RAW_CHANNELS}.")
    data = rows.T
    return {"time": np.arange(data.shape[0]) / float(sample_rate), "data": data}


def hard_x_rays(
    ods: Any,
    shot: int,
    *,
    data_root: str | Path | None = None,
    flux_file: str | Path | None = None,
    time_offset: float | None = None,
    time_reference: str = "auto",
    trigger_settings_path: str | Path | None = None,
) -> None:
    """Map one archived VEST HXR Eflux record into ``hard_x_rays``.

    Fills a single channel whose ``radiance.data`` is ``(energy, time)``, the
    IDS coordinate order.  ``time_offset`` (s) overrides the trigger settings;
    ``time_reference="archive"`` keeps the axis trigger-relative.
    """
    if time_reference not in {"auto", "archive"}:
        raise ValueError("time_reference must be 'auto' or 'archive'.")
    source = _resolve_hxr_file(int(shot), "Eflux", data_root, flux_file)
    flux = load_hxr_flux_csv(source)

    if time_offset is not None:
        alignment = HXRTimeAlignment(float(time_offset), "explicit", "Caller-provided time_offset.")
    elif time_reference == "auto":
        alignment = resolve_hxr_time_alignment(shot, trigger_settings_path=trigger_settings_path)
    else:
        alignment = HXRTimeAlignment(0.0, "trigger_relative", "Archive trigger-relative time requested.")
    # Row i averages [i, i+1) ms after the trigger; the IDS time is its centre.
    time = alignment.offset_seconds + (np.arange(flux.shape[0]) + 0.5) * HXR_BIN_WIDTH_S
    # Data Dictionary codes, as vaft.validation.validity names them:
    # 0 valid, -1 suspect. A negative unfolded flux is a processing artefact.
    validity = np.where(np.any(flux < 0.0, axis=1), -1, 0).astype(int)

    set_path(ods, "hard_x_rays.ids_properties.homogeneous_time", 1)
    set_path(ods, "hard_x_rays.ids_properties.name", "VEST hard X-ray spectrum")
    set_path(ods, "hard_x_rays.ids_properties.comment", f"{_COMMENT} time_alignment={alignment.source} ({alignment.detail}).")
    set_path(ods, "hard_x_rays.ids_properties.source", str(source))
    set_path(ods, "hard_x_rays.ids_properties.creation_date", datetime.now(timezone.utc).isoformat())
    set_path(ods, "hard_x_rays.time", time)

    prefix = "hard_x_rays.channel.0"
    set_path(ods, f"{prefix}.name", "HXR unfolded spectrum")
    set_path(ods, f"{prefix}.identifier", f"{HXR_DAQ_LABEL}:hxr:Eflux")
    for index, energy_kev in enumerate(HXR_FLUX_ENERGIES_KEV):
        set_path(ods, f"{prefix}.energy_band.{index}.energies", np.array([energy_kev * 1e3]))
    set_path(ods, f"{prefix}.radiance.data", flux.T)
    set_path(ods, f"{prefix}.radiance.time", time)
    set_path(ods, f"{prefix}.radiance.validity_timed", validity)
    set_path(ods, f"{prefix}.radiance.validity", int(validity.min()))

    set_path(ods, "hard_x_rays.code.name", "vaft.machine_mapping.hard_x_rays")
    set_path(
        ods,
        "hard_x_rays.code.parameters",
        json.dumps(
            {
                "product": "sample_v4 Eflux",
                "energy_labels_keV": list(HXR_FLUX_ENERGIES_KEV),
                "dropped_energy_keV": [210.0],
                "bin_width_s": HXR_BIN_WIDTH_S,
                "time_convention": "bin centre after trigger",
                "time_alignment": alignment.source,
                "unfolding_model": HXR_UNFOLDING_MODEL,
                "units": "relative intensity [a.u.]",
            },
            sort_keys=True,
        ),
    )


def hard_x_rays_from_flux_csv(shot: int, *, consistency_check: bool = True, **kwargs: Any):
    """Create a shot-level ODS holding one HXR Eflux record."""
    from omas import ODS

    ods = ODS(consistency_check=consistency_check)
    hard_x_rays(ods, shot, **kwargs)
    return ods


__all__ = [
    "HXR_BIN_WIDTH_S",
    "HXR_DAQ_LABEL",
    "HXR_FLUX_ENERGIES_KEV",
    "HXR_MODEL_ENERGIES_KEV",
    "HXR_SCINTILLATOR_THICKNESS_M",
    "HXR_UNFOLDING_MODEL",
    "HXRTimeAlignment",
    "hard_x_rays",
    "hard_x_rays_from_flux_csv",
    "hxr_file_candidates",
    "load_hxr_flux_csv",
    "load_hxr_raw_csv",
    "resolve_hxr_time_alignment",
]
