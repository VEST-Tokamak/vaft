"""Synthetic plasma-timing products shared by the #409 tests.

Not a test module (the leading underscore keeps pytest from collecting it):
the builders here make an ODS on the analysis grid with an H-alpha-like
emission, a plasma-current pulse, or coil-firing pickup alone, so the omas
timing tests and the consumers' tests describe the same shots.
"""
from __future__ import annotations

import gzip
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from omas import ODS, load_omas_json

DT = 4e-5
RNG = np.random.default_rng(409)


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


def pipeline_ods(shot: int) -> ODS:
    """The packaged ``pipeline-until-efit`` product of ``shot``, or a skip."""
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


