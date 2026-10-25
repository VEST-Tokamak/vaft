"""The archived legacy pipeline against the current one: slice pairing and paired statistics.

The legacy archive is the shot-first FileDB ``public/<shot>/`` that held
pre-VAFT reconstructions (EFIT with legacy weights, CHEASE, DCON/RDCON). The
current pipeline is HSDS ``main`` with the ``statistical_891`` EFIT preset and
the stability atlas. Both are scored by the same validation kernels. This
module only decides which slices are comparable and how to summarise the
differences; it computes no physics.

Paired by time, never by position
---------------------------------

The two pipelines chose their own slice times. A legacy shot can have 8 slices
where the current one has 17, so zipping the two lists pairs unrelated
equilibria and is the repository's most repeated defect. :func:`pair_slices`
matches each legacy slice to the nearest current slice within a tolerance,
one to one, closest first. Slices left over are reported, never dropped, so
a population view can still use them.

What "better" means
-------------------

Every metric declares its direction in :data:`METRICS`: a residual is better
lower, an agreement ratio is better closer to zero in log, a stability energy
has no direction (it is a physical result, not a quality). The paired
statistics use that direction to report how often the current pipeline
improved a slice. A metric without a direction reports the shift only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "METRICS",
    "PAIR_TOLERANCE_S",
    "SlicePair",
    "SlicePairing",
    "metric_direction",
    "pair_slices",
    "paired_summary",
    "population_summary",
]

#: Default pairing tolerance [s]: half an EFIT slice spacing of 1 ms.
PAIR_TOLERANCE_S = 5.0e-4

#: Metric directions and labels. ``better`` is ``"lower"``, ``"zero"`` (|value|
#: lower is better), or ``None`` for a physical result with no quality order.
METRICS: dict[str, dict[str, Any]] = {
    "vacuum_normalized_residual": {"better": "lower", "label": "vacuum residual / dynamic range",
                                   "check": "vacuum"},
    "vacuum_improvement": {"better": None, "label": "eddy-term RMS improvement", "check": "vacuum"},
    "probe_relative_rms": {"better": "lower", "label": "probe |m - r| RMS / |m| RMS", "check": "fit"},
    "loop_relative_rms": {"better": "lower", "label": "flux-loop |m - r| RMS / |m| RMS", "check": "fit"},
    "probe_relative_rms_common": {"better": "lower", "label": "probe relative RMS, channels both fitted",
                                  "check": "fit"},
    "loop_relative_rms_common": {"better": "lower", "label": "flux-loop relative RMS, channels both fitted",
                                 "check": "fit"},
    "ip_relative_error": {"better": "lower", "label": "|I_p,m - I_p,r| / |I_p,m|", "check": "fit"},
    "dia_relative_error": {"better": "lower", "label": "diamagnetic |m - r| / |m|", "check": "fit"},
    "virial_log_ratio": {"better": "zero", "label": "ln(beta_p,int / beta_p,virial)", "check": "virial"},
    "li_log_ratio": {"better": "zero", "label": "ln(l_i,int / l_i,virial)", "check": "virial"},
    "gs_residual": {"better": "lower", "label": "Grad-Shafranov residual (whole)", "check": "virial"},
    "thomson_log_ratio": {"better": "zero", "label": "ln(sum p_e / sum p_EFIT)", "check": "thomson"},
    "w_t": {"better": None, "label": "DCON W_t (least stable)", "check": "stability"},
    "delta_prime_max": {"better": None, "label": "max Delta' over surfaces", "check": "stability"},
}


@dataclass(frozen=True)
class SlicePair:
    """One legacy slice and the current slice nearest it in time."""

    legacy_index: int
    current_index: int
    legacy_time: float
    current_time: float

    @property
    def offset(self) -> float:
        return self.current_time - self.legacy_time


@dataclass(frozen=True)
class SlicePairing:
    """Pairs within tolerance and the slices of each side that found no partner."""

    pairs: tuple[SlicePair, ...]
    unpaired_legacy: tuple[int, ...]
    unpaired_current: tuple[int, ...]
    tolerance: float


def pair_slices(legacy_times: Iterable[float], current_times: Iterable[float], *,
                tolerance: float = PAIR_TOLERANCE_S) -> SlicePairing:
    """Match slices by time, one to one, closest pairs first.

    Every candidate pair within ``tolerance`` is ranked by ``|dt|`` and taken
    greedily while neither slice is used, so a current slice that sits between
    two legacy ones goes to the nearer. Non-finite times never pair.
    """
    a = np.asarray(list(legacy_times), dtype=float).reshape(-1)
    b = np.asarray(list(current_times), dtype=float).reshape(-1)
    tolerance = float(tolerance)
    if not (tolerance >= 0.0):
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    candidates = []
    for i, ta in enumerate(a):
        if not math.isfinite(ta):
            continue
        for j, tb in enumerate(b):
            if math.isfinite(tb) and abs(tb - ta) <= tolerance:
                candidates.append((abs(tb - ta), i, j))
    candidates.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = []
    for _, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append(SlicePair(i, j, float(a[i]), float(b[j])))
    pairs.sort(key=lambda p: p.legacy_time)
    return SlicePairing(
        pairs=tuple(pairs),
        unpaired_legacy=tuple(i for i in range(a.size) if i not in used_a),
        unpaired_current=tuple(j for j in range(b.size) if j not in used_b),
        tolerance=tolerance,
    )


def _oriented(values: np.ndarray, better: str | None) -> np.ndarray:
    """Values on a scale where smaller is better (``|x|`` for a ``"zero"`` metric)."""
    return np.abs(values) if better == "zero" else values


def _bootstrap_median_ci(values: np.ndarray, *, n_boot: int, confidence: float,
                         rng: np.random.Generator) -> tuple[float, float]:
    if values.size < 2:
        return (float("nan"), float("nan"))
    draws = rng.integers(0, values.size, size=(int(n_boot), values.size))
    medians = np.median(values[draws], axis=1)
    tail = 0.5 * (1.0 - float(confidence))
    low, high = np.quantile(medians, [tail, 1.0 - tail])
    return (float(low), float(high))


def paired_summary(legacy: Sequence[float], current: Sequence[float], *, better: str | None = "lower",
                   n_boot: int = 2000, confidence: float = 0.95, seed: int = 0) -> dict[str, Any]:
    """Medians, the paired shift and its significance for one metric on matched slices.

    ``legacy[k]`` and ``current[k]`` must be the same slice pair. Pairs where
    either value is not finite are left out and counted. ``shift`` is
    ``current - legacy`` on the oriented scale (``|x|`` for a ``"zero"``
    metric), so a negative shift is an improvement whenever ``better`` is set.
    The p-value is the two-sided Wilcoxon signed-rank test of that shift
    (``nan`` below 6 pairs or when every shift is zero). The confidence
    interval is a percentile bootstrap of the median shift.
    """
    from scipy.stats import wilcoxon

    a = np.asarray(legacy, dtype=float).reshape(-1)
    b = np.asarray(current, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"paired metrics need equal lengths, got {a.size} and {b.size}")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    oa, ob = _oriented(a, better), _oriented(b, better)
    shift = ob - oa
    rng = np.random.default_rng(seed)
    p_value = float("nan")
    if shift.size >= 6 and np.any(shift != 0.0):
        p_value = float(wilcoxon(shift, zero_method="wilcox", alternative="two-sided").pvalue)
    improved = None
    if better is not None and shift.size:
        improved = float(np.mean(shift < 0.0))
    return {
        "n_pairs": int(shift.size),
        "n_dropped": int((~keep).sum()),
        "better": better,
        "legacy_median": float(np.median(oa)) if oa.size else float("nan"),
        "current_median": float(np.median(ob)) if ob.size else float("nan"),
        "shift_median": float(np.median(shift)) if shift.size else float("nan"),
        "shift_ci": _bootstrap_median_ci(shift, n_boot=n_boot, confidence=confidence, rng=rng),
        "confidence": float(confidence),
        "wilcoxon_p": p_value,
        "improved_fraction": improved,
    }


def population_summary(legacy: Sequence[float], current: Sequence[float], *, better: str | None = "lower",
                       n_boot: int = 2000, confidence: float = 0.95, seed: int = 0) -> dict[str, Any]:
    """Two independent populations: medians with bootstrap intervals and a Mann-Whitney test.

    For the full archive against the full current set, where most slices have
    no partner. Says nothing about any one slice; read it beside
    :func:`paired_summary`, because the two populations differ in which shots
    reached a reconstruction at all.
    """
    from scipy.stats import mannwhitneyu

    a = _oriented(np.asarray(legacy, dtype=float).reshape(-1), better)
    b = _oriented(np.asarray(current, dtype=float).reshape(-1), better)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    rng = np.random.default_rng(seed)
    p_value = float("nan")
    if a.size >= 3 and b.size >= 3:
        p_value = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return {
        "n_legacy": int(a.size),
        "n_current": int(b.size),
        "better": better,
        "legacy_median": float(np.median(a)) if a.size else float("nan"),
        "current_median": float(np.median(b)) if b.size else float("nan"),
        "legacy_ci": _bootstrap_median_ci(a, n_boot=n_boot, confidence=confidence, rng=rng),
        "current_ci": _bootstrap_median_ci(b, n_boot=n_boot, confidence=confidence, rng=rng),
        "confidence": float(confidence),
        "mannwhitney_p": p_value,
    }


def metric_direction(metric: str, metrics: Mapping[str, Mapping[str, Any]] = METRICS) -> str | None:
    """The declared direction of ``metric``; an unknown name is an error, not a guess."""
    try:
        return metrics[metric]["better"]
    except KeyError:
        raise KeyError(f"unknown comparison metric {metric!r}; known: {sorted(metrics)}") from None
