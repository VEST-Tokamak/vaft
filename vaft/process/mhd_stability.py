"""Post-processing of DCON stability results read back from ``mhd_linear`` (#940).

The inputs are the rows of
:func:`vaft.machine_mapping.mhd_linear.extract_dcon_stability`, so everything
here works from the final ODS without reopening DCON's own files (#940's
scientific-closure contract).

Three questions, kept apart:

* :func:`criterion_intervals`: where a local criterion profile is on its
  unstable side, as intervals in psi_N with interpolated zero crossings. Pure
  numerics, no solver knowledge.
* :func:`dcon_local_stability`: that, applied to DCON's three local criteria
  with DCON's sign conventions (``D_I > 0`` Mercier unstable, ``D_R > 0``
  resistive-interchange unstable, ``C_A < 0`` high-n ballooning unstable). A
  criterion the run did not evaluate stays ``None``, never a marginal zero.
* :func:`dcon_edge_scan` and :func:`dcon_edge_comparison`: DCON's edge scan,
  its peak and truncation, and the full-edge minus truncated W_t of one case.

None of these is a verdict on the plasma. They locate where a solver's
criterion is met; the stability atlas's interpretation layer decides what that
means (#939).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "criterion_intervals",
    "dcon_local_stability",
    "dcon_edge_scan",
    "dcon_edge_comparison",
]

#: DCON's sign conventions: which side of zero each local criterion is unstable on.
DCON_UNSTABLE_SIDE = {"D_I": "positive", "D_R": "positive", "C_A": "negative"}

#: DCON's ``nperq_edge`` default (``dcon_mod.f:124``); VAFT's templates leave it.
DCON_NPERQ_EDGE = 20


def criterion_intervals(psi_n, values, *, unstable_side: str, evaluated=None) -> dict[str, Any]:
    """Where a radial criterion profile is strictly on its unstable side.

    Parameters
    ----------
    psi_n : array_like
        Normalized poloidal flux of each sample, increasing [-].
    values : array_like
        The criterion at each sample [criterion units].
    unstable_side : {"positive", "negative"}
        Which sign is unstable; zero is marginal and never counts as unstable [-].
    evaluated : array_like of bool or None, optional
        Samples the solver actually computed; ``None`` means all of them [-].

    Returns
    -------
    result : dict
        The located unstable side of the profile, keyed as below [-].

        ``unstable_intervals``: list of ``[start, end]`` in psi_N [-], one per
        run of consecutive unstable samples, each end moved to the interpolated
        zero crossing when the neighbouring sample is evaluated and on the
        stable side, otherwise left at the last unstable sample.
        ``zero_crossings``: psi_N of every sign change between two adjacent
        evaluated samples, linearly interpolated [-].
        ``unstable_fraction``: unstable samples over evaluated samples [-].
        ``extremum`` and ``psi_n_at_extremum``: the most unstable value and its
        location [criterion units, -]; the largest value for ``positive``, the
        smallest for ``negative``.
        ``n_evaluated`` [-]. Every entry is ``None`` or empty when no sample is
        evaluated.

    Assumptions
    -----------
    ``psi_n`` increases. Unevaluated or non-finite samples break an interval:
    nothing is interpolated across them.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Linear interpolation between samples; a sign change and back between two
    samples is invisible. The resolution is the solver's radial grid.
    """
    if unstable_side not in ("positive", "negative"):
        raise ValueError(f"unstable_side must be 'positive' or 'negative', not {unstable_side!r}")
    psi = np.asarray(psi_n, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    if psi.shape != v.shape:
        raise ValueError(f"psi_n and values differ in length: {psi.size} vs {v.size}")
    keep = np.isfinite(v) & np.isfinite(psi)
    if evaluated is not None:
        mask = np.asarray(evaluated, dtype=bool).reshape(-1)
        if mask.shape != v.shape:
            raise ValueError(f"evaluated and values differ in length: {mask.size} vs {v.size}")
        keep &= mask
    if psi.size > 1 and np.any(np.diff(psi[np.isfinite(psi)]) < 0):
        raise ValueError("psi_n must increase")
    if not keep.any():
        return {"unstable_intervals": [], "zero_crossings": [], "unstable_fraction": None,
                "extremum": None, "psi_n_at_extremum": None, "n_evaluated": 0}
    signed = v if unstable_side == "positive" else -v
    unstable = keep & (signed > 0)

    def crossing(i: int) -> float:
        # Zero of the line through samples i and i+1 (both kept, opposite signs or one zero).
        a, b = signed[i], signed[i + 1]
        return float(psi[i] + (psi[i + 1] - psi[i]) * a / (a - b)) if a != b else float(psi[i])

    # Strict sign changes between neighbours, plus samples exactly at zero (once each).
    crossings = sorted(
        [crossing(i) for i in range(v.size - 1) if keep[i] and keep[i + 1] and signed[i] * signed[i + 1] < 0]
        + [float(psi[i]) for i in np.flatnonzero(keep & (signed == 0))]
    )
    intervals = []
    i = 0
    while i < v.size:
        if not unstable[i]:
            i += 1
            continue
        j = i
        while j + 1 < v.size and unstable[j + 1]:
            j += 1
        start = crossing(i - 1) if i > 0 and keep[i - 1] else float(psi[i])
        end = crossing(j) if j + 1 < v.size and keep[j + 1] else float(psi[j])
        intervals.append([start, end])
        i = j + 1
    index = np.flatnonzero(keep)[int(np.argmax(signed[keep]))]
    return {
        "unstable_intervals": intervals,
        "zero_crossings": crossings,
        "unstable_fraction": float(unstable.sum() / keep.sum()),
        "extremum": float(v[index]),
        "psi_n_at_extremum": float(psi[index]),
        "n_evaluated": int(keep.sum()),
    }


def dcon_local_stability(row: Mapping[str, Any]) -> dict[str, Optional[dict[str, Any]]]:
    """DCON's three local criteria, located: Mercier, resistive interchange, ballooning.

    Parameters
    ----------
    row : mapping
        One row of :func:`vaft.machine_mapping.mhd_linear.extract_dcon_stability`
        (``psi_n``, ``D_I``, ``D_R``, ``C_A``, ``mercier_evaluated``,
        ``ballooning_evaluated``) [-].

    Returns
    -------
    result : dict
        One located criterion per DCON local criterion, keyed as below [-].

        ``{"mercier": ..., "resistive_interchange": ..., "ballooning": ...}``, each
        the :func:`criterion_intervals` result for ``D_I`` (unstable > 0),
        ``D_R`` (unstable > 0) and ``C_A`` (unstable < 0) [-], plus
        ``criterion`` and ``unstable_side``. ``None`` for a criterion the run
        did not evaluate or whose profile is absent.

    Convention
    ----------
    DCON's signs (``mercier.f``, ``bal.f``): ``D_I > 0`` Mercier unstable,
    ``D_R > 0`` resistive-interchange criterion unstable, ``C_A < 0`` high-n
    ideal ballooning unstable. ``C_A`` is read only where the payload marks it
    evaluated: the NaN entries of an unevaluated surface are skipped, not zero.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The payload carries no per-surface mask for ``D_I``/``D_R``; ``mer_flag``
    evaluates all of them, so the flag alone decides.

    Provenance
    ----------
    .. [940] Issue #940, section 8 (local-stability post-processing).
    """
    psi = row.get("psi_n")
    flags = {"D_I": row.get("mercier_evaluated"), "D_R": row.get("mercier_evaluated"),
             "C_A": row.get("ballooning_evaluated")}
    names = {"D_I": "mercier", "D_R": "resistive_interchange", "C_A": "ballooning"}
    out: dict[str, Optional[dict[str, Any]]] = {}
    for criterion, name in names.items():
        values = row.get(criterion)
        if flags[criterion] is not True or values is None or psi is None:
            out[name] = None
            continue
        side = DCON_UNSTABLE_SIDE[criterion]
        out[name] = {"criterion": criterion, "unstable_side": side,
                     **criterion_intervals(psi, values, unstable_side=side)}
    return out


def dcon_edge_scan(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """DCON's edge scan: where Re dW_edge peaks, where it changes sign, where DCON truncated.

    Parameters
    ----------
    row : mapping
        One row of :func:`vaft.machine_mapping.mhd_linear.extract_dcon_stability`
        (``edge_scan``, ``requested_psiedge``, ``psilim``, ``qlim``) [-].

    Returns
    -------
    result : dict or None
        ``None`` for a full-edge run (no scan). Otherwise:
        ``psi_n_at_peak``, ``q_at_peak``, ``dW_at_peak``: the maximum of
        Re dW_edge over the filled scan entries [-, -, normalized];
        ``psilim``, ``qlim``: the truncation DCON used [-];
        ``truncated_at_peak``: whether ``psilim`` equals the peak's psi_N
        within 1e-9 [-];
        ``zero_crossings_psi_n``: psi_N where Re dW_edge changes sign [-];
        ``negative_intervals``: psi_N intervals with Re dW_edge < 0 [-];
        ``n_points`` [-]; ``peak_search_start``: index of the first searched
        entry [-].

    Convention
    ----------
    dW_edge is DCON's least-stable total energy against the edge truncation,
    normalized as W_t is (not Joules). Negative is unstable. The peak is taken
    as ``dcon.F:253`` does, ``MAXLOC(REAL(dw_edge(pre_edge:i_edge)))``: from
    ``pre_edge`` on and over filled entries only (unfilled entries keep
    ``psi_edge = 0``, ``sing.f:232``). ``pre_edge`` skips the nominal grid
    points ``INT(q(psiedge)) + i/(nperq_edge*n)`` below ``q(psiedge)``
    (``sing.f:226-238``); those entries are filled but never searched. The
    zero crossings and negative intervals cover every filled entry.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    One scan per run; DCON keeps only the truncated solution's eigenvectors.
    The payload has no q profile, so ``q(psiedge)`` is the q of the first
    filled entry, one ODE step past psiedge; a nominal grid point inside that
    step would shift ``pre_edge`` by one. ``nperq_edge`` is DCON's default 20,
    which VAFT's templates never change.

    Provenance
    ----------
    .. [940] Issue #940, section 10 (edge-scan post-processing).
    """
    scan = row.get("edge_scan")
    if scan is None:
        return None
    psi = np.asarray(scan["psi_n"], dtype=float)
    q = np.asarray(scan["q"], dtype=float)
    dw = np.asarray(scan["dW"])
    filled = (psi > 0) & np.isfinite(np.real(dw))
    n_tor = row.get("n_tor")
    if not filled.any() or not n_tor or n_tor <= 0:
        return None
    q_at_psiedge = float(q[np.flatnonzero(filled)[0]])
    nominal = int(q_at_psiedge) + np.arange(psi.size) / (DCON_NPERQ_EDGE * int(n_tor))
    first = int(np.count_nonzero(nominal < q_at_psiedge))
    searched = filled & (np.arange(psi.size) >= first)
    if not searched.any():
        return None
    peak = np.flatnonzero(searched)[int(np.argmax(np.real(dw[searched])))]
    psilim, qlim = row.get("psilim"), row.get("qlim")
    signs = criterion_intervals(psi[filled], np.real(dw[filled]), unstable_side="negative")
    return {
        "psi_n_at_peak": float(psi[peak]),
        "q_at_peak": float(q[peak]),
        "dW_at_peak": complex(dw[peak]),
        "psilim": psilim,
        "qlim": qlim,
        "truncated_at_peak": None if psilim is None else math.isclose(float(psi[peak]), psilim, rel_tol=0, abs_tol=1e-9),
        "zero_crossings_psi_n": signs["zero_crossings"],
        "negative_intervals": signs["unstable_intervals"],
        "n_points": int(filled.sum()),
        "peak_search_start": first,
    }


def dcon_edge_comparison(full: Mapping[str, Any], truncated: Mapping[str, Any]) -> dict[str, Any]:
    """Full-edge minus peak-dW-truncated least-stable W_t of one case (#792, #940).

    Parameters
    ----------
    full : mapping
        The full-edge row of
        :func:`vaft.machine_mapping.mhd_linear.extract_dcon_stability` [-].
    truncated : mapping
        The peak-dW-truncated row of the same case and n [-].

    Returns
    -------
    result : dict
        ``n_tor`` [-]; ``W_t_full``, ``W_t_truncated`` and ``delta_W_edge`` =
        ``Re W_t_full - Re W_t_truncated`` [normalized, not Joules];
        ``sign_agreement``: whether both real parts have the same strict sign
        [-]; ``psilim_full``, ``psilim_truncated`` [-].

    Convention
    ----------
    The difference is of two separate solutions, each with its own boundary:
    DCON's full re-run after truncation discards the full-edge eigenvectors, so
    nothing here is a mode-by-mode comparison. Both energies share DCON's
    normalization only because they come from the same equilibrium and n.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The rows must be the same case: same ``n_tor`` and treatments
    ``full_edge``/``peak_dw_truncated``; anything else raises. Same equilibrium
    is the caller's to ensure (the payload has no equilibrium hash).

    Provenance
    ----------
    .. [792] Issue #792, the full-edge vs truncated comparison.
    .. [940] Issue #940, ``DconEdgeAnalysis`` follow-up.
    """
    if full.get("edge_treatment") != "full_edge" or truncated.get("edge_treatment") != "peak_dw_truncated":
        raise ValueError("expected one full_edge and one peak_dw_truncated row, got "
                         f"{full.get('edge_treatment')!r} and {truncated.get('edge_treatment')!r}")
    if full.get("n_tor") != truncated.get("n_tor"):
        raise ValueError(f"rows are for different n: {full.get('n_tor')} vs {truncated.get('n_tor')}")
    a, b = full.get("W_t"), truncated.get("W_t")
    delta = None if a is None or b is None else float(np.real(a) - np.real(b))
    agree = None if delta is None else bool(np.real(a) * np.real(b) > 0)
    return {
        "n_tor": full.get("n_tor"),
        "W_t_full": a,
        "W_t_truncated": b,
        "delta_W_edge": delta,
        "sign_agreement": agree,
        "psilim_full": full.get("psilim"),
        "psilim_truncated": truncated.get("psilim"),
    }
