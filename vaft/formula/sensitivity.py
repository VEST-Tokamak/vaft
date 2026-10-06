"""Generic kernels for local sensitivity, linear uncertainty propagation and identifiability (issue #1642).

The mathematics shared by every VAFT domain that linearizes something: a
Jacobian by finite differences, covariance pushed through a Jacobian, the
same push by sampling, the singular-value spectrum of a Jacobian, and a test
of whether the linearization still describes a finite step.

Conventions
-----------
* A forward model ``f`` maps a 1-D parameter vector ``x`` [n] to a 1-D output
  vector ``y`` [m]; a scalar output is treated as ``m = 1``.  The Jacobian is
  ``J[i, j] = dy_i/dx_j`` with shape ``(m, n)``.
* Nothing here knows what the parameters mean.  Which perturbation is
  physical, which step is admissible and what the derivative is *for* belong to
  the domain that owns ``f`` (#1642 s12); interpreting the result belongs to
  :mod:`vaft.validation.sensitivity`.
* Every function is cheap except :func:`monte_carlo_propagation`, which calls
  ``f`` once per sample and is therefore never invoked by anything else in
  VAFT: it exists to be called explicitly (#1639 s8).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "finite_difference_jacobian",
    "linear_covariance_propagation",
    "linearity_ratio",
    "monte_carlo_propagation",
    "singular_value_spectrum",
]


def _vector(value) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=float)).ravel()


def finite_difference_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    step: float | np.ndarray | None = None,
    scheme: str = "central",
) -> np.ndarray:
    r"""Jacobian of a vector function by forward or central differences.

    $$J_{ij} \approx \frac{f_i(x + h_j e_j) - f_i(x - h_j e_j)}{2 h_j}
      \quad\text{(central)}, \qquad
      J_{ij} \approx \frac{f_i(x + h_j e_j) - f_i(x)}{h_j} \quad\text{(forward)}$$

    Parameters
    ----------
    f : callable
        Forward model taking a parameter vector and returning an output vector [any].
    x : array-like
        Point of linearization [any].
    step : float or array-like, optional
        Absolute step per parameter; default
        $h_j = \epsilon^{1/3}\max(1, |x_j|)$ (central) or
        $\epsilon^{1/2}\max(1, |x_j|)$ (forward) [any].
    scheme : str, optional
        ``"central"`` (second order) or ``"forward"`` (first order); either
        costs n + 1 or 2n + 1 evaluations, the extra one fixing the output
        size [-].

    Returns
    -------
    numpy.ndarray
        ``(m, n)`` Jacobian [any].

    Raises
    ------
    ValueError
        On an unknown scheme or a non-positive step.

    Numerical notes
    ---------------
    The default steps balance truncation against round-off for a model
    evaluated to machine precision.  They are absolute and floored at one
    unit, so a parameter far below unit magnitude (``1e-8``) is stepped far
    beyond its own size -- possibly across a domain edge such as a sign or a
    logarithm; give such a parameter its own ``step``.  A model with its own solver tolerance
    (an iterative equilibrium, an external code) needs a step well above that
    tolerance, chosen by its owner: a finite difference across a convergence
    threshold measures the solver, not the physics (#1642 s18).

    References
    ----------
    .. [1] J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed.,
           Springer (2006), Sec. 8.1.
    """
    if scheme not in ("central", "forward"):
        raise ValueError(f"scheme must be 'central' or 'forward', got {scheme!r}")
    x0 = _vector(x)
    eps = np.finfo(float).eps
    if step is None:
        h = (eps ** (1 / 3) if scheme == "central" else eps ** 0.5) * np.maximum(1.0, np.abs(x0))
    else:
        h = np.broadcast_to(np.asarray(step, dtype=float), x0.shape).copy()
    if np.any(~np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("finite-difference steps must be finite and positive")
    f0 = _vector(f(x0.copy()))
    jac = np.empty((f0.size, x0.size))
    for j in range(x0.size):
        up = x0.copy()
        up[j] += h[j]
        if scheme == "central":
            down = x0.copy()
            down[j] -= h[j]
            jac[:, j] = (_vector(f(up)) - _vector(f(down))) / (2.0 * h[j])
        else:
            jac[:, j] = (_vector(f(up)) - f0) / h[j]
    return jac


def _covariance(covariance, n: int) -> np.ndarray:
    """A square, symmetric (to round-off) covariance of size ``n``."""
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim == 1:
        cov = np.diag(cov)
    if cov.shape != (n, n):
        raise ValueError(f"covariance {cov.shape} does not match {n} parameters")
    scale = float(np.max(np.abs(cov))) if cov.size else 0.0
    if not np.allclose(cov, cov.T, rtol=0.0, atol=1e-12 * max(scale, np.finfo(float).tiny)):
        raise ValueError("input covariance must be symmetric")
    return 0.5 * (cov + cov.T)


def linear_covariance_propagation(jacobian: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    r"""Output covariance of a linearized model, correlations included.

    $$\Sigma_y = J\,\Sigma_x\,J^{\mathsf T}$$

    Parameters
    ----------
    jacobian : array-like
        ``(m, n)`` sensitivity matrix [any].
    covariance : array-like
        ``(n, n)`` input covariance; a 1-D array is read as independent
        variances [any].

    Returns
    -------
    numpy.ndarray
        ``(m, m)`` output covariance, symmetrized [any].

    Raises
    ------
    ValueError
        When the shapes do not agree or the input covariance is not symmetric
        to round-off (relative ``1e-12`` of its largest entry).

    Assumptions
    -----------
    The model is linear over the spread of the inputs; :func:`linearity_ratio`
    and :func:`monte_carlo_propagation` test that.

    References
    ----------
    .. [1] JCGM 100:2008, *Evaluation of measurement data -- Guide to the
           expression of uncertainty in measurement*, Sec. 5.2, Eq. (13).
    """
    jac = np.atleast_2d(np.asarray(jacobian, dtype=float))
    cov = _covariance(covariance, jac.shape[1])
    out = jac @ cov @ jac.T
    return 0.5 * (out + out.T)


def monte_carlo_propagation(
    f: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    covariance: np.ndarray,
    samples: int = 2000,
    seed: int = 0,
) -> dict:
    r"""Output mean and covariance by sampling Gaussian inputs through the full model.

    $$x^{(k)} \sim \mathcal N(\mu, \Sigma_x), \qquad
      \hat\Sigma_y = \frac{1}{K-1}\sum_k (y^{(k)} - \bar y)(y^{(k)} - \bar y)^{\mathsf T}$$

    Parameters
    ----------
    f : callable
        Forward model, called once per sample [any].
    mean : array-like
        Input mean [any].
    covariance : array-like
        Input covariance, positive semi-definite; 1-D means independent
        variances [any].
    samples : int, optional
        Number of draws $K$ [-].
    seed : int, optional
        Seed of the NumPy generator, so a result is reproducible [-].

    Returns
    -------
    dict
        ``mean`` and ``covariance`` of the finite outputs, ``samples`` used,
        ``rejected`` (draws whose output was not finite or whose evaluation
        raised) [any].

    Raises
    ------
    ValueError
        When ``samples`` is below 2, or the covariance does not match the
        mean, is not symmetric, or is not positive semi-definite.

    Assumptions
    -----------
    Gaussian inputs.  Expensive: ``samples`` model evaluations, so it is never
    called implicitly (#1639 s8); the cost is the caller's decision.

    Limitations
    -----------
    The statistics are over the *accepted* draws only, so a model that fails
    on part of the input distribution yields a covariance conditioned on
    success; ``rejected`` says how much was conditioned away.  An output with
    infinite variance (a pole inside the input distribution) has a sample
    covariance that does not converge with ``samples``.

    References
    ----------
    .. [1] JCGM 101:2008, *Propagation of distributions using a Monte Carlo
           method*, Sec. 7.
    """
    if samples < 2:
        raise ValueError("Monte Carlo propagation needs at least two samples")
    mu = _vector(mean)
    cov = _covariance(covariance, mu.size)
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues.size and eigenvalues.min() < -1e-12 * max(abs(eigenvalues.max()), np.finfo(float).tiny):
        raise ValueError("input covariance must be positive semi-definite")
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mu, cov, size=int(samples), method="eigh")
    outputs, rejected = [], 0
    for draw in draws:
        try:
            value = _vector(f(draw))
        except Exception:  # a failed evaluation is a rejected draw, counted, not fatal
            rejected += 1
            continue
        if np.all(np.isfinite(value)):
            outputs.append(value)
        else:
            rejected += 1
    if len(outputs) < 2:
        width = outputs[0].size if outputs else 0
        return {"mean": np.full(width, np.nan), "covariance": np.full((width, width), np.nan),
                "samples": len(outputs), "rejected": rejected}
    kept = np.array(outputs)
    return {"mean": kept.mean(axis=0), "covariance": np.atleast_2d(np.cov(kept, rowvar=False)),
            "samples": int(kept.shape[0]), "rejected": rejected}


def singular_value_spectrum(
    jacobian: np.ndarray,
    column_scale: np.ndarray | None = None,
    rtol: float | None = None,
) -> dict:
    r"""Singular values, numerical rank and null space of a (column-scaled) Jacobian.

    $$J D = U \Sigma V^{\mathsf T}, \qquad
      \operatorname{rank} = \#\{\sigma_k > \tau\,\sigma_1\}, \qquad
      \kappa = \sigma_1/\sigma_{\min}$$

    Parameters
    ----------
    jacobian : array-like
        ``(m, n)`` sensitivity matrix, ideally already weighted by the inverse
        measurement uncertainty so its rows are commensurate [any].
    column_scale : array-like, optional
        Diagonal of $D$, e.g. a typical magnitude per parameter, making the
        columns commensurate; default no scaling [any].
    rtol : float, optional
        Relative rank tolerance $\tau$; default $\max(m, n)\,\epsilon$ [-].

    Returns
    -------
    dict
        ``singular_values`` (descending), ``rank``, ``nullity`` (n - rank),
        ``condition_number`` (inf when rank deficient), ``null_space``
        (``(n, nullity)`` basis in the *unscaled* parameters) [any].

    Raises
    ------
    ValueError
        When the matrix is not finite or ``column_scale`` has a zero.

    Numerical notes
    ---------------
    The SVD is taken of $JD$ itself, never of $J^{\mathsf T}J$, which would
    square the condition number -- the practice of
    :func:`vaft.code.efit.analyze_efit_identifiability`.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed.,
           Johns Hopkins (2013), Secs. 2.4 and 5.4.
    """
    jac = np.atleast_2d(np.asarray(jacobian, dtype=float))
    if not np.all(np.isfinite(jac)):
        raise ValueError("Jacobian must be finite")
    n = jac.shape[1]
    scale = np.ones(n) if column_scale is None else _vector(column_scale)
    if scale.shape != (n,) or np.any(scale == 0) or not np.all(np.isfinite(scale)):
        raise ValueError("column_scale must be finite, non-zero and one per column")
    _, sigma, vt = np.linalg.svd(jac * scale, full_matrices=True)
    tol = (max(jac.shape) * np.finfo(float).eps) if rtol is None else float(rtol)
    rank = int(np.sum(sigma > tol * sigma[0])) if sigma.size and sigma[0] > 0 else 0
    null = (vt[rank:].T * scale[:, None]) if rank < n else np.zeros((n, 0))
    if null.shape[1]:
        null = null / np.linalg.norm(null, axis=0)
    condition = float(sigma[0] / sigma[n - 1]) if rank == n else float("inf")
    return {"singular_values": sigma, "rank": rank, "nullity": n - rank,
            "condition_number": condition, "null_space": null}


def linearity_ratio(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    step: np.ndarray,
    jacobian: np.ndarray,
    output_scale: np.ndarray | None = None,
) -> float:
    r"""How much of a finite response the linearization misses.

    $$r = \frac{\lVert W(f(x + \delta) - f(x) - J\delta)\rVert}{\lVert W(f(x + \delta) - f(x))\rVert},
      \qquad W = \operatorname{diag}(1/s_i)$$

    Parameters
    ----------
    f : callable
        Forward model [any].
    x : array-like
        Point of linearization [any].
    step : array-like
        Finite perturbation $\delta$, typically one standard deviation along
        a direction of interest [any].
    jacobian : array-like
        $J$ at ``x`` [any].
    output_scale : array-like, optional
        Typical magnitude $s_i$ per output, so outputs in different units
        weigh alike; default unweighted [any].

    Returns
    -------
    float
        $r$; 0 for an exactly linear response, of order 1 when the local
        Jacobian no longer describes the step; ``nan`` when the response is
        zero [-].

    Assumptions
    -----------
    One direction is tested per call; a model can be linear along one and not
    another.
    """
    x0 = _vector(x)
    delta = _vector(step)
    f0 = _vector(f(x0.copy()))
    response = _vector(f(x0 + delta)) - f0
    predicted = np.atleast_2d(np.asarray(jacobian, dtype=float)) @ delta
    weight = np.ones_like(response) if output_scale is None else 1.0 / _vector(output_scale)
    norm = float(np.linalg.norm(weight * response))
    if norm == 0.0 or not np.isfinite(norm):
        return float("nan")
    return float(np.linalg.norm(weight * (response - predicted)) / norm)
