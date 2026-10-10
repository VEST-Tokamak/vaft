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
* :func:`propagate_formula_uncertainty` connects these kernels to one public
  ``vaft.formula`` function by its parameter names (#1874).  It is opt-in: no
  formula calls it, and a formula's own signature, values and cost do not
  change.  A formula may carry a domain-owned analytic Jacobian
  (:mod:`vaft.formula._propagation`); otherwise the Jacobian is a scale-aware
  finite difference.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "FormulaPropagation",
    "finite_difference_jacobian",
    "linear_covariance_propagation",
    "linearity_ratio",
    "monte_carlo_propagation",
    "propagate_formula_uncertainty",
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


# ---------------------------------------------------------------------------
# per-formula propagation (#1874)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormulaPropagation:
    """The result of :func:`propagate_formula_uncertainty`: values with their provenance.

    ``value`` is the formula evaluated at the input means, in its own shape;
    ``covariance`` is the output covariance over the flattened outputs, rows
    and columns in ``value.ravel()`` order; ``std`` its diagonal square root.
    ``jacobian`` rows are outputs and columns follow ``input_names``; it is
    ``None`` for a Monte Carlo propagation.  ``method`` is how the covariance
    was obtained (``"linear"`` or ``"monte_carlo"``), ``derivative_method``
    how the Jacobian was (``"analytic"`` or ``"finite_difference"``, ``None``
    for Monte Carlo).  ``linearity`` is the largest :func:`linearity_ratio`
    over a one-sigma step along each input (``None`` for Monte Carlo): the
    covariance is a local first-order result, and a ratio near 1 means that
    approximation does not describe the input spread.  ``mc_mean``,
    ``samples`` and ``rejected`` are the Monte Carlo statistics; the
    covariance then describes only the accepted draws.
    """

    value: np.ndarray
    covariance: np.ndarray
    std: np.ndarray
    input_names: tuple
    input_covariance: np.ndarray
    method: str
    jacobian: Optional[np.ndarray] = None
    derivative_method: Optional[str] = None
    linearity: Optional[float] = None
    mc_mean: Optional[np.ndarray] = None
    samples: Optional[int] = None
    rejected: Optional[int] = None
    fixed: Mapping[str, object] = field(default_factory=dict)


def _relative_steps(x: np.ndarray, names: Sequence[str], fd_step) -> np.ndarray:
    """A step per input scaled to its own magnitude, never one step for every unit."""
    if fd_step is not None:
        if isinstance(fd_step, Mapping):
            missing = [name for name in names if name not in fd_step]
            extra = [name for name in fd_step if name not in names]
            if missing or extra:
                raise ValueError(f"fd_step must name exactly the uncertain inputs {', '.join(names)}")
            return np.array([float(fd_step[name]) for name in names])
        steps = np.asarray(fd_step, dtype=float)
        if steps.ndim == 0:
            return np.full(x.shape, float(steps))
        if steps.shape != x.shape:
            raise ValueError(f"fd_step has {steps.size} entries for {x.size} inputs")
        return steps.copy()
    zero = [name for name, value in zip(names, x) if value == 0.0]
    if zero:
        raise ValueError(f"no scale for a finite-difference step at {', '.join(zero)} = 0; give fd_step")
    return np.finfo(float).eps ** (1 / 3) * np.abs(x)


def _linearity(evaluate, x0: np.ndarray, f0: np.ndarray, sigma: np.ndarray, jac: np.ndarray) -> Optional[float]:
    """Largest linearization miss (as :func:`linearity_ratio`) over a +/- one-sigma step along each input.

    A step that leaves the formula's domain (outside its declared domain, an
    exception or a non-finite output) makes the linearization meaningless
    there: ``inf``.  A step whose response is zero while the Jacobian predicts
    one is ``inf`` too.  ``None`` when every input is exact, so nothing was
    tested.  One model call per step.
    """
    worst: Optional[float] = None
    for j, s in enumerate(sigma):
        if s == 0:
            continue
        for sign in (1.0, -1.0):
            step = np.zeros_like(x0)
            step[j] = sign * s
            try:
                shifted = _vector(evaluate(x0 + step))
            except Exception:  # noqa: BLE001 - leaving the domain is the finding
                return float("inf")
            if not np.all(np.isfinite(shifted)):
                return float("inf")
            response = shifted - f0
            predicted = jac @ step
            norm = float(np.linalg.norm(response))
            if norm == 0.0:
                ratio = 0.0 if float(np.linalg.norm(predicted)) == 0.0 else float("inf")
            else:
                ratio = float(np.linalg.norm(response - predicted) / norm)
            worst = ratio if worst is None else max(worst, ratio)
    return worst


class _OutsideDomain(ValueError):
    """An evaluation point the formula declares itself undefined at."""


def propagate_formula_uncertainty(
    formula: Callable,
    inputs: Mapping[str, object],
    *,
    input_names: Optional[Sequence[str]] = None,
    covariance=None,
    std=None,
    method: str = "linear",
    derivative: str = "auto",
    fd_step=None,
    domain: Optional[Callable[..., bool]] = None,
    samples: int = 2000,
    seed: int = 0,
) -> FormulaPropagation:
    r"""Propagate input uncertainty through one ``vaft.formula`` function, correlations included.

    $$\Sigma_y \approx J\,\Sigma_x\,J^{\mathsf T}, \qquad J_{ij} = \partial f_i/\partial x_j$$

    Parameters
    ----------
    formula : callable
        A public formula, called with keyword arguments [any].
    inputs : mapping
        Every argument the call needs, by parameter name; uncertain inputs at
        their mean values [any].
    input_names : sequence of str, optional
        The uncertain inputs, in the order of the covariance axes; every other
        entry of ``inputs`` is fixed configuration. Default: the numeric
        scalar entries of ``inputs`` in the formula's parameter order [-].
    covariance : array-like, optional
        ``(n, n)`` covariance of the inputs in ``input_names`` order, or a 1-D
        array of *variances*. Exactly one of ``covariance`` and ``std`` [any].
    std : array-like, optional
        1-D *standard deviations* of independent inputs, ``input_names``
        order [any].
    method : str, optional
        ``"linear"`` (first order, $J\Sigma J^{\mathsf T}$) or
        ``"monte_carlo"`` (Gaussian sampling through the formula) [-].
    derivative : str, optional
        ``"auto"`` (analytic when the formula carries one, else finite
        difference), ``"analytic"`` (fail if it carries none) or
        ``"finite_difference"`` [-].
    fd_step : float, array-like or mapping, optional
        Absolute finite-difference step per input; default
        $\epsilon^{1/3}|x_j|$, scaled to each input's own magnitude [any].
    domain : callable, optional
        Predicate on the call's arguments, ``True`` where the formula is
        physically defined; default the domain the formula declares
        (:mod:`vaft.formula._propagation`), else none [-].
    samples : int, optional
        Monte Carlo draws [-].
    seed : int, optional
        Monte Carlo seed [-].

    Returns
    -------
    FormulaPropagation
        Value, covariance, standard deviation and their provenance [any].

    Raises
    ------
    ValueError
        No uncertainty given (unknown is not zero), both ``covariance`` and
        ``std``, an uncertain input missing, non-scalar, unknown to the
        formula or non-finite, an uncertainty that is NaN, infinite, negative,
        not symmetric or not positive semi-definite, input means outside the
        formula's domain (declared, or a raised or non-finite evaluation), a
        non-finite Jacobian, a Monte Carlo run that accepts fewer than two
        draws, or an unknown or inconsistent ``method``/``derivative``.

    Assumptions
    -----------
    ``"linear"`` is a *local* first-order result at the input means; it is
    not a verified global uncertainty and carries no model-form or
    fitted-coefficient uncertainty. ``"monte_carlo"`` assumes Gaussian
    inputs (no other input distribution is supported), and its statistics
    cover the accepted draws only: a draw outside the domain is rejected and
    counted in ``rejected``.

    Limitations
    -----------
    Uncertain inputs must be scalars; profile- and field-valued covariance is
    out of scope (#1874). A finite difference is not a solver-native
    linearization.

    References
    ----------
    .. [1] JCGM 100:2008, *Guide to the expression of uncertainty in
           measurement*, Sec. 5.2.
    .. [2] JCGM 101:2008, *Propagation of distributions using a Monte Carlo
           method*, Sec. 7.
    """
    from ._propagation import analytic_jacobian, formula_domain

    if method not in ("linear", "monte_carlo"):
        raise ValueError(f"method must be 'linear' or 'monte_carlo', got {method!r}")
    if derivative not in ("auto", "analytic", "finite_difference"):
        raise ValueError(f"derivative must be 'auto', 'analytic' or 'finite_difference', got {derivative!r}")
    if method == "monte_carlo" and derivative != "auto":
        raise ValueError("a Monte Carlo propagation takes no derivative: leave derivative='auto'")
    parameters = list(inspect.signature(formula).parameters)
    unknown = [name for name in inputs if name not in parameters]
    if unknown:
        raise ValueError(f"{formula.__name__} has no parameter {', '.join(unknown)}")
    if input_names is None:
        input_names = [name for name in parameters if name in inputs and np.ndim(inputs[name]) == 0
                       and isinstance(inputs[name], (int, float, np.integer, np.floating))
                       and not isinstance(inputs[name], bool)]
    names = tuple(input_names)
    if not names:
        raise ValueError("no uncertain inputs: name them in input_names")
    missing = [name for name in names if name not in inputs]
    if missing:
        raise ValueError(f"uncertain inputs missing from inputs: {', '.join(missing)}")
    if len(set(names)) != len(names):
        raise ValueError("input_names repeats an input")
    for name in names:
        if np.ndim(inputs[name]) != 0:
            raise ValueError(f"{name} is not a scalar; array-valued uncertain inputs are not supported")
    x0 = np.array([float(inputs[name]) for name in names])
    if not np.all(np.isfinite(x0)):
        raise ValueError("uncertain inputs must be finite")
    fixed = {name: value for name, value in inputs.items() if name not in names}

    if (covariance is None) == (std is None):
        raise ValueError("give exactly one of covariance and std: an unstated uncertainty is unknown, not zero")
    if std is not None:
        sigma = np.asarray(std, dtype=float)
        if sigma.ndim == 0 and len(names) == 1:
            sigma = sigma.reshape(1)
        if sigma.shape != (len(names),):
            raise ValueError(f"std must be 1-D with one entry per input: shape {sigma.shape} for {len(names)} inputs")
        if np.any(np.isnan(sigma)) or np.any(sigma < 0) or not np.all(np.isfinite(sigma)):
            raise ValueError("standard deviations must be known (not NaN), finite and non-negative")
        cov = np.diag(sigma ** 2)
    else:
        raw = np.asarray(covariance, dtype=float)
        if raw.ndim == 0 and len(names) == 1:
            raw = raw.reshape(1, 1)
        if np.any(np.isnan(raw)):
            raise ValueError("covariance must be known: NaN is an unknown uncertainty, not a value")
        if not np.all(np.isfinite(raw)):
            raise ValueError("covariance must be finite")
        if raw.ndim == 1 and np.any(raw < 0):
            raise ValueError("variances must be non-negative")
        if raw.ndim > 2:
            raise ValueError(f"covariance must be (n, n) or 1-D variances, got shape {raw.shape}")
        cov = _covariance(raw, len(names))
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues.size and eigenvalues.min() < -1e-12 * max(abs(eigenvalues.max()), np.finfo(float).tiny):
        raise ValueError("input covariance must be positive semi-definite")

    defined = domain if domain is not None else formula_domain(formula)

    def arguments_at(x: np.ndarray) -> dict:
        arguments = dict(fixed)
        arguments.update({name: float(value) for name, value in zip(names, x)})
        return arguments

    def evaluate(x: np.ndarray):
        arguments = arguments_at(x)
        if defined is not None and not defined(**arguments):
            raise _OutsideDomain(f"{formula.__name__} is not defined at {arguments}")
        return np.asarray(formula(**arguments), dtype=float)

    try:
        value = evaluate(x0)
    except _OutsideDomain:
        raise ValueError(f"the input means are outside the declared domain of {formula.__name__}") from None
    except (ArithmeticError, ValueError) as error:
        raise ValueError(f"{formula.__name__} cannot be evaluated at the input means: outside its domain "
                         f"({error})") from None
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{formula.__name__} is not finite at the input means: outside its domain")

    if method == "monte_carlo":
        mc = monte_carlo_propagation(lambda x: evaluate(x).ravel(), x0, cov, samples=samples, seed=seed)
        if mc["samples"] < 2:
            raise ValueError(f"Monte Carlo accepted {mc['samples']} of {samples} draws: the input distribution lies "
                             f"outside the domain of {formula.__name__}")
        out = np.atleast_2d(mc["covariance"])
        return FormulaPropagation(value=value, covariance=out, std=np.sqrt(np.clip(np.diag(out), 0, None)),
                                  input_names=names, input_covariance=cov, method="monte_carlo",
                                  mc_mean=mc["mean"], samples=mc["samples"], rejected=mc["rejected"],
                                  fixed=fixed)

    carried = analytic_jacobian(formula)
    if derivative == "analytic" and carried is None:
        raise ValueError(f"{formula.__name__} carries no analytic Jacobian")
    if carried is not None and derivative != "finite_difference":
        function, wrt = carried
        absent = [name for name in names if name not in wrt]
        if absent:
            raise ValueError(f"the analytic Jacobian of {formula.__name__} has no column for {', '.join(absent)}")
        # the derivative sees the whole call, fixed configuration included
        full = np.atleast_2d(np.asarray(function(**arguments_at(x0)), dtype=float))
        jac = full[:, [wrt.index(name) for name in names]]
        derivative_method = "analytic"
    else:
        steps = _relative_steps(x0, names, fd_step)
        jac = finite_difference_jacobian(lambda x: evaluate(x).ravel(), x0, step=steps)
        derivative_method = "finite_difference"
    if jac.shape != (value.size, len(names)):
        raise ValueError(f"Jacobian has shape {jac.shape}, expected {(value.size, len(names))}")
    if not np.all(np.isfinite(jac)):
        raise ValueError(f"the Jacobian of {formula.__name__} is not finite at the input means")
    out = linear_covariance_propagation(jac, cov)
    linearity = _linearity(evaluate, x0, value.ravel(), np.sqrt(np.diag(cov)), jac)
    return FormulaPropagation(value=value, covariance=out, std=np.sqrt(np.clip(np.diag(out), 0, None)),
                              input_names=names, input_covariance=cov, method="linear", jacobian=jac,
                              derivative_method=derivative_method, linearity=linearity, fixed=fixed)
