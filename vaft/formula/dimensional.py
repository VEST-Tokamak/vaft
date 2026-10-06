r"""Dimensional analysis: dimension matrices, Buckingham-Pi groups and similarity constraints.

A power-law monomial $\Pi = \prod_j x_j^{a_j}$ of physical variables $x_j$ is
dimensionless exactly when $D\mathbf a = 0$, where the columns of the
dimension matrix $D$ hold the exponents of each variable's base dimensions.
The admissible Buckingham-Pi groups therefore span $\ker D$ (issue #1621,
Stage II).  Everything here is exact: dimensions and exponents are
:class:`fractions.Fraction`, the null space comes from Gauss-Jordan
elimination over the rationals, and no floating-point tolerance decides a
rank.

A null-space basis is not unique.  :func:`express_in_basis` shows that the
conventional plasma groups ($\Omega_i\tau_E$, $\rho_*$, $\beta$, $\nu_*$, ...)
are one basis among many by writing them in another.

A *similarity* assumption is stronger than dimensional invariance: it keeps
only some of the groups (Connor and Taylor drop the Debye-length group, the
quasi-neutral limit of the governing equations).  Over the variables that
vary, the exponent vector of a law that depends on the kept groups only must
lie in their span; :func:`similarity_constraint_vectors` returns the
orthogonal complement, the linear constraints that span imposes.  For the
collisional, finite-$\beta$ set it is the Connor-Kadomtsev constraint.

Notation
--------
D        : dimension matrix, base dimensions x variables                   [-]
a        : exponent vector of a monomial over the variables                [-]
L,M,T,I  : SI base dimensions length, mass, time, current                  [-]
n        : particle density                                                [m^-3]
T_e      : temperature in energy units                                     [J]
B        : magnetic field                                                  [T]
R        : major radius (any length scale at fixed shape)                  [m]
tau      : time                                                            [s]
e        : elementary charge                                               [C]
m_i      : ion mass                                                        [kg]
mu0      : vacuum permeability                                             [H/m]
eps0     : vacuum permittivity                                             [F/m]

Conventions
-----------
**Temperature is an energy.**  Plasma temperatures enter as $k_B T$ in
joules, so the base dimensions are the four mechanical-electrical ones
$(L, M, T, I)$ and no temperature dimension appears.

**Engineering exponents use the confinement order.**  An engineering scaling
$\tau_E \propto I_p^{\alpha_I} B^{\alpha_B} P^{\alpha_P} n^{\alpha_n}
R^{\alpha_R}$ is mapped onto the state variables $(n, T, B, R)$ at fixed
shape ($\epsilon$, $\kappa$), safety factor and ion mass, with
$I_p \propto B R$ and $P = W/\tau_E$, $W \propto n T R^3$.

References
----------
.. [1] E. Buckingham, Phys. Rev. 4 (1914) 345.
.. [2] J. W. Connor and J. B. Taylor, Nucl. Fusion 17 (1977) 1047.
.. [3] B. B. Kadomtsev, Sov. J. Plasma Phys. 1 (1975) 295.
"""

from __future__ import annotations

from fractions import Fraction
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "BASE_DIMENSIONS",
    "PHYSICAL_DIMENSIONS",
    "dimension_matrix",
    "rational_null_space",
    "dimensionless_groups",
    "monomial_dimension",
    "express_in_basis",
    "similarity_constraint_vectors",
    "state_exponents_from_engineering_exponents",
]

#: The base dimensions, in the row order of :func:`dimension_matrix`.
BASE_DIMENSIONS = ("L", "M", "T", "I")

#: Base-dimension exponents (L, M, T, I) of the variables plasma scaling laws use.
PHYSICAL_DIMENSIONS: Mapping[str, tuple] = MappingProxyType({
    "n": (-3, 0, 0, 0),          # density [m^-3]
    "T": (2, 1, -2, 0),          # temperature as energy [J]
    "W": (2, 1, -2, 0),          # stored energy [J]
    "B": (0, 1, -2, -1),         # magnetic field [T] = kg s^-2 A^-1
    "R": (1, 0, 0, 0),           # major radius [m]
    "a": (1, 0, 0, 0),           # minor radius [m]
    "I_p": (0, 0, 0, 1),         # plasma current [A]
    "P": (2, 1, -3, 0),          # power [W]
    "tau": (0, 0, 1, 0),         # time [s]
    "e": (0, 0, 1, 1),           # elementary charge [C] = A s
    "m_i": (0, 1, 0, 0),         # ion mass [kg]
    "mu0": (1, 1, -2, -2),       # vacuum permeability [H/m] = kg m s^-2 A^-2
    "eps0": (-3, -1, 4, 2),      # vacuum permittivity [F/m] = A^2 s^4 kg^-1 m^-3
})


def _frac(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, str):
        return Fraction(value)
    f = float(value)
    if not np.isfinite(f):
        raise ValueError(f"non-finite exponent {value!r}")
    # A float exponent is taken at face value only when it is a short rational.
    return Fraction(f).limit_denominator(10**6)


def _matrix(rows) -> list[list[Fraction]]:
    out = [[_frac(v) for v in row] for row in rows]
    if not out or len({len(r) for r in out}) != 1 or not out[0]:
        raise ValueError("a non-empty rectangular matrix is needed")
    return out


def _rref(rows: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    m = [r[:] for r in rows]
    pivots: list[int] = []
    r = 0
    for c in range(len(m[0])):
        p = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        piv = m[r][c]
        m[r] = [v / piv for v in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
        if r == len(m):
            break
    return m, pivots


def _integer_scaled(vector: list[Fraction]) -> np.ndarray:
    den = 1
    for v in vector:
        den = den * v.denominator // np.gcd(den, v.denominator)
    ints = [int(v * den) for v in vector]
    g = 0
    for v in ints:
        g = int(np.gcd(g, abs(v)))
    g = g or 1
    first = next((v for v in ints if v != 0), 1)
    sign = -1 if first < 0 else 1
    return np.array([Fraction(sign * v // g) for v in ints], dtype=object)


def dimension_matrix(variables: Sequence[str], dimensions: Mapping[str, Sequence] = PHYSICAL_DIMENSIONS) -> np.ndarray:
    r"""Dimension matrix $D$ of a list of variables, exactly.

    $$D_{kj} = \text{exponent of base dimension } k \text{ in variable } j$$

    Parameters
    ----------
    variables : sequence of str
        Variable names, keys of ``dimensions`` [-].
    dimensions : Mapping, optional
        Base-dimension exponents $(L, M, T, I)$ per name; default
        :data:`PHYSICAL_DIMENSIONS` [-].

    Returns
    -------
    numpy.ndarray
        Object array of :class:`fractions.Fraction`, one row per base
        dimension of :data:`BASE_DIMENSIONS` and one column per variable [-].

    Raises
    ------
    KeyError
        A variable with no declared dimensions.
    ValueError
        No variable, or a dimension vector of the wrong length.

    Convention
    ----------
    Rows are $(L, M, T, I)$; temperature is an energy, so there is no
    temperature row.

    References
    ----------
    .. [1] E. Buckingham, Phys. Rev. 4 (1914) 345.
    """
    names = list(variables)
    if not names:
        raise ValueError("at least one variable is needed")
    cols = []
    for name in names:
        dim = tuple(dimensions[name])
        if len(dim) != len(BASE_DIMENSIONS):
            raise ValueError(f"{name}: {len(dim)} base-dimension exponents, expected {len(BASE_DIMENSIONS)}")
        cols.append([_frac(v) for v in dim])
    return np.array([[col[k] for col in cols] for k in range(len(BASE_DIMENSIONS))], dtype=object)


def rational_null_space(matrix) -> list[np.ndarray]:
    r"""Exact basis of the null space $\{\mathbf a : M\mathbf a = 0\}$ of a rational matrix.

    $$M\mathbf a = 0$$

    Parameters
    ----------
    matrix : array-like
        Rows of integers, fractions or short-rational floats [-].

    Returns
    -------
    list of numpy.ndarray
        One object array of integer-valued :class:`fractions.Fraction` per
        basis vector, coprime and with a positive first non-zero entry; empty
        when $M$ has full column rank [-].

    Raises
    ------
    ValueError
        An empty or ragged matrix, or a non-finite entry.

    Convention
    ----------
    The basis is the one Gauss-Jordan elimination gives (one vector per free
    column, that column set to one); any invertible combination of it spans
    the same space.

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 4th ed.,
           Thomson (2006), Sec. 2.2.
    """
    m = _matrix(np.asarray(matrix, dtype=object).tolist())
    ncol = len(m[0])
    reduced, pivots = _rref(m)
    basis = []
    for free in (c for c in range(ncol) if c not in pivots):
        v = [Fraction(0)] * ncol
        v[free] = Fraction(1)
        for row, pc in zip(reduced, pivots):
            v[pc] = -row[free]
        basis.append(_integer_scaled(v))
    return basis


def dimensionless_groups(variables: Sequence[str], dimensions: Mapping[str, Sequence] = PHYSICAL_DIMENSIONS) -> list[np.ndarray]:
    r"""A basis of the Buckingham-Pi groups of a set of variables: $\ker D$.

    $$\Pi = \prod_j x_j^{a_j}\ \text{dimensionless} \iff D\mathbf a = 0$$

    Parameters
    ----------
    variables : sequence of str
        Variable names, keys of ``dimensions`` [-].
    dimensions : Mapping, optional
        Base-dimension exponents per name; default :data:`PHYSICAL_DIMENSIONS` [-].

    Returns
    -------
    list of numpy.ndarray
        Integer exponent vectors over ``variables``, one per group; there are
        ``len(variables) - rank(D)`` of them [-].

    Raises
    ------
    KeyError
        A variable with no declared dimensions.

    Convention
    ----------
    The basis is not unique; :func:`express_in_basis` changes it to a
    conventional one.

    References
    ----------
    .. [1] E. Buckingham, Phys. Rev. 4 (1914) 345.
    """
    return rational_null_space(dimension_matrix(variables, dimensions))


def monomial_dimension(exponents: Sequence, variables: Sequence[str],
                       dimensions: Mapping[str, Sequence] = PHYSICAL_DIMENSIONS) -> np.ndarray:
    r"""Base dimensions of the monomial $\prod_j x_j^{a_j}$: $D\mathbf a$.

    $$\dim\Big(\prod_j x_j^{a_j}\Big) = D\mathbf a$$

    Parameters
    ----------
    exponents : sequence
        Exponent $a_j$ per variable [-].
    variables : sequence of str
        Variable names, keys of ``dimensions`` [-].
    dimensions : Mapping, optional
        Base-dimension exponents per name; default :data:`PHYSICAL_DIMENSIONS` [-].

    Returns
    -------
    numpy.ndarray
        Object array of :class:`fractions.Fraction`, one entry per base
        dimension; all zero for a dimensionless monomial [-].

    Raises
    ------
    ValueError
        ``exponents`` and ``variables`` of different lengths.

    Convention
    ----------
    Base dimensions in the order of :data:`BASE_DIMENSIONS`.

    References
    ----------
    .. [1] E. Buckingham, Phys. Rev. 4 (1914) 345.
    """
    a = [_frac(v) for v in exponents]
    names = list(variables)
    if len(a) != len(names):
        raise ValueError(f"{len(a)} exponents for {len(names)} variables")
    d = dimension_matrix(names, dimensions)
    return np.array([sum((d[k][j] * a[j] for j in range(len(a))), Fraction(0)) for k in range(d.shape[0])],
                    dtype=object)


def express_in_basis(target: Sequence, basis: Sequence[Sequence]) -> np.ndarray:
    r"""Exact coefficients $\mathbf c$ with $\mathbf t = \sum_k c_k \mathbf b_k$.

    $$\mathbf t = \sum_k c_k\,\mathbf b_k$$

    Parameters
    ----------
    target : sequence
        Exponent vector to express, e.g. a conventional group [-].
    basis : sequence of sequences
        Basis vectors of the same length, e.g. from :func:`dimensionless_groups` [-].

    Returns
    -------
    numpy.ndarray
        Object array of :class:`fractions.Fraction`, one coefficient per basis
        vector [-].

    Raises
    ------
    ValueError
        Lengths that disagree, linearly dependent basis vectors, or a target
        outside their span.

    Convention
    ----------
    In multiplicative form $\Pi_t = \prod_k \Pi_k^{c_k}$: the coefficients
    are the powers of the basis groups that build the target group.

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 4th ed.,
           Thomson (2006), Sec. 2.3.
    """
    b = _matrix([list(v) for v in basis])
    t = [_frac(v) for v in target]
    if len(t) != len(b[0]):
        raise ValueError(f"target of length {len(t)} for basis vectors of length {len(b[0])}")
    k = len(b)
    # Solve B^T c = t as an augmented system over the rationals.
    aug = [[b[j][i] for j in range(k)] + [t[i]] for i in range(len(t))]
    reduced, pivots = _rref(aug)
    if len(pivots) and pivots[-1] == k:
        raise ValueError("target is not in the span of the basis")
    if [p for p in pivots if p < k] != list(range(k)):
        raise ValueError("basis vectors are linearly dependent")
    return np.array([reduced[i][k] for i in range(k)], dtype=object)


def similarity_constraint_vectors(kept_groups: Sequence[Sequence]) -> list[np.ndarray]:
    r"""Linear constraints a law depending only on ``kept_groups`` imposes on its exponents.

    $$\mathbf e \in \mathrm{span}\{\mathbf g_k\} \iff \mathbf c\cdot\mathbf e = 0\ \ \forall\,\mathbf c \perp \mathbf g_k$$

    Parameters
    ----------
    kept_groups : sequence of sequences
        Exponent vectors, over the varied variables, of the dimensionless
        groups the similarity assumption keeps [-].

    Returns
    -------
    list of numpy.ndarray
        Integer constraint vectors $\mathbf c$, a basis of the orthogonal
        complement of the groups' span; empty when the groups span the whole
        space (no constraint) [-].

    Raises
    ------
    ValueError
        An empty or ragged group list.

    Convention
    ----------
    $\mathbf e$ is the exponent vector of the dimensionless response (e.g.
    $\Omega_i\tau_E$) over the *varied* variables only; the constants and the
    variables held fixed (shape, $q$, ion mass) are not coordinates.

    References
    ----------
    .. [1] J. W. Connor and J. B. Taylor, Nucl. Fusion 17 (1977) 1047.
    .. [2] B. B. Kadomtsev, Sov. J. Plasma Phys. 1 (1975) 295.
    """
    return rational_null_space([list(g) for g in kept_groups])


def state_exponents_from_engineering_exponents(a_I, a_B, a_P, a_n=0, a_R=0) -> np.ndarray:
    r"""Exponents of $\tau_E$ over $(n, T, B, R)$ implied by an engineering scaling, exactly.

    $$\tau_E^{1+\alpha_P} \propto n^{\alpha_n+\alpha_P}\,T^{\alpha_P}\,
      B^{\alpha_I+\alpha_B}\,R^{\alpha_I+3\alpha_P+\alpha_R}$$

    Parameters
    ----------
    a_I : float or Fraction
        Plasma-current exponent $\alpha_I$ [-].
    a_B : float or Fraction
        Toroidal-field exponent $\alpha_B$ [-].
    a_P : float or Fraction
        Loss-power exponent $\alpha_P$ [-].
    a_n : float or Fraction, optional
        Density exponent $\alpha_n$, default 0 [-].
    a_R : float or Fraction, optional
        Size exponent $\alpha_R$, default 0 [-].

    Returns
    -------
    numpy.ndarray
        The exponents of $n$, $T$, $B$ and $R$ in $\tau_E$: exact
        :class:`fractions.Fraction` when every input is an int, Fraction or
        rational string, float otherwise [-].

    Raises
    ------
    ValueError
        $\alpha_P = -1$, where $\tau_E$ cancels and the map is singular.

    Convention
    ----------
    At fixed $\epsilon$, $\kappa$, $q$ and ion mass: $I_p \propto B R$
    (cylindrical $q$) and $P = W/\tau_E$ with $W \propto n T R^3$.  Every
    exponent carries the factor $1/(1+\alpha_P)$, the conditioning of the
    whole engineering-to-dimensionless map.

    References
    ----------
    .. [1] J. W. Connor and J. B. Taylor, Nucl. Fusion 17 (1977) 1047.
    .. [2] ITER Physics Expert Groups, Nucl. Fusion 39 (1999) 2175, Ch. 2, Sec. 6.
    """
    values = (a_I, a_B, a_P, a_n, a_R)
    exact = all(isinstance(v, (int, np.integer, Fraction, str)) for v in values)
    if exact:
        i, b, p, n, r = (_frac(v) for v in values)
    else:
        i, b, p, n, r = (float(v) for v in values)
        if not np.all(np.isfinite([i, b, p, n, r])):
            raise ValueError("non-finite exponent")
    d = 1 + p
    if d == 0:
        raise ValueError("alpha_P = -1: tau_E cancels from P = W/tau_E and the map is singular")
    out = [(n + p) / d, p / d, (i + b) / d, (i + 3 * p + r) / d]
    return np.array(out, dtype=object if exact else float)
