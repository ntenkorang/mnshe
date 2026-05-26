"""
Spline basis functions for MNSHE.

Provides M-spline and I-spline basis matrices used
to represent the hazard and cumulative hazard functions.
"""

import numpy as np
from scipy.interpolate import BSpline


def make_knots(internal_knots, degree, t_min, t_max):
    """Construct augmented knot sequence with boundary repeats."""
    return np.concatenate([
        np.repeat(t_min, degree + 1),
        internal_knots,
        np.repeat(t_max, degree + 1)
    ])


def bspline_basis_matrix(t, internal_knots, degree, t_min, t_max):
    """Evaluate B-spline basis matrix at time points t."""
    t     = np.asarray(t, dtype=float)
    knots = make_knots(internal_knots, degree, t_min, t_max)
    K     = len(knots) - degree - 1
    B     = np.zeros((len(t), K))
    for j in range(K):
        c       = np.zeros(K)
        c[j]    = 1.0
        spl     = BSpline(knots, c, degree, extrapolate=False)
        B[:, j] = spl(t)
    return np.nan_to_num(B, nan=0.0)


def mspline_basis(t, internal_knots, degree, t_min, t_max):
    """
    Evaluate M-spline basis matrix at time points t.

    M-splines are nonnegative scaled B-splines:
        M_j(t) = (degree + 1) / (kappa_{j+d+1} - kappa_j) * B_j(t)
    """
    knots  = make_knots(internal_knots, degree, t_min, t_max)
    B      = bspline_basis_matrix(t, internal_knots, degree,
                                   t_min, t_max)
    K      = B.shape[1]
    scales = np.zeros(K)
    for j in range(K):
        denom     = knots[j + degree + 1] - knots[j]
        scales[j] = (degree + 1) / denom if denom > 0 else 0.0
    return np.nan_to_num(B * scales[np.newaxis, :], nan=0.0)


def ispline_basis(t, internal_knots, degree, t_min, t_max):
    """
    Evaluate I-spline basis matrix at time points t.

    I-splines are integrals of M-splines:
        I_j(t) = integral_0^t M_j(s) ds

    Computed analytically via the B-spline antiderivative.
    Each I_j is nondecreasing with I_j(t_min) = 0.
    """
    t     = np.asarray(t, dtype=float)
    knots = make_knots(internal_knots, degree, t_min, t_max)
    K     = len(knots) - degree - 1
    I     = np.zeros((len(t), K))
    for j in range(K):
        c       = np.zeros(K)
        c[j]    = 1.0
        b       = BSpline(knots, c, degree, extrapolate=False)
        denom   = knots[j + degree + 1] - knots[j]
        scale   = (degree + 1) / denom if denom > 0 else 0.0
        b_int   = b.antiderivative()
        I[:, j] = scale * (b_int(t) - b_int(t_min))
    I = np.nan_to_num(I, nan=0.0)
    I[I < 0] = 0.0
    return I


def quantile_knots(event_times, K_internal):
    """Place K_internal knots at empirical quantiles of event times."""
    probs = np.linspace(0, 1, K_internal + 2)[1:-1]
    return np.quantile(event_times, probs)
