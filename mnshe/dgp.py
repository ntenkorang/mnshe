"""
Data-generating processes for the MNSHE simulation study.

Three DGPs spanning proportional hazards, non-proportional
hazards, and a local hazard spike structure.

All DGPs use per-call seeding via numpy.random.default_rng
for full reproducibility across replications.
"""

import numpy as np


def simulate_weibull_ph(n, p, shape=1.5, scale=5.0,
                        censoring_strength=0.3, seed=42):
    """
    DGP 1: Weibull proportional hazards.

    Hazard:
        h(t|x) = (shape/scale) * (t/scale)^(shape-1) * exp(beta^T x)

    Parameters: shape=1.5, scale=5.0, beta ~ N(0, 0.25 * I_p)
    Censoring:  exponential with mean scale/censoring_strength
    Event rate: approximately 73%

    Cox PH is correctly specified under this DGP.
    """
    rng   = np.random.default_rng(seed)
    X     = rng.normal(size=(n, p))
    beta  = rng.normal(size=p) * 0.5
    risk  = np.exp(X @ beta)
    U     = rng.uniform(size=n)
    T     = scale * (-np.log(U) / risk) ** (1 / shape)
    C     = rng.exponential(scale / censoring_strength, size=n)
    t_obs = np.minimum(T, C)
    delta = (T <= C).astype(float)
    meta  = {"dgp": "Weibull PH", "shape": shape,
              "scale": scale, "beta": beta}
    return t_obs, delta, X, meta


def simulate_weibull_nonph(n, p, shape=1.5, scale=5.0,
                           gamma=0.80, censoring_strength=0.3,
                           seed=42):
    """
    DGP 2: Non-proportional hazards Weibull.

    Cumulative hazard:
        H(t|x) = (t/scale)^shape * exp(beta^T x + gamma*x1*t/scale)

    The time-varying coefficient gamma*x1*t/scale violates
    the proportional hazards assumption. Event times simulated
    by numerical inversion of H via bisection (80 iterations).

    Parameters: shape=1.5, scale=5.0, gamma=0.80,
                beta ~ N(0, 0.04 * I_p)
    Censoring:  exponential with mean scale/censoring_strength

    Primary evaluation setting for MNSHE.
    """
    rng    = np.random.default_rng(seed)
    X      = rng.normal(size=(n, p))
    beta   = rng.normal(size=p) * 0.20
    U      = rng.uniform(size=n)
    target = -np.log(U)
    lo     = np.zeros(n)
    hi     = np.repeat(30.0, n)
    xbeta  = X @ beta
    x1     = X[:, 0]

    def H_func(t):
        return ((t / scale) ** shape *
                np.exp(xbeta + gamma * x1 * t / scale))

    for _ in range(10):
        mask = H_func(hi) < target
        if not np.any(mask):
            break
        hi[mask] *= 2.0

    for _ in range(80):
        mid   = 0.5 * (lo + hi)
        H_mid = H_func(mid)
        left  = H_mid < target
        lo[left]  = mid[left]
        hi[~left] = mid[~left]

    T     = 0.5 * (lo + hi)
    C     = rng.exponential(scale / censoring_strength, size=n)
    t_obs = np.minimum(T, C)
    delta = (T <= C).astype(float)
    meta  = {"dgp": "Non-PH Weibull", "shape": shape,
              "scale": scale, "beta": beta, "gamma": gamma}
    return t_obs, delta, X, meta


def simulate_local_spike(n, p, censoring_strength=0.3, seed=42):
    """
    DGP 3: Local hazard spike.

    Hazard:
        h(t|x) = 0.08 + 0.04*t
                 + 0.65*exp(-(t-3)^2/0.80) * exp(0.6*x1)

    A Gaussian spike at t=3 creates a period of elevated risk.
    Event times simulated by numerical integration and inversion
    on a grid of 5000 points over [0.001, 25].

    Censoring: exponential with mean 8.0/censoring_strength,
               longer horizon consistent with extended time domain.
    """
    rng    = np.random.default_rng(seed)
    X      = rng.normal(size=(n, p))
    beta   = np.zeros(p)
    beta[0] = 0.6
    t_grid = np.linspace(0.001, 25.0, 5000)
    dt     = np.diff(t_grid, prepend=0.0)
    T      = np.zeros(n)
    for i in range(n):
        x1     = X[i, 0]
        hazard = (
            0.08
            + 0.04 * t_grid
            + 0.65 * np.exp(-((t_grid - 3.0) ** 2) / 0.80)
            * np.exp(0.6 * x1)
        )
        cumhaz = np.cumsum(hazard * dt)
        target = -np.log(rng.uniform())
        idx    = np.searchsorted(cumhaz, target)
        T[i]   = t_grid[-1] if idx >= len(t_grid) else t_grid[idx]
    C     = rng.exponential(8.0 / censoring_strength, size=n)
    t_obs = np.minimum(T, C)
    delta = (T <= C).astype(float)
    meta  = {"dgp": "Local Spike", "beta": beta}
    return t_obs, delta, X, meta
