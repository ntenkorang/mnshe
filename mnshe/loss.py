"""
Loss function for MNSHE.

Penalised negative log-likelihood for right-censored
survival data. The L2 penalty is applied directly to
the output spline coefficients alpha(x).
"""

import torch


def mnshe_nll(h, H, delta, alpha=None, lambda_reg=1e-3):
    """
    Penalised negative log-likelihood.

    Parameters
    ----------
    h          : (n,) hazard values
    H          : (n,) cumulative hazard values
    delta      : (n,) event indicators (1=event, 0=censored)
    alpha      : (n, K) spline coefficients for regularisation
    lambda_reg : float  regularisation parameter lambda

    Returns
    -------
    loss : scalar tensor
    """
    nll = -(delta * torch.log(h + 1e-8) - H).mean()
    if alpha is not None:
        nll = nll + lambda_reg * torch.mean(alpha ** 2)
    return nll
