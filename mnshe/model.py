"""
MNSHE model architecture.

Implements the scale-shape neural parameterisation that
maps covariates to strictly positive spline coefficients,
guaranteeing a nondecreasing cumulative hazard by construction.

Paper: Monotonic Neural Spline Hazard Estimation for
       Right-Censored Survival Data
"""

import torch
import torch.nn as nn


class MNSHE(nn.Module):
    """
    Monotonic Neural Spline Hazard Estimation model.

    Architecture:
        - Shared feedforward feature network with ReLU activations
        - Scale head: softplus output -> strictly positive scalar
        - Shape head: softmax output -> simplex-valued weights
        - Spline coefficients: alpha_j(x) = scale(x) * weight_j(x)

    The nonnegative coefficients guarantee:
        (i)  h(t|x) >= 0 for all t, x
        (ii) H(t|x) nondecreasing in t for all x
        (iii) S(t|x) = exp(-H(t|x)) is a valid survival function
    """

    def __init__(self, input_dim, hidden_dim, n_layers, K):
        """
        Parameters
        ----------
        input_dim  : int  — number of covariates p
        hidden_dim : int  — width of hidden layers q
        n_layers   : int  — number of hidden layers L
        K          : int  — number of spline basis functions J
        """
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
            ])
        self.feature_net = nn.Sequential(*layers)
        self.shape_head  = nn.Linear(hidden_dim, K)
        self.scale_head  = nn.Linear(hidden_dim, 1)
        self.K           = K

        # Initialisation: small weights, scale bias -> ~0.31 at start
        nn.init.normal_(self.shape_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.shape_head.bias, 0.0)
        nn.init.normal_(self.scale_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.scale_head.bias, -1.0)

    def forward(self, x, M_basis, I_basis):
        """
        Forward pass.

        Parameters
        ----------
        x       : (n, p) covariate tensor
        M_basis : (n, K) M-spline basis matrix
        I_basis : (n, K) I-spline basis matrix

        Returns
        -------
        h     : (n,) hazard values
        H     : (n,) cumulative hazard values
        alpha : (n, K) spline coefficients
        """
        features     = self.feature_net(x)
        shape_logits = self.shape_head(features)
        weights      = torch.softmax(shape_logits, dim=1)
        scale        = torch.nn.functional.softplus(
                           self.scale_head(features)
                       ) + 1e-6
        alpha = scale * weights
        h     = (alpha * M_basis).sum(dim=1)
        H     = (alpha * I_basis).sum(dim=1)
        return h, H, alpha
