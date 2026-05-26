"""
Hyperparameter selection for MNSHE via 5-fold cross-validation.

Grid search over 16 configurations. Within each fold,
a separate validation set is used for early stopping.
The holdout fold is used for C-index scoring.

Selected hyperparameters are saved to results/best_hparams.json
and used unchanged for all subsequent experiments.
"""

import os
import json
import numpy as np
from itertools import product
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch

from mnshe.dgp import simulate_weibull_ph
from mnshe.splines import mspline_basis, ispline_basis, quantile_knots
from mnshe.training import fit_mnshe, get_risk_scores
from mnshe.metrics import concordance_index_manual


def select_hyperparameters(n_cv=500, p=5, seed=0, n_folds=5,
                            save_path="results/best_hparams.json"):
    """
    5-fold CV hyperparameter selection.

    Search grid:
        hidden_dim  : {64, 128}
        n_layers    : {2, 3}
        lambda_reg  : {1e-4, 1e-3}
        learning_rate: {1e-3, 3e-4}

    Total: 16 configurations.
    Selection criterion: mean holdout C-index across 5 folds.
    """
    print("=" * 60)
    print("HYPERPARAMETER SELECTION via 5-fold CV")
    print("=" * 60)

    param_grid = list(product(
        [64, 128],
        [2, 3],
        [1e-4, 1e-3],
        [1e-3, 3e-4],
    ))

    t_obs, delta, X, _ = simulate_weibull_ph(
        n=n_cv, p=p, seed=seed
    )

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    kf          = KFold(n_splits=n_folds, shuffle=True,
                        random_state=seed)
    best_score  = -np.inf
    best_params = None

    for i, (hd, nl, lam, lr) in enumerate(param_grid):
        fold_scores = []

        for tr_va_idx, ho_idx in kf.split(X_sc):

            delta_trva = delta[tr_va_idx]
            tr_idx, va_idx = train_test_split(
                tr_va_idx,
                test_size=0.20,
                random_state=seed,
                stratify=delta_trva
            )

            sc_fold = StandardScaler()
            X_tr    = sc_fold.fit_transform(X[tr_idx])
            X_va    = sc_fold.transform(X[va_idx])
            X_ho    = sc_fold.transform(X[ho_idx])

            t_tr = t_obs[tr_idx]
            t_va = t_obs[va_idx]
            t_ho = t_obs[ho_idx]
            d_tr = delta[tr_idx]
            d_va = delta[va_idx]
            d_ho = delta[ho_idx]

            t_min   = 0.0
            t_max   = np.quantile(t_tr, 0.98)
            t_tr_c  = np.clip(t_tr, t_min, t_max)
            t_va_c  = np.clip(t_va, t_min, t_max)
            t_ho_c  = np.clip(t_ho, t_min, t_max)
            knots   = quantile_knots(t_tr_c[d_tr == 1], 5)

            setup_cv = {
                "X_train": X_tr, "X_val": X_va, "X_test": X_ho,
                "t_train": t_tr_c, "t_val": t_va_c,
                "t_test":  t_ho_c,
                "d_train": d_tr,   "d_val": d_va, "d_test": d_ho,
                "M_train": mspline_basis(t_tr_c, knots, 3,
                                         t_min, t_max),
                "I_train": ispline_basis(t_tr_c, knots, 3,
                                         t_min, t_max),
                "M_val":   mspline_basis(t_va_c, knots, 3,
                                         t_min, t_max),
                "I_val":   ispline_basis(t_va_c, knots, 3,
                                         t_min, t_max),
                "M_test":  mspline_basis(t_ho_c, knots, 3,
                                         t_min, t_max),
                "I_test":  ispline_basis(t_ho_c, knots, 3,
                                         t_min, t_max),
                "knots": knots, "degree": 3,
                "t_min": t_min, "t_max": t_max,
            }

            try:
                model, _, _, _ = fit_mnshe(
                    setup_cv,
                    hidden_dim=hd, n_layers=nl,
                    epochs=500, lr=lr,
                    weight_decay=5e-4,
                    lambda_reg=lam,
                    patience=50
                )
                X_ho_t = torch.tensor(X_ho, dtype=torch.float32)
                risk   = get_risk_scores(model, X_ho_t, setup_cv)
                c      = concordance_index_manual(t_ho_c, risk, d_ho)
                fold_scores.append(c)
            except Exception as e:
                fold_scores.append(np.nan)

        mean_c = np.nanmean(fold_scores)
        print(f"  [{i+1:2d}/{len(param_grid)}] "
              f"hd={hd} nl={nl} lam={lam:.0e} lr={lr:.0e} "
              f"-> CV C = {mean_c:.4f}")

        if mean_c > best_score:
            best_score  = mean_c
            best_params = {
                "hidden_dim":   hd,
                "n_layers":     nl,
                "lambda_reg":   lam,
                "lr":           lr,
                "weight_decay": 5e-4,
                "patience":     150,
                "epochs":       2000,
                "K_internal":   5,
            }

    print(f"\nBest CV C-index: {best_score:.4f}")
    print(f"Best params:     {best_params}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved to {save_path}")

    return best_params


if __name__ == "__main__":
    select_hyperparameters()
