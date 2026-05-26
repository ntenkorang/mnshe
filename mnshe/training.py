"""
Training utilities for MNSHE.

Provides:
    - Three-way stratified data splitting
    - Model fitting with early stopping on validation loss
    - Risk score computation
    - Integrated Brier Score computation
    - Monotonicity verification
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mnshe.splines import mspline_basis, ispline_basis, quantile_knots
from mnshe.model import MNSHE
from mnshe.loss import mnshe_nll


def prepare_train_test(t_obs, delta, X,
                       test_size=0.15,
                       val_size=0.15,
                       seed=42,
                       degree=3,
                       K_internal=5):
    """
    Three-way stratified split: 70% train / 15% val / 15% test.

    Design principles:
        - Scaler fitted on training data only
        - Knots placed on training event times only
        - Validation used exclusively for early stopping
        - Test used exclusively for final evaluation
        - No information from val or test leaks into training
    """
    idx = np.arange(len(t_obs))

    # Split off test set
    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=delta
    )

    # Split remaining into train and validation
    delta_trainval = delta[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
        stratify=delta_trainval
    )

    # Scaler fitted on training only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X[idx_train])
    X_val   = scaler.transform(X[idx_val])
    X_test  = scaler.transform(X[idx_test])

    t_train = t_obs[idx_train]
    t_val   = t_obs[idx_val]
    t_test  = t_obs[idx_test]
    d_train = delta[idx_train]
    d_val   = delta[idx_val]
    d_test  = delta[idx_test]

    # Knots from training event times only
    t_min     = 0.0
    t_max     = np.quantile(t_train, 0.98)
    t_train_c = np.clip(t_train, t_min, t_max)
    t_val_c   = np.clip(t_val,   t_min, t_max)
    t_test_c  = np.clip(t_test,  t_min, t_max)
    knots     = quantile_knots(t_train_c[d_train == 1], K_internal)

    # Basis matrices from training knots only
    M_train = mspline_basis(t_train_c, knots, degree, t_min, t_max)
    I_train = ispline_basis(t_train_c, knots, degree, t_min, t_max)
    M_val   = mspline_basis(t_val_c,   knots, degree, t_min, t_max)
    I_val   = ispline_basis(t_val_c,   knots, degree, t_min, t_max)
    M_test  = mspline_basis(t_test_c,  knots, degree, t_min, t_max)
    I_test  = ispline_basis(t_test_c,  knots, degree, t_min, t_max)

    print(f"  Split: train={len(t_train)} | "
          f"val={len(t_val)} | test={len(t_test)} | "
          f"event rates: "
          f"tr={d_train.mean():.2f} "
          f"va={d_val.mean():.2f} "
          f"te={d_test.mean():.2f}")

    return {
        "X_train": X_train, "X_val": X_val,   "X_test": X_test,
        "t_train": t_train_c, "t_val": t_val_c, "t_test": t_test_c,
        "d_train": d_train,   "d_val": d_val,   "d_test": d_test,
        "M_train": M_train,   "I_train": I_train,
        "M_val":   M_val,     "I_val":   I_val,
        "M_test":  M_test,    "I_test":  I_test,
        "knots": knots, "degree": degree,
        "t_min": t_min, "t_max":  t_max,
        "scaler": scaler,
    }


def fit_mnshe(setup, hidden_dim=128, n_layers=3, epochs=2000,
              lr=1e-3, weight_decay=5e-4, lambda_reg=1e-3,
              patience=150):
    """
    Fit MNSHE with early stopping on validation loss.

    Early stopping monitors validation loss only.
    Test set is never seen during fitting.

    Returns
    -------
    model        : fitted MNSHE model (float32)
    train_losses : list of training losses per epoch
    val_losses   : list of validation losses per epoch
    best_epoch   : epoch at which early stopping triggered
    """

    def to_t(arr):
        return torch.tensor(arr, dtype=torch.float32)

    X_tr = to_t(setup["X_train"])
    M_tr = to_t(setup["M_train"])
    I_tr = to_t(setup["I_train"])
    d_tr = to_t(setup["d_train"])

    X_va = to_t(setup["X_val"])
    M_va = to_t(setup["M_val"])
    I_va = to_t(setup["I_val"])
    d_va = to_t(setup["d_val"])

    K     = setup["M_train"].shape[1]
    model = MNSHE(setup["X_train"].shape[1],
                  hidden_dim, n_layers, K)
    opt   = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_loss = np.inf
    best_epoch    = 0
    best_state    = None
    train_losses  = []
    val_losses    = []

    for epoch in range(epochs):

        model.train()
        opt.zero_grad()
        h, H, alpha = model(X_tr, M_tr, I_tr)
        loss = mnshe_nll(h, H, d_tr, alpha, lambda_reg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            h_v, H_v, a_v = model(X_va, M_va, I_va)
            val_loss = mnshe_nll(h_v, H_v, d_va, a_v, lambda_reg)
        val_losses.append(val_loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch    = epoch + 1
            best_state    = {
                k: v.clone()
                for k, v in model.state_dict().items()
            }

        if (epoch + 1) - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_epoch


def get_risk_scores(model, X_t, setup):
    """
    Compute risk scores as cumulative hazard at t_max.

    Higher score indicates higher predicted risk.
    """
    n     = len(X_t)
    t_e   = np.repeat(setup["t_max"], n)
    M_e   = mspline_basis(t_e, setup["knots"], setup["degree"],
                           setup["t_min"], setup["t_max"])
    I_e   = ispline_basis(t_e, setup["knots"], setup["degree"],
                           setup["t_min"], setup["t_max"])
    M_e_t = torch.tensor(M_e, dtype=torch.float32)
    I_e_t = torch.tensor(I_e, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        _, H_e, _ = model(X_t, M_e_t, I_e_t)
    return H_e.numpy()


def integrated_brier_score(model, setup, n_time_points=50):
    """
    Compute Integrated Brier Score for MNSHE on the test set.

    Uses IPCW weighting via Kaplan-Meier on the training
    censoring distribution. KM is fitted on training data only.
    """
    from lifelines import KaplanMeierFitter

    t_min    = setup["t_min"]
    t_max    = setup["t_max"]
    t_grid   = np.linspace(t_min, t_max, n_time_points)
    X_test_t = torch.tensor(setup["X_test"], dtype=torch.float32)
    n_test   = len(setup["X_test"])
    t_test   = setup["t_test"]
    d_test   = setup["d_test"]

    kmf = KaplanMeierFitter()
    kmf.fit(setup["t_train"], event_observed=1 - setup["d_train"])

    brier_scores = []

    for t_star in t_grid:
        t_e   = np.repeat(t_star, n_test)
        M_e   = mspline_basis(t_e, setup["knots"],
                               setup["degree"],
                               t_min, t_max)
        I_e   = ispline_basis(t_e, setup["knots"],
                               setup["degree"],
                               t_min, t_max)
        M_e_t = torch.tensor(M_e, dtype=torch.float32)
        I_e_t = torch.tensor(I_e, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            _, H_e, _ = model(X_test_t, M_e_t, I_e_t)
        S_hat = np.exp(-H_e.numpy())

        G_t  = max(float(
            kmf.survival_function_at_times([t_star]).iloc[0]
        ), 1e-8)
        G_ti = np.maximum(np.array([
            float(kmf.survival_function_at_times([ti]).iloc[0])
            for ti in t_test
        ]), 1e-8)

        I1 = ((t_test <= t_star) & (d_test == 1)).astype(float)
        I2 = (t_test > t_star).astype(float)

        bs = (np.mean(I1 * (0 - S_hat) ** 2 / G_ti) +
              np.mean(I2 * (1 - S_hat) ** 2 / G_t))
        brier_scores.append(bs)

    return np.trapezoid(brier_scores, t_grid) / (t_max - t_min)


def check_monotonicity(model, setup, n_subjects=10,
                       grid_size=300, seed=0):
    """
    Check monotonicity of H(t|x) on a fine time grid.

    Uses float64 precision to avoid float32 rounding artefacts
    near t_max where H is nearly flat. Threshold of -1e-6 is
    appropriate for double precision arithmetic.

    Returns
    -------
    H_viol : int  total number of monotonicity violations
    """
    rng    = np.random.default_rng(seed)
    X_test = setup["X_test"]
    chosen = rng.choice(len(X_test),
                         min(n_subjects, len(X_test)),
                         replace=False)

    t_grid = np.linspace(setup["t_min"], setup["t_max"],
                          grid_size)
    M_g    = mspline_basis(t_grid, setup["knots"],
                            setup["degree"],
                            setup["t_min"], setup["t_max"])
    I_g    = ispline_basis(t_grid, setup["knots"],
                            setup["degree"],
                            setup["t_min"], setup["t_max"])

    M_g_t  = torch.tensor(M_g, dtype=torch.float64)
    I_g_t  = torch.tensor(I_g, dtype=torch.float64)

    model_d = model.double()
    H_viol  = 0

    model_d.eval()
    for idx in chosen:
        x_i = torch.tensor(
            X_test[idx], dtype=torch.float64
        ).unsqueeze(0).repeat(grid_size, 1)

        with torch.no_grad():
            _, H_g, _ = model_d(x_i, M_g_t, I_g_t)

        H_np   = H_g.numpy()
        diffs  = np.diff(H_np)
        H_viol += int(np.sum(diffs < -1e-6))

    model.float()
    return H_viol
