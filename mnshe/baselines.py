"""
Baseline survival models for comparison with MNSHE.

Provides:
    - Cox proportional hazards model (via lifelines)
    - DeepSurv deep Cox model (PyTorch)
    - Integrated Brier Score for both baselines
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Cox PH
# ============================================================

def fit_cox_ph(X_train, t_train, d_train, X_test):
    """
    Fit Cox PH model via penalised partial likelihood.

    Penalty parameter 0.01 applied for numerical stability.
    Returns risk scores and fitted model for IBS computation.
    """
    try:
        from lifelines import CoxPHFitter
        cols     = [f"x{j+1}" for j in range(X_train.shape[1])]
        train_df = pd.DataFrame(X_train, columns=cols)
        train_df["time"]  = t_train
        train_df["event"] = d_train
        test_df  = pd.DataFrame(X_test, columns=cols)
        cph      = CoxPHFitter(penalizer=0.01)
        cph.fit(train_df, duration_col="time", event_col="event")
        risk = cph.predict_partial_hazard(
            test_df
        ).values.reshape(-1)
        return risk, cph
    except Exception as e:
        print(f"  Cox PH failed: {e}")
        return None, None


def cox_integrated_brier_score(cox_model, setup,
                                n_time_points=50):
    """
    Integrated Brier Score for Cox PH on the test set.

    Uses IPCW weighting via Kaplan-Meier on the training
    censoring distribution.
    """
    from lifelines import KaplanMeierFitter

    t_min   = setup["t_min"]
    t_max   = setup["t_max"]
    t_grid  = np.linspace(t_min, t_max, n_time_points)
    t_test  = setup["t_test"]
    d_test  = setup["d_test"]

    cols    = [f"x{j+1}" for j in range(setup["X_test"].shape[1])]
    test_df = pd.DataFrame(setup["X_test"], columns=cols)

    kmf = KaplanMeierFitter()
    kmf.fit(setup["t_train"], event_observed=1 - setup["d_train"])

    brier_scores = []

    for t_star in t_grid:
        S_hat = cox_model.predict_survival_function(
            test_df, times=[t_star]
        ).values.flatten()

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


# ============================================================
# DeepSurv
# ============================================================

class DeepSurvNet(nn.Module):
    """
    DeepSurv: deep Cox proportional hazards model.

    Maps covariates to a scalar log-risk score via a
    fully connected network. Trained by maximising the
    Cox partial likelihood.
    """

    def __init__(self, input_dim, hidden_dim=64, n_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
            ])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x):
        return self.net(x).squeeze()


def cox_partial_likelihood_loss(eta, times, events):
    """Cox partial likelihood loss for DeepSurv training."""
    order          = torch.argsort(times, descending=True)
    eta_sorted     = eta[order]
    events_sorted  = events[order]
    log_cumsum_exp = torch.logcumsumexp(eta_sorted, dim=0)
    event_mask     = events_sorted == 1
    if event_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return -(eta_sorted[event_mask] -
             log_cumsum_exp[event_mask]).mean()


def fit_deepsurv(X_train, t_train, d_train, X_test,
                 epochs=500, hidden_dim=64, n_layers=2,
                 lr=1e-3, weight_decay=1e-4):
    """
    Fit DeepSurv model.

    Uses published default configuration:
        hidden_dim=64, n_layers=2, epochs=500.
    No hyperparameter tuning applied.
    """
    X_tr  = torch.tensor(X_train, dtype=torch.float32)
    t_tr  = torch.tensor(t_train, dtype=torch.float32)
    d_tr  = torch.tensor(d_train, dtype=torch.float32)
    X_te  = torch.tensor(X_test,  dtype=torch.float32)
    model = DeepSurvNet(X_train.shape[1], hidden_dim, n_layers)
    opt   = torch.optim.Adam(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = cox_partial_likelihood_loss(model(X_tr), t_tr, d_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        eta_test = model(X_te).numpy()
    return model, eta_test


def deepsurv_ibs(ds_model, X_train, X_test,
                 t_train, d_train, t_test, d_test,
                 n_time_points=50):
    """
    Integrated Brier Score for DeepSurv via Breslow estimator.

    Uses IPCW weighting via Kaplan-Meier on the training
    censoring distribution. Breslow baseline cumulative hazard
    is computed from training data only.
    """
    from lifelines import KaplanMeierFitter

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32)

    ds_model.eval()
    with torch.no_grad():
        eta_train = ds_model(X_tr_t).numpy()
        eta_test  = ds_model(X_te_t).numpy()

    t_min  = 0.0
    t_max  = np.quantile(t_train, 0.98)
    t_grid = np.linspace(t_min, t_max, n_time_points)

    order      = np.argsort(t_train)
    t_sorted   = t_train[order]
    d_sorted   = d_train[order]

    exp_eta_train = np.exp(eta_train)
    exp_eta_test  = np.exp(eta_test)

    kmf = KaplanMeierFitter()
    kmf.fit(t_train, event_observed=1 - d_train)

    brier_scores = []

    for t_star in t_grid:
        mask = t_sorted <= t_star
        if mask.sum() == 0:
            brier_scores.append(0.0)
            continue

        at_risk_sum = np.array([
            exp_eta_train[t_train >= t_sorted[k]].sum()
            for k in range(len(t_sorted))
        ])
        at_risk_sum = np.where(at_risk_sum == 0, 1e-8, at_risk_sum)

        dN      = d_sorted.astype(float)
        H0_star = np.sum((dN / at_risk_sum)[mask])
        S_hat   = np.exp(-H0_star * exp_eta_test)

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
