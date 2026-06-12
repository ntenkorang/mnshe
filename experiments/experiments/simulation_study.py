"""
Monotonic Neural Spline Hazard Estimation for Right-Censored Survival Data
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import BSpline
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from itertools import product


def make_knots(internal_knots, degree, t_min, t_max):
    return np.concatenate([
        np.repeat(t_min, degree + 1),
        internal_knots,
        np.repeat(t_max, degree + 1)
    ])


def bspline_basis_matrix(t, internal_knots, degree, t_min, t_max):
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
    knots  = make_knots(internal_knots, degree, t_min, t_max)
    B      = bspline_basis_matrix(t, internal_knots, degree, t_min, t_max)
    K      = B.shape[1]
    scales = np.zeros(K)
    for j in range(K):
        denom     = knots[j + degree + 1] - knots[j]
        scales[j] = (degree + 1) / denom if denom > 0 else 0.0
    return np.nan_to_num(B * scales[np.newaxis, :], nan=0.0)


def ispline_basis(t, internal_knots, degree, t_min, t_max):
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
    probs = np.linspace(0, 1, K_internal + 2)[1:-1]
    return np.quantile(event_times, probs)


def concordance_index_manual(times, scores, events):
    n          = len(times)
    concordant = 0
    tied       = 0
    comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
            if events[i] == 1 and times[i] < times[j]:
                comparable += 1
                if scores[i] > scores[j]:
                    concordant += 1
                elif scores[i] == scores[j]:
                    tied += 1
            elif events[j] == 1 and times[j] < times[i]:
                comparable += 1
                if scores[j] > scores[i]:
                    concordant += 1
                elif scores[j] == scores[i]:
                    tied += 1
    if comparable == 0:
        return np.nan
    return (concordant + 0.5 * tied) / comparable


def simulate_weibull_ph(n, p, shape=1.5, scale=5.0,
                        censoring_strength=0.3, seed=42):
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


class MNSHE(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, K):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.feature_net = nn.Sequential(*layers)
        self.shape_head  = nn.Linear(hidden_dim, K)
        self.scale_head  = nn.Linear(hidden_dim, 1)
        self.K           = K
        nn.init.normal_(self.shape_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.shape_head.bias, 0.0)
        nn.init.normal_(self.scale_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.scale_head.bias, -1.0)

    def forward(self, x, M_basis, I_basis):
        features     = self.feature_net(x)
        shape_logits = self.shape_head(features)
        weights      = torch.softmax(shape_logits, dim=1)
        scale        = torch.nn.functional.softplus(
                           self.scale_head(features)) + 1e-6
        alpha = scale * weights
        h     = (alpha * M_basis).sum(dim=1)
        H     = (alpha * I_basis).sum(dim=1)
        return h, H, alpha


def mnshe_nll(h, H, delta, alpha=None, lambda_reg=1e-3):
    nll = -(delta * torch.log(h + 1e-8) - H).mean()
    if alpha is not None:
        nll = nll + lambda_reg * torch.mean(alpha ** 2)
    return nll


class DeepSurvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x):
        return self.net(x).squeeze()


def cox_partial_likelihood_loss(eta, times, events):
    order          = torch.argsort(times, descending=True)
    eta_sorted     = eta[order]
    events_sorted  = events[order]
    log_cumsum_exp = torch.logcumsumexp(eta_sorted, dim=0)
    event_mask     = events_sorted == 1
    if event_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return -(eta_sorted[event_mask] - log_cumsum_exp[event_mask]).mean()


def fit_deepsurv(X_train, t_train, d_train, X_test,
                 epochs=500, hidden_dim=64, n_layers=2,
                 lr=1e-3, weight_decay=1e-4):
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
    order         = np.argsort(t_train)
    t_sorted      = t_train[order]
    d_sorted      = d_train[order]
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
        G_t  = max(float(kmf.survival_function_at_times([t_star]).iloc[0]), 1e-8)
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


def select_deepsurv_hyperparameters(t_obs, delta, X, seed=0, n_folds=5):
    param_grid = list(product([64, 128], [2, 3], [1e-3, 3e-4]))
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    kf          = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_score  = -np.inf
    best_params = None
    for hd, nl, lr in param_grid:
        fold_scores = []
        for tr_idx, ho_idx in kf.split(X_sc):
            sc_fold = StandardScaler()
            X_tr    = sc_fold.fit_transform(X[tr_idx])
            X_ho    = sc_fold.transform(X[ho_idx])
            t_tr    = t_obs[tr_idx]
            t_ho    = t_obs[ho_idx]
            d_tr    = delta[tr_idx]
            d_ho    = delta[ho_idx]
            try:
                _, eta = fit_deepsurv(X_tr, t_tr, d_tr, X_ho,
                                      epochs=500, hidden_dim=hd,
                                      n_layers=nl, lr=lr)
                c = concordance_index_manual(t_ho, eta, d_ho)
                fold_scores.append(c)
            except Exception:
                fold_scores.append(np.nan)
        mean_c = np.nanmean(fold_scores)
        if mean_c > best_score:
            best_score  = mean_c
            best_params = {"hidden_dim": hd, "n_layers": nl,
                           "lr": lr, "epochs": 500}
    return best_params


def fit_rsf(X_train, t_train, d_train, X_test,
            n_estimators=100, random_state=42):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    y_train = Surv.from_arrays(event=d_train.astype(bool), time=t_train)
    rsf = RandomSurvivalForest(n_estimators=n_estimators,
                                random_state=random_state, n_jobs=-1)
    rsf.fit(X_train, y_train)
    risk = rsf.predict(X_test)
    return rsf, risk


def rsf_integrated_brier_score(rsf_model, X_train, X_test,
                                t_train, d_train, t_test, d_test,
                                n_time_points=50):
    from lifelines import KaplanMeierFitter
    t_min  = 0.0
    t_max  = np.quantile(t_train, 0.98)
    t_grid = np.linspace(t_min, t_max, n_time_points)
    surv_funcs = rsf_model.predict_survival_function(X_test, return_array=False)
    kmf = KaplanMeierFitter()
    kmf.fit(t_train, event_observed=1 - d_train)
    brier_scores = []
    for t_star in t_grid:
        t_star_c = min(t_star, t_max * 0.99)
        S_hat = np.array([float(fn(t_star_c)) for fn in surv_funcs])
        S_hat = np.clip(S_hat, 1e-8, 1.0)
        G_t  = max(float(kmf.survival_function_at_times([t_star]).iloc[0]), 1e-8)
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


def fit_cox_ph(X_train, t_train, d_train, X_test):
    try:
        from lifelines import CoxPHFitter
        cols     = [f"x{j+1}" for j in range(X_train.shape[1])]
        train_df = pd.DataFrame(X_train, columns=cols)
        train_df["time"]  = t_train
        train_df["event"] = d_train
        test_df  = pd.DataFrame(X_test, columns=cols)
        cph      = CoxPHFitter(penalizer=0.01)
        cph.fit(train_df, duration_col="time", event_col="event")
        risk = cph.predict_partial_hazard(test_df).values.reshape(-1)
        return risk, cph
    except Exception as e:
        print(f"  Cox PH failed: {e}")
        return None, None


def cox_integrated_brier_score(cox_model, setup, n_time_points=50):
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
            test_df, times=[t_star]).values.flatten()
        G_t  = max(float(kmf.survival_function_at_times([t_star]).iloc[0]), 1e-8)
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


def prepare_train_test(t_obs, delta, X, test_size=0.15,
                       val_size=0.15, seed=42, degree=3, K_internal=5):
    idx = np.arange(len(t_obs))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=delta)
    delta_trainval = delta[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_size / (1.0 - test_size),
        random_state=seed, stratify=delta_trainval)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X[idx_train])
    X_val   = scaler.transform(X[idx_val])
    X_test  = scaler.transform(X[idx_test])
    t_train = t_obs[idx_train]; t_val = t_obs[idx_val]; t_test = t_obs[idx_test]
    d_train = delta[idx_train]; d_val = delta[idx_val]; d_test = delta[idx_test]
    t_min     = 0.0
    t_max     = np.quantile(t_train, 0.98)
    t_train_c = np.clip(t_train, t_min, t_max)
    t_val_c   = np.clip(t_val,   t_min, t_max)
    t_test_c  = np.clip(t_test,  t_min, t_max)
    knots     = quantile_knots(t_train_c[d_train == 1], K_internal)
    M_train = mspline_basis(t_train_c, knots, degree, t_min, t_max)
    I_train = ispline_basis(t_train_c, knots, degree, t_min, t_max)
    M_val   = mspline_basis(t_val_c,   knots, degree, t_min, t_max)
    I_val   = ispline_basis(t_val_c,   knots, degree, t_min, t_max)
    M_test  = mspline_basis(t_test_c,  knots, degree, t_min, t_max)
    I_test  = ispline_basis(t_test_c,  knots, degree, t_min, t_max)
    print(f"  Split: train={len(t_train)} | val={len(t_val)} | "
          f"test={len(t_test)} | event rates: "
          f"tr={d_train.mean():.2f} va={d_val.mean():.2f} te={d_test.mean():.2f}")
    return {
        "X_train": X_train, "X_val": X_val,   "X_test": X_test,
        "t_train": t_train_c, "t_val": t_val_c, "t_test": t_test_c,
        "d_train": d_train,   "d_val": d_val,   "d_test": d_test,
        "M_train": M_train,   "I_train": I_train,
        "M_val":   M_val,     "I_val":   I_val,
        "M_test":  M_test,    "I_test":  I_test,
        "knots": knots, "degree": degree,
        "t_min": t_min, "t_max":  t_max, "scaler": scaler,
    }


def fit_mnshe(setup, hidden_dim=128, n_layers=3, epochs=2000,
              lr=1e-3, weight_decay=5e-4, lambda_reg=1e-3, patience=150):
    def to_t(arr):
        return torch.tensor(arr, dtype=torch.float32)
    X_tr = to_t(setup["X_train"]); M_tr = to_t(setup["M_train"])
    I_tr = to_t(setup["I_train"]); d_tr = to_t(setup["d_train"])
    X_va = to_t(setup["X_val"]);   M_va = to_t(setup["M_val"])
    I_va = to_t(setup["I_val"]);   d_va = to_t(setup["d_val"])
    K     = setup["M_train"].shape[1]
    model = MNSHE(setup["X_train"].shape[1], hidden_dim, n_layers, K)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = np.inf; best_epoch = 0; best_state = None
    train_losses  = []; val_losses = []
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        h, H, alpha = model(X_tr, M_tr, I_tr)
        loss = mnshe_nll(h, H, d_tr, alpha, lambda_reg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            h_v, H_v, a_v = model(X_va, M_va, I_va)
            val_loss = mnshe_nll(h_v, H_v, d_va, a_v, lambda_reg)
        val_losses.append(val_loss.item())
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item(); best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_epoch


def get_risk_scores(model, X_t, setup):
    n     = len(X_t)
    t_e   = np.repeat(setup["t_max"], n)
    M_e   = mspline_basis(t_e, setup["knots"], setup["degree"], setup["t_min"], setup["t_max"])
    I_e   = ispline_basis(t_e, setup["knots"], setup["degree"], setup["t_min"], setup["t_max"])
    M_e_t = torch.tensor(M_e, dtype=torch.float32)
    I_e_t = torch.tensor(I_e, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        _, H_e, _ = model(X_t, M_e_t, I_e_t)
    return H_e.numpy()


def integrated_brier_score(model, setup, n_time_points=50):
    from lifelines import KaplanMeierFitter
    t_min    = setup["t_min"]; t_max = setup["t_max"]
    t_grid   = np.linspace(t_min, t_max, n_time_points)
    X_test_t = torch.tensor(setup["X_test"], dtype=torch.float32)
    n_test   = len(setup["X_test"]); t_test = setup["t_test"]; d_test = setup["d_test"]
    kmf = KaplanMeierFitter()
    kmf.fit(setup["t_train"], event_observed=1 - setup["d_train"])
    brier_scores = []
    for t_star in t_grid:
        t_e   = np.repeat(t_star, n_test)
        M_e   = mspline_basis(t_e, setup["knots"], setup["degree"], t_min, t_max)
        I_e   = ispline_basis(t_e, setup["knots"], setup["degree"], t_min, t_max)
        M_e_t = torch.tensor(M_e, dtype=torch.float32)
        I_e_t = torch.tensor(I_e, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            _, H_e, _ = model(X_test_t, M_e_t, I_e_t)
        S_hat = np.exp(-H_e.numpy())
        G_t  = max(float(kmf.survival_function_at_times([t_star]).iloc[0]), 1e-8)
        G_ti = np.maximum(np.array([
            float(kmf.survival_function_at_times([ti]).iloc[0]) for ti in t_test
        ]), 1e-8)
        I1 = ((t_test <= t_star) & (d_test == 1)).astype(float)
        I2 = (t_test > t_star).astype(float)
        bs = (np.mean(I1 * (0 - S_hat) ** 2 / G_ti) +
              np.mean(I2 * (1 - S_hat) ** 2 / G_t))
        brier_scores.append(bs)
    return np.trapezoid(brier_scores, t_grid) / (t_max - t_min)


def check_monotonicity(model, setup, n_subjects=10, grid_size=300, seed=0):
    rng    = np.random.default_rng(seed)
    X_test = setup["X_test"]
    chosen = rng.choice(len(X_test), min(n_subjects, len(X_test)), replace=False)
    t_grid = np.linspace(setup["t_min"], setup["t_max"], grid_size)
    M_g    = mspline_basis(t_grid, setup["knots"], setup["degree"], setup["t_min"], setup["t_max"])
    I_g    = ispline_basis(t_grid, setup["knots"], setup["degree"], setup["t_min"], setup["t_max"])
    M_g_t  = torch.tensor(M_g, dtype=torch.float64)
    I_g_t  = torch.tensor(I_g, dtype=torch.float64)
    model_d = model.double(); H_viol = 0
    model_d.eval()
    for idx in chosen:
        x_i = torch.tensor(X_test[idx], dtype=torch.float64).unsqueeze(0).repeat(grid_size, 1)
        with torch.no_grad():
            _, H_g, _ = model_d(x_i, M_g_t, I_g_t)
        H_np   = H_g.numpy(); diffs = np.diff(H_np)
        H_viol += int(np.sum(diffs < -1e-6))
    model.float()
    return H_viol


def select_hyperparameters(n_cv=500, p=5, seed=0, n_folds=5):
    print("=" * 60)
    print("MNSHE HYPERPARAMETER SELECTION via 5-fold CV")
    print("=" * 60)
    param_grid = list(product([64, 128], [2, 3], [1e-4, 1e-3], [1e-3, 3e-4]))
    t_obs, delta, X, _ = simulate_weibull_ph(n=n_cv, p=p, seed=seed)
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_score = -np.inf; best_params = None
    for i, (hd, nl, lam, lr) in enumerate(param_grid):
        fold_scores = []
        for tr_va_idx, ho_idx in kf.split(X_sc):
            delta_trva = delta[tr_va_idx]
            tr_idx, va_idx = train_test_split(
                tr_va_idx, test_size=0.20, random_state=seed, stratify=delta_trva)
            sc_fold = StandardScaler()
            X_tr = sc_fold.fit_transform(X[tr_idx]); X_va = sc_fold.transform(X[va_idx])
            X_ho = sc_fold.transform(X[ho_idx])
            t_tr = t_obs[tr_idx]; t_va = t_obs[va_idx]; t_ho = t_obs[ho_idx]
            d_tr = delta[tr_idx]; d_va = delta[va_idx]; d_ho = delta[ho_idx]
            t_min = 0.0; t_max = np.quantile(t_tr, 0.98)
            t_tr_c = np.clip(t_tr, t_min, t_max); t_va_c = np.clip(t_va, t_min, t_max)
            t_ho_c = np.clip(t_ho, t_min, t_max)
            knots = quantile_knots(t_tr_c[d_tr == 1], 5)
            setup_cv = {
                "X_train": X_tr, "X_val": X_va, "X_test": X_ho,
                "t_train": t_tr_c, "t_val": t_va_c, "t_test": t_ho_c,
                "d_train": d_tr, "d_val": d_va, "d_test": d_ho,
                "M_train": mspline_basis(t_tr_c, knots, 3, t_min, t_max),
                "I_train": ispline_basis(t_tr_c, knots, 3, t_min, t_max),
                "M_val":   mspline_basis(t_va_c, knots, 3, t_min, t_max),
                "I_val":   ispline_basis(t_va_c, knots, 3, t_min, t_max),
                "M_test":  mspline_basis(t_ho_c, knots, 3, t_min, t_max),
                "I_test":  ispline_basis(t_ho_c, knots, 3, t_min, t_max),
                "knots": knots, "degree": 3, "t_min": t_min, "t_max": t_max,
            }
            try:
                model, _, _, _ = fit_mnshe(setup_cv, hidden_dim=hd, n_layers=nl,
                                            epochs=500, lr=lr, weight_decay=5e-4,
                                            lambda_reg=lam, patience=50)
                X_ho_t = torch.tensor(X_ho, dtype=torch.float32)
                risk = get_risk_scores(model, X_ho_t, setup_cv)
                c = concordance_index_manual(t_ho_c, risk, d_ho)
                fold_scores.append(c)
            except Exception:
                fold_scores.append(np.nan)
        mean_c = np.nanmean(fold_scores)
        print(f"  [{i+1:2d}/{len(param_grid)}] hd={hd} nl={nl} lam={lam:.0e} lr={lr:.0e} -> CV C = {mean_c:.4f}")
        if mean_c > best_score:
            best_score  = mean_c
            best_params = {"hidden_dim": hd, "n_layers": nl, "lambda_reg": lam,
                           "lr": lr, "weight_decay": 5e-4, "patience": 150,
                           "epochs": 2000, "K_internal": 5}
    print(f"\nBest CV C-index: {best_score:.4f}")
    print(f"Best params:     {best_params}")
    return best_params


def run_one_rep(dgp_name, simulator, n, p, seed, hparams, ds_hparams):
    t_obs, delta, X, _ = simulator(n=n, p=p, seed=seed)
    setup = prepare_train_test(t_obs=t_obs, delta=delta, X=X,
                                test_size=0.15, val_size=0.15, seed=seed,
                                degree=3, K_internal=hparams["K_internal"])
    model, _, _, best_epoch = fit_mnshe(
        setup, hidden_dim=hparams["hidden_dim"], n_layers=hparams["n_layers"],
        epochs=hparams["epochs"], lr=hparams["lr"], weight_decay=hparams["weight_decay"],
        lambda_reg=hparams["lambda_reg"], patience=hparams["patience"])
    X_tr_t = torch.tensor(setup["X_train"], dtype=torch.float32)
    X_te_t = torch.tensor(setup["X_test"],  dtype=torch.float32)
    risk_tr = get_risk_scores(model, X_tr_t, setup)
    risk_te = get_risk_scores(model, X_te_t, setup)
    mnshe_train_c = concordance_index_manual(setup["t_train"], risk_tr, setup["d_train"])
    mnshe_test_c  = concordance_index_manual(setup["t_test"],  risk_te, setup["d_test"])
    mnshe_ibs = integrated_brier_score(model, setup)
    h_viol    = check_monotonicity(model, setup)
    cox_risk, cox_model = fit_cox_ph(setup["X_train"], setup["t_train"],
                                      setup["d_train"], setup["X_test"])
    cox_c = concordance_index_manual(setup["t_test"], cox_risk, setup["d_test"]) \
            if cox_risk is not None else np.nan
    cox_ibs_val = cox_integrated_brier_score(cox_model, setup) \
                  if cox_model is not None else np.nan
    ds_model, ds_risk = fit_deepsurv(
        setup["X_train"], setup["t_train"], setup["d_train"], setup["X_test"],
        epochs=ds_hparams["epochs"], hidden_dim=ds_hparams["hidden_dim"],
        n_layers=ds_hparams["n_layers"], lr=ds_hparams["lr"])
    ds_c = concordance_index_manual(setup["t_test"], ds_risk, setup["d_test"])
    ds_ibs_val = deepsurv_ibs(ds_model, setup["X_train"], setup["X_test"],
                               setup["t_train"], setup["d_train"],
                               setup["t_test"],  setup["d_test"])
    rsf_model, rsf_risk = fit_rsf(setup["X_train"], setup["t_train"],
                                   setup["d_train"], setup["X_test"])
    rsf_c = concordance_index_manual(setup["t_test"], rsf_risk, setup["d_test"])
    rsf_ibs_val = rsf_integrated_brier_score(rsf_model, setup["X_train"], setup["X_test"],
                                              setup["t_train"], setup["d_train"],
                                              setup["t_test"],  setup["d_test"])
    return {"dgp": dgp_name, "n": n, "rep": seed,
            "mnshe_train_c": mnshe_train_c, "mnshe_test_c": mnshe_test_c,
            "mnshe_ibs": mnshe_ibs, "cox_c": cox_c, "cox_ibs": cox_ibs_val,
            "ds_c": ds_c, "ds_ibs": ds_ibs_val, "rsf_c": rsf_c, "rsf_ibs": rsf_ibs_val,
            "H_violations": h_viol, "best_epoch": best_epoch}


def run_simulation_study(hparams, ds_hparams, n_reps=100,
                         sample_sizes=[500, 1000], seed_start=0, save_dir="results_revised"):
    os.makedirs(save_dir, exist_ok=True)
    dgps = [("Weibull PH", simulate_weibull_ph),
            ("Non-PH Weibull", simulate_weibull_nonph),
            ("Local Spike", simulate_local_spike)]
    all_results = []
    for n in sample_sizes:
        for dgp_name, simulator in dgps:
            print(f"\n{'='*60}\nDGP: {dgp_name}  |  n={n}\n{'='*60}")
            rep_results = []
            for rep in range(n_reps):
                seed = seed_start + rep * 100
                try:
                    result = run_one_rep(dgp_name, simulator, n=n, p=5, seed=seed,
                                         hparams=hparams, ds_hparams=ds_hparams)
                    rep_results.append(result)
                    if (rep + 1) % 10 == 0:
                        mean_c = np.mean([r["mnshe_test_c"] for r in rep_results])
                        print(f"  Rep {rep+1:3d}/{n_reps} | Running mean MNSHE C: {mean_c:.4f}")
                except Exception as e:
                    print(f"  Rep {rep} failed: {e}")
                    continue
            fname = os.path.join(save_dir, f"{dgp_name.replace(' ', '_')}_{n}.json")
            with open(fname, "w") as f:
                json.dump(rep_results, f)
            print(f"  Saved {len(rep_results)} reps to {fname}")
            all_results.extend(rep_results)
    return all_results


def print_simulation_table(all_results):
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 150)
    print("SIMULATION STUDY — Mean (SD) across replications")
    print("=" * 150)
    header = (f"{'DGP':<18} {'n':>5}{'MNSHE C':>12}{'Cox C':>12}"
              f"{'DS C':>10}{'RSF C':>10}{'MNSHE IBS':>12}"
              f"{'Cox IBS':>12}{'DS IBS':>10}{'RSF IBS':>10}{'H Viol':>8}")
    print(header)
    print("-" * 150)
    for n in [500, 1000]:
        for dgp in ["Weibull PH", "Non-PH Weibull", "Local Spike"]:
            sub = df[(df["dgp"] == dgp) & (df["n"] == n)]
            if len(sub) == 0:
                continue
            def fmt(col):
                m = sub[col].mean(); s = sub[col].std()
                return f"{m:.3f}({s:.3f})"
            print(f"{dgp:<18} {n:>5}{fmt('mnshe_test_c'):>12}{fmt('cox_c'):>12}"
                  f"{fmt('ds_c'):>10}{fmt('rsf_c'):>10}{fmt('mnshe_ibs'):>12}"
                  f"{fmt('cox_ibs'):>12}{fmt('ds_ibs'):>10}{fmt('rsf_ibs'):>10}"
                  f"{int(sub['H_violations'].sum()):>8}")
        print()


if __name__ == "__main__":
    if os.path.exists("best_hparams.json"):
        with open("best_hparams.json") as f:
            hparams = json.load(f)
        print("Loaded MNSHE hyperparameters from best_hparams.json")
    else:
        hparams = select_hyperparameters(n_cv=500, p=5, seed=0, n_folds=5)
        with open("best_hparams.json", "w") as f:
            json.dump(hparams, f, indent=2)

    if os.path.exists("best_ds_hparams.json"):
        with open("best_ds_hparams.json") as f:
            ds_hparams = json.load(f)
        print("Loaded DeepSurv hyperparameters from best_ds_hparams.json")
    else:
        print("\nSelecting DeepSurv hyperparameters via 5-fold CV...")
        t_cv, d_cv, X_cv, _ = simulate_weibull_ph(n=500, p=5, seed=0)
        ds_hparams = select_deepsurv_hyperparameters(t_cv, d_cv, X_cv, seed=0, n_folds=5)
        with open("best_ds_hparams.json", "w") as f:
            json.dump(ds_hparams, f, indent=2)
        print(f"DeepSurv best params: {ds_hparams}")

    all_results = run_simulation_study(
        hparams=hparams, ds_hparams=ds_hparams,
        n_reps=100, sample_sizes=[500, 1000], seed_start=0, save_dir="results_revised")

    print_simulation_table(all_results)
    print("\nSimulation study complete.")
