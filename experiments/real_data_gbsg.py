"""
Real data analysis on the GBSG breast cancer dataset.

Uses three-way 70/15/15 train/val/test split consistent
with the simulation study. Hyperparameters loaded from
results/best_hparams.json without modification.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from mnshe.training import (
    prepare_train_test,
    fit_mnshe,
    get_risk_scores,
    integrated_brier_score,
    check_monotonicity,
)
from mnshe.baselines import (
    fit_cox_ph,
    cox_integrated_brier_score,
    fit_deepsurv,
    deepsurv_ibs,
)
from mnshe.metrics import concordance_index_manual
from mnshe.splines import mspline_basis, ispline_basis


def run_real_data_analysis(
    hparams_path="results/best_hparams.json",
    results_path="results/gbsg_results.json",
    figures_dir="figures"
):
    """
    Real data analysis on GBSG breast cancer dataset.

    Validation used for early stopping only.
    Test used for final evaluation only.
    Hyperparameters transferred unchanged from simulation study.
    """

    print("=" * 60)
    print("REAL DATA ANALYSIS: GBSG Breast Cancer Dataset")
    print("=" * 60)

    # Load GBSG data
    try:
        from pycox.datasets import gbsg
        df = gbsg.read_df()
        print(f"Loaded GBSG via pycox: {df.shape}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        print("Install pycox with: pip install pycox")
        return None

    duration_col = "duration"
    event_col    = "event"

    X     = df.drop(
        columns=[duration_col, event_col]
    ).values.astype(float)
    t_obs = df[duration_col].values.astype(float)
    delta = df[event_col].values.astype(float)
    p     = X.shape[1]

    print(f"n = {len(t_obs)}")
    print(f"p = {p}")
    print(f"Event rate: {delta.mean():.3f}")
    print(f"Time range: [{t_obs.min():.1f}, {t_obs.max():.1f}]")

    # Three-way split consistent with simulation study
    setup = prepare_train_test(
        t_obs=t_obs, delta=delta, X=X,
        test_size=0.15,
        val_size=0.15,
        seed=42, degree=3,
        K_internal=5
    )
    print(f"Basis dimension K: {setup['M_train'].shape[1]}")

    # Load CV-selected hyperparameters
    try:
        with open(hparams_path) as f:
            hparams = json.load(f)
        print(f"\nUsing CV-selected hyperparameters from {hparams_path}")
    except FileNotFoundError:
        hparams = {
            "hidden_dim":   128,
            "n_layers":     3,
            "lambda_reg":   1e-3,
            "lr":           1e-3,
            "weight_decay": 5e-4,
            "patience":     150,
            "epochs":       2000,
            "K_internal":   5,
        }
        print(f"  {hparams_path} not found. Using defaults.")

    # Fit MNSHE
    print("\nFitting MNSHE...")
    model, train_losses, val_losses, best_epoch = fit_mnshe(
        setup,
        hidden_dim=hparams["hidden_dim"],
        n_layers=hparams["n_layers"],
        epochs=hparams["epochs"],
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
        lambda_reg=hparams["lambda_reg"],
        patience=hparams["patience"]
    )
    print(f"  Early stopping at epoch {best_epoch}")

    # MNSHE metrics on test set only
    X_tr_t = torch.tensor(setup["X_train"], dtype=torch.float32)
    X_te_t = torch.tensor(setup["X_test"],  dtype=torch.float32)

    risk_tr = get_risk_scores(model, X_tr_t, setup)
    risk_te = get_risk_scores(model, X_te_t, setup)

    mnshe_train_c = concordance_index_manual(
        setup["t_train"], risk_tr, setup["d_train"]
    )
    mnshe_test_c = concordance_index_manual(
        setup["t_test"], risk_te, setup["d_test"]
    )
    mnshe_ibs = integrated_brier_score(model, setup)
    h_viol    = check_monotonicity(
        model, setup,
        n_subjects=len(setup["X_test"]),
        grid_size=300, seed=42
    )

    print(f"\n  MNSHE Train C-index: {mnshe_train_c:.4f}")
    print(f"  MNSHE Test  C-index: {mnshe_test_c:.4f}")
    print(f"  MNSHE IBS:           {mnshe_ibs:.4f}")
    print(f"  H violations:        {h_viol}")

    # Cox PH baseline
    print("\nFitting Cox PH...")
    cox_risk, cox_model = fit_cox_ph(
        setup["X_train"], setup["t_train"],
        setup["d_train"], setup["X_test"]
    )
    if cox_risk is not None:
        cox_c       = concordance_index_manual(
            setup["t_test"], cox_risk, setup["d_test"]
        )
        cox_ibs_val = cox_integrated_brier_score(cox_model, setup)
        print(f"  Cox PH C-index: {cox_c:.4f}")
        print(f"  Cox PH IBS:     {cox_ibs_val:.4f}")
    else:
        cox_c       = np.nan
        cox_ibs_val = np.nan

    # DeepSurv baseline
    print("\nFitting DeepSurv...")
    ds_model, ds_risk = fit_deepsurv(
        setup["X_train"], setup["t_train"],
        setup["d_train"], setup["X_test"],
        epochs=500, hidden_dim=64, n_layers=2
    )
    ds_c = concordance_index_manual(
        setup["t_test"], ds_risk, setup["d_test"]
    )
    ds_ibs_val = deepsurv_ibs(
        ds_model,
        setup["X_train"], setup["X_test"],
        setup["t_train"], setup["d_train"],
        setup["t_test"],  setup["d_test"]
    )
    print(f"  DeepSurv C-index: {ds_c:.4f}")
    print(f"  DeepSurv IBS:     {ds_ibs_val:.4f}")

    # Results summary
    print("\n" + "=" * 60)
    print("GBSG RESULTS TABLE")
    print("=" * 60)
    print(f"{'Model':<15} {'C-index':>10} {'IBS':>10}")
    print("-" * 60)
    print(f"{'MNSHE':<15} {mnshe_test_c:>10.4f} {mnshe_ibs:>10.4f}")
    print(f"{'Cox PH':<15} {cox_c:>10.4f} {cox_ibs_val:>10.4f}")
    print(f"{'DeepSurv':<15} {ds_c:>10.4f} {ds_ibs_val:>10.4f}")
    print("=" * 60)

    os.makedirs(figures_dir, exist_ok=True)

    # Training loss figure
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train",      linewidth=1.5)
    plt.plot(val_losses,   label="Validation", linewidth=1.5,
             linestyle="--")
    plt.axvline(x=best_epoch, color="red", linestyle=":",
                alpha=0.7, label=f"Best epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("MNSHE Training Loss — GBSG")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/gbsg_training_loss.png", dpi=150)
    plt.show()

    # Individual survival curves
    print("\nPlotting individual survival curves...")

    risk_percentiles = [10, 30, 50, 70, 90]
    sorted_idx       = np.argsort(risk_te)
    n_test           = len(risk_te)

    selected = {
        f"p{pct}": sorted_idx[int(n_test * pct / 100)]
        for pct in risk_percentiles
    }

    t_grid = np.linspace(setup["t_min"], setup["t_max"], 300)
    M_g    = mspline_basis(t_grid, setup["knots"],
                           setup["degree"],
                           setup["t_min"], setup["t_max"])
    I_g    = ispline_basis(t_grid, setup["knots"],
                           setup["degree"],
                           setup["t_min"], setup["t_max"])
    M_g_t  = torch.tensor(M_g, dtype=torch.float32)
    I_g_t  = torch.tensor(I_g, dtype=torch.float32)

    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(selected)))

    plt.figure(figsize=(9, 5))
    model.eval()

    for (label, idx), color in zip(selected.items(), colors):
        x_i = torch.tensor(
            setup["X_test"][idx], dtype=torch.float32
        ).unsqueeze(0).repeat(len(t_grid), 1)

        with torch.no_grad():
            _, H_g, _ = model(x_i, M_g_t, I_g_t)

        S_g = np.exp(-H_g.numpy())
        plt.plot(t_grid, S_g, label=f"Percentile {label}",
                 color=color, linewidth=2)

    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Estimated Survival $S(t \\mid x)$", fontsize=12)
    plt.title(
        "MNSHE Individual Survival Curves — GBSG\n"
        "Subjects selected at risk percentiles 10, 30, 50, 70, 90",
        fontsize=11
    )
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/gbsg_survival_curves.png", dpi=300)
    plt.show()

    # Monotonicity verification
    print("\nMonotonicity verification on GBSG test set...")

    rng     = np.random.default_rng(42)
    n_check = min(20, n_test)
    chosen  = rng.choice(n_test, n_check, replace=False)

    all_violations = 0
    min_diffs      = []

    for idx in chosen:
        x_i = torch.tensor(
            setup["X_test"][idx], dtype=torch.float32
        ).unsqueeze(0).repeat(len(t_grid), 1)

        with torch.no_grad():
            _, H_g, _ = model(x_i, M_g_t, I_g_t)

        H_np  = H_g.numpy()
        diffs = np.diff(H_np)
        all_violations += int(np.sum(diffs < -1e-6))
        min_diffs.append(diffs.min())

    print(f"  Subjects checked: {n_check}")
    print(f"  Total H violations: {all_violations}")
    print(f"  Min diff: {min(min_diffs):.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    check_ids = chosen[:3]

    for ax, idx in zip(axes, check_ids):
        x_i = torch.tensor(
            setup["X_test"][idx], dtype=torch.float32
        ).unsqueeze(0).repeat(len(t_grid), 1)

        with torch.no_grad():
            _, H_g, _ = model(x_i, M_g_t, I_g_t)

        H_np = H_g.numpy()
        S_np = np.exp(-H_np)

        ax2 = ax.twinx()
        ax.plot(t_grid, H_np, color="steelblue",
                linewidth=2, label="$H(t|x)$")
        ax2.plot(t_grid, S_np, color="darkorange",
                 linewidth=2, linestyle="--", label="$S(t|x)$")

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Cumulative Hazard $H(t|x)$",
                      color="steelblue")
        ax2.set_ylabel("Survival $S(t|x)$", color="darkorange")
        ax.set_title(f"Subject {idx} (test set)")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=8, loc="center right")

    plt.suptitle(
        "Monotonicity Verification — GBSG Test Set\n"
        "$H(t|x)$ non-decreasing, $S(t|x)$ non-increasing "
        "by construction",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/gbsg_monotonicity.png", dpi=300)
    plt.show()

    # Save results
    gbsg_results = {
        "dataset":       "GBSG",
        "n_total":       len(t_obs),
        "n_train":       len(setup["t_train"]),
        "n_val":         len(setup["t_val"]),
        "n_test":        len(setup["t_test"]),
        "p":             p,
        "event_rate":    float(delta.mean()),
        "mnshe_train_c": mnshe_train_c,
        "mnshe_test_c":  mnshe_test_c,
        "mnshe_ibs":     mnshe_ibs,
        "cox_c":         cox_c,
        "cox_ibs":       cox_ibs_val,
        "ds_c":          ds_c,
        "ds_ibs":        ds_ibs_val,
        "H_violations":  h_viol,
        "best_epoch":    best_epoch,
    }

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(gbsg_results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Figures saved to {figures_dir}/")

    return gbsg_results


if __name__ == "__main__":
    run_real_data_analysis()
