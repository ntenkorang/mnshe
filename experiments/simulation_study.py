"""
Simulation study for MNSHE.

Evaluates MNSHE, Cox PH, and DeepSurv across three
data-generating processes and two sample sizes over
100 independent replications each.

Pre-computed results are provided in results/simulation/.
To run a quick sanity check, set n_reps=5.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from mnshe.dgp import (
    simulate_weibull_ph,
    simulate_weibull_nonph,
    simulate_local_spike,
)
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


def run_one_rep(dgp_name, simulator, n, p, seed, hparams):
    """Run a single replication for one DGP and sample size."""

    t_obs, delta, X, _ = simulator(n=n, p=p, seed=seed)

    setup = prepare_train_test(
        t_obs=t_obs, delta=delta, X=X,
        test_size=0.15, val_size=0.15,
        seed=seed, degree=3,
        K_internal=hparams["K_internal"]
    )

    model, _, _, best_epoch = fit_mnshe(
        setup,
        hidden_dim=hparams["hidden_dim"],
        n_layers=hparams["n_layers"],
        epochs=hparams["epochs"],
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
        lambda_reg=hparams["lambda_reg"],
        patience=hparams["patience"]
    )

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
    h_viol    = check_monotonicity(model, setup)

    cox_risk, cox_model = fit_cox_ph(
        setup["X_train"], setup["t_train"],
        setup["d_train"], setup["X_test"]
    )
    cox_c = concordance_index_manual(
        setup["t_test"], cox_risk, setup["d_test"]
    ) if cox_risk is not None else np.nan
    cox_ibs_val = cox_integrated_brier_score(
        cox_model, setup
    ) if cox_model is not None else np.nan

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

    return {
        "dgp":           dgp_name,
        "n":             n,
        "rep":           seed,
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


def run_simulation_study(hparams, n_reps=100,
                         sample_sizes=[500, 1000],
                         seed_start=0,
                         save_dir="results/simulation"):
    """
    Full simulation study.

    Results are saved per DGP per sample size as JSON files
    after each condition completes, preserving partial results
    if the run is interrupted.
    """
    os.makedirs(save_dir, exist_ok=True)

    dgps = [
        ("Weibull PH",     simulate_weibull_ph),
        ("Non-PH Weibull", simulate_weibull_nonph),
        ("Local Spike",    simulate_local_spike),
    ]

    all_results = []

    for n in sample_sizes:
        for dgp_name, simulator in dgps:
            print(f"\n{'='*60}")
            print(f"DGP: {dgp_name}  |  n={n}")
            print(f"{'='*60}")

            rep_results = []

            for rep in range(n_reps):
                seed = seed_start + rep * 100
                try:
                    result = run_one_rep(
                        dgp_name, simulator,
                        n=n, p=5, seed=seed,
                        hparams=hparams
                    )
                    rep_results.append(result)

                    if (rep + 1) % 10 == 0:
                        mean_c = np.mean([
                            r["mnshe_test_c"]
                            for r in rep_results
                        ])
                        print(f"  Rep {rep+1:3d}/{n_reps} "
                              f"| Running mean MNSHE C: "
                              f"{mean_c:.4f}")

                except Exception as e:
                    print(f"  Rep {rep} failed: {e}")
                    continue

            fname = os.path.join(
                save_dir,
                f"{dgp_name.replace(' ', '_')}_{n}.json"
            )
            with open(fname, "w") as f:
                json.dump(rep_results, f)
            print(f"  Saved {len(rep_results)} reps to {fname}")

            all_results.extend(rep_results)

    return all_results


def print_simulation_table(all_results):
    """Print results table to console."""
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 130)
    print("SIMULATION STUDY — Mean (SD) across replications")
    print("=" * 130)

    header = (
        f"{'DGP':<18} {'n':>5}"
        f"{'MNSHE C':>12}"
        f"{'Cox C':>12}"
        f"{'DS C':>10}"
        f"{'MNSHE IBS':>12}"
        f"{'Cox IBS':>12}"
        f"{'DS IBS':>10}"
        f"{'H Viol':>8}"
    )
    print(header)
    print("-" * 130)

    for n in [500, 1000]:
        for dgp in ["Weibull PH", "Non-PH Weibull", "Local Spike"]:
            sub = df[(df["dgp"] == dgp) & (df["n"] == n)]
            if len(sub) == 0:
                continue

            def fmt(col):
                m = sub[col].mean()
                s = sub[col].std()
                return f"{m:.3f}({s:.3f})"

            print(
                f"{dgp:<18} {n:>5}"
                f"{fmt('mnshe_test_c'):>12}"
                f"{fmt('cox_c'):>12}"
                f"{fmt('ds_c'):>10}"
                f"{fmt('mnshe_ibs'):>12}"
                f"{fmt('cox_ibs'):>12}"
                f"{fmt('ds_ibs'):>10}"
                f"{int(sub['H_violations'].sum()):>8}"
            )
        print()


def plot_cindex_distributions(all_results,
                               save_path="figures/cindex_distributions.png"):
    """Plot C-index distributions across replications."""
    df   = pd.DataFrame(all_results)
    dgps = ["Weibull PH", "Non-PH Weibull", "Local Spike"]
    ns   = sorted(df["n"].unique())

    fig, axes = plt.subplots(
        len(ns), len(dgps),
        figsize=(5 * len(dgps), 4 * len(ns))
    )

    if len(ns) == 1:
        axes = axes[np.newaxis, :]

    for row, n in enumerate(ns):
        for col, dgp in enumerate(dgps):
            ax  = axes[row, col]
            sub = df[(df["dgp"] == dgp) & (df["n"] == n)]

            for label, col_name, color in [
                ("MNSHE",    "mnshe_test_c", "steelblue"),
                ("Cox PH",   "cox_c",        "darkorange"),
                ("DeepSurv", "ds_c",         "seagreen"),
            ]:
                vals = sub[col_name].dropna()
                ax.hist(vals, bins=20, alpha=0.5,
                        label=label, color=color)

            ax.set_title(f"{dgp}  |  n={n}")
            ax.set_xlabel("C-index")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

    plt.suptitle(
        "C-index Distributions Across Replications",
        fontsize=14
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


def plot_nonph_survival_curves(hparams,
                                save_path="figures/nonph_crossing_curves.png"):
    """
    Plot MNSHE survival curves for Non-PH DGP.
    Three subjects selected at extreme x1 values.
    """
    print("\nFitting demo model for Non-PH survival curve figure...")
    t_obs, delta, X, _ = simulate_weibull_nonph(
        n=1000, p=5, seed=999
    )
    setup_demo = prepare_train_test(
        t_obs, delta, X,
        test_size=0.15, val_size=0.15,
        seed=999, K_internal=hparams["K_internal"]
    )
    model_demo, _, _, _ = fit_mnshe(
        setup_demo,
        hidden_dim=hparams["hidden_dim"],
        n_layers=hparams["n_layers"],
        epochs=hparams["epochs"],
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
        lambda_reg=hparams["lambda_reg"],
        patience=hparams["patience"]
    )

    x1_vals     = setup_demo["X_test"][:, 0]
    idx_high_x1 = np.argmax(x1_vals)
    idx_low_x1  = np.argmin(x1_vals)
    idx_mid_x1  = np.argsort(x1_vals)[len(x1_vals) // 2]

    selected_ids = {
        "High $x_1$ (early risk)": idx_high_x1,
        "Medium $x_1$":            idx_mid_x1,
        "Low $x_1$ (late risk)":   idx_low_x1,
    }

    t_grid   = np.linspace(
        setup_demo["t_min"], setup_demo["t_max"], 300
    )
    M_grid   = mspline_basis(
        t_grid, setup_demo["knots"], setup_demo["degree"],
        setup_demo["t_min"], setup_demo["t_max"]
    )
    I_grid   = ispline_basis(
        t_grid, setup_demo["knots"], setup_demo["degree"],
        setup_demo["t_min"], setup_demo["t_max"]
    )
    M_grid_t = torch.tensor(M_grid, dtype=torch.float32)
    I_grid_t = torch.tensor(I_grid, dtype=torch.float32)

    plt.figure(figsize=(8, 5))
    model_demo.eval()
    colors = ["steelblue", "darkorange", "seagreen"]

    for (label, idx), color in zip(selected_ids.items(), colors):
        x_i = torch.tensor(
            setup_demo["X_test"][idx], dtype=torch.float32
        ).unsqueeze(0).repeat(len(t_grid), 1)

        with torch.no_grad():
            _, H_g, _ = model_demo(x_i, M_grid_t, I_grid_t)

        S_g = np.exp(-H_g.numpy())
        plt.plot(t_grid, S_g, label=label,
                 color=color, linewidth=2)

    plt.xlabel("Time", fontsize=12)
    plt.ylabel("$S(t \\mid x)$", fontsize=12)
    plt.title(
        "MNSHE Survival Curves: Non-PH Weibull DGP\n"
        "Subjects selected at extreme $x_1$ values",
        fontsize=11
    )
    plt.legend(fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved to {save_path}")


if __name__ == "__main__":

    hparams_path = "results/best_hparams.json"
    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hparams = json.load(f)
        print(f"Loaded hyperparameters from {hparams_path}")
    else:
        raise FileNotFoundError(
            f"{hparams_path} not found. "
            "Run experiments/hyperparameter_cv.py first."
        )

    # Full simulation study — set n_reps=5 for quick check
    all_results = run_simulation_study(
        hparams=hparams,
        n_reps=100,
        sample_sizes=[500, 1000],
        seed_start=0,
        save_dir="results/simulation"
    )

    print_simulation_table(all_results)
    plot_cindex_distributions(all_results)
    plot_nonph_survival_curves(hparams)

    print("\nSimulation study complete.")
