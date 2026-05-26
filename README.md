# MNSHE — Monotonic Neural Spline Hazard Estimation for Right-Censored Survival Data

Code for the paper:

**"Monotonic Neural Spline Hazard Estimation for
Right-Censored Survival Data"**

---

## Overview

MNSHE is a continuous-time deep survival modelling framework
that guarantees a nondecreasing cumulative hazard by
architectural construction. The hazard is modelled as a
covariate-dependent nonnegative combination of M-spline
basis functions, with the cumulative hazard represented
analytically through the corresponding I-spline expansion.
A scale-shape neural parameterisation maps covariates to
strictly positive spline coefficients, ensuring structural
validity for every individual without post-hoc corrections.

---

## Repository Structure

```
mnshe/
├── mnshe/                  # Core package
│   ├── splines.py          # M-spline and I-spline basis functions
│   ├── model.py            # MNSHE architecture
│   ├── loss.py             # Penalised negative log-likelihood
│   ├── training.py         # Data splitting, fitting, IBS, monotonicity
│   ├── baselines.py        # Cox PH and DeepSurv baselines
│   ├── metrics.py          # C-index
│   └── dgp.py              # Data-generating processes
├── experiments/
│   ├── hyperparameter_cv.py    # 5-fold CV hyperparameter selection
│   ├── simulation_study.py     # Full simulation study
│   └── real_data_gbsg.py       # GBSG real data analysis
├── results/
│   ├── best_hparams.json       # CV-selected hyperparameters
│   ├── simulation/             # Pre-computed simulation results
│   │   ├── Weibull_PH_500.json
│   │   ├── Weibull_PH_1000.json
│   │   ├── Non-PH_Weibull_500.json
│   │   ├── Non-PH_Weibull_1000.json
│   │   ├── Local_Spike_500.json
│   │   └── Local_Spike_1000.json
│   └── gbsg_results.json       # Pre-computed GBSG results
├── figures/                    # Generated figures
├── requirements.txt
└── README.md
```

---

## Requirements

```
python >= 3.11
torch >= 2.0.0
numpy >= 1.24.0
scipy >= 1.10.0
scikit-learn >= 1.2.0
lifelines >= 0.27.0
pycox >= 0.2.3
pandas >= 1.5.0
matplotlib >= 3.7.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

The GBSG dataset is loaded automatically via pycox:

```python
from pycox.datasets import gbsg
df = gbsg.read_df()
```

No manual download is required.

---

## Reproducing Results

### Step 1 — Hyperparameter selection

```bash
python experiments/hyperparameter_cv.py
```

Saves `results/best_hparams.json`. Skip this step if the file
already exists — the pre-computed configuration is provided.

### Step 2 — Simulation study

```bash
python experiments/simulation_study.py
```

To run a quick sanity check with 5 replications, edit
`n_reps=5` in the `__main__` block of `simulation_study.py`.

Pre-computed results for all 600 model fits are provided
in `results/simulation/` and correspond exactly to the
tables reported in the paper.

### Step 3 — Real data analysis

```bash
python experiments/real_data_gbsg.py
```

Requires `results/best_hparams.json` from Step 1.

---

## Pre-Computed Results

All simulation results and GBSG results are provided in
`results/` and correspond exactly to the numbers reported
in the paper. These allow verification of reported results
without rerunning the full simulation study.

---

## Notes on Reproducibility

Results were produced using Python 3.11 and PyTorch 2.0.
Minor numerical differences may occur across platforms and
PyTorch versions due to floating point implementation
differences. The pre-computed results in `results/`
correspond to the exact numbers reported in the paper.

All random seeds are specified per-function via
`numpy.random.default_rng(seed)` and `seed` arguments,
ensuring that individual replications are independently
reproducible given the same seed.

---

## Citation

If you use this code please cite:

```
@article{mnshe2025,
  title   = {Monotonic Neural Spline Hazard Estimation
             for Right-Censored Survival Data},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]}
}
```

---

## License

MIT License. See LICENSE file for details.
