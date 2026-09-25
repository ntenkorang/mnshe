# MNSHE — Monotonic Neural Spline Hazard Estimation for Right-Censored Survival Data

Code for the paper:

**"Monotonic Neural Spline Hazard Estimation for Right-Censored Survival Data"**

## Models

- MNSHE (proposed)
- Cox Proportional Hazards
- DeepSurv
- Random Survival Forest

## Requirements

```
pip install torch lifelines scikit-survival pycox scikit-learn scipy pandas numpy
```

## Repository Structure

```
mnshe/
├── experiments/
│   ├── simulation_study.py
│   ├── real_data_gbsg.py
│   └── real_data_flchain.py
├── results/
│   └── simulation/
└── README.md
```

## Usage

Each script can be run directly:

```
python experiments/simulation_study.py
python experiments/real_data_gbsg.py
python experiments/real_data_flchain.py
```

Hyperparameters are selected once by cross-validation and cached
to `best_hparams.json` / `best_ds_hparams.json`, then reused across
all experiments.
