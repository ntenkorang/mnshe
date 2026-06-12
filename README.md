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
pip install torch lifelines scikit-survival pycox scikit-learn scipy pandas matplotlib
```

## Repository Structure

```
mnshe/
├── experiments/
│   ├── simulation_study.py
│   ├── real_data_gbsg.py
│   └── real_data_flchain.py
├── results/
│   ├── best_hparams.json
│   ├── best_ds_hparams.json
│   ├── gbsg_results.json
│   ├── flchain_results.json
│   └── simulation/
└── README.md
```

## Citation

```
@article{mnshe2025,
  title   = {Monotonic Neural Spline Hazard Estimation
             for Right-Censored Survival Data},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]}
}
```
