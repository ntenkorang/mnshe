"""
MNSHE — Monotonic Neural Spline Hazard Estimation for
Right-Censored Survival Data

A deep learning framework for valid and flexible survival analysis.
"""

from mnshe.model import MNSHE
from mnshe.splines import mspline_basis, ispline_basis, quantile_knots
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
from mnshe.dgp import (
    simulate_weibull_ph,
    simulate_weibull_nonph,
    simulate_local_spike,
)
