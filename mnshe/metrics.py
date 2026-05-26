"""
Evaluation metrics for survival analysis.

Provides Harrell's C-index computed from first principles
to avoid external dependency differences across versions.
"""

import numpy as np


def concordance_index_manual(times, scores, events):
    """
    Harrell's C-index.

    Computes the probability that a subject who experiences
    the event earlier receives a higher predicted risk score.

    Parameters
    ----------
    times  : array-like  observed times
    scores : array-like  predicted risk scores (higher = more risk)
    events : array-like  event indicators (1=event, 0=censored)

    Returns
    -------
    float : C-index in [0, 1], or nan if no comparable pairs
    """
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
