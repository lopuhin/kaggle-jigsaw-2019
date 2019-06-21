import numpy as np
from scipy.special import expit, logit

from .metrics import compute_bias_metrics_for_model, IDENTITY_COLUMNS


def improve_predictions(preds, iterative=True, min_gain=1e-5):
    # TODO handle intersecting identities?
    preds = preds.copy()
    base_auc = compute_bias_metrics_for_model(preds, 'prediction')['auc']
    results = []
    for subgroup in IDENTITY_COLUMNS:
        best_auc = base_auc
        best_delta = 0
        best_preds = preds
        for delta in sorted(set([0] + list(np.linspace(-0.4, 0.4, 40)))):
            hacked = preds.copy()
            _apply_delta(hacked, subgroup, delta)
            auc = compute_bias_metrics_for_model(hacked, 'prediction')['auc']
            if auc > best_auc:
                best_delta = delta
                best_auc = auc
                best_preds = hacked
        gain = best_auc - base_auc
        results.append((subgroup, gain, best_delta))
        print(results[-1])
        if iterative and gain >= min_gain:
            preds = best_preds
            base_auc = best_auc
    if not iterative:
        for subgroup, gain, delta in results:
            if gain >= min_gain:
                _apply_delta(preds, subgroup, delta)
    return preds


def _apply_delta(df, subgroup, delta):
    subgroup_probs = df[subgroup] >= 0.5, 'prediction'
    df.loc[subgroup_probs] = expit(delta + logit(df.loc[subgroup_probs]))
