import numpy as np

from .metrics import compute_bias_metrics_for_model, IDENTITY_COLUMNS


def improve_predictions(preds, iterative=False, min_gain=1e-4):
    preds = preds.copy()
    base_auc = compute_bias_metrics_for_model(preds, 'prediction')['auc']
    results = []
    for subgroup in IDENTITY_COLUMNS:
        best_auc = base_auc
        best_delta = 0
        best_preds = preds
        for delta in sorted(set([0] + list(np.linspace(-0.2, 0.2, 20)))):
            hacked = preds.copy()
            hacked.loc[hacked[subgroup] >= 0.5, 'prediction'] *= 1 + delta
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
                preds.loc[preds[subgroup] >= 0.5, 'prediction'] *= 1 + delta
    return preds
