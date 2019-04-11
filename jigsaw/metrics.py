from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # background negative, subgroup positive
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
POWER = -5
OVERALL_MODEL_WEIGHT = 0.25
MAIN_METRICS = [
    'auc', 'bias_score', 'overall_auc', SUBGROUP_AUC, BPSN_AUC, BNSP_AUC,
    'valid_loss']


def compute_bias_metrics_for_model(df: pd.DataFrame, pred_col: str) -> Dict:
    """ Computes per-subgroup metrics for all subgroups and one model.
    """
    df = _convert_dataframe_to_bool(df)
    records = []
    metrics = {}
    for subgroup in IDENTITY_COLUMNS:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(df[df[subgroup]]),
            SUBGROUP_AUC: _compute_subgroup_auc(df, subgroup, pred_col),
            BPSN_AUC: _compute_bpsn_auc(df, subgroup, pred_col),
            BNSP_AUC: _compute_bnsp_auc(df, subgroup, pred_col),
        }
        records.append(record)
        metrics.update({f'{subgroup}_{k}': v for k, v in record.items()})
    bias_df = pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
    metrics.update({
        SUBGROUP_AUC: _power_mean(bias_df[SUBGROUP_AUC], POWER),
        BPSN_AUC: _power_mean(bias_df[BPSN_AUC], POWER),
        BNSP_AUC: _power_mean(bias_df[BNSP_AUC], POWER),
    })

    bias_score = np.average([
        metrics[k] for k in [SUBGROUP_AUC, BPSN_AUC, BNSP_AUC]])
    overall_auc = _calculate_overall_auc(df, pred_col)
    auc = ((OVERALL_MODEL_WEIGHT * overall_auc) +
           ((1 - OVERALL_MODEL_WEIGHT) * bias_score))
    metrics.update({
        'bias_score': bias_score,
        'overall_auc': overall_auc,
        'auc': auc,
    })
    return metrics


def _convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)


def _convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col in ['target'] + IDENTITY_COLUMNS:
        _convert_to_bool(bool_df, col)
    return bool_df


def _compute_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def _compute_subgroup_auc(df, subgroup, pred_col):
    subgroup_examples = df[df[subgroup]]
    return _compute_auc(subgroup_examples['target'],
                        subgroup_examples[pred_col])


def _compute_bpsn_auc(df, subgroup, pred_col):
    """ Computes the AUC of the within-subgroup negative examples
    and the background positive examples.
    """
    subgroup_negative_examples = df[df[subgroup] & ~df['target']]
    non_subgroup_positive_examples = df[~df[subgroup] & df['target']]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return _compute_auc(examples['target'], examples[pred_col])


def _compute_bnsp_auc(df, subgroup, pred_col):
    """ Computes the AUC of the within-subgroup positive examples and
    the background negative examples.
    """
    subgroup_positive_examples = df[df[subgroup] & df['target']]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df['target']]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return _compute_auc(examples['target'], examples[pred_col])


def _calculate_overall_auc(df, pred_col):
    true_labels = df['target']
    predicted_labels = df[pred_col]
    return roc_auc_score(true_labels, predicted_labels)


def _power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def _get_final_metric(bias_df, overall_auc):
    bias_score = np.average([
        _power_mean(bias_df[SUBGROUP_AUC], POWER),
        _power_mean(bias_df[BPSN_AUC], POWER),
        _power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return ((OVERALL_MODEL_WEIGHT * overall_auc) +
            ((1 - OVERALL_MODEL_WEIGHT) * bias_score))
