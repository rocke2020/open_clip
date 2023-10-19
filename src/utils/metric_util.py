from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score
)
from .log_util import logger
import numpy as np
from scipy import stats


def calc_metrics(y_true, y_score, threshold = 0.5):
    """ NB: return order is accuracy, f1, mcc, precision, recall
    -
    if threshold > 0: y_score = y_score >= threshold; else, directly uses y_score, that's treat y_score as integer """
    if threshold > 0:
        if not isinstance(y_score, np.ndarray):
            y_score = np.array(y_score)
        y_pred_id = y_score >= threshold
    else:
        y_pred_id = y_score
    accuracy = accuracy_score(y_true, y_pred_id)
    f1 = f1_score(y_true, y_pred_id)
    mcc = matthews_corrcoef(y_true, y_pred_id)
    precision = precision_score(y_true, y_pred_id)
    recall = recall_score(y_true, y_pred_id)
    return accuracy, f1, mcc, precision, recall


def calc_f1_precision_recall(y_true, y_predict):
    """  """
    # accuracy = accuracy_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    return f1, precision, recall


def find_threshold(y_true, y_score, alpha = 0.05):
    """ return threshold when fpr <= 0.05 """
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    for i, _fpr in enumerate(fpr):
        if _fpr > alpha:
            return thresh[i-1]


def roc(y_true, y_score):
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    roc = roc_auc_score(y_true, y_score)
    return roc, fpr, tpr


def calc_metrics_at_thresholds(
        y_true, y_pred_probability, thresholds=None, default_threshold=None):
    """  """
    performances = {}
    if default_threshold:
        accuracy, f1, mcc, precision, recall = calc_metrics(
            y_true, y_pred_probability, default_threshold)
        default_threshold
        performances[f'default_threshold_{default_threshold}'] = {
            'accuracy': accuracy, 'f1': f1, 'mcc': mcc, 
            'precision': precision, 'recall': recall}

    if not thresholds: 
        thresholds = []
    thresholds = sorted(thresholds, reverse=True)
    if 0.5 not in thresholds:
        thresholds.append(0.5)
    for threshold in thresholds:
        accuracy, f1, mcc, precision, recall = calc_metrics(
            y_true, y_pred_probability, threshold)
        performances[f'threshold_{threshold}'] = {
            'accuracy': accuracy, 'f1': f1, 'mcc': mcc, 
            'precision': precision, 'recall': recall}
    return performances


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """
    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (stats.entropy(p, m) + stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def calc_spearmanr(x, y):
    """  """
    res = stats.spearmanr(x, y)
    spearman_ratio = float(res.statistic)
    return spearman_ratio