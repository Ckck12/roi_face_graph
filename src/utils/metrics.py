# src/utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    pred_bin = (y_pred_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, pred_bin)
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = 0.0
    return acc, auc
