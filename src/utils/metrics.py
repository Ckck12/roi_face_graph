# src/utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    정확도(Accuracy), AUC, Recall을 계산하는 함수.

    Args:
        y_true (np.ndarray): 실제 레이블 (0 또는 1).
        y_pred_probs (np.ndarray): 양성 클래스(1)의 예측 확률.
        threshold (float): 클래스를 결정할 임계값.

    Returns:
        tuple: (accuracy, auc, recall)
    """
    # 1) 예측 레이블 생성
    preds = (y_pred_probs >= threshold).astype(int)

    # 2) 정확도 계산
    accuracy = accuracy_score(y_true, preds)

    # 3) AUC 계산
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        # 모든 레이블이 동일(전부0 or 전부1)하면 roc_auc_score 계산 불가
        auc = float('nan')

    # 4) Recall(재현율) 계산
    #   = TP / (TP + FN)
    recall = recall_score(y_true, preds)  

    return accuracy, auc, recall
