# src/utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    정확도(Accuracy)와 AUC를 계산하는 함수.

    Args:
        y_true (np.ndarray): 실제 레이블 (0 또는 1).
        y_pred_probs (np.ndarray): 양성 클래스의 예측 확률.
        threshold (float): 클래스를 결정할 임계값.

    Returns:
        tuple: (accuracy, auc)
    """
    # 예측 레이블 생성
    preds = (y_pred_probs >= threshold).astype(int)
    
    # 정확도 계산
    accuracy = accuracy_score(y_true, preds)
    
    # AUC 계산 (레이블에 불균형이 없는지 확인 필요)
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = float('nan')  # AUC 계산 불가 시 NaN 반환
    
    return accuracy, auc
