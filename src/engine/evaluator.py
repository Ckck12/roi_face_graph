# src/engine/evaluator.py

import torch
import numpy as np
from tqdm import tqdm
from src.utils.metrics import compute_metrics
        
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    epoch=0,
    mode="val",
    wandb=None
):
    model.eval()
    running_loss = 0.0
    y_true_list = []
    y_pred_probs_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating({mode}) Epoch {epoch+1}"):
            frames = batch["frames"].to(device)
            bboxes = batch["bboxes"].to(device)
            labels = batch["label"].to(device)

            outputs = model(frames, bboxes)  # (B, num_classes)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * frames.size(0)  # 배치 손실 * 배치 크기

            probs = torch.softmax(outputs, dim=1)[:,1].detach().cpu().numpy()  # 양성 클래스 확률 추출
            y_true_list.extend(labels.cpu().numpy())        # 실제 레이블 추가
            y_pred_probs_list.extend(probs)                 # 예측 확률 추가

    avg_loss = running_loss / len(dataloader.dataset)  # 전체 샘플 수로 나누기
    acc, auc = compute_metrics(np.array(y_true_list), np.array(y_pred_probs_list))


    if wandb is not None:
        wandb.log({f"{mode}_loss": avg_loss, f"{mode}_acc": acc, f"{mode}_auc": auc}, step=epoch)

    return avg_loss, acc, auc
