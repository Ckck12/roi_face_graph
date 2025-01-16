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
    """
    *비디오 레벨* 평가 루프.
    - frames: (B, T, 3, H, W)
    - bboxes: (B, T, N, 4)
    - labels: (B,)
    => model(...) -> (B, num_classes)
    => label도 (B,)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    y_true_list = []
    y_pred_probs_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating({mode}) Epoch {epoch+1}"):
            frames = batch["frames"].to(device)   # (B,T,3,H,W)
            bboxes = batch["bboxes"].to(device)   # (B,T,N,4)
            labels = batch["label"].to(device)    # (B,)

            B = frames.size(0)

            # Forward => (B, num_classes)
            outputs = model(frames, bboxes)
            loss = criterion(outputs, labels)

            # 배치 손실 * B
            running_loss += loss.item() * B
            total_samples += B

            # 양성 클래스 확률 => (B,)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            # 누적
            y_true_list.extend(labels.cpu().numpy())        # (B,)
            y_pred_probs_list.extend(probs)                 # (B,)

    # 최종 손실
    avg_loss = running_loss / total_samples

    # 비디오 단위 Acc/AUC
    acc, auc = compute_metrics(np.array(y_true_list), np.array(y_pred_probs_list))

    if wandb is not None:
        wandb.log({f"{mode}_loss": avg_loss, f"{mode}_acc": acc, f"{mode}_auc": auc}, step=epoch)

    return avg_loss, acc, auc
