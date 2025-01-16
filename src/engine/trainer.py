# src/engine/trainer.py

import os
import torch
import numpy as np
from tqdm import tqdm
from src.utils.metrics import compute_metrics

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch=0,
    wandb=None
):
    model.train()
    running_loss = 0.0
    y_true_list = []
    y_pred_probs_list = []

    for batch in tqdm(dataloader, desc=f"Train Epoch {epoch+1}"):
        frames = batch["frames"].to(device)
        bboxes = batch["bboxes"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(frames, bboxes)  # (B, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 배치 손실 합산
        running_loss += loss.item() * frames.size(0)

        # 양성 클래스(1) 확률
        probs = torch.softmax(outputs, dim=1)[:,1].detach().cpu().numpy()
        y_true_list.extend(labels.cpu().numpy())
        y_pred_probs_list.extend(probs)

    avg_loss = running_loss / len(dataloader.dataset)
    # compute_metrics가 (acc, auc, recall) 반환
    acc, auc, recall = compute_metrics(np.array(y_true_list), np.array(y_pred_probs_list))

    if wandb is not None:
        wandb.log({
            "train_loss": avg_loss,
            "train_acc": acc,
            "train_auc": auc,
            "train_recall": recall
        }, step=epoch)

    # ★ 반환값을 4개로: (loss, acc, auc, recall)
    return avg_loss, acc, auc, recall
