# src/engine/trainer.py

import torch
import numpy as np
from tqdm import tqdm
from ..utils.metrics import compute_metrics

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
    y_pred_list = []

    for batch in tqdm(dataloader, desc=f"Train Epoch {epoch+1}"):
        frames = batch["frames"].to(device)   # (B,32,C,H,W)
        bboxes = batch["bboxes"].to(device)   # (B,32,6,4)
        labels = batch["label"].to(device)    # (B,)

        optimizer.zero_grad()
        outputs = model(frames, bboxes)       # (B, num_classes=2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:,1].detach().cpu().numpy()
        y_true_list.extend(labels.cpu().numpy())
        y_pred_list.extend(probs)

    avg_loss = running_loss / len(dataloader)
    acc, auc = compute_metrics(np.array(y_true_list), np.array(y_pred_list))

    if wandb is not None:
        wandb.log({"train_loss": avg_loss, "train_acc": acc, "train_auc": auc}, step=epoch)

    return avg_loss, acc, auc
