# src/engine/evaluator.py

import torch
import numpy as np
from tqdm import tqdm
from ..utils.metrics import compute_metrics

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
    y_pred_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating({mode}) Epoch {epoch+1}"):
            frames = batch["frames"].to(device)
            bboxes = batch["bboxes"].to(device)
            labels = batch["label"].to(device)

            outputs = model(frames, bboxes)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:,1].detach().cpu().numpy()
            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(probs)

    avg_loss = running_loss / len(dataloader)
    acc, auc = compute_metrics(np.array(y_true_list), np.array(y_pred_list))

    if wandb is not None:
        wandb.log({f"{mode}_loss": avg_loss, f"{mode}_acc": acc, f"{mode}_auc": auc}, step=epoch)

    return avg_loss, acc, auc
