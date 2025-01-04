# src/train.py

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from .data import create_dataloader
from .engine import train_one_epoch, evaluate
from .models.final_model import FullPipelineModel
from .utils import set_seed, get_device, init_wandb, finish_wandb

def main_train(config):
    seed = config.get("seed", 42)
    set_seed(seed)

    device = get_device(0)

    # wandb
    wandb_run = init_wandb(config)

    # DataLoader
    train_loader = create_dataloader(
        csv_path=config["csv_file"],
        shape_predictor_path=config["shape_predictor_path"],
        dataset_type="train",
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        image_size=config["model"]["image_size"]
    )
    val_loader = create_dataloader(
        csv_path=config["csv_file"],
        shape_predictor_path=config["shape_predictor_path"],
        dataset_type="val",
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        image_size=config["model"]["image_size"]
    )

    # 모델
    m_cfg = config["model"]
    model = FullPipelineModel(
        image_size=m_cfg["image_size"],
        patch_size=m_cfg["patch_size"],
        hidden_dim=m_cfg["hidden_dim"],
        num_classes=m_cfg["num_classes"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    best_auc = 0.0
    epochs = config["epochs"]

    for epoch in range(epochs):
        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, wandb=wandb_run
        )
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device, epoch, "val", wandb=wandb_run
        )

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"TrainLoss={train_loss:.4f} Acc={train_acc:.4f} AUC={train_auc:.4f} | "
              f"ValLoss={val_loss:.4f} Acc={val_acc:.4f} AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pt")
            print("[안내] 새로운 최고 AUC 갱신! 모델 저장 완료.")

    finish_wandb()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/default.yaml")
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main_train(config)

if __name__ == "__main__":
    main()
