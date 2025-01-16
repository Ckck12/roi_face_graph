# src/train.py

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataloader import create_dataloader
from src.engine import train_one_epoch, evaluate
from src.utils import set_seed, get_device, init_wandb, finish_wandb

# ★ 실제 위치와 클래스명에 맞게 import
from src.models.full_gat_gru_pipeline import FullGATGRUPipelineModel


def main_train(config):
    # (1) DDP 여부 판단
    if config.get("ddp", False):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        torch.distributed.init_process_group(
            backend=config["dist_backend"],
            init_method=config["dist_url"],
            world_size=config["world_size"],
            rank=config["rank"]
        )
    else:
        device = get_device(0)

    # (2) 시드 고정
    seed = config.get("seed", 42)
    set_seed(seed)

    # (3) WandB init
    wandb_run = init_wandb(config)

    # (4) DataLoader 생성
    train_loader = create_dataloader(
        csv_path=config["csv_file"],
        dataset_type="train",
        batch_size=config["batch_size"],
        shuffle=not config.get("ddp", False),
        num_workers=config["num_workers"],
        image_size=config["model"]["image_size"],
        distributed=config.get("ddp", False),
        rank=config.get("rank", 0),
        world_size=config.get("world_size", 1)
    )
    val_loader = create_dataloader(
        csv_path=config["csv_file"],
        dataset_type="val",
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        image_size=config["model"]["image_size"],
        distributed=False
    )

    # (5) 모델 설정
    m_cfg = config["model"]

    # 예: model_name, device, image_size, patch_size, hidden_dim 등등
    # GAT/GRU 세부 hyperparam도 넘겨줄 수 있음
    model = FullGATGRUPipelineModel(
        model_name="ViT-B/32",
        device=device,
        image_size=m_cfg["image_size"],
        patch_size=m_cfg["patch_size"],
        hidden_dim=m_cfg["hidden_dim"],
        gat_hidden=128,
        gat_heads=2,
        gat_dropout=0.3,
        gru_hidden_dim=512,
        gru_dropout=0.3,
        num_classes=m_cfg["num_classes"]
    ).to(device)

    # (6) DDP 래핑
    if config.get("ddp", False):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # (7) criterion, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    best_auc = 0.0
    epochs = config["epochs"]

    # (8) 학습 루프
    for epoch in range(epochs):
        if config.get("ddp", False):
            train_loader.sampler.set_epoch(epoch)

        # (loss, acc, auc, recall)
        train_loss, train_acc, train_auc, train_recall = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, wandb=wandb_run
        )
        val_loss, val_acc, val_auc, val_recall = evaluate(
            model, val_loader, criterion, device, epoch, "val", wandb=wandb_run
        )

        if config.get("ddp", False):
            torch.distributed.barrier()

        if config.get("rank", 0) == 0:
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"TrainLoss={train_loss:.4f} Acc={train_acc:.4f} AUC={train_auc:.4f} Recall={train_recall:.4f} | "
                  f"ValLoss={val_loss:.4f} Acc={val_acc:.4f} AUC={val_auc:.4f} Recall={val_recall:.4f}")

            # 기존 AUC로 best check 시
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.module.state_dict(), "best_model.pt")
                print("[안내] 새로운 최고 AUC 갱신! 모델 저장 완료.")

    # (9) 정리
    if config.get("ddp", False):
        torch.distributed.destroy_process_group()

    finish_wandb()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config["ddp"] = args.ddp
    if args.ddp:
        config["rank"] = int(os.environ.get("RANK", 0))
    else:
        config["rank"] = 0

    main_train(config)

if __name__ == "__main__":
    main()
