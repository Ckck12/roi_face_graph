# src/train.py

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.data import create_dataloader
from src.engine import train_one_epoch, evaluate
from src.models.final_model import FullPipelineModel
from src.utils import set_seed, get_device, init_wandb, finish_wandb

def main_train(config):
    # DDP 설정 여부 확인
    if config.get("ddp", False):
        # torchrun이 설정한 LOCAL_RANK 환경 변수에서 local_rank 가져오기
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        # 분산 프로세스 그룹 초기화
        torch.distributed.init_process_group(
            backend=config["dist_backend"],
            init_method=config["dist_url"],
            world_size=config["world_size"],
            rank=config["rank"]
        )
    else:
        device = get_device(0)

    # 시드 설정
    seed = config.get("seed", 42)
    set_seed(seed)

    # WandB 초기화
    wandb_run = init_wandb(config)

    # DataLoader 설정
    train_loader = create_dataloader(
        csv_path=config["csv_file"],
        shape_predictor_path=config["shape_predictor_path"],
        dataset_type="train",
        batch_size=config["batch_size"],
        shuffle=not config.get("ddp", False),  # DDP일 때는 DistributedSampler에서 shuffle 처리
        num_workers=config["num_workers"],
        image_size=config["model"]["image_size"],
        distributed=config.get("ddp", False),
        rank=config.get("rank", 0),
        world_size=config.get("world_size", 1)
    )
    val_loader = create_dataloader(
        csv_path=config["csv_file"],
        shape_predictor_path=config["shape_predictor_path"],
        dataset_type="val",
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        image_size=config["model"]["image_size"],
        distributed=False
    )

    # 모델 설정
    m_cfg = config["model"]
    model = FullPipelineModel(
        image_size=m_cfg["image_size"],
        patch_size=m_cfg["patch_size"],
        hidden_dim=m_cfg["hidden_dim"],
        num_classes=m_cfg["num_classes"]
    ).to(device)

    # DDP로 모델 래핑
    if config.get("ddp", False):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    best_auc = 0.0
    epochs = config["epochs"]

    for epoch in range(epochs):
        if config.get("ddp", False):
            train_loader.sampler.set_epoch(epoch)  # DistributedSampler에서 epoch 설정

        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, wandb=wandb_run
        )
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device, epoch, "val", wandb=wandb_run
        )

        # 모든 프로세스가 동기화
        if config.get("ddp", False):
            torch.distributed.barrier()

        # 로그 및 모델 저장 (rank 0 프로세스만 수행)
        if config.get("rank", 0) == 0:
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"TrainLoss={train_loss:.4f} Acc={train_acc:.4f} AUC={train_auc:.4f} | "
                  f"ValLoss={val_loss:.4f} Acc={val_acc:.4f} AUC={val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), "best_model.pt")
                print("[안내] 새로운 최고 AUC 갱신! 모델 저장 완료.")

    # 프로세스 그룹 정리
    if config.get("ddp", False):
        torch.distributed.destroy_process_group()

    # WandB 종료
    finish_wandb()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # DDP 설정
    config["ddp"] = args.ddp
    if args.ddp:
        # torchrun이 설정한 RANK 환경 변수에서 rank 가져오기
        config["rank"] = int(os.environ.get("RANK", 0))
    else:
        config["rank"] = 0

    main_train(config)

if __name__ == "__main__":
    main()
