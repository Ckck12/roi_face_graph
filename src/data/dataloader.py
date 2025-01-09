# src/data/dataloader.py

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from src.data.dataset import FFPlusDataset  # 절대 import로 변경


def create_dataloader(
    csv_path: str,
    dataset_type: str = "train",
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 2,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    image_size=224
):
    """
    FFPlusDataset을 이용해 DataLoader를 생성.
    또한 데이터셋 초기 로드 시 tqdm을 이용해 진행상황을 출력.
    """
    print(f"[로딩 알림] create_dataloader() 시작... CSV={csv_path}, type={dataset_type}")

    # 이미지 변환(전처리) 정의
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # FFPlusDataset 생성
    dataset = FFPlusDataset(
        csv_path=csv_path,
        dataset_type=dataset_type,
        transform=tfm,
        image_size=image_size
    )

    print(f"[로딩 알림] FFPlusDataset 생성 완료. dataset 길이: {len(dataset)}")

    # DDP 설정 시 분산 샘플러
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
    else:
        sampler = None

    # DataLoader 생성
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        sampler=sampler
    )

    print("[로딩 알림] DataLoader 생성 완료!")
    return loader
