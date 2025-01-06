# src/data/dataloader.py

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from src.data.dataset import FFPlusDataset  # 절대 import로 변경

def create_dataloader(
    csv_path: str,
    shape_predictor_path: str,
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
    """
    tfm = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = FFPlusDataset(
        csv_path=csv_path,
        shape_predictor_path=shape_predictor_path,
        dataset_type=dataset_type,
        transform=tfm,
        image_size=image_size
    )

    if distributed:
        sampler = DistributedSampler(
            dataset,
            rank=rank,
            num_replicas=world_size,
            shuffle=shuffle
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        sampler=sampler
    )
    return loader
