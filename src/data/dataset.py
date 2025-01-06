# src/data/dataset.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

def calc_face_part_bbox(landmarks, indices, image_size=224):
    """
    indices에 해당하는 랜드마크 점들에 대해 x_min, y_min, x_max, y_max를 구한다.
    """
    part_points = landmarks[list(indices)]  # shape: (len(indices), 2)
    x_min = max(np.min(part_points[:, 0]), 0)
    y_min = max(np.min(part_points[:, 1]), 0)
    x_max = min(np.max(part_points[:, 0]), image_size)
    y_max = min(np.max(part_points[:, 1]), image_size)
    return [x_min, y_min, x_max, y_max]

class FFPlusDataset(Dataset):
    """
    ff++_c23_landmark.csv를 읽고, 각 row에 대해
    1) video_npy_path에서 16프레임 로드
    2) landmark_npy_path에서 16프레임의 랜드마크 로드
    3) 랜드마크를 이용해 ROI bbox 생성
    => frames(16,3,224,224), bboxes(16,9,4), label
    """

    def __init__(self, csv_path, dataset_type="train", transform=None, image_size=224):
        super().__init__()
        self.csv_path = csv_path
        self.dataset_type = dataset_type
        self.transform = transform
        self.image_size = image_size

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"[에러] CSV 파일이 존재하지 않습니다: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if "train_type" in self.df.columns:
            self.df = self.df[self.df["train_type"] == self.dataset_type].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"[에러] dataset_type='{self.dataset_type}'에 해당하는 데이터가 없습니다.")

        # ROI 정의
        self.roi_indices = {
            "jawline": range(0, 17),            # [0, 16]
            "left_eyebrow": range(17, 22),      # [17, 21]
            "right_eyebrow": range(22, 27),     # [22, 26]
            "nose": range(27, 36),              # [27, 35]
            "left_eye": range(36, 42),          # [36, 41]
            "right_eye": range(42, 48),         # [42, 47]
            "outer_lip": range(48, 60),         # [48, 59]
            "inner_lip": range(60, 68),         # [60, 67]
            "whole_face": None                   # 전체 이미지
        }
        self.num_rois = len(self.roi_indices)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["label"])
        video_npy_path = row["video_npy_path"]
        landmark_npy_path = row["landmark_npy_path"]

        # 1) 프레임 로드
        if not os.path.exists(video_npy_path):
            raise FileNotFoundError(f"[에러] 비디오 npy 파일이 존재하지 않습니다: {video_npy_path}")
        frames = np.load(video_npy_path)  # (16, 3, 224, 224)

        if frames.shape != (16, 3, 224, 224):
            raise ValueError(f"[에러] 비디오 npy 파일의 형태가 올바르지 않습니다: {video_npy_path}")

        # 2) 랜드마크 로드
        if not os.path.exists(landmark_npy_path):
            raise FileNotFoundError(f"[에러] 랜드마크 npy 파일이 존재하지 않습니다: {landmark_npy_path}")
        landmarks = np.load(landmark_npy_path)  # (16, 1, 68, 2)

        if landmarks.shape != (16, 1, 68, 2):
            raise ValueError(f"[에러] 랜드마크 npy 파일의 형태가 올바르지 않습니다: {landmark_npy_path}")

        # 3) 프레임 변환
        frames_tensor = []
        bboxes_16 = []
        for t in range(16):
            frame = frames[t]  # (3, 224, 224)

            if self.transform:
                # PIL 이미지로 변환
                frame_pil = Image.fromarray(np.transpose(frame, (1, 2, 0)).astype(np.uint8))
                frame_t = self.transform(frame_pil)  # (C, H, W)
            else:
                # Tensor 변환
                frame_t = torch.from_numpy(frame).float()

            frames_tensor.append(frame_t)

            # 4) ROI bbox 계산
            landmarks_frame = landmarks[t, 0, :, :]  # (68, 2)

            part_bboxes = []
            for roi, indices in self.roi_indices.items():
                if roi == "whole_face":
                    bbox = [0, 0, self.image_size, self.image_size]
                else:
                    bbox = calc_face_part_bbox(landmarks_frame, indices, image_size=self.image_size)
                part_bboxes.append(bbox)

            bboxes_16.append(part_bboxes)

        frames_tensor = torch.stack(frames_tensor, dim=0)  # (16, 3, 224, 224)
        bboxes_16 = torch.tensor(bboxes_16, dtype=torch.float32)  # (16, 9, 4)

        return {
            "frames": frames_tensor,     # (16, 3, 224, 224)
            "bboxes": bboxes_16,        # (16, 9, 4)
            "label": torch.tensor(label, dtype=torch.long),
            "video_path": video_npy_path
        }
