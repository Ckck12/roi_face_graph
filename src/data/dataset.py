# src/data/dataset.py
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from .frame_extractor import extract_32_frames
from .landmark_detector import Landmark68Detector

def calc_face_part_bbox(landmarks, indices):
    """
    indices에 해당하는 랜드마크 점들에 대해 x_min,y_min,x_max,y_max를 구한다.
    """
    part_points = landmarks[indices]  # shape: (len(indices), 2)
    x_min = np.min(part_points[:,0])
    y_min = np.min(part_points[:,1])
    x_max = np.max(part_points[:,0])
    y_max = np.max(part_points[:,1])
    return [x_min,y_min,x_max,y_max]

class FFPlusDataset(Dataset):
    """
    ff++_c23.csv를 읽고, 각 row에 대해
    1) video_path => 32프레임 추출
    2) dlib 68랜드마크 => 왼눈, 오른눈, 코, 입, 머리카락 등 bbox
    3) 전체 이미지 bbox(= [0,0, W,H])도 추가
    => frames(32,C,H,W), bboxes(32,N,4), label
    """

    def __init__(self, csv_path, shape_predictor_path, dataset_type="train", transform=None, image_size=224):
        super().__init__()
        self.csv_path = csv_path
        self.shape_predictor_path = shape_predictor_path
        self.dataset_type = dataset_type
        self.transform = transform
        self.image_size = image_size

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"[에러] CSV 파일이 존재하지 않습니다: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if "type" in self.df.columns:
            self.df = self.df[self.df["type"] == self.dataset_type].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"[에러] dataset_type='{self.dataset_type}'에 해당하는 데이터가 없습니다.")

        if not os.path.exists(self.shape_predictor_path):
            raise FileNotFoundError(f"[에러] dlib 랜드마크 모델이 존재하지 않습니다: {self.shape_predictor_path}")

        self.landmark_detector = Landmark68Detector(self.shape_predictor_path)

        # dlib 68점에서 각 부위 인덱스
        self.left_eye_idx = range(36,42)
        self.right_eye_idx = range(42,48)
        self.nose_idx = range(27,36)
        self.mouth_idx = range(48,68)
        # 머리카락은 없음 => 예시로, jaw(0~16)를 "머리"라고 가정?
        self.hair_idx = range(0,17)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        label = int(row["label"])

        # 1) 32프레임 추출
        frames = extract_32_frames(video_path)  # list of PIL, length=32

        # 2) 각 프레임 -> transform -> (C,H,W)
        frames_tensor = []
        bboxes_32 = []
        for img_pil in frames:
            w,h = img_pil.size
            if self.transform:
                img_t = self.transform(img_pil)  # (C,H,W)
            else:
                # transform이 없으면 Tensor 변환
                img_t = torch.from_numpy(np.array(img_pil)).permute(2,0,1).float()

            frames_tensor.append(img_t)

            # 랜드마크 검출
            img_array = np.array(img_pil)
            landmarks_68 = self.landmark_detector.detect_landmarks(img_array)
            if landmarks_68 is None:
                # 얼굴 없음 => bbox 전부 0
                n_parts = 6  # 왼눈/오른눈/코/입/머리(가정)/전체
                part_bboxes = [[0,0,0,0] for _ in range(n_parts)]
            else:
                # 부위별 bbox 계산
                left_eye_box = calc_face_part_bbox(landmarks_68, self.left_eye_idx)
                right_eye_box = calc_face_part_bbox(landmarks_68, self.right_eye_idx)
                nose_box = calc_face_part_bbox(landmarks_68, self.nose_idx)
                mouth_box = calc_face_part_bbox(landmarks_68, self.mouth_idx)
                hair_box = calc_face_part_bbox(landmarks_68, self.hair_idx)
                # 전체 이미지 bbox
                full_box = [0, 0, w, h]

                part_bboxes = [
                    left_eye_box,
                    right_eye_box,
                    nose_box,
                    mouth_box,
                    hair_box,
                    full_box
                ]

            bboxes_32.append(part_bboxes)

        frames_tensor = torch.stack(frames_tensor, dim=0)    # (32,C,H,W)
        bboxes_32 = torch.tensor(bboxes_32, dtype=torch.float32)  # (32,6,4)

        return {
            "frames": frames_tensor,     # (32,C,H,W)
            "bboxes": bboxes_32,        # (32,6,4)
            "label": torch.tensor(label, dtype=torch.long),
            "video_path": video_path
        }
