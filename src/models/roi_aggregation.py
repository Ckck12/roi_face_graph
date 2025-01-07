# src/models/roi_aggregation.py

import torch

def compute_overlap_ratio(bbox, patch_box):
    """
    목적:
    - bbox(ROI)와 patch_box(이미지의 패치)가 얼마나 겹치는지를 계산
    데이터 shape:
    - bbox, patch_box: (x1, y1, x2, y2)
    반환:
    - overlap / area(patch_box), float
    """
    # 입력으로 들어오는 두 박스의 x1, y1, x2, y2 중 겹치는 부분의 좌표 계산
    x1 = max(bbox[0], patch_box[0])
    y1 = max(bbox[1], patch_box[1])
    x2 = min(bbox[2], patch_box[2])
    y2 = min(bbox[3], patch_box[3])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    overlap = w * h  # 실제 겹치는 면적

    patch_area = (patch_box[2] - patch_box[0]) * (patch_box[3] - patch_box[1])
    if patch_area <= 0:
        return 0.0
    # 전체 패치 면적 대비 겹치는 비율을 반환
    return overlap / patch_area
