# src/models/roi_aggregation.py

import torch

def compute_overlap_ratio(bboxes, patch_boxes):
    """
    여러 개의 바운딩 박스와 패치 박스 간의 겹침 비율을 계산합니다.
    
    Args:
        bboxes (Tensor): (N, 4) - N개의 ROI 바운딩 박스
        patch_boxes (Tensor): (num_patches, 4) - 패치의 바운딩 박스
    
    Returns:
        Tensor: (N, num_patches) - 각 ROI와 패치 간의 겹침 비율
    """
    N = bboxes.size(0)
    num_patches = patch_boxes.size(0)
    
    # Expand bboxes and patch_boxes for broadcasting
    bboxes_exp = bboxes.unsqueeze(1).expand(-1, num_patches, -1)  # (N, num_patches, 4)
    patch_boxes_exp = patch_boxes.unsqueeze(0).expand(N, -1, -1)  # (N, num_patches, 4)
    
    # 교집합 좌표 계산
    x1_i = torch.max(bboxes_exp[:, :, 0], patch_boxes_exp[:, :, 0])
    y1_i = torch.max(bboxes_exp[:, :, 1], patch_boxes_exp[:, :, 1])
    x2_i = torch.min(bboxes_exp[:, :, 2], patch_boxes_exp[:, :, 2])
    y2_i = torch.min(bboxes_exp[:, :, 3], patch_boxes_exp[:, :, 3])
    
    # 교집합 면적 계산
    inter_width = (x2_i - x1_i).clamp(min=0)
    inter_height = (y2_i - y1_i).clamp(min=0)
    inter_area = inter_width * inter_height  # (N, num_patches)
    
    # 패치 면적 계산
    patch_area = (patch_boxes[:, 2] - patch_boxes[:, 0]) * (patch_boxes[:, 3] - patch_boxes[:, 1])  # (num_patches,)
    patch_area = patch_area.unsqueeze(0).expand(N, -1)  # (N, num_patches)
    
    # 겹침 비율 계산
    overlap_ratios = inter_area / patch_area  # (N, num_patches)
    overlap_ratios = overlap_ratios.clamp(max=1.0)  # 1을 초과하지 않도록 클램핑
    
    return overlap_ratios
