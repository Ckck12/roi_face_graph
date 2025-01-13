# src/models/roi_aggregation.py

import torch

def compute_overlap_ratio(bboxes, patch_boxes):
    """
    여러 개의 바운딩 박스와 패치 박스 간의 겹침 비율을 계산하되,
    각 바운딩 박스별로 합이 1이 되도록 정규화합니다.
    
    Args:
        bboxes (Tensor): (N, 4) - N개의 ROI 바운딩 박스
        patch_boxes (Tensor): (num_patches, 4) - 패치의 바운딩 박스
    
    Returns:
        Tensor: (N, num_patches) - 각 ROI와 패치 간의 [0~1] 정규화된 겹침 비율
                                  => 한 ROI row별 합이 1
    """
    N = bboxes.size(0)
    num_patches = patch_boxes.size(0)
    
    # (1) 브로드캐스팅을 통해 교집합 영역(inter_area)을 구함
    bboxes_exp = bboxes.unsqueeze(1).expand(-1, num_patches, -1)   # (N, num_patches, 4)
    patch_boxes_exp = patch_boxes.unsqueeze(0).expand(N, -1, -1)   # (N, num_patches, 4)
    
    x1_i = torch.max(bboxes_exp[..., 0], patch_boxes_exp[..., 0])
    y1_i = torch.max(bboxes_exp[..., 1], patch_boxes_exp[..., 1])
    x2_i = torch.min(bboxes_exp[..., 2], patch_boxes_exp[..., 2])
    y2_i = torch.min(bboxes_exp[..., 3], patch_boxes_exp[..., 3])
    
    inter_width = (x2_i - x1_i).clamp(min=0)
    inter_height = (y2_i - y1_i).clamp(min=0)
    inter_area = inter_width * inter_height               # (N, num_patches)
    
    # (2) 패치 영역 계산
    patch_area = ((patch_boxes[:, 2] - patch_boxes[:, 0]) *
                  (patch_boxes[:, 3] - patch_boxes[:, 1]))  # (num_patches,)
    patch_area = patch_area.unsqueeze(0).expand(N, -1)      # (N, num_patches)
    
    # (3) 패치 단위 겹침 비율(= inter_area / patch_area)
    overlap_ratios = inter_area / patch_area
    # 굳이 1.0 초과를 막고 싶다면 아래 .clamp(max=1.0) 추가 가능
    # overlap_ratios = overlap_ratios.clamp(max=1.0)
    
    # (4) 각 바운딩 박스별 합을 1이 되도록 정규화
    #     sum_i = sum_j(overlap[i,j])가 0이 아닐 때만 나눠줌
    for i in range(N):
        row_sum = overlap_ratios[i].sum()
        if row_sum > 0:
            overlap_ratios[i] /= row_sum
    print(f"overlap_ratios: {overlap_ratios}")
    return overlap_ratios
