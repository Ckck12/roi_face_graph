# src/models/roi_vit_extractor.py

import torch
import torch.nn as nn
from src.models.vit_backbone import CLIPViTBackbone
from src.models.roi_aggregation import compute_overlap_ratio

class ROIViTExtractor(nn.Module):
    """
    한 프레임 (C, H, W)을 ViT 백본에 통과 -> patch_tokens
    bboxes => M_i = Overlap/Area -> weighted sum
    """
    def __init__(self, model_name="ViT-B/32", device='cuda', image_size=224, patch_size=32, hidden_dim=512):
        super(ROIViTExtractor, self).__init__()
        self.vit_backbone = CLIPViTBackbone(model_name=model_name, device=device)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

    def _get_patch_coords(self):
        """
        이미지 패치의 바운딩 박스를 계산합니다.
        """
        coords = []
        num_side = self.image_size // self.patch_size
        for r in range(num_side):
            for c in range(num_side):
                x1 = c * self.patch_size
                y1 = r * self.patch_size
                x2 = x1 + self.patch_size
                y2 = y1 + self.patch_size
                coords.append([x1, y1, x2, y2])
        return coords

    def forward(self, frame, bboxes):
        """
        Args:
            frame: (C, H, W)
            bboxes: (N, 4) => [x1, y1, x2, y2] for each ROI
        Returns:
            node_embs: (N+1, hidden_dim)
        """
        # ViT 백본을 통해 CLS 토큰과 패치 토큰을 추출
        cls_token, patch_tokens = self.vit_backbone(frame.unsqueeze(0))  # (1, d_v), (1, num_patches, d_v)
        cls_token = cls_token.squeeze(0)  # (d_v)
        patch_tokens = patch_tokens.squeeze(0)  # (num_patches, d_v)

        # 패치 좌표 계산
        patch_coords = self._get_patch_coords()  # list of [x1, y1, x2, y2], length=num_patches

        # 노드 임베딩 초기화 (CLS 토큰 추가)
        node_embs = [cls_token]  # list of Tensors, each (d_v)

        # 각 ROI에 대해 가중 평균 수행
        for bbox in bboxes:
            # 각 패치에 대한 겹침 비율 계산
            overlap_ratios = []
            for patch_box in patch_coords:
                ratio = compute_overlap_ratio(bbox, patch_box)
                overlap_ratios.append(ratio)
            overlap_ratios = torch.tensor(overlap_ratios, dtype=torch.float32, device=patch_tokens.device)  # (num_patches,)

            # 가중 평균을 통해 ROI 특징 계산
            weighted_patch_tokens = patch_tokens * overlap_ratios.unsqueeze(-1)  # (num_patches, d_v)
            roi_feat = weighted_patch_tokens.mean(dim=0)  # (d_v)

            node_embs.append(roi_feat)

        node_embs = torch.stack(node_embs, dim=0)  # (N+1, d_v)

        return node_embs
