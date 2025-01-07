# src/models/roi_vit_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.vit_backbone import CLIPViTBackbone
from src.models.roi_aggregation import compute_overlap_ratio

class ROIViTExtractor(nn.Module):
    """
    한 프레임 (C, H, W)을 ViT 백본에 통과하여 CLS/patch 토큰을 구하고,
    bboxes(ROI) 정보를 이용해 각 ROI의 임베딩을 계산합니다.
    
    데이터 shape:
        - frame: (C, H, W)
        - bboxes: (N, 4)
    
    결과 shape:
        - node_embs: (N+1, hidden_dim)
    """
    def __init__(self, model_name="ViT-B/32", device='cuda', image_size=224, patch_size=32, hidden_dim=768):
        super(ROIViTExtractor, self).__init__()
        self.vit_backbone = CLIPViTBackbone(model_name=model_name, device=device)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Transformer 블록을 CustomResidualAttentionBlock으로 교체
        transformer = self.vit_backbone.model.visual.transformer
        custom_resblocks = nn.ModuleList([
            CustomResidualAttentionBlock(block) for block in transformer.resblocks
        ])
        transformer.resblocks = custom_resblocks

    def _get_patch_coords(self):
        """
        이미지 패치의 바운딩 박스를 계산합니다.
        이미지 크기를 patch_size로 나누어 각 패치의 좌표 범위 리스트를 생성합니다.
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
        return torch.tensor(coords, dtype=torch.float32, device=self.vit_backbone.device)  # (num_patches, 4)

    def _build_attention_mask(self, Mi):
        """
        Mi를 기반으로 Attention Mask를 생성합니다.
        
        Args:
            Mi (Tensor): (N, num_patches) tensor
        
        Returns:
            attn_mask (Tensor): (N+1, 1 + num_patches) tensor
        """
        N, num_patches = Mi.shape
        # +1은 CLS 토큰을 위한 자리
        # CLS 토큰은 모든 패치와 상호작용할 수 있도록 1로 설정
        # ROI 토큰은 Mi 값을 사용하여 패치와의 상호작용을 설정
        attn_mask = torch.ones((N + 1, 1 + num_patches), device=Mi.device)  # (N+1, 1 + num_patches)
        attn_mask[1:, 1:] = Mi  # ROI와 패치 간의 가중치 적용
        return attn_mask

    def forward(self, frame, bboxes):
        """
        Args:
            frame (Tensor): (C, H, W)
            bboxes (Tensor): (N, 4)
        
        Returns:
            node_embs (Tensor): (N+1, hidden_dim)
        """
        # ViT 백본을 통해 CLS 토큰과 패치 토큰을 추출
        visual = self.vit_backbone.model.visual

        # frame: (C, H, W) -> (1, C, H, W)
        x = frame.unsqueeze(0).to(next(self.vit_backbone.parameters()).device)  # (1, C, H, W)
        x = visual.conv1(x)     # (1, embed_dim, H/32, W/32)
        x = F.relu(x)           # ReLU 활성화 함수 적용
        x = x.flatten(2).transpose(1, 2)  # (1, num_patches, embed_dim)

        cls_token = visual.class_embedding.unsqueeze(0).expand(x.size(0), -1, -1)  # (1, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (1, 1 + num_patches, embed_dim)

        # 패치 좌표 계산
        patch_coords = self._get_patch_coords()  # (num_patches, 4)

        N = bboxes.shape[0]
        num_patches = patch_coords.size(0)

        # Mi 매트릭스 벡터화된 계산
        Mi = compute_overlap_ratio(bboxes, patch_coords)  # (N, num_patches)

        # Attention Mask 생성: (N+1, 1 + num_patches)
        attn_mask = self._build_attention_mask(Mi)  # (N+1, 1 + num_patches)

        # Transformer 블록을 통과하여 Attention Mask 적용
        # x는 (1, 1 + num_patches, embed_dim)
        for blk in visual.transformer.resblocks:
            x = blk(x, attn_mask=attn_mask)  # (1, 1 + num_patches, embed_dim)

        # 최종 출력
        # x: (1, 1 + num_patches, embed_dim)
        cls_out = x[:, 0, :]  # (1, embed_dim)
        roi_cls = x[:, 1:, :]  # (1, num_patches, embed_dim)

        # Mi 매트릭스를 사용하여 ROI 임베딩 계산
        # Mi: (N, num_patches)
        # roi_cls: (1, num_patches, embed_dim) -> (num_patches, embed_dim)
        roi_cls = roi_cls.squeeze(0)  # (num_patches, embed_dim)
        roi_embeddings = Mi @ roi_cls  # (N, embed_dim)

        # node_embs: [CLS] + [ROI 임베딩]
        node_embs = torch.cat((cls_out, roi_embeddings), dim=0)  # (N+1, embed_dim)

        return node_embs
