# src/models/roi_vit_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.vit_backbone import CLIPViTBackbone
from src.models.roi_aggregation import compute_overlap_ratio


class ModifiedMultiheadAttention(nn.Module):
    """
    CLIP의 MultiheadAttention을 수정하여 attn_mask를 지원하도록 함.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(ModifiedMultiheadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
    
    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: (L, N, E)
            key: (S, N, E)
            value: (S, N, E)
            attn_mask: (L, S)
        Returns:
            attn_output: (L, N, E)
            attn_weights: (N*num_heads, L, S)
        """
        attn_output, attn_weights = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        return attn_output, attn_weights


class CustomResidualAttentionBlock(nn.Module):
    """
    CLIP의 ResidualAttentionBlock을 수정하여 attn_mask를 지원하도록 함.
    """
    def __init__(self, block):
        super(CustomResidualAttentionBlock, self).__init__()
        self.ln_1 = block.ln_1
        self.ln_2 = block.ln_2
        self.attn = ModifiedMultiheadAttention(block.attn.embed_dim, block.attn.num_heads, dropout=block.attn.dropout)
        self.mlp = block.mlp

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: (N, 1 + num_patches, E)
            attn_mask: (N+1, 1 + num_patches)
        Returns:
            x: (N, 1 + num_patches, E)
        """
        # LayerNorm
        x_norm = self.ln_1(x)
        # Self-Attention with attn_mask
        # PyTorch의 MultiheadAttention은 (L, N, E) 형태를 기대하므로, Transpose 필요
        attn_output, _ = self.attn(
            x_norm.transpose(0, 1),  # (1 + num_patches, N, E)
            x_norm.transpose(0, 1),  # (1 + num_patches, N, E)
            x_norm.transpose(0, 1),  # (1 + num_patches, N, E)
            attn_mask=attn_mask
        )
        attn_output = attn_output.transpose(0, 1)  # (N, 1 + num_patches, E)
        # Residual Connection
        x = x + attn_output
        # LayerNorm
        x_norm = self.ln_2(x)
        # MLP
        mlp_output = self.mlp(x_norm)
        # Residual Connection
        x = x + mlp_output
        return x


class ROIViTExtractor(nn.Module):
    """
    한 프레임 (C, H, W)을 ViT 백본에 통과하여 CLS/patch 토큰을 구하고,
    bboxes(ROI) 정보를 이용해 가중 평균된 ROI 특징을 계산합니다.
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
        이미지 크기를 patch_size로 나누어 각 패치의 좌표 범위 리스트를 생성
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

    def _build_attention_mask(self, Mi):
        """
        Mi를 기반으로 Attention Mask를 생성합니다.

        Args:
            Mi: (N, num_patches) tensor

        Returns:
            attn_mask: (N+1, 1 + num_patches) tensor
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
            frame: (C, H, W)
            bboxes: (N, 4)
        반환:
            node_embs: (N+1, hidden_dim)
        목적:
            ViT backbone에서 patch_tokens를 받아,
            각 ROI의 overlap ratio를 가중치로 하여 ROI 특징을 추출합니다.
        """
        # 1) ViT 백본을 통해 CLS 토큰과 패치 토큰을 추출
        visual = self.vit_backbone.model.visual

        # frame: (C, H, W) -> (1, C, H, W)
        x = frame.unsqueeze(0).to(next(self.vit_backbone.parameters()).device)  # (1, C, H, W)
        x = visual.conv1(x)     # (1, embed_dim, H/32, W/32)
        x = F.relu(x)           # ReLU 활성화 함수 적용
        x = x.flatten(2).transpose(1, 2)  # (1, num_patches, embed_dim)

        cls_token = visual.class_embedding.unsqueeze(0).expand(x.size(0), -1, -1)  # (1, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (1, 1 + num_patches, embed_dim)

        # 패치 좌표 계산
        patch_coords = self._get_patch_coords()  # list of [x1, y1, x2, y2], length=num_patches
        num_patches = len(patch_coords)

        N = bboxes.shape[0]
        Mi = torch.zeros((N, num_patches), device=x.device)

        # Mi 매트릭스 계산: Overlap(bi, bp) / Area(bp)
        for i, bbox in enumerate(bboxes):
            for j, patch_box in enumerate(patch_coords):
                overlap = compute_overlap_ratio(bbox, patch_box)
                Mi[i, j] = overlap  # Mi[i][j] = Overlap(bi, bp) / Area(bp)

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

        # node_embs: [global CLS] + [ROI CLS]
        # 각 ROI 토큰은 Mi 매트릭스를 통해 가중이 부여된 상태
        # cls_out은 (1, embed_dim), roi_cls.mean(dim=1, keepdim=True)은 (1, embed_dim)
        node_embs = torch.cat((cls_out, roi_cls.mean(dim=1, keepdim=True)), dim=0)  # (N+1, embed_dim)

        return node_embs
