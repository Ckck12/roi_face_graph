# roi_vit_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

# 가정: vit_backbone.py와 roi_aggregation.py는 같은 프로젝트 내에 존재하며, 필요한 클래스와 함수를 포함하고 있습니다.
# vit_backbone.py에는 CLIPViTBackbone 클래스가 정의되어 있어야 합니다.
# roi_aggregation.py에는 compute_overlap_ratio 함수가 정의되어 있어야 합니다.
from vit_backbone import CLIPViTBackbone
from roi_aggregation import compute_overlap_ratio


class CustomMultiheadAttention(nn.Module):
    """
    기존 MultiheadAttention에 Attention Bias를 추가로 지원하는 클래스입니다.
    Attention Bias는 Self-Attention 스코어에 더해져 특정 패치에 더 높은 주목을 유도합니다.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Q, K, V를 위한 선형 변환 계층
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query (Tensor): (B, N, E)
            key (Tensor): (B, N, E)
            value (Tensor): (B, N, E)
            attn_bias (Tensor, optional): (B, N, N) 또는 (B, N+1, N+1) 형태의 Attention Bias

        Returns:
            attn_output (Tensor): (B, N, E)
            attn_weights (Tensor): (B, num_heads, N, N)
        """
        B, N, E = query.size()

        # Q, K, V 계산
        Q = self.q_proj(query)  # (B, N, E)
        K = self.k_proj(key)    # (B, N, E)
        V = self.v_proj(value)  # (B, N, E)

        # Multihead Attention을 위해 Head 단위로 분리
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)

        # Scaled Dot-Product Attention 계산
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, N, N)

        if attn_bias is not None:
            # Attention Bias를 Attention 스코어에 더함
            # attn_bias: (B, N, N) 또는 (B, N+1, N+1)
            # num_heads 차원에 브로드캐스팅
            attn_scores = attn_scores + attn_bias.unsqueeze(1)  # (B, num_heads, N, N)

        # Softmax를 통해 Attention 가중치 계산
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, N, N)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Attention 가중치를 V에 적용
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, E)  # (B, N, E)

        # 최종 출력 선형 변환
        attn_output = self.out_proj(attn_output)  # (B, N, E)

        return attn_output, attn_weights


class CustomResidualAttentionBlock(nn.Module):
    """
    기존 Residual Block에 CustomMultiheadAttention을 통합한 클래스입니다.
    Attention Bias를 통해 특정 ROI에 더 집중할 수 있도록 Self-Attention을 조절합니다.
    """
    def __init__(
        self,
        block: nn.Module,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super(CustomResidualAttentionBlock, self).__init__()
        self.ln_1 = block.ln_1  # LayerNorm 1
        self.ln_2 = block.ln_2  # LayerNorm 2
        self.attn = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)  # Custom Multihead Attention
        self.mlp = block.mlp  # MLP (Feed-Forward Network)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): (B, N, E)
            attn_bias (Tensor, optional): (B, N, N) 또는 (B, N+1, N+1)

        Returns:
            x (Tensor): (B, N, E)
        """
        # 첫 번째 Residual 연결: Self-Attention
        x_norm = self.ln_1(x)  # LayerNorm
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_bias=attn_bias)  # Custom Attention
        x = x + attn_output  # Residual 연결

        # 두 번째 Residual 연결: MLP
        x_norm = self.ln_2(x)  # LayerNorm
        mlp_output = self.mlp(x_norm)  # MLP
        x = x + mlp_output  # Residual 연결

        return x


class ROIViTExtractor(nn.Module):
    """
    ROI Aggregation 모듈을 구현한 클래스입니다.
    Vision Transformer(ViT) 기반의 Self-Attention에서
    특정 ROI에 해당하는 패치들에 Attention Bias를 적용하여
    <CLS> 토큰 임베딩을 추출합니다.
    
    Args:
        model_name (str): CLIP 모델 이름, 예: "ViT-B/32"
        device (str): "cuda" 또는 "cpu"
        image_size (int): 입력 이미지 크기 (예: 224)
        patch_size (int): 패치 크기 (예: 32)
        hidden_dim (int): ViT 임베딩 차원 (예: 768)
    """
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = 'cuda',
        image_size: int = 224,
        patch_size: int = 32,
        hidden_dim: int = 768
    ):
        super(ROIViTExtractor, self).__init__()
        self.vit_backbone = CLIPViTBackbone(
            model_name=model_name,
            device=device,
            num_finetune_blocks=2  # 마지막 2블록만 fine-tuning
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.device = device

        # Transformer ResBlocks을 CustomResidualAttentionBlock으로 교체
        visual = self.vit_backbone.model.visual
        transformer = visual.transformer
        custom_resblocks = []
        for block in transformer.resblocks:
            custom_resblocks.append(
                CustomResidualAttentionBlock(
                    block=block,
                    embed_dim=transformer.width,
                    num_heads=block.attn.num_heads,
                    dropout=block.attn.dropout
                )
            )
        transformer.resblocks = nn.ModuleList(custom_resblocks)

    def _get_patch_coords(self) -> torch.Tensor:
        """
        패치의 좌표를 계산하여 반환합니다.
        
        Returns:
            coords (Tensor): (num_patches, 4) [x1, y1, x2, y2]
        """
        num_side = self.image_size // self.patch_size  # 예: 224//32 = 7
        coords = []
        for r in range(num_side):
            for c in range(num_side):
                x1 = c * self.patch_size
                y1 = r * self.patch_size
                x2 = x1 + self.patch_size
                y2 = y1 + self.patch_size
                coords.append([x1, y1, x2, y2])
        coords = torch.tensor(coords, dtype=torch.float32, device=self.device)  # (num_patches, 4)
        return coords  # (num_patches, 4)

    def _expand_mask(self, mask_1d: torch.Tensor) -> torch.Tensor:
        """
        Overlap 마스크 M_i를 Attention Bias로 확장합니다.
        
        Args:
            mask_1d (Tensor): (B, N, num_patches) 각 패치와의 Overlap 비율
        
        Returns:
            attn_bias (Tensor): (B, N, num_patches + 1, num_patches + 1)
        """
        B, N, num_patches = mask_1d.shape
        L = num_patches + 1  # +1 for CLS
        attn_bias = torch.zeros(B, N, L, L, device=mask_1d.device)  # (B, N, L, L)

        # CLS <-> Patch 간 Attention Bias 설정
        attn_bias[:, :, 0, 1:] = mask_1d       # CLS -> patches
        attn_bias[:, :, 1:, 0] = mask_1d       # patches -> CLS

        return attn_bias  # (B, N, L, L)

    def _build_single_roi_cls(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        한 ROI에 대해 Transformer를 통과시켜 <CLS> 임베딩을 추출합니다.
        
        Args:
            feat (Tensor): (B, N, num_patches, E) 패치 임베딩
            mask (Tensor): (B, N, num_patches) Overlap 비율
        
        Returns:
            cls_out (Tensor): (B, N, E) 업데이트된 <CLS> 임베딩
        """
        B, N, num_patches, E = feat.shape

        # CLS 토큰을 각 ROI마다 추가
        # class_embedding: (1, E)
        cls_token = self.vit_backbone.model.visual.class_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(1).expand(B, N, -1, -1)  # (B, N, 1, E)
        x = torch.cat((cls_token, feat), dim=2)  # (B, N, 1 + num_patches, E)

        # Attention Bias 생성
        attn_2d = self._expand_mask(mask)  # (B, N, L, L)

        # Transformer ResBlocks 통과
        for blk in self.vit_backbone.model.visual.transformer.resblocks:
            x = blk(x, attn_bias=attn_2d)  # (B, N, L, E)

        # 업데이트된 CLS 토큰 추출
        cls_out = x[:, :, 0, :]  # (B, N, E)
        return cls_out  # (B, N, E)

    def forward(self, frames: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        """
        모델의 순전파를 정의합니다.
        
        Args:
            frames (Tensor): (B, 3, H, W) 입력 이미지
            bboxes (Tensor): (B, N, 4) 각 ROI의 바운딩 박스 [x1, y1, x2, y2]
        
        Returns:
            all_roi_cls (Tensor): (B, N, E) 각 ROI에 대한 <CLS> 임베딩
        """
        B, C, H, W = frames.shape
        _, N, _ = bboxes.shape

        # 1. ViT Backbone을 통해 패치 임베딩 획득
        # vit_backbone.forward returns (cls_tokens, patch_tokens)
        _, feat = self.vit_backbone(frames)  # feat: (B, num_patches, E)

        # 2. Overlap 비율 계산
        patch_coords = self._get_patch_coords()  # (num_patches, 4)
        # compute_overlap_ratio는 바운딩 박스와 패치 간 Overlap 비율을 계산
        # 바운딩 박스: (B, N, 4), 패치 좌표: (num_patches, 4)
        M = compute_overlap_ratio(bboxes, patch_coords)  # (B, N, num_patches)

        # 3. ROI별로 <CLS> 토큰을 붙여 Self-Attention 수행하여 업데이트된 <CLS> 임베딩 획득
        # feat: (B, num_patches, E) -> reshape to (B, N, num_patches, E)
        feat = feat.unsqueeze(1).expand(B, N, -1, -1)  # (B, N, num_patches, E)

        # <CLS> 토큰을 추가하고 Attention Bias 적용
        all_roi_cls = self._build_single_roi_cls(feat, M)  # (B, N, E)

        return all_roi_cls  # (B, N, E)