# src/models/roi_vit_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.vit_backbone import CLIPViTBackbone
from src.models.roi_aggregation import compute_overlap_ratio

class CustomMultiheadAttention(nn.Module):
    """
    기존과 동일: Overlap Bias(= attn_bias)를 Self-Attention 스코어에 추가하는 MultiheadAttention 구현.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_bias=None):
        """
        query, key, value: (L, N, E)
        attn_bias: (L, L) 형태(= (H'W'+1, H'W'+1))를 예상.
        """
        Q = self.q_proj(query)  # (L, N, E)
        K = self.k_proj(key)    # (L, N, E)
        V = self.v_proj(value)  # (L, N, E)

        L, N, E = Q.size()

        # Reshape for multi-head
        Q = Q.view(L, N, self.num_heads, self.head_dim).permute(2,1,0,3)  # (h, N, L, d)
        K = K.view(L, N, self.num_heads, self.head_dim).permute(2,1,0,3)  # (h, N, L, d)
        V = V.view(L, N, self.num_heads, self.head_dim).permute(2,1,0,3)  # (h, N, L, d)

        # QK^T => (h, N, L, L)
        # 여기서 K를 transpose(-2,-1) or einsum 적절히 사용
        # 아래 einsum: 'hnld,hnad->hnla', a가 L
        attn_scores = torch.einsum('hnld,hnad->hnla', Q, K) / math.sqrt(self.head_dim)
        # attn_scores: (h, N, L, L)

        if attn_bias is not None:
            # attn_bias: (L,L) -> 확장 필요: (h, N, L, L)
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(1)  # (1,1,L,L)
            attn_bias = attn_bias.expand(self.num_heads, N, L, L)
            attn_scores = attn_scores + attn_bias
            # print(f"attn_bias: {attn_bias}")
            # print(f"attn_scores {attn_scores}")

        attn_weights = F.softmax(attn_scores, dim=-1)  # (h, N, L, L)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_output => (h, N, L, d)
        attn_output = torch.einsum('hnla,hnad->hnld', attn_weights, V)
        # => (h,N,L,d)

        # 다시 (L,N,E)로
        attn_output = attn_output.permute(2,1,0,3).contiguous()  # (L,N,h,d)
        attn_output = attn_output.view(L, N, self.embed_dim)      # (L,N,E)

        attn_output = self.out_proj(attn_output)  # (L, N, E)
        # attn_weights => (h*N, L, L) 등으로 변형 가능

        return attn_output, attn_weights


class CustomResidualAttentionBlock(nn.Module):
    """
    CLIP의 ResidualAttentionBlock을 수정하여, attn_bias=(L,L)을 입력받아 Self-Attention에 추가.
    """
    def __init__(self, block, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln_1 = block.ln_1
        self.ln_2 = block.ln_2
        self.attn = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = block.mlp

    def forward(self, x, attn_bias=None):
        """
        x: (L, N, E)
        attn_bias: (L, L)
        """
        x_norm = self.ln_1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_bias=attn_bias)
        x = x + attn_output

        x_norm = self.ln_2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        return x


class ROIViTExtractor(nn.Module):
    """
    논문 방식으로 'ROI마다 별도의 Self-Attention'을 수행하여,
    ROI별 <CLS> 토큰 임베딩을 직접 얻는 구조.

    - frame: (C,H,W)
    - bboxes: (N,4)
    - Return: (N+1, hidden_dim)  # N개 ROI + whole_face(글로벌)
    """
    def __init__(self, model_name="ViT-B/32", device='cuda', image_size=224, patch_size=32, hidden_dim=768):
        super().__init__()
        # (1) CLIP ViT 백본
        from src.models.vit_backbone import CLIPViTBackbone
        self.vit_backbone = CLIPViTBackbone(model_name=model_name, device=device)

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.device = device

        # (2) CLIP 내부 Transformer blocks -> CustomResidualAttentionBlock 교체
        visual = self.vit_backbone.model.visual
        transformer = visual.transformer
        custom_resblocks = nn.ModuleList([
            CustomResidualAttentionBlock(block, transformer.width, block.attn.num_heads, block.attn.dropout)
            for block in transformer.resblocks
        ])
        transformer.resblocks = custom_resblocks

        # (3) 미리 conv1 + relu + flatten => patch 임베딩
        #     => 논문은 ROI마다 다시 ViT 사용? or conv1까지만 공통 사용?
        #     여기서는 "conv1 결과"를 공통 추출해 patch_tokens로 씀.

    def _get_patch_coords(self):
        coords = []
        num_side = self.image_size // self.patch_size
        for r in range(num_side):
            for c in range(num_side):
                x1 = c * self.patch_size
                y1 = r * self.patch_size
                x2 = x1 + self.patch_size
                y2 = y1 + self.patch_size
                coords.append([x1,y1,x2,y2])
        coords = torch.tensor(coords, dtype=torch.float32, device=self.device)
        return coords

    def _expand_mask(self, mask_1d):
        """
        mask_1d: (num_patches,) => 2D 확장 (num_patches+1, num_patches+1)
        (논문) M_i ∈ R^(H'W') => \tilde{M}_i ∈ R^((H'W'+1)x(H'W'+1))
        """
        L = mask_1d.shape[0] + 1
        attn_bias = torch.zeros(L, L, device=mask_1d.device)
        # <CLS> -> patch 구간을 mask_1d로
        attn_bias[1:,1:] = torch.diag(mask_1d)  # diag로 처리 (혹은 broadcasting)
        # 논문에서 "Bias"라고 하면, QK^T + bias 형태
        return attn_bias  # (L,L)

    def _build_single_roi_cls(self, patch_tokens, mask_1d):
        """
        ROI i에 대해:
          - <CLS> + patch_tokens
          - attn_bias = expand(mask_1d)
          - Transformer -> 최종 <CLS> out
        """
        L_pt = patch_tokens.size(1)  # num_patches
        N_bt = patch_tokens.size(0)  # batch=1(항상)
        E = patch_tokens.size(2)     # embed_dim

        # (1) shape = (1, 1+num_patches, E)
        visual = self.vit_backbone.model.visual
        cls_token = visual.class_embedding.unsqueeze(0).expand(N_bt, -1, -1)  # (1,1,E)
        x = torch.cat((cls_token, patch_tokens), dim=1)  # (1, 1+patches, E)

        # (2) attn_bias => (1+patches, 1+patches)
        attn_2d = self._expand_mask(mask_1d)  # (L,L)

        # (3) L->(1+patches), N->1, E->embed_dim => x.transpose(0,1)
        x = x.transpose(0,1)  # => (1+patches,1,E)

        for blk in visual.transformer.resblocks:
            x = blk(x, attn_bias=attn_2d)  # (L,N,E)

        x = x.transpose(0,1)  # (1,L,E)
        cls_out = x[:,0,:].squeeze(0)  # => (E,) ROI 임베딩
        return cls_out  # shape=(embed_dim,)

    def forward(self, frame, bboxes):
        """
        frame: (C,H,W)
        bboxes: (N,4), 마지막 하나는 whole_face라고 가정
        => Return: (N+1, embed_dim)
        """
        # (1) 먼저 "conv1 -> relu -> flatten"까지만 공통 추출
        # => patch_tokens: (1, num_patches, embed_dim)
        visual = self.vit_backbone.model.visual
        x_in = frame.unsqueeze(0).to(self.device)     # (1,C,H,W)
        feat = visual.conv1(x_in)                    # (1,embed_dim,H/32,W/32)
        feat = F.relu(feat)
        feat = feat.flatten(2).transpose(1,2)        # (1,num_patches,embed_dim)

        # (2) Patch 좌표
        patch_coords = self._get_patch_coords()       # (num_patches,4)
        num_patches = patch_coords.size(0)            # = H'W'

        # (3) Overlap Ratio => shape=(N, num_patches)
        # bboxes shape=(N,4)
        M = compute_overlap_ratio(bboxes, patch_coords)  # (N, num_patches)

        # (4) ROI마다 <CLS> Self-Attn
        # => ROI i => M[i], => <CLS> i
        # => stack => (N, embed_dim)
        all_roi_cls = []
        N = M.size(0)  # 실제 ROI 수

        for i in range(N):
            # M[i] => shape=(num_patches,)
            cls_i = self._build_single_roi_cls(feat, M[i])
            all_roi_cls.append(cls_i)

        # (5) shape => (N, embed_dim)
        all_roi_cls = torch.stack(all_roi_cls, dim=0)  # (N, embed_dim)
        return all_roi_cls
