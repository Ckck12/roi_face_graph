# src/models/roi_vit_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.vit_backbone import CLIPViTBackbone  # 수정된 finetune 버전
from src.models.roi_aggregation import compute_overlap_ratio

class CustomMultiheadAttention(nn.Module):
    """
    기존과 동일한 MultiheadAttention + attn_bias.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_bias=None):
        # query,key,value: (L, N, E)
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        L, N, E = Q.size()
        Q = Q.view(L, N, self.num_heads, self.head_dim).permute(2,1,0,3)  # (h,N,L,d)
        K = K.view(L, N, self.num_heads, self.head_dim).permute(2,1,0,3)
        V = V.view(L, N, self.num_heads, self.head_dim).permute(2,1,0,3)

        # QK^T => (h,N,L,L)
        attn_scores = torch.einsum("hnld,hnad->hnla", Q, K) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            # attn_bias: (L,L) => expand (h,N,L,L)
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(1).expand(self.num_heads, N, L, L)
            attn_scores = attn_scores + attn_bias

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # (h,N,L,d)
        attn_output = torch.einsum("hnla,hnad->hnld", attn_weights, V)
        attn_output = attn_output.permute(2,1,0,3).contiguous().view(L, N, E)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights

class CustomResidualAttentionBlock(nn.Module):
    """
    CLIP의 Residual Block에 attn_bias 추가.
    """
    def __init__(self, block, embed_dim, num_heads, dropout=0.5):
        super().__init__()
        self.ln_1 = block.ln_1
        self.ln_2 = block.ln_2
        self.attn = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = block.mlp

    def forward(self, x, attn_bias=None):
        # x: (L, N, E)
        x_norm = self.ln_1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_bias=attn_bias)
        x = x + attn_output

        x_norm = self.ln_2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        return x

class ROIViTExtractor(nn.Module):
    """
    (1) conv1 ~ flatten까지만 공통 추출
    (2) Overlap Ratio 구해 => mask_1d
    (3) _expand_mask(...) -> CLS↔patch 위치에 mask_1d 반영
    (4) 각 ROI마다 Self-Attn -> <CLS> 임베딩
    """
    def __init__(self, model_name="ViT-B/32", device='cuda',
                 image_size=224, patch_size=32, hidden_dim=768):
        super().__init__()
        # 수정된 CLIPViTBackbone (마지막 2블록 finetune)
        self.vit_backbone = CLIPViTBackbone(model_name=model_name,
                                            device=device,
                                            num_finetune_blocks=2)

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.device = device

        # CLIP 내부 Transformer -> CustomResidualAttentionBlock 교체
        visual = self.vit_backbone.model.visual
        transformer = visual.transformer
        custom_resblocks = []
        for block in transformer.resblocks:
            custom_resblocks.append(
                CustomResidualAttentionBlock(block,
                                             transformer.width,
                                             block.attn.num_heads,
                                             block.attn.dropout)
            )
        transformer.resblocks = nn.ModuleList(custom_resblocks)

    def _get_patch_coords(self):
        coords = []
        num_side = self.image_size // self.patch_size  # e.g., 7
        for r in range(num_side):
            for c in range(num_side):
                x1 = c * self.patch_size
                y1 = r * self.patch_size
                x2 = x1 + self.patch_size
                y2 = y1 + self.patch_size
                coords.append([x1, y1, x2, y2])
        return torch.tensor(coords, dtype=torch.float32, device=self.device)

    def _expand_mask(self, mask_1d: torch.Tensor):
        """
        mask_1d: (num_patches,) → attn_bias: (1+num_patches, 1+num_patches)

        여기서는 CLS->patch & patch->CLS 경로에 mask_1d를 더해
        overlap이 높은 patch에 더 집중하도록.
        """
        L = mask_1d.shape[0] + 1  # +1 for CLS
        attn_bias = torch.zeros(L, L, device=mask_1d.device)
        # - row=0 => CLS -> patch k (col k) = mask_1d[k-1]
        # - col=0 => patch k -> CLS (row k) = mask_1d[k-1]
        # => 양방향 반영
        attn_bias[0, 1:] = mask_1d       # CLS->patch
        attn_bias[1:, 0] = mask_1d       # patch->CLS

        return attn_bias  # (L,L)

    def _build_single_roi_cls(self, patch_tokens, mask_1d):
        """
        한 ROI에 대해
         - patch_tokens: (1, num_patches, E)
         - mask_1d: (num_patches,)
         => Self-Attn -> <CLS> out
        """
        visual = self.vit_backbone.model.visual
        B, P, E = patch_tokens.shape  # B=1, P=num_patches
        cls_token = visual.class_embedding.unsqueeze(0).expand(B, -1, -1)  # (1,1,E)

        x = torch.cat((cls_token, patch_tokens), dim=1)  # (1, P+1, E)
        attn_2d = self._expand_mask(mask_1d)            # (P+1, P+1)

        # => (L,N,E) = (P+1,1,E)
        x = x.transpose(0,1)  # (P+1,1,E)

        # Transformer resblocks
        for blk in visual.transformer.resblocks:
            x = blk(x, attn_bias=attn_2d)

        # => (1,P+1,E)
        x = x.transpose(0,1)
        cls_out = x[:,0,:]  # (1,E)
        return cls_out.squeeze(0)  # (E,)

    def forward(self, frame: torch.Tensor, bboxes: torch.Tensor):
        """
        frame: (C,H,W)
        bboxes: (N,4)  # N개 ROI
        Returns: (N, E)
        """
        visual = self.vit_backbone.model.visual

        # 1) conv1 ~ flatten
        x_in = frame.unsqueeze(0).to(self.device)  # (1,C,H,W)
        feat = visual.conv1(x_in)                 # (1,embed_dim,H/32,W/32)
        feat = F.relu(feat)
        feat = feat.flatten(2).transpose(1,2)     # (1,num_patches,embed_dim)

        # 2) Overlap ratio => shape=(N,num_patches)
        patch_coords = self._get_patch_coords()    # (num_patches,4)
        M = compute_overlap_ratio(bboxes, patch_coords)  # (N, num_patches)

        # 3) ROI마다 <CLS> 토큰으로 Self-Attn -> (N, E)
        roi_cls_list = []
        for i in range(M.size(0)):
            cls_i = self._build_single_roi_cls(feat, M[i])  # (embed_dim,)
            roi_cls_list.append(cls_i)

        all_roi_cls = torch.stack(roi_cls_list, dim=0)  # (N, E) 모든 roi에 해당하는 임베딩
        return all_roi_cls
