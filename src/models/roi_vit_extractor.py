# # # src/models/roi_vit_extractor.py

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from src.models.vit_backbone import CLIPViTBackbone
# # from src.models.roi_aggregation import compute_overlap_ratio  # 함수 이름 일치시킴

# # class ModifiedMultiheadAttention(nn.Module):
# #     """
# #     PyTorch의 MultiheadAttention을 수정하여 attn_bias를 지원하도록 합니다.
# #     Attention 점수(QK^T)에 attn_bias를 더한 후 softmax를 적용합니다.
# #     """
# #     def __init__(self, embed_dim, num_heads, dropout=0.0):
# #         super(ModifiedMultiheadAttention, self).__init__()
# #         self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
    
# #     def forward(self, query, key, value, attn_bias=None):
# #         """
# #         Args:
# #             query: (L, N, E)
# #             key: (S, N, E)
# #             value: (S, N, E)
# #             attn_bias: (L, S) tensor
        
# #         Returns:
# #             attn_output: (L, N, E)
# #             attn_weights: (N*num_heads, L, S)
# #         """
# #         # 기본적인 MultiheadAttention의 Q, K, V 투영 수행
# #         # 그러나 여기서는 attn_bias를 직접 추가
# #         # MultiheadAttention의 내부 구현을 그대로 사용하는 것이 어려우므로, QK^T 계산을 직접 수행
        
# #         # Q, K, V 계산
# #         Q = self.multihead_attn.in_proj_weight[:self.multihead_attn.embed_dim, :].mm(query.view(-1, query.size(-1)).T).view(query.size(0), query.size(1), -1)
# #         K = self.multihead_attn.in_proj_weight[self.multihead_attn.embed_dim:2*self.multihead_attn.embed_dim, :].mm(key.view(-1, key.size(-1)).T).view(key.size(0), key.size(1), -1)
# #         V = self.multihead_attn.in_proj_weight[2*self.multihead_attn.embed_dim:, :].mm(value.view(-1, value.size(-1)).T).view(value.size(0), value.size(1), -1)
        
# #         # Attention 점수 계산: QK^T / sqrt(d_k) + attn_bias
# #         d_k = Q.size(-1)
# #         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (L, N, S)
        
# #         if attn_bias is not None:
# #             attn_scores = attn_scores + attn_bias.unsqueeze(1)  # (L, 1, S) Broadcast
        
# #         # Softmax 적용
# #         attn_weights = F.softmax(attn_scores, dim=-1)  # (L, N, S)
# #         attn_weights = F.dropout(attn_weights, p=self.multihead_attn.dropout, training=self.training)
        
# #         # Attention 출력 계산
# #         attn_output = torch.matmul(attn_weights, V)  # (L, N, E)
        
# #         return attn_output, attn_weights

# # class CustomResidualAttentionBlock(nn.Module):
# #     """
# #     CLIP의 ResidualAttentionBlock을 수정하여 attn_bias를 지원하도록 합니다.
# #     """
# #     def __init__(self, block):
# #         super(CustomResidualAttentionBlock, self).__init__()
# #         self.ln_1 = block.ln_1
# #         self.ln_2 = block.ln_2
# #         self.attn = ModifiedMultiheadAttention(block.attn.embed_dim, block.attn.num_heads, dropout=block.attn.dropout)
# #         self.mlp = block.mlp

# #     def forward(self, x, attn_bias=None):
# #         """
# #         Args:
# #             x: (L, N, E)
# #             attn_bias: (L, S) tensor
        
# #         Returns:
# #             x: (L, N, E)
# #         """
# #         # LayerNorm
# #         x_norm = self.ln_1(x)
# #         # Self-Attention with attn_bias
# #         attn_output, _ = self.attn(
# #             x_norm,  # (L, N, E)
# #             x_norm,  # (S, N, E)
# #             x_norm,  # (S, N, E)
# #             attn_bias=attn_bias  # (L, S)
# #         )
# #         # Residual Connection
# #         x = x + attn_output
# #         # LayerNorm
# #         x_norm = self.ln_2(x)
# #         # MLP
# #         mlp_output = self.mlp(x_norm)
# #         # Residual Connection
# #         x = x + mlp_output
# #         return x

# # class ROIViTExtractor(nn.Module):
# #     """
# #     한 프레임 (C, H, W)을 ViT 백본에 통과하여 CLS/patch 토큰을 구하고,
# #     bboxes(ROI) 정보를 이용해 각 ROI의 임베딩을 계산합니다.
    
# #     데이터 shape:
# #         - frame: (C, H, W)
# #         - bboxes: (N, 4)
    
# #     결과 shape:
# #         - node_embs: (N+1, hidden_dim)
# #     """
# #     def __init__(self, model_name="ViT-B/32", device='cuda', image_size=224, patch_size=32, hidden_dim=768):
# #         super(ROIViTExtractor, self).__init__()
# #         self.vit_backbone = CLIPViTBackbone(model_name=model_name, device=device)
# #         self.image_size = image_size
# #         self.patch_size = patch_size
# #         self.hidden_dim = hidden_dim

# #         # Transformer 블록을 CustomResidualAttentionBlock으로 교체
# #         transformer = self.vit_backbone.model.visual.transformer
# #         custom_resblocks = nn.ModuleList([
# #             CustomResidualAttentionBlock(block) for block in transformer.resblocks
# #         ])
# #         transformer.resblocks = custom_resblocks

# #     def _get_patch_coords(self):
# #         """
# #         이미지 패치의 바운딩 박스를 계산합니다.
# #         이미지 크기를 patch_size로 나누어 각 패치의 좌표 범위 리스트를 생성합니다.
# #         """
# #         coords = []
# #         num_side = self.image_size // self.patch_size
# #         for r in range(num_side):
# #             for c in range(num_side):
# #                 x1 = c * self.patch_size
# #                 y1 = r * self.patch_size
# #                 x2 = x1 + self.patch_size
# #                 y2 = y1 + self.patch_size
# #                 coords.append([x1, y1, x2, y2])
# #         return torch.tensor(coords, dtype=torch.float32, device=self.vit_backbone.device)  # (49, 4)

# #     def _build_attention_bias(self, Mi):
# #         """
# #         Mi를 기반으로 Attention Bias를 생성합니다.
        
# #         Args:
# #             Mi (Tensor): (N, num_patches) tensor
        
# #         Returns:
# #             attn_bias (Tensor): (1 + num_patches, 1 + num_patches) tensor
# #         """
# #         N, num_patches = Mi.shape  # N=9, num_patches=49

# #         # [CLS] 토큰은 모든 패치와 상호작용할 수 있도록 0 Bias 설정
# #         cls_bias = torch.zeros(1, num_patches, device=Mi.device)  # (1, 49)

# #         # 각 ROI에 대한 Mi를 평균하여 [CLS] 토큰의 Bias로 사용
# #         # 이는 [CLS] 토큰이 ROI가 있는 패치에 더 많은 주의를 기울이도록 함
# #         cls_roi_bias = Mi.mean(dim=0, keepdim=True)  # (1, 49)

# #         # 전체 Attention Bias는 [CLS] + 패치에 대한 Bias
# #         attn_bias = torch.cat((cls_bias, cls_roi_bias), dim=0)  # (2, 49)

# #         return attn_bias  # (2, 49)

# #     def forward(self, frame, bboxes):
# #         """
# #         Args:
# #             frame (Tensor): (C, H, W)
# #             bboxes (Tensor): (N, 4)
        
# #         Returns:
# #             node_embs (Tensor): (N+1, hidden_dim)
# #         """
# #         # ViT 백본을 통해 CLS 토큰과 패치 토큰을 추출
# #         visual = self.vit_backbone.model.visual

# #         # frame: (C, H, W) -> (1, C, H, W)
# #         x = frame.unsqueeze(0).to(next(self.vit_backbone.parameters()).device)  # (1, C, H, W)
# #         x = visual.conv1(x)     # (1, embed_dim, H/32, W/32)
# #         x = F.relu(x)           # ReLU 활성화 함수 적용
# #         x = x.flatten(2).transpose(1, 2)  # (1, num_patches, embed_dim)

# #         cls_token = visual.class_embedding.unsqueeze(0).expand(x.size(0), -1, -1)  # (1, 1, embed_dim)
# #         x = torch.cat((cls_token, x), dim=1)  # (1, 1 + num_patches, embed_dim)

# #         # 패치 좌표 계산
# #         patch_coords = self._get_patch_coords()  # (49, 4)

# #         num_patches = patch_coords.size(0)  # 49

# #         # Mi 매트릭스 벡터화된 계산
# #         Mi = compute_overlap_ratio(bboxes, patch_coords)  # (N, 49)

# #         # Attention Bias 생성: [CLS] + 패치에 대한 Bias
# #         attn_bias = self._build_attention_bias(Mi)  # (2, 49)

# #         # Transformer 블록을 통과하여 Attention Bias 적용
# #         # x는 (1, 50, embed_dim)
# #         x = x.transpose(0, 1)  # (50, 1, embed_dim)
# #         for blk in visual.transformer.resblocks:
# #             x = blk(x, attn_bias=attn_bias)  # (50, 1, embed_dim)

# #         # 최종 출력
# #         # x: (50, 1, embed_dim)
# #         x = x.transpose(0, 1)  # (1, 50, embed_dim)
# #         cls_out = x[:, 0, :]  # (1, embed_dim)
# #         patch_out = x[:, 1:, :]  # (1, 49, embed_dim)

# #         # ROI 임베딩 계산
# #         # Mi: (N, 49)
# #         roi_embeddings = Mi @ patch_out.squeeze(0)  # (N, embed_dim)

# #         # node_embs: [CLS] + [ROI 임베딩]
# #         node_embs = torch.cat((cls_out, roi_embeddings), dim=0)  # (N+1, embed_dim)

# #         return node_embs


# # src/models/roi_vit_extractor.py
# # src/models/roi_vit_extractor.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from src.models.vit_backbone import CLIPViTBackbone
# from src.models.roi_aggregation import compute_overlap_ratio  # 함수 이름 일치시킴

# class CustomMultiheadAttention(nn.Module):
#     """
#     Custom MultiheadAttention that adds attn_bias to the attention scores before softmax.
#     """
#     def __init__(self, embed_dim, num_heads, dropout=0.0):
#         super(CustomMultiheadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

#         # Define projection layers
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)

#     def forward(self, query, key, value, attn_bias=None):
#         """
#         Args:
#             query: (L, N, E)
#             key: (S, N, E)
#             value: (S, N, E)
#             attn_bias: (L, S) tensor
        
#         Returns:
#             attn_output: (L, N, E)
#             attn_weights: (N*num_heads, L, S)
#         """
#         # Project Q, K, V
#         Q = self.q_proj(query)  # (L, N, E)
#         K = self.k_proj(key)    # (S, N, E)
#         V = self.v_proj(value)  # (S, N, E)

#         # Reshape for multi-head
#         L, N, E = Q.size()
#         S, _, _ = K.size()

#         Q = Q.view(L, N, self.num_heads, self.head_dim).transpose(1,2)  # (L, num_heads, N, head_dim)
#         K = K.view(S, N, self.num_heads, self.head_dim).transpose(1,2)  # (S, num_heads, N, head_dim)
#         V = V.view(S, N, self.num_heads, self.head_dim).transpose(1,2)  # (S, num_heads, N, head_dim)

#         # Compute QK^T
#         Q = Q.permute(1, 2, 0, 3)  # (num_heads, N, L, head_dim)
#         K = K.permute(1, 2, 3, 0)  # (num_heads, N, head_dim, S)
#         attn_scores = torch.matmul(Q, K) / (self.head_dim ** 0.5)  # (num_heads, N, L, S)

#         if attn_bias is not None:
#             # attn_bias: (L, S)
#             # Expand to (num_heads, N, L, S)
#             attn_bias = attn_bias.unsqueeze(0).unsqueeze(1)  # (1,1,L,S)
#             attn_bias = attn_bias.expand(self.num_heads, N, L, S)  # (num_heads, N, L, S)
#             attn_scores = attn_scores + attn_bias  # (num_heads, N, L, S)

#         # Apply softmax
#         attn_weights = F.softmax(attn_scores, dim=-1)  # (num_heads, N, L, S)
#         attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

#         # Compute attention output
#         attn_output = torch.matmul(attn_weights, V)  # (num_heads, N, L, head_dim)

#         # Reshape and project out
#         attn_output = attn_output.permute(2, 1, 0, 3).contiguous().view(L, N, self.embed_dim)  # (L, N, E)
#         attn_output = self.out_proj(attn_output)  # (L, N, E)

#         return attn_output, attn_weights.view(self.num_heads * N, L, S)

# class CustomResidualAttentionBlock(nn.Module):
#     """
#     CLIP의 ResidualAttentionBlock을 수정하여 attn_bias를 지원하도록 합니다.
#     """
#     def __init__(self, block, embed_dim, num_heads, dropout=0.0):
#         super(CustomResidualAttentionBlock, self).__init__()
#         self.ln_1 = block.ln_1
#         self.ln_2 = block.ln_2
#         self.attn = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.mlp = block.mlp

#     def forward(self, x, attn_bias=None):
#         """
#         Args:
#             x: (L, N, E)
#             attn_bias: (L, S) tensor
#         Returns:
#             x: (L, N, E)
#         """
#         # LayerNorm
#         x_norm = self.ln_1(x)
#         # Self-Attention with attn_bias
#         attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_bias=attn_bias)  # (L, N, E), (num_heads*N, L, S)
#         # Residual Connection
#         x = x + attn_output
#         # LayerNorm
#         x_norm = self.ln_2(x)
#         # MLP
#         mlp_output = self.mlp(x_norm)
#         # Residual Connection
#         x = x + mlp_output
#         return x

# class ROIViTExtractor(nn.Module):
#     """
#     한 프레임 (C, H, W)을 ViT 백본에 통과하여 CLS/patch 토큰을 구하고,
#     bboxes(ROI) 정보를 이용해 각 ROI의 임베딩을 계산합니다.
    
#     데이터 shape:
#         - frame: (C, H, W)
#         - bboxes: (N, 4)
    
#     결과 shape:
#         - node_embs: (N+1, hidden_dim)
#     """
#     def __init__(self, model_name="ViT-B/32", device='cuda', image_size=224, patch_size=32, hidden_dim=768):
#         super(ROIViTExtractor, self).__init__()
#         self.vit_backbone = CLIPViTBackbone(model_name=model_name, device=device)
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.hidden_dim = hidden_dim

#         # Transformer 블록을 CustomResidualAttentionBlock으로 교체
#         transformer = self.vit_backbone.model.visual.transformer
#         custom_resblocks = nn.ModuleList([
#             CustomResidualAttentionBlock(block, transformer.width, block.attn.num_heads, block.attn.dropout)
#             for block in transformer.resblocks
#         ])
#         transformer.resblocks = custom_resblocks

#     def _get_patch_coords(self):
#         """
#         이미지 패치의 바운딩 박스를 계산합니다.
#         이미지 크기를 patch_size로 나누어 각 패치의 좌표 범위 리스트를 생성합니다.
#         """
#         coords = []
#         num_side = self.image_size // self.patch_size  # e.g., 224//32=7
#         for r in range(num_side):
#             for c in range(num_side):
#                 x1 = c * self.patch_size
#                 y1 = r * self.patch_size
#                 x2 = x1 + self.patch_size
#                 y2 = y1 + self.patch_size
#                 coords.append([x1, y1, x2, y2])
#         return torch.tensor(coords, dtype=torch.float32, device=self.vit_backbone.device)  # (49, 4)

#     def _build_attention_bias(self, Mi):
#         """
#         Mi를 기반으로 Attention Bias를 생성합니다.
        
#         Args:
#             Mi (Tensor): (N, num_patches) tensor
        
#         Returns:
#             attn_bias (Tensor): (L, S) tensor, where L = S = 1 + num_patches
#         """
#         N, num_patches = Mi.shape  # 예: N=9, num_patches=49

#         # Calculate per-patch bias by summing Mi over all ROIs
#         # 이는 각 패치가 여러 ROI와 얼마나 겹치는지를 나타냅니다.
#         patch_bias = Mi.sum(dim=0)  # (49,)

#         # Normalize patch_bias
#         patch_bias = patch_bias / (patch_bias.max(dim=0, keepdim=True)[0] + 1e-6)  # (49,)

#         # Create attention bias matrix
#         # Sequence length L = S = 1 + num_patches (CLS + patches)
#         L = 1 + num_patches
#         S = 1 + num_patches
#         attn_bias = torch.zeros(L, S, device=Mi.device)  # (50,50)

#         # [CLS] 토큰은 모든 패치에 주의를 기울일 수 있도록 패치 Bias 추가
#         attn_bias[0, 1:] = patch_bias  # [CLS] -> 패치에 대한 Bias

#         # 패치들은 [CLS]에만 접근할 수 있도록 Bias를 0으로 유지
#         # 이미 초기화된 상태이므로 추가 설정 필요 없음

#         return attn_bias  # (50,50)

#     def forward(self, frame, bboxes):
#         """
#         Args:
#             frame (Tensor): (C, H, W)
#             bboxes (Tensor): (N, 4)
        
#         Returns:
#             node_embs (Tensor): (N+1, hidden_dim)
#         """
#         # ViT 백본을 통해 CLS 토큰과 패치 토큰을 추출
#         visual = self.vit_backbone.model.visual

#         # frame: (C, H, W) -> (1, C, H, W)
#         x = frame.unsqueeze(0).to(next(self.vit_backbone.parameters()).device)  # (1, C, H, W)
#         x = visual.conv1(x)     # (1, embed_dim, H/32, W/32)
#         x = F.relu(x)           # ReLU 활성화 함수 적용
#         x = x.flatten(2).transpose(1, 2)  # (1, num_patches, embed_dim)

#         cls_token = visual.class_embedding.unsqueeze(0).expand(x.size(0), -1, -1)  # (1, 1, embed_dim)
#         x = torch.cat((cls_token, x), dim=1)  # (1, 1 + num_patches, embed_dim)

#         # 패치 좌표 계산
#         patch_coords = self._get_patch_coords()  # (49, 4)

#         num_patches = patch_coords.size(0)  # 49

#         # Mi 매트릭스 벡터화된 계산
#         Mi = compute_overlap_ratio(bboxes, patch_coords)  # (N, 49)

#         # Attention Bias 생성: (50,50)
#         attn_bias = self._build_attention_bias(Mi)  # (50,50)

#         # Transformer 블록을 통과하여 Attention Bias 적용
#         # x는 (1, 50, embed_dim)
#         x = x.transpose(0, 1)  # (50, 1, embed_dim)
#         for blk in visual.transformer.resblocks:
#             x = blk(x, attn_bias=attn_bias)  # (50,1,embed_dim)

#         # 최종 출력
#         # x: (50,1,embed_dim)
#         x = x.transpose(0, 1)  # (1,50,embed_dim)
#         cls_out = x[:, 0, :]  # (1, embed_dim)
#         patch_out = x[:, 1:, :]  # (1,49,embed_dim)

#         # ROI 임베딩 계산
#         # Mi: (N,49)
#         # patch_out: (1,49,embed_dim) -> (49, embed_dim)
#         patch_out = patch_out.squeeze(0)  # (49, embed_dim)
#         roi_embeddings = Mi @ patch_out  # (N, embed_dim)

#         # node_embs: [CLS] + [ROI 임베딩]
#         node_embs = torch.cat((cls_out, roi_embeddings), dim=0)  # (N+1, embed_dim)

#         return node_embs

# src/models/roi_vit_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.vit_backbone import CLIPViTBackbone
from src.models.roi_aggregation import compute_overlap_ratio
import math

###############################################
# 1) 수정된 MultiheadAttention
###############################################
class CustomMultiheadAttention(nn.Module):
    """
    Custom MultiheadAttention that adds attn_bias to the attention scores before softmax.
    Uses torch.einsum for clear tensor operations.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Define projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_bias=None):
        """
        Args:
            query: (L, N, E)
            key: (S, N, E)
            value: (S, N, E)
            attn_bias: (L, S) tensor  (optional)

        Returns:
            attn_output: (L, N, E)
            attn_weights: (num_heads * N, L, S)
        """
        # Q, K, V shape: (L, N, E) or (S, N, E)
        # L, S: sequence lengths (ex: 50 each if self-attention)
        # N: batch size
        # E: embed_dim
        Q = self.q_proj(query)  # (L, N, E)
        K = self.k_proj(key)    # (S, N, E)
        V = self.v_proj(value)  # (S, N, E)

        L, N, E = Q.shape
        S = K.shape[0]

        # Reshape for multi-head: -> (L, N, num_heads, head_dim)
        Q = Q.view(L, N, self.num_heads, self.head_dim)
        K = K.view(S, N, self.num_heads, self.head_dim)
        V = V.view(S, N, self.num_heads, self.head_dim)

        # Permute to put head_dim last for Q, but for K we want head_dim at -2
        # Common approach:
        #   Q: (num_heads, N, L, head_dim) = 'hnld'
        #   K: (num_heads, N, head_dim, S) = 'hnd s'
        #   V: (num_heads, N, S, head_dim) = 'hns d'
        # So that we do: 'hnld,hnds->hnls' for Q*K
        Q = Q.permute(2, 1, 0, 3).contiguous()  # (h, N, L, d)
        K = K.permute(2, 1, 3, 0).contiguous()  # (h, N, d, S)
        V = V.permute(2, 1, 0, 3).contiguous()  # (h, N, S, d)

        # Compute QK^T: attn_scores shape = (h, N, L, S)
        attn_scores = torch.einsum('hnld,hnds->hnls', Q, K) / math.sqrt(self.head_dim)

        if attn_bias is not None:
            # attn_bias: (L, S) => expand to (h, N, L, S)
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(1)  # (1,1,L,S)
            attn_bias = attn_bias.expand(self.num_heads, N, L, S)
            attn_scores = attn_scores + attn_bias  # (h, N, L, S)

        # softmax over last dim (S)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (h, N, L, S)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Multiply by V => attn_output shape = (h, N, L, d)
        attn_output = torch.einsum('hnls,hnsd->hnld', attn_weights, V)
        # Rearrange back to (L, N, E)
        attn_output = attn_output.permute(2, 1, 0, 3).contiguous()  # (L, N, h, d)
        attn_output = attn_output.view(L, N, self.embed_dim)         # (L, N, E)

        # final linear
        attn_output = self.out_proj(attn_output)  # (L, N, E)

        # Flatten attn_weights
        attn_weights = attn_weights.view(self.num_heads * N, L, S)  # (h*N, L, S)
        return attn_output, attn_weights

###############################################
# 2) ResidualAttentionBlock (동일)
###############################################
class CustomResidualAttentionBlock(nn.Module):
    def __init__(self, block, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln_1 = block.ln_1
        self.ln_2 = block.ln_2
        self.attn = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = block.mlp

    def forward(self, x, attn_bias=None):
        """
        x: (L, N, E)
        """
        x_norm = self.ln_1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_bias=attn_bias)
        x = x + attn_output
        x_norm = self.ln_2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        return x

###############################################
# 3) ROIViTExtractor (나머지 동일)
###############################################
class ROIViTExtractor(nn.Module):
    def __init__(self, model_name="ViT-B/32", device='cuda', image_size=224, patch_size=32, hidden_dim=768):
        super().__init__()
        self.vit_backbone = CLIPViTBackbone(model_name=model_name, device=device)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        transformer = self.vit_backbone.model.visual.transformer
        custom_resblocks = nn.ModuleList([
            CustomResidualAttentionBlock(block, transformer.width, block.attn.num_heads, block.attn.dropout)
            for block in transformer.resblocks
        ])
        transformer.resblocks = custom_resblocks

    def _get_patch_coords(self):
        coords = []
        num_side = self.image_size // self.patch_size  # 224//32=7
        for r in range(num_side):
            for c in range(num_side):
                x1 = c * self.patch_size
                y1 = r * self.patch_size
                x2 = x1 + self.patch_size
                y2 = y1 + self.patch_size
                coords.append([x1, y1, x2, y2])
        device_ = next(self.vit_backbone.parameters()).device
        return torch.tensor(coords, dtype=torch.float32, device=device_)

    def _build_attention_bias(self, Mi):
        N, num_patches = Mi.shape
        patch_bias = Mi.sum(dim=0)  # (num_patches,)
        patch_bias = patch_bias / (patch_bias.max(dim=0, keepdim=True)[0] + 1e-6)
        L = 1 + num_patches
        S = 1 + num_patches
        attn_bias = torch.zeros(L, S, device=Mi.device)
        attn_bias[0, 1:] = patch_bias
        return attn_bias

    def forward(self, frame, bboxes):
        visual = self.vit_backbone.model.visual

        # (C,H,W)->(1,C,H,W)
        x = frame.unsqueeze(0).to(next(self.vit_backbone.parameters()).device)
        x = visual.conv1(x)   # (1, embed_dim, H/32, W/32)
        x = F.relu(x)
        x = x.flatten(2).transpose(1,2)  # (1, num_patches, embed_dim)

        cls_token = visual.class_embedding.unsqueeze(0).expand(x.size(0), -1, -1)  # (1,1,embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (1, 1+num_patches, embed_dim)

        patch_coords = self._get_patch_coords()
        num_patches = patch_coords.size(0)
        Mi = compute_overlap_ratio(bboxes, patch_coords)  # (N, num_patches)

        attn_bias = self._build_attention_bias(Mi)  # (1+num_patches,1+num_patches)=(50,50)

        # Transformer blocks
        x = x.transpose(0,1)  # (1+num_patches,1,embed_dim)->(L,N,E)= (50,1,E)
        for blk in visual.transformer.resblocks:
            x = blk(x, attn_bias=attn_bias)  # (50,1,E)
        x = x.transpose(0,1)  # (1,50,E)

        cls_out = x[:,0,:]       # (1,E)
        patch_out = x[:,1:,:]    # (1,49,E)->(49,E)
        patch_out = patch_out.squeeze(0)

        roi_embeddings = Mi @ patch_out  # (N,E)
        node_embs = torch.cat((cls_out, roi_embeddings), dim=0) # (N+1,E)
        return node_embs
