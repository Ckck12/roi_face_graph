# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from src.models.vit_backbone import CLIPViTBackbone
# from src.models.roi_aggregation import compute_overlap_ratio
# import math

# ###############################################
# # 1) 수정된 MultiheadAttention
# ###############################################
# class CustomMultiheadAttention(nn.Module):
#     """
#     Custom MultiheadAttention that adds attn_bias to the attention scores before softmax.
#     Uses torch.einsum for clear tensor operations.
#     """
#     def __init__(self, embed_dim, num_heads, dropout=0.0):
#         super().__init__()
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
#             attn_bias: (L, S) tensor  (optional)

#         Returns:
#             attn_output: (L, N, E)
#             attn_weights: (num_heads * N, L, S)
#         """
#         # Q, K, V shape: (L, N, E) or (S, N, E)
#         # L, S: sequence lengths (ex: 50 each if self-attention)
#         # N: batch size
#         # E: embed_dim
#         Q = self.q_proj(query)  # (L, N, E)
#         K = self.k_proj(key)    # (S, N, E)
#         V = self.v_proj(value)  # (S, N, E)

#         L, N, E = Q.shape
#         S = K.shape[0]

#         # Reshape for multi-head: -> (L, N, num_heads, head_dim)
#         Q = Q.view(L, N, self.num_heads, self.head_dim)
#         K = K.view(S, N, self.num_heads, self.head_dim)
#         V = V.view(S, N, self.num_heads, self.head_dim)

#         # Permute to put head_dim last for Q, but for K we want head_dim at -2
#         # Common approach:
#         #   Q: (num_heads, N, L, head_dim) = 'hnld'
#         #   K: (num_heads, N, head_dim, S) = 'hnd s'
#         #   V: (num_heads, N, S, head_dim) = 'hns d'
#         # So that we do: 'hnld,hnds->hnls' for Q*K
#         Q = Q.permute(2, 1, 0, 3).contiguous()  # (h, N, L, d)
#         K = K.permute(2, 1, 3, 0).contiguous()  # (h, N, d, S)
#         V = V.permute(2, 1, 0, 3).contiguous()  # (h, N, S, d)

#         # Compute QK^T: attn_scores shape = (h, N, L, S)
#         attn_scores = torch.einsum('hnld,hnds->hnls', Q, K) / math.sqrt(self.head_dim)

#         if attn_bias is not None:
#             # attn_bias: (L, S) => expand to (h, N, L, S)
#             attn_bias = attn_bias.unsqueeze(0).unsqueeze(1)  # (1,1,L,S)
#             attn_bias = attn_bias.expand(self.num_heads, N, L, S)
#             attn_scores = attn_scores + attn_bias  # (h, N, L, S)

#         # softmax over last dim (S)
#         attn_weights = F.softmax(attn_scores, dim=-1)  # (h, N, L, S)
#         attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

#         # Multiply by V => attn_output shape = (h, N, L, d)
#         attn_output = torch.einsum('hnls,hnsd->hnld', attn_weights, V)
#         # Rearrange back to (L, N, E)
#         attn_output = attn_output.permute(2, 1, 0, 3).contiguous()  # (L, N, h, d)
#         attn_output = attn_output.view(L, N, self.embed_dim)         # (L, N, E)

#         # final linear
#         attn_output = self.out_proj(attn_output)  # (L, N, E)

#         # Flatten attn_weights
#         attn_weights = attn_weights.view(self.num_heads * N, L, S)  # (h*N, L, S)
#         return attn_output, attn_weights

# ###############################################
# # 2) ResidualAttentionBlock (동일)
# ###############################################
# class CustomResidualAttentionBlock(nn.Module):
#     def __init__(self, block, embed_dim, num_heads, dropout=0.0):
#         super().__init__()
#         self.ln_1 = block.ln_1
#         self.ln_2 = block.ln_2
#         self.attn = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.mlp = block.mlp

#     def forward(self, x, attn_bias=None):
#         """
#         x: (L, N, E)
#         """
#         x_norm = self.ln_1(x)
#         attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_bias=attn_bias)
#         x = x + attn_output
#         x_norm = self.ln_2(x)
#         mlp_output = self.mlp(x_norm)
#         x = x + mlp_output
#         return x

# ###############################################
# # 3) ROIViTExtractor (나머지 동일)
# ###############################################
# class ROIViTExtractor(nn.Module):
#     def __init__(self, model_name="ViT-B/32", device='cuda', image_size=224, patch_size=32, hidden_dim=768):
#         super().__init__()
#         self.vit_backbone = CLIPViTBackbone(model_name=model_name, device=device)
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.hidden_dim = hidden_dim

#         transformer = self.vit_backbone.model.visual.transformer
#         custom_resblocks = nn.ModuleList([
#             CustomResidualAttentionBlock(block, transformer.width, block.attn.num_heads, block.attn.dropout)
#             for block in transformer.resblocks
#         ])
#         transformer.resblocks = custom_resblocks

#     def _get_patch_coords(self):
#         coords = []
#         num_side = self.image_size // self.patch_size  # 224//32=7
#         for r in range(num_side):
#             for c in range(num_side):
#                 x1 = c * self.patch_size
#                 y1 = r * self.patch_size
#                 x2 = x1 + self.patch_size
#                 y2 = y1 + self.patch_size
#                 coords.append([x1, y1, x2, y2])
#         device_ = next(self.vit_backbone.parameters()).device
#         return torch.tensor(coords, dtype=torch.float32, device=device_)

#     def _build_attention_bias(self, Mi):
#         N, num_patches = Mi.shape
#         patch_bias = Mi.sum(dim=0)  # (num_patches,)
#         patch_bias = patch_bias / (patch_bias.max(dim=0, keepdim=True)[0] + 1e-6)
#         L = 1 + num_patches
#         S = 1 + num_patches
#         attn_bias = torch.zeros(L, S, device=Mi.device)
#         attn_bias[0, 1:] = patch_bias
#         return attn_bias

#     def forward(self, frame, bboxes):
#         visual = self.vit_backbone.model.visual

#         # (C,H,W)->(1,C,H,W)
#         x = frame.unsqueeze(0).to(next(self.vit_backbone.parameters()).device)
#         x = visual.conv1(x)   # (1, embed_dim, H/32, W/32)
#         x = F.relu(x)
#         x = x.flatten(2).transpose(1,2)  # (1, num_patches, embed_dim)

#         cls_token = visual.class_embedding.unsqueeze(0).expand(x.size(0), -1, -1)  # (1,1,embed_dim)
#         x = torch.cat((cls_token, x), dim=1)  # (1, 1+num_patches, embed_dim)

#         patch_coords = self._get_patch_coords()
#         num_patches = patch_coords.size(0)
#         Mi = compute_overlap_ratio(bboxes, patch_coords)  # (N, num_patches)

#         attn_bias = self._build_attention_bias(Mi)  # (1+num_patches,1+num_patches)=(50,50)

#         # Transformer blocks
#         x = x.transpose(0,1)  # (1+num_patches,1,embed_dim)->(L,N,E)= (50,1,E)
#         for blk in visual.transformer.resblocks:
#             x = blk(x, attn_bias=attn_bias)  # (50,1,E)
#         x = x.transpose(0,1)  # (1,50,E)

#         cls_out = x[:,0,:]       # (1,E)
#         patch_out = x[:,1:,:]    # (1,49,E)->(49,E)
#         patch_out = patch_out.squeeze(0)

#         roi_embeddings = Mi @ patch_out  # (N,E)
#         node_embs = torch.cat((cls_out, roi_embeddings), dim=0) # (N+1,E)
#         return node_embs


# src/models/roi_vit_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.vit_backbone import CLIPViTBackbone
from src.models.roi_aggregation import compute_overlap_ratio


##############################################################################
# 1) Custom MultiheadAttention (Overlap => Attention Bias)
##############################################################################
class CustomMultiheadAttention(nn.Module):
    """
    기존 PyTorch MultiheadAttention과 달리, attn_bias(Overlap 정보)를
    스코어 계산 시 추가할 수 있도록 만든 클래스입니다.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        assert (embed_dim % num_heads) == 0, "embed_dim must be divisible by num_heads."
        self.head_dim = embed_dim // num_heads

        # Q, K, V projection
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_bias=None):
        """
        Args:
            query, key, value: (seq_len, batch_size, embed_dim)
            attn_bias (optional): (seq_len, seq_len) - Overlap mask 등에 사용
                                  broadcast 시 (1, 1, seq_len, seq_len)
        Returns:
            out: (seq_len, batch_size, embed_dim)
            attn_weights: (num_heads, batch_size, seq_len, seq_len)
        """

        seq_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim, "Input embed_dim != defined embed_dim"

        # 1) Q, K, V projection
        Q = self.q_proj(query)  # (seq_len, bsz, embed_dim)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 2) Reshape to (num_heads, bsz, seq_len, head_dim)
        #    - 먼저 (seq_len, bsz, num_heads, head_dim)로 만든 뒤,
        #      permute(0, 2) → (num_heads, bsz, seq_len, head_dim)가 아님을 주의.
        #    - seq_len=0축, bsz=1축 → 이걸 (num_heads=0축, bsz=1축, seq_len=2축, head_dim=3축)으로 바꾸려면
        #      transpose + view 로 처리해야 함.
        def _reshape(x):
            # x: (seq_len, bsz, embed_dim)
            # -> view(seq_len, bsz, num_heads, head_dim)
            x = x.view(seq_len, bsz, self.num_heads, self.head_dim)
            # -> (seq_len, bsz, num_heads, head_dim)
            # permute to (num_heads, bsz, seq_len, head_dim)
            x = x.permute(2, 1, 0, 3).contiguous()
            # shape=(n, b, l, d)
            return x

        Q = _reshape(Q)  # (n, b, l, d)
        K = _reshape(K)  # (n, b, l, d)
        V = _reshape(V)  # (n, b, l, d)

        # 3) Q*K^T => attn_scores
        #    Q: (n, b, l, d), K^T: (n, b, d, l)
        #    => (n, b, l, l)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4) attn_bias 추가
        #    attn_bias: (seq_len, seq_len) => (1,1,seq_len,seq_len) => shape match
        #    => broadcast to (n,b,l,l)
        if attn_bias is not None:
            scores = scores + attn_bias.unsqueeze(0).unsqueeze(0)

        # 5) softmax & dropout
        attn_weights = F.softmax(scores, dim=-1)  # (n,b,l,l)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 6) attn_weights x V => (n,b,l,d)
        #    V: (n,b,l,d)
        out = torch.einsum('nbll,nbld->nbld', attn_weights, V)

        # 7) reshape back => (seq_len, bsz, embed_dim)
        #    out shape=(n,b,l,d) => permute -> (l,b,n,d) => view -> (l,b,E)
        out = out.permute(2, 1, 0, 3).contiguous()  # (l,b,n,d)
        out = out.view(seq_len, bsz, embed_dim)     # (l,b,E)

        # 8) final projection
        out = self.out_proj(out)  # (l,b,E)

        return out, attn_weights  # (seq_len,b,embed_dim), (n,b,l,l)


##############################################################################
# 2) ResidualAttentionBlock
##############################################################################
class CustomResidualAttentionBlock(nn.Module):
    """
    CLIP의 ResBlock을 확장하여 attn_bias를 받도록 수정한 클래스
    """
    def __init__(self, original_block, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.ln_1 = original_block.ln_1
        self.ln_2 = original_block.ln_2
        self.attn = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = original_block.mlp

    def forward(self, x, attn_bias=None):
        """
        x: (seq_len, batch_size, embed_dim)
        attn_bias: (seq_len, seq_len) attention bias
        """
        # 1) Self-Attention
        x_norm = self.ln_1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_bias=attn_bias)
        x = x + attn_out

        # 2) MLP
        x_norm = self.ln_2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


##############################################################################
# 3) ROI Aggregation: ROI마다 별도의 <CLS>
##############################################################################
class ROIViTExtractor(nn.Module):
    """
    논문처럼 "ROI마다 별도의 <CLS>"를 만들어,
    Overlap 정보(M_i)를 self-attn bias로 추가하는 구조.

    절차:
      1) frame -> conv1 -> patch_tokens (B=1 가정)
      2) for each ROI:
         - <CLS> + patch_tokens
         - attn_bias = expand(M_i)
         - Transformer 통과 -> 해당 ROI에 대한 <CLS>만 추출
      3) ROI별 <CLS>를 쌓아서 (N, hidden_dim) 형태 반환
    """
    def __init__(
        self,
        model_name="ViT-B/32",
        device="cuda",
        image_size=224,
        patch_size=32,
        hidden_dim=768,
        num_finetune_blocks=2
    ):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # (A) CLIP ViT 백본
        from src.models.vit_backbone import CLIPViTBackbone
        self.vit_backbone = CLIPViTBackbone(
            model_name=model_name,
            device=device,
            num_finetune_blocks=num_finetune_blocks
        )

        # (B) CLIP 내부 Transformer 블록을 CustomResidualAttentionBlock으로 교체
        visual = self.vit_backbone.model.visual
        transformer = visual.transformer
        custom_blocks = []
        for blk in transformer.resblocks:
            custom_blocks.append(
                CustomResidualAttentionBlock(
                    original_block=blk,
                    embed_dim=transformer.width,
                    num_heads=blk.attn.num_heads,
                    dropout=blk.attn.dropout
                )
            )
        transformer.resblocks = nn.ModuleList(custom_blocks)

    def _get_patch_coords(self):
        """
        patch 좌표 계산 => (num_patches, 4)
        """
        coords = []
        num_side = self.image_size // self.patch_size  # 예: 224//32=7
        for r in range(num_side):
            for c in range(num_side):
                x1 = c * self.patch_size
                y1 = r * self.patch_size
                x2 = x1 + self.patch_size
                y2 = y1 + self.patch_size
                coords.append([x1, y1, x2, y2])
        coords = torch.tensor(coords, dtype=torch.float32, device=self.device)
        return coords  # (num_patches,4)

    def _build_attn_bias(self, mask_1d: torch.Tensor) -> torch.Tensor:
        """
        Overlap mask(M_i: (num_patches,)) => 2D attn_bias: (num_patches+1, num_patches+1)

        - row=0 => <CLS> -> patch i
        - col=0 => patch i -> <CLS>
        """
        L = mask_1d.size(0) + 1  # +1 for <CLS>
        attn_bias = torch.zeros(L, L, device=mask_1d.device)

        # <CLS>(0)->patch(i=1..L-1)
        attn_bias[0, 1:] = mask_1d
        # patch(i)-><CLS>(0)
        attn_bias[1:, 0] = mask_1d

        return attn_bias

    def forward(self, frame: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        """
        frame: (3,H,W)
        bboxes: (N,4)
        return: (N, hidden_dim)
        """
        # 1) (3,H,W)->(1,3,H,W)
        x_in = frame.unsqueeze(0).to(self.device)  # (1,3,H,W)

        # 2) ViT의 conv1 -> flatten -> (1, num_patches, embed_dim)
        visual = self.vit_backbone.model.visual
        feat = visual.conv1(x_in)
        feat = F.relu(feat)
        feat = feat.flatten(2).transpose(1,2)  # => (1, num_patches, embed_dim)

        # 3) patch 좌표 -> overlap ratio
        from src.models.roi_aggregation import compute_overlap_ratio
        patch_coords = self._get_patch_coords()  # (num_patches,4)
        M = compute_overlap_ratio(bboxes, patch_coords)  # (N, num_patches)

        # 4) (N, hidden_dim) => ROI별로 <CLS> Attention
        #    => ROI 갯수(N)만큼 for-loop
        #    => Transformer resblocks는 'roi별'로 새로 통과
        #    => <CLS> output을 stack
        out_list = []
        for i in range(M.size(0)):
            # i번째 ROI에 대한 overlap mask
            mask_1d = M[i]  # shape=(num_patches,)

            # (A) seq = <CLS> + patch_tokens
            #     => shape: (1, 1+num_patches, E)
            cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0)  # (1,1,E)
            cls_token = cls_token.expand(1, -1, feat.size(-1))  # (1,1,E) 그대로
            x_seq = torch.cat((cls_token, feat), dim=1)  # (1, 1+num_patches, E)

            # (B) attn_bias: (1+num_patches, 1+num_patches)
            attn_bias = self._build_attn_bias(mask_1d)

            # (C) Transformer Forward
            x_seq = x_seq.transpose(0,1)  # => (L=1+patches, B=1, E)
            for blk in visual.transformer.resblocks:
                x_seq = blk(x_seq, attn_bias=attn_bias)
            x_seq = x_seq.transpose(0,1)  # => (1, 1+patches, E)

            # (D) <CLS> 추출 => (1,E)
            roi_cls = x_seq[:,0,:]  # => (1,E)
            out_list.append(roi_cls.squeeze(0))  # => (E,)

        # => (N,E)
        all_roi_cls = torch.stack(out_list, dim=0)
        return all_roi_cls
