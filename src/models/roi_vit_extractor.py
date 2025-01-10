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
