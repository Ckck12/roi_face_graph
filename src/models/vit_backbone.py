# src/models/vit_backbone.py

import torch
import torch.nn as nn

class SimpleViTBackbone(nn.Module):
    """
    patch_size=16, hidden_dim=768 등으로
    (B, C, H, W) -> (B, num_patches+1, hidden_dim)
    """
    def __init__(self, image_size=224, patch_size=16, hidden_dim=768, num_layers=2, nhead=8):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1,1,hidden_dim))

    def forward(self, x):
        # x: (B,3,H,W)
        B = x.size(0)
        x = self.patch_embed(x)    # (B, hidden_dim, H/16, W/16)
        x = x.flatten(2).transpose(1,2) # (B, num_patches, hidden_dim)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1) # (B, 1+num_patches, hidden_dim)
        x = self.transformer(x)
        return x
