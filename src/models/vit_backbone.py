# src/models/vit_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class CLIPViTBackbone(nn.Module):
    """
    CLIP의 ViT 백본에서, 마지막 2개 Transformer 블록만 Unfreeze하고
    나머지는 Freeze 상태로 유지하는 예시.
    """
    def __init__(self, model_name="ViT-B/32", device='cuda', num_finetune_blocks=2):
        super(CLIPViTBackbone, self).__init__()
        self.device = device

        # (1) CLIP 모델 로드
        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.model.eval()  # 기본 eval 모드

        # (2) 전체 파라미터를 일단 requires_grad=False → Freeze
        for param in self.model.parameters():
            param.requires_grad = False

        # (3) float32 변환
        self.model = self.model.float()
        for param in self.model.parameters():
            param.data = param.data.float()
        for buffer in self.model.buffers():
            buffer.data = buffer.data.float()

        # (4) 마지막 num_finetune_blocks개 Transformer 블록만 Unfreeze
        visual = self.model.visual
        total_blocks = len(visual.transformer.resblocks)  # 예: 12
        start_idx = max(0, total_blocks - num_finetune_blocks)
        
        for i in range(start_idx, total_blocks):
            for param in visual.transformer.resblocks[i].parameters():
                param.requires_grad = True
        
        # 만약 <CLS> 토큰 관련이나, conv1 등을 같이 학습하려면 이 부분도 unfreeze
        # 예: visual.class_embedding.requires_grad = True
        #     visual.conv1.weight.requires_grad = True (optional)
        
    def forward(self, x):
        """
        Forward 시, 마지막 2개 블록만 gradient가 생기고,
        나머지는 gradient가 0인 상태(Freeze)로 남음.
        """
        # grad 계산 위해 with torch.no_grad():는 쓰지 않음
        visual = self.model.visual

        # conv1 -> ReLU -> flatten
        x = visual.conv1(x)                   # (B, embed_dim, H/32, W/32)
        x = F.relu(x)
        x = x.flatten(2).transpose(1,2)       # (B, num_patches, embed_dim)

        # CLS 토큰 붙이기
        cls_token = visual.class_embedding.unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, 1+patches, E)

        # Transformer 통과 (여기서 마지막 2개 블록만 학습 가능)
        x = visual.transformer(x)            # (B, 1+patches, E)

        cls_tokens = x[:, 0, :]              # (B, embed_dim)
        patch_tokens = x[:, 1:, :]           # (B, num_patches, embed_dim)
        return cls_tokens, patch_tokens
