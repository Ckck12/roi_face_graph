# src/models/vit_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class CLIPViTBackbone(nn.Module):
    """
    CLIP의 ViT 백본에서, 마지막 N개의 Transformer 블록만 재학습하도록 설정하는 클래스.
    
    Args:
        model_name (str): CLIP 모델 이름 (예: "ViT-B/32")
        device (str): 학습/추론 디바이스 ("cuda" 또는 "cpu")
        num_finetune_blocks (int): 재학습할 마지막 Transformer 블록의 개수
    """
    def __init__(self, model_name="ViT-B/32", device='cuda', num_finetune_blocks=2):
        super(CLIPViTBackbone, self).__init__()
        self.device = device
        
        # (1) CLIP 모델 로드
        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.model.eval()  # 기본적으로 eval 모드
        # 전체 파라미터 Freeze
        for param in self.model.parameters():
            param.requires_grad = False

        # float32 변환
        self.model = self.model.float()
        for param in self.model.parameters():
            param.data = param.data.float()
        for buffer in self.model.buffers():
            buffer.data = buffer.data.float()

        # (2) 마지막 num_finetune_blocks개 Transformer 블록만 Unfreeze
        visual = self.model.visual
        transformer = visual.transformer
        total_blocks = len(transformer.resblocks)  # 예: ViT-B/32는 12블록
        start_idx = max(0, total_blocks - num_finetune_blocks)
        
        # 마지막 N개 블록만 requires_grad=True
        for i in range(start_idx, total_blocks):
            for param in transformer.resblocks[i].parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        입력 x: (B, C, H, W)
        출력:
            - cls_tokens: (B, d_v)
            - patch_tokens: (B, num_patches, d_v)
        """
        # (주의) 여기서는 일부 블록만 grad 계산, 나머지는 고정
        # grad 계산을 위해 with torch.no_grad():는 사용하지 않음
        visual = self.model.visual

        # conv1 -> ReLU -> flatten
        x = visual.conv1(x)       # (B, embed_dim, H/32, W/32)
        x = F.relu(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # CLS 토큰 붙이기
        cls_token = visual.class_embedding.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (B, 1 + num_patches, embed_dim)
        
        # Transformer 통과 (마지막 num_finetune_blocks 개는 학습 가능)
        x = visual.transformer(x)  # (B, 1 + num_patches, embed_dim)
        
        cls_tokens = x[:, 0, :]
        patch_tokens = x[:, 1:, :]
        return cls_tokens, patch_tokens
