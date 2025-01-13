# src/models/vit_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class CLIPViTBackbone(nn.Module):
    """
    CLIP의 사전 학습된 ViT 백본을 사용하는 클래스입니다.
    입력 이미지 패치를 임베딩하여 시퀀스 형태의 임베딩을 반환합니다.
    
    출력:
        - CLS 토큰: (B, d_v)
        - 패치 토큰: (B, num_patches, d_v)
    """
    def __init__(self, model_name="ViT-B/32", device='cuda'):
        super(CLIPViTBackbone, self).__init__()
        self.device = device  # 이 줄을 추가
        
        # CLIP 모델 로드
        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.model.eval() # 학습 모드로 전환
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 모델을 float32로 변환
        self.model = self.model.float()
        for param in self.model.parameters():
            param.data = param.data.float()
        for buffer in self.model.buffers():
            buffer.data = buffer.data.float()
                
    def forward(self, x):
        """
        ViT 백본을 통해 입력 이미지를 처리합니다.
        
        Args:
            x: (B, C, H, W) 텐서
        
        Returns:
            cls_tokens: (B, d_v) - 클래스 토큰
            patch_tokens: (B, num_patches, d_v) - 패치 토큰
        """
        with torch.no_grad():
            # CLIP 모델의 visual 모듈을 통해 이미지 임베딩 추출
            visual = self.model.visual
            x = visual.conv1(x)       # (B, embed_dim, H/32, W/32)
            x = F.relu(x)
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
            cls_token = visual.class_embedding.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, 1, embed_dim)
            x = torch.cat((cls_token, x), dim=1)  # (B, 1 + num_patches, embed_dim)
            
            x = visual.transformer(x)  # (B, 1 + num_patches, embed_dim)
            
            cls_tokens = x[:, 0, :]     # 클래스 토큰 (글로벌 특징)
            patch_tokens = x[:, 1:, :]  # 패치 토큰 (로컬 특징)
            
        return cls_tokens, patch_tokens
