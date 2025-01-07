# src/models/vit_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class CLIPViTBackbone(nn.Module):
    """
    CLIP의 사전 학습된 ViT 백본을 사용하는 클래스.
    입력 이미지 패치를 임베딩하여 시퀀스 형태의 임베딩을 반환합니다.
    
    출력:
        - CLS 토큰: (B, d_v)
        - 패치 토큰: (B, H'*W', d_v)
    """
    def __init__(self, model_name="ViT-B/32", device='cuda'):
        super(CLIPViTBackbone, self).__init__()
        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.model.eval()  # 백본은 고정
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
        Args:
            x: (B, 3, H, W)
        Returns:
            cls_tokens: (B, d_v)
            patch_tokens: (B, num_patches, d_v)
        """
        with torch.no_grad():
            # CLIP 모델의 visual module을 통해 이미지 임베딩을 추출
            visual = self.model.visual
            x = visual.conv1(x)  # (B, embed_dim, H/32, W/32)
            x = F.relu(x)        # ReLU 활성화 함수 적용
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
            cls_token = visual.class_embedding.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, 1, embed_dim)
            x = torch.cat((cls_token, x), dim=1)  # (B, 1 + num_patches, embed_dim)
            
            x = visual.transformer(x)  # (B, 1 + num_patches, embed_dim)
            
            cls_tokens = x[:, 0, :]    # (B, embed_dim)
            patch_tokens = x[:, 1:, :]  # (B, num_patches, embed_dim)
            
        print(cls_tokens.shape, patch_tokens.shape)
        return cls_tokens, patch_tokens
