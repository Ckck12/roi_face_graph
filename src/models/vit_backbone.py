# src/models/vit_backbone.py (부분 레이어만 재학습 예시)

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class CLIPViTBackbone(nn.Module):
    """
    CLIP의 ViT 백본에서, 마지막 2개 Transformer 블록 + 분류 헤드만 재학습하고
    나머지는 Freeze하는 버전.
    """
    def __init__(self, model_name="ViT-B/32", device='cuda', num_finetune_blocks=1):
        super(CLIPViTBackbone, self).__init__()
        self.device = device

        # CLIP 모델 로드
        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.model.eval()  # 기본적으로 eval()

        # (1) 전체 파라미터를 먼저 requires_grad=False
        for param in self.model.parameters():
            param.requires_grad = False

        # (2) float32 변환
        self.model = self.model.float()
        for param in self.model.parameters():
            param.data = param.data.float()
        for buffer in self.model.buffers():
            buffer.data = buffer.data.float()

        # (3) 특정 레이어만 Unfreeze:
        #     예) visual.transformer.resblocks => 여러 개 중 마지막 num_finetune_blocks개만
        visual = self.model.visual
        total_blocks = len(visual.transformer.resblocks)  # 예: 12
        start_idx = max(0, total_blocks - num_finetune_blocks)  # 마지막 2개 -> start_idx=10

        for i in range(start_idx, total_blocks):
            for param in visual.transformer.resblocks[i].parameters():
                param.requires_grad = True  # ★ 마지막 num_finetune_blocks만 학습

        # (추가) CLS 토큰이나 class_embedding 등 특정 부분도 학습할지 여부
        # 이 예시는 block 부분만 업데이트. 필요하면 아래 부분도 unfreeze
        # visual.class_embedding.requires_grad = True (optional)

    def forward(self, x):
        """
        마지막 몇 개 Transformer 블록만 업데이트되며,
        나머지는 Freeze 상태로 forward.
        """
        # gradient는 unfreeze된 레이어에서만 발생
        # with torch.no_grad()는 쓰지 않음 -> forward 시 gradient 계산
        visual = self.model.visual

        x = visual.conv1(x)
        x = F.relu(x)
        x = x.flatten(2).transpose(1,2)

        cls_token = visual.class_embedding.unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, 1 + patches, E)

        # Transformer 통과
        # (Freeze된 블록은 grad가 0이지만, 마지막 n개 블록은 grad 발생)
        x = visual.transformer(x)

        cls_tokens = x[:,0,:]
        patch_tokens = x[:,1:,:]
        return cls_tokens, patch_tokens
