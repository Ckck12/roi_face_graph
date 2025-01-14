# src/models/full_gru_pipeline.py

import torch
import torch.nn as nn
from src.models.roi_vit_extractor import ROIViTExtractor
from src.models.gru_mlp import GRUMLPClassifier

class FullGRUPipelineModel(nn.Module):
    """
    ROI 추출기와 GRU 기반 MLP 분류기를 통합한 파이프라인 모델
    """
    def __init__(
        self,
        model_name="ViT-B/32",
        device="cuda",
        image_size=224,
        patch_size=32,
        hidden_dim=768,
        gru_hidden_dim=256,
        num_classes=2
    ):
        super(FullGRUPipelineModel, self).__init__()
        self.device = device

        # ROI 추출 모델
        self.roi_extractor = ROIViTExtractor(
            model_name=model_name,
            device=device,
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim
        )

        # GRU + MLP 기반 분류기
        self.classifier = GRUMLPClassifier(
            hidden_dim=hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            num_classes=num_classes
        )

    def forward(self, frames, bboxes):
        """
        Args:
            frames: (B, T, 3, H, W) 입력 이미지 시퀀스
            bboxes: (B, T, N, 4) 각 프레임의 ROI 바운딩 박스
        Returns:
            logits: (B, num_classes) 분류 결과
        """
        B, T, _, _, _ = frames.shape
        _, _, N, _ = bboxes.shape

        # ROI 임베딩 생성
        batch_all_roi_cls = []
        for b_idx in range(B):
            all_roi_cls = []
            for t_idx in range(T):
                frame_t = frames[b_idx, t_idx]  # (3, H, W)
                bbox_t = bboxes[b_idx, t_idx]  # (N, 4)
                roi_cls = self.roi_extractor(frame_t, bbox_t)  # (N, hidden_dim)
                all_roi_cls.append(roi_cls.unsqueeze(0))  # (1, N, hidden_dim)

            # 프레임별 ROI 임베딩을 결합
            batch_all_roi_cls.append(torch.cat(all_roi_cls, dim=0).unsqueeze(0))  # (1, T, N, hidden_dim)

        # 배치별 데이터 결합
        batch_all_roi_cls = torch.cat(batch_all_roi_cls, dim=0).to(self.device)  # (B, T, N, hidden_dim)

        # GRU 기반 분류기 통과
        logits = self.classifier(batch_all_roi_cls)  # (B, num_classes)

        return logits


# 이 최종모델의 파라미터수 계산
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
