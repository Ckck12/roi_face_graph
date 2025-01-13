# src/models/final_model.py

import torch
import torch.nn as nn
from src.models.roi_vit_extractor import ROIViTExtractor
from src.models.gat_classifier import FacePartGAT

class FullPipelineModel(nn.Module):
    def __init__(self, model_name="ViT-B/32", device='cuda', image_size=224, patch_size=32, hidden_dim=768, num_classes=2):
        super(FullPipelineModel, self).__init__()
        self.roi_extractor = ROIViTExtractor(model_name=model_name, device=device, image_size=image_size, patch_size=patch_size, hidden_dim=hidden_dim)
        self.gat = FacePartGAT(hidden_dim=hidden_dim, gat_hidden=128, heads=4)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, frames, bboxes):
        """
        Args:
            frames: (B, 32, 3, 224, 224)
            bboxes: (B, 32, N, 4)
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = frames.shape
        _, _, N, _ = bboxes.shape

        # 각 배치(B)에 대해 모든 프레임(T)에 대해 ROI 추출 -> GAT으로 관계 학습
        out_logits = []
        for b_idx in range(B):
            frame_emb_list = []
            for t_idx in range(T):
                frame_t = frames[b_idx, t_idx]  # (3, 224, 224)
                boxes_t = bboxes[b_idx, t_idx]  # (N, 4)

                # 1) ROI 추출(패치 기반)
                # 2) GAT 통해 노드 임베딩을 통합
                # ROI Extract -> (N+1, hidden_dim)
                node_feats = self.roi_extractor(frame_t, boxes_t)  # (N+1, hidden_dim=768)

                # GAT -> (hidden_dim=768)
                frame_emb = self.gat(node_feats)  # (768)
                frame_emb_list.append(frame_emb)

            # 모든 프레임 임베딩을 평균 내어 하나의 비디오 임베딩 생성
            # 16 프레임의 임베딩을 평균하여 비디오 임베딩 생성
            video_emb = torch.stack(frame_emb_list, dim=0).mean(dim=0)  # (768)

            # 분류
            logit = self.classifier(video_emb)  # (num_classes)
            out_logits.append(logit)

        out_logits = torch.stack(out_logits, dim=0)  # (B, num_classes)
        return out_logits  # 최종 로짓(클래스별 점수)
