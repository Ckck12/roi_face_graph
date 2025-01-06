# src/models/final_model.py

import torch
import torch.nn as nn
from src.models.roi_vit_extractor import ROIViTExtractor
from src.models.gat_classifier import FacePartGAT

class FullPipelineModel(nn.Module):
    """
    (B, 16, 3, 224, 224), (B, 16, N, 4) -> loop 프레임별 ViT+ROI -> GAT -> pooling -> 분류
    """
    def __init__(self, model_name="ViT-B/32", device='cuda', image_size=224, patch_size=32, hidden_dim=512, num_classes=2):
        super(FullPipelineModel, self).__init__()
        self.roi_extractor = ROIViTExtractor(model_name=model_name, device=device, image_size=image_size, patch_size=patch_size, hidden_dim=hidden_dim)
        self.gat = FacePartGAT(hidden_dim=hidden_dim, gat_hidden=128, heads=4)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, frames, bboxes):
        """
        Args:
            frames: (B, 16, 3, 224, 224)
            bboxes: (B, 16, N, 4)
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = frames.shape
        _, _, N, _ = bboxes.shape

        out_logits = []
        for b_idx in range(B):
            frame_emb_list = []
            for t_idx in range(T):
                frame_t = frames[b_idx, t_idx]  # (3, 224, 224)
                boxes_t = bboxes[b_idx, t_idx]  # (N, 4)

                # ROI Extract -> (N+1, hidden_dim)
                node_feats = self.roi_extractor(frame_t, boxes_t)  # (N+1, hidden_dim)

                # GAT -> (hidden_dim)
                frame_emb = self.gat(node_feats)  # (hidden_dim)
                frame_emb_list.append(frame_emb)

            # 16 프레임의 임베딩을 평균하여 비디오 임베딩 생성
            video_emb = torch.stack(frame_emb_list, dim=0).mean(dim=0)  # (hidden_dim)

            # 분류
            logit = self.classifier(video_emb)  # (num_classes)
            out_logits.append(logit)

        out_logits = torch.stack(out_logits, dim=0)  # (B, num_classes)
        return out_logits
