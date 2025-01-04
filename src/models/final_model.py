# src/models/final_model.py

import torch
import torch.nn as nn
from .roi_vit_extractor import ROIViTExtractor
from .gat_classifier import FacePartGAT

class FullPipelineModel(nn.Module):
    """
    (B,32,C,H,W), (B,32,N,4) -> loop 프레임별 ViT+ROI -> GAT -> pooling -> 분류
    """
    def __init__(self, image_size=224, patch_size=16, hidden_dim=768, num_classes=2):
        super().__init__()
        self.roi_extractor = ROIViTExtractor(image_size, patch_size, hidden_dim)
        self.gat = FacePartGAT(hidden_dim, gat_hidden=128, heads=4)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, frames, bboxes):
        """
        frames: (B,32,C,H,W)
        bboxes: (B,32,N,4)
        Return: (B,num_classes)
        """
        B, T, C, H, W = frames.shape
        _, _, N, _ = bboxes.shape

        out_logits = []
        for b_idx in range(B):
            frame_logit_list = []
            for t_idx in range(T):
                # 1) 한 프레임
                frame_t = frames[b_idx,t_idx]    # (C,H,W)
                boxes_t = bboxes[b_idx,t_idx]    # (N,4)

                # 2) ROI Extract -> (N+1, hidden_dim)
                node_feats = self.roi_extractor(frame_t, boxes_t)

                # 3) GAT -> (hidden_dim)
                frame_emb = self.gat(node_feats)

                frame_logit_list.append(frame_emb)

            # 4) T=32프레임 -> 평균 or 다른 pooling
            video_emb = torch.stack(frame_logit_list, dim=0).mean(dim=0)  # (hidden_dim)

            # 5) 분류
            logit = self.classifier(video_emb)  # (num_classes)
            out_logits.append(logit)

        out_logits = torch.stack(out_logits, dim=0)  # (B,num_classes)
        return out_logits
