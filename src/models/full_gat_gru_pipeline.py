# src/models/full_gat_gru_pipeline.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.roi_vit_extractor import ROIViTExtractor
from src.models.gat_classifier import FacePartGAT

class FullGATGRUPipelineModel(nn.Module):
    """
    1) 프레임별 (N개 ROI 임베딩) -> GAT -> (프레임 임베딩)
    2) 프레임 임베딩 시퀀스 -> GRU -> 최종 비디오 임베딩
    3) MLP 분류 -> (B, num_classes)
    """
    def __init__(
        self,
        model_name="ViT-B/32",
        device="cuda",
        image_size=224,
        patch_size=32,
        hidden_dim=768,
        gat_hidden=128,
        gat_heads=2,
        gat_dropout=0.1,
        gru_hidden_dim=512,
        gru_dropout=0.1,
        num_classes=2
    ):
        super().__init__()
        self.device = device
        
        # (A) ROI Extractor (ViT + etc.)
        self.roi_extractor = ROIViTExtractor(
            model_name=model_name,
            device=device,
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim
        )

        # (B) GAT for frame-level ROI aggregation
        self.gat = FacePartGAT(
            hidden_dim=hidden_dim,
            gat_hidden=gat_hidden,
            heads=gat_heads,
            dropout=gat_dropout
        )

        # (C) GRU for temporal (frame) sequence
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True
        )
        self.dropout = nn.Dropout(gru_dropout)

        # (D) MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=gru_dropout),
            nn.Linear(gru_hidden_dim // 2, num_classes)
        )

    def forward(self, frames: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames:  (B, T, 3, H, W)
            bboxes:  (B, T, N, 4)
        Returns:
            logits:  (B, num_classes)
        """
        B, T, _, _, _ = frames.shape
        # bboxes shape: (B, T, N, 4)
        _, _, N, _ = bboxes.shape

        # 1) ROI 임베딩 생성 (=> shape: (B, T, N, hidden_dim))
        all_roi_cls_list = []
        for b_idx in range(B):
            frames_roi_list = []
            for t_idx in range(T):
                frame_t = frames[b_idx, t_idx]    # (3,H,W)
                bbox_t  = bboxes[b_idx, t_idx]    # (N,4)

                roi_feats_t = self.roi_extractor(frame_t, bbox_t)  # (N, hidden_dim)
                frames_roi_list.append(roi_feats_t.unsqueeze(0))   # (1, N, E)
            # (T, N, E)
            frames_roi_stack = torch.cat(frames_roi_list, dim=0)
            all_roi_cls_list.append(frames_roi_stack.unsqueeze(0)) # (1, T, N, E)

        # => (B, T, N, E)
        all_roi_cls_batch = torch.cat(all_roi_cls_list, dim=0).to(self.device)

        # 2) 프레임 단위로 GAT -> (B, T, E)
        frame_embs = []
        for b_idx in range(B):
            frame_emb_list = []
            for t_idx in range(T):
                roi_feats = all_roi_cls_batch[b_idx, t_idx]  # (N, E)
                # GAT => (E,)
                frame_emb = self.gat(roi_feats)
                frame_emb_list.append(frame_emb.unsqueeze(0))  # (1, E)

            # (T, E)
            frame_emb_tensor = torch.cat(frame_emb_list, dim=0)
            frame_embs.append(frame_emb_tensor.unsqueeze(0))  # (1, T, E)

        # => (B, T, E)
        frame_embs = torch.cat(frame_embs, dim=0)

        # 3) GRU 시퀀스 처리
        gru_out, _ = self.gru(frame_embs)  # (B, T, gru_hidden_dim)
        gru_out = self.dropout(gru_out)
        last_hidden = gru_out[:, -1, :]    # (B, gru_hidden_dim)

        # 4) MLP 분류
        logits = self.mlp(last_hidden)     # (B, num_classes)
        return logits
