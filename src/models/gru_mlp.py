# src/models/gru_mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUMLPClassifier(nn.Module):
    """
    GRU와 MLP 기반의 비디오 수준 분류기
    """
    def __init__(self, hidden_dim=768, gru_hidden_dim=512, num_classes=2):
        super(GRUMLPClassifier, self).__init__()
        # GRU for temporal processing
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=gru_hidden_dim, batch_first=True)

        # MLP layers for classification
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(gru_hidden_dim//2),
            nn.Dropout(p=0.5),
            nn.Linear(gru_hidden_dim//2, gru_hidden_dim//4),
            nn.ReLU(),
            nn.BatchNorm1d(gru_hidden_dim//4),
            nn.Dropout(p=0.5),
            nn.Linear(gru_hidden_dim//4, num_classes)
        )
        
    def forward(self, all_roi_cls_batch):
        """
        Args:
            all_roi_cls_batch: (B, T, N, hidden_dim) tensor
              - B: batch size
              - T: number of frames
              - N: number of ROIs (마지막 ROI는 whole_image에 해당)
              - hidden_dim: 각 ROI 임베딩의 차원
        Returns:
            logits: (B, num_classes)
        """
        B, T, N, hidden_dim = all_roi_cls_batch.shape

        # 마지막 ROI (whole_image CLS)를 추출
        whole_image_cls = all_roi_cls_batch[:, :, -1, :]  # (B, T, hidden_dim)

        # GRU를 통해 시퀀스 처리
        gru_out, _ = self.gru(whole_image_cls)  # (B, T, gru_hidden_dim)

        # 시퀀스의 마지막 hidden state를 가져옴
        last_hidden_state = gru_out[:, -1, :]  # (B, gru_hidden_dim)

        # MLP를 통해 분류 수행
        logits = self.mlp(last_hidden_state)  # (B, num_classes)
        return logits
