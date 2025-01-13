# src/models/final_model_temporal_gat.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from src.models.roi_vit_extractor import ROIViTExtractor

##############################################################################
# 1) GRU를 이용한 Temporal Aggregation
##############################################################################
class GRUTemporalAggregator(nn.Module):
    """
    각 ROI마다 T개의 임베딩을 입력받아 GRU를 통해 하나의 벡터로 요약.
    """
    def __init__(self, input_dim=768, hidden_dim=768, num_layers=1, bidirectional=False):
        super(GRUTemporalAggregator, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,    # (B, T, input_dim)
            bidirectional=bidirectional
        )
        # bidirectional이면 hidden_dim이 2배
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, seq_embs):
        """
        Args:
            seq_embs: (B, T, input_dim) 또는 (T, input_dim)
        Returns:
            (B, output_dim) 또는 (output_dim,)
        """
        if seq_embs.dim() == 2:
            # (T, input_dim) -> (1, T, input_dim)
            seq_embs = seq_embs.unsqueeze(0)
            batch_first = True
        else:
            batch_first = True  # 이미 (B, T, input_dim)

        out, h_n = self.gru(seq_embs)  # out: (B, T, hidden_dim * num_directions)
                                       # h_n: (num_layers * num_directions, B, hidden_dim)
        # 최종 hidden state 가져오기
        if self.gru.bidirectional:
            # 양방향이면 마지막 두 hidden state를 concat
            h_n = h_n.view(self.gru.num_layers, 2, -1, self.gru.hidden_size)  # (num_layers, 2, B, hidden_dim)
            last_h = torch.cat((h_n[-1,0], h_n[-1,1]), dim=-1)  # (B, hidden_dim * 2)
        else:
            last_h = h_n[-1]  # (B, hidden_dim)

        if seq_embs.dim() == 2:
            # (1, B, output_dim) -> (output_dim,)
            last_h = last_h.squeeze(0)

        return last_h  # (B, output_dim) 또는 (output_dim,)

##############################################################################
# 2) GAT를 이용한 ROI 관계 학습
##############################################################################
class FacePartGAT(nn.Module):
    """
    ROI들(N개) 간 fully-connected 그래프에 GATConv 2단 적용.
    """
    def __init__(self, hidden_dim=768, gat_hidden=128, heads=4):
        super(FacePartGAT, self).__init__()
        self.gat1 = GATConv(hidden_dim, gat_hidden, heads=heads, concat=True)
        out_dim = gat_hidden * heads
        self.gat2 = GATConv(out_dim, gat_hidden, heads=1, concat=False)
        self.final_fc = nn.Linear(gat_hidden, hidden_dim)

    def forward(self, roi_feats):
        """
        Args:
            roi_feats: (N, hidden_dim)
        Returns:
            (hidden_dim,)
        """
        N = roi_feats.size(0)
        device = roi_feats.device

        # 1) ROI 간 fully-connected 엣지 구성
        src = []
        dst = []
        for i in range(N):
            for j in range(N):
                src.append(i)
                dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)

        # 2) GATConv 1
        x = self.gat1(roi_feats, edge_index)   # (N, gat_hidden * heads)
        x = F.elu(x)

        # 3) GATConv 2
        x = self.gat2(x, edge_index)           # (N, gat_hidden)
        x = F.elu(x)

        # 4) Mean Pooling
        x = x.mean(dim=0)                      # (gat_hidden,)

        # 5) Final Linear Layer
        x = self.final_fc(x)                   # (hidden_dim,)
        return x

##############################################################################
# 3) 전체 모델: GRU + GAT + Classifier
##############################################################################
class FullTemporalGraphModel(nn.Module):
    """
    1) 각 프레임별 ROI 임베딩을 GRU로 시계열 정보 요약
    2) 요약된 ROI 임베딩을 GAT로 관계 학습
    3) 최종적으로 이진 분류 수행
    """
    def __init__(
        self,
        model_name="ViT-B/32",
        device="cuda",
        image_size=224,
        patch_size=32,
        hidden_dim=768,
        num_classes=2,
        gru_hidden=768,
        gat_hidden=128,
        num_gru_layers=1,
        bidirectional=False
    ):
        super(FullTemporalGraphModel, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 1) ROI 추출기 (부분 finetuning된 ViT 백본 포함)
        self.roi_extractor = ROIViTExtractor(
            model_name=model_name,
            device=device,
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim
        )

        # 2) Temporal Aggregator (GRU)
        self.gru_agg = GRUTemporalAggregator(
            input_dim=hidden_dim,
            hidden_dim=gru_hidden,
            num_layers=num_gru_layers,
            bidirectional=bidirectional
        )
        self.temporal_out_dim = self.gru_agg.output_dim  # (gru_hidden * num_directions)

        # 3) GAT (ROI 관계 학습)
        self.gat = FacePartGAT(
            hidden_dim=self.temporal_out_dim,
            gat_hidden=gat_hidden,
            heads=4
        )

        # 4) 최종 분류기
        self.classifier = nn.Linear(self.temporal_out_dim, num_classes)

    def _build_roi_connections(self, N):
        """
        ROI 간의 연결을 fully-connected로 설정
        Args:
            N (int): ROI의 수
        Returns:
            edge_index (Tensor): (2, N*N)
        """
        src = []
        dst = []
        for i in range(N):
            for j in range(N):
                src.append(i)
                dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        return edge_index

    def forward(self, frames, bboxes):
        """
        Args:
            frames: (B, T, 3, H, W)
            bboxes: (B, T, N, 4)
        Returns:
            logits: (B, num_classes)
        """
        B, T, _, _, _ = frames.shape
        _, _, N, _ = bboxes.shape

        # 1) ROI 임베딩 추출
        # node_feats_list: list of length T, each element is (B, N, hidden_dim)
        node_feats_list = []
        for t in range(T):
            frame_t = frames[:, t, :, :, :]     # (B, 3, H, W)
            bbox_t = bboxes[:, t, :, :]        # (B, N, 4)
            roi_feats_t = self.roi_extractor(frame_t, bbox_t)  # (B, N, hidden_dim)
            node_feats_list.append(roi_feats_t)

        # 2) ROI별로 GRU를 통해 시계열 정보 요약
        # node_feats_list: list of (B, N, hidden_dim) -> stack to (B, T, N, hidden_dim)
        node_feats_tensor = torch.stack(node_feats_list, dim=1)  # (B, T, N, hidden_dim)

        # Reshape to (B*N, T, hidden_dim) to process each ROI across the batch
        node_feats_reshaped = node_feats_tensor.permute(0, 2, 1, 3).contiguous().view(B*N, T, self.hidden_dim)  # (B*N, T, hidden_dim)

        # Pass through GRU
        gru_out = self.gru_agg(node_feats_reshaped)  # (B*N, temporal_out_dim)

        # Reshape back to (B, N, temporal_out_dim)
        gru_out = gru_out.view(B, N, self.temporal_out_dim)  # (B, N, temporal_out_dim)

        # 3) GAT을 이용한 ROI 관계 학습
        # GAT은 각 샘플(B)마다 별도로 처리
        gat_out_list = []
        for b in range(B):
            roi_feats = gru_out[b]  # (N, temporal_out_dim)
            gat_out = self.gat(roi_feats)  # (hidden_dim)
            gat_out_list.append(gat_out)

        # Stack gat_out_list to (B, hidden_dim)
        gat_out_tensor = torch.stack(gat_out_list, dim=0)  # (B, hidden_dim)

        # 4) 최종 분류
        logits = self.classifier(gat_out_tensor)  # (B, num_classes)
        return logits
