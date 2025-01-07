# src/models/gat_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class FacePartGAT(nn.Module):
    def __init__(self, hidden_dim=768, gat_hidden=128, heads=4):
        super(FacePartGAT, self).__init__()
        self.gat1 = GATConv(hidden_dim, gat_hidden, heads=heads, concat=True)
        out_dim = gat_hidden * heads  # 128 * 4 = 512
        self.gat2 = GATConv(out_dim, gat_hidden, heads=1, concat=False)
        self.final_fc = nn.Linear(gat_hidden, hidden_dim)

    def _build_fc_edges(self, num_nodes, device):
        # 모든 노드를 연결하는 fully-connected 그래프를 GATConv에 전달
        src = []
        dst = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                src.append(i)
                dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        return edge_index

    def forward(self, node_feats):
        # GATConv를 2단으로 적용해 시각적 ROI 특징들 간의 관계를 학습
        device = node_feats.device
        num_nodes = node_feats.size(0)
        edge_index = self._build_fc_edges(num_nodes, device)

        x = self.gat1(node_feats, edge_index)  # (N+1, gat_hidden * heads) = (N+1, 512)
        x = F.elu(x)  # 활성화 함수 적용
        x = self.gat2(x, edge_index)  # (N+1, gat_hidden) = (N+1, 128)
        x = F.elu(x)  # 활성화 함수 적용

        # mean pooling
        x = x.mean(dim=0)  # 전체 노드에 대해 평균 pooling
        x = self.final_fc(x)  # (768)
        return x