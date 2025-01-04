# src/models/gat_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class FacePartGAT(nn.Module):
    """
    (N+1, hidden_dim) 노드를 fully-connected 그래프로 GATConv
    """
    def __init__(self, hidden_dim=768, gat_hidden=128, heads=4):
        super().__init__()
        self.gat1 = GATConv(hidden_dim, gat_hidden, heads=heads, concat=True)
        out_dim = gat_hidden * heads
        self.gat2 = GATConv(out_dim, gat_hidden, heads=1, concat=False)
        self.final_fc = nn.Linear(gat_hidden, hidden_dim)

    def _build_fc_edges(self, num_nodes, device):
        src, dst = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                src.append(i)
                dst.append(j)
        edge_index = torch.tensor([src,dst], dtype=torch.long, device=device)
        return edge_index

    def forward(self, node_feats):
        """
        node_feats: (N+1, hidden_dim)
        return: (hidden_dim) => pooling 결과
        """
        device = node_feats.device
        num_nodes = node_feats.size(0)
        edge_index = self._build_fc_edges(num_nodes, device)

        x = self.gat1(node_feats, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # mean pooling
        x = x.mean(dim=0)          # (gat_hidden)
        x = self.final_fc(x)       # (hidden_dim)
        return x
