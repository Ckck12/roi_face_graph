# src/models/gat_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MultiHeadAttentionPooling(nn.Module):
    """
    노드 임베딩들을 입력받아 Multi-Head Attention으로 풀링(집약)하는 모듈.
    """
    def __init__(self, input_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Args:
            input_dim (int): 노드 임베딩 차원(= GATConv의 출력 채널)
            num_heads (int): 멀티헤드 어텐션의 헤드 수
            dropout (float): 어텐션 연산 시 적용할 드롭아웃 비율 (optional)
        """
        super(MultiHeadAttentionPooling, self).__init__()
        # PyTorch의 nn.MultiheadAttention 사용
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # 풀링 결과를 후처리할 FC
        self.fc = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape = (N, input_dim)
               그래프의 N개 노드 임베딩 (GATConv 결과 등)

        Returns:
            pooled: shape = (input_dim,)
               멀티헤드 어텐션으로 풀링된 그래프 임베딩
        """
        # nn.MultiheadAttention는 (batch, seq_len, embed_dim) 형태를 기대
        # => x를 (1, N, input_dim)로 변환 (batch=1, seq_len=N)
        x = x.unsqueeze(0)  # (1, N, input_dim)

        # Self-Attention: query=key=value=x
        attn_output, _ = self.attention(x, x, x)  # (1, N, input_dim)
        # 노드별 결과를 평균 풀링 -> 그래프 단위 임베딩
        pooled = attn_output.mean(dim=1)  # (1, input_dim)
        pooled = self.dropout(pooled)

        # FC로 후처리
        pooled = self.fc(pooled)         # (1, input_dim)
        pooled = self.dropout(pooled)

        # (input_dim,) 형태로 반환
        return pooled.squeeze(0)


class FacePartGAT(nn.Module):
    """
    1) GATConv를 2단으로 적용해 노드 임베딩 계산
    2) Multi-Head Attention Pooling으로 글로벌 풀링
    3) 최종 FC로 임베딩 차원 원상 복구 (hidden_dim)
    """
    def __init__(
        self,
        hidden_dim=768,
        gat_hidden=128,
        heads=4,
        dropout=0.0,
        pooling_heads=4
    ):
        """
        Args:
            hidden_dim (int): 입력 및 최종 임베딩 차원 (예: 768)
            gat_hidden (int): GATConv의 out_channels
            heads (int): GATConv의 multi-head 수
            dropout (float): GATConv와 MHA에 적용할 Dropout 비율
            pooling_heads (int): MultiHeadAttentionPooling에 사용할 head 수
        """
        super(FacePartGAT, self).__init__()

        # (1) GATConv 1
        self.gat1 = GATConv(
            in_channels=hidden_dim,
            out_channels=gat_hidden,
            heads=heads,
            concat=True,    # 출력 차원 = gat_hidden * heads
            dropout=dropout
        )
        out_dim = gat_hidden * heads

        # (2) GATConv 2
        self.gat2 = GATConv(
            in_channels=out_dim,
            out_channels=gat_hidden,
            heads=1,        # single-head
            concat=False,   # => shape=(N, gat_hidden)
            dropout=dropout
        )

        # (3) Multi-Head Attention Pooling (노드 임베딩들을 한번 더)
        self.pooling = MultiHeadAttentionPooling(
            input_dim=gat_hidden,
            num_heads=pooling_heads,
            dropout=dropout
        )

        # (4) 최종 FC: (gat_hidden) -> (hidden_dim)
        self.final_fc = nn.Linear(gat_hidden, hidden_dim)

        # 필요 시 추가 드롭아웃
        self.final_dropout = nn.Dropout(p=dropout)


    def _build_fc_edges(self, num_nodes: int, device) -> torch.Tensor:
        """
        모든 노드를 연결하는 fully-connected 그래프를 GATConv에 전달하기 위해
        edge_index를 구성. (2, N*N) 형태
        """
        src = []
        dst = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                src.append(i)
                dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        return edge_index


    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feats (Tensor): (N, hidden_dim) 노드 임베딩 (예: ROI별 임베딩)

        Returns:
            graph_emb (Tensor): (hidden_dim,) 그래프 임베딩 (ex: frame 임베딩)
        """
        device = node_feats.device
        num_nodes = node_feats.size(0)

        # 1) fully-connected edge
        edge_index = self._build_fc_edges(num_nodes, device)  # (2, N*N)

        # 2) GATConv 1
        x = self.gat1(node_feats, edge_index)  # => (N, gat_hidden * heads)
        x = F.elu(x)
        x = self.final_dropout(x)

        # 3) GATConv 2
        x = self.gat2(x, edge_index)          # => (N, gat_hidden)
        x = F.elu(x)
        x = self.final_dropout(x)

        # 4) Multi-Head Attention Pooling => (gat_hidden,)
        x = self.pooling(x)  # (gat_hidden,)

        # 5) 최종 FC => (hidden_dim,)
        x = self.final_fc(x)
        x = self.final_dropout(x)

        return x
