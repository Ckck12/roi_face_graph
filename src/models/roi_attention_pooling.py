import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ROIAttentionPooling(nn.Module):
    """
    단일 ROI에 대해, 여러 프레임(t=1..T)에서 나온 노드 임베딩(T, hidden_dim)을
    self-attention 기반으로 하나의 벡터(hidden_dim)로 Pooling하는 모듈.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # 예시로 Q, K, V를 동일 linear로 구축
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, roi_nodes):
        """
        roi_nodes: (T, hidden_dim)
        반환: (hidden_dim,)
        """
        # (T, hidden_dim) -> (T, 1, hidden_dim) 배치=1로 가정
        x = roi_nodes.unsqueeze(1)  # (T,1,H)
        Q = self.W_q(x)  # (T,1,H)
        K = self.W_k(x)  # (T,1,H)
        V = self.W_v(x)  # (T,1,H)

        # shape 변환: (T,1,H) -> (T,1, num_heads=1, head_dim=H) 라든지...
        # 여기선 간단히 einsum로 계산
        dim = Q.size(-1)
        # 어텐션 스코어: (T,1,H) x (T,1,H)^T => (T,T)
        #   다만 배치축 고려하면 조금 복잡, 여기서는 배치=1 가정
        #   Q: (T,1,H),  K: (T,1,H)
        #   matmul 시 (T,1,H) x (T,H,1) => (T,T)
        #   여기서는 einsum으로 간단히
        attn_scores = torch.einsum("tbd,sbd->ts", Q, K) / math.sqrt(dim)  # (T,T)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (T,T)

        # attn_output = (T,T) x (T,1,H) => (T,1,H)
        # 단, einsum("ts,sbd->tbd")로 계산
        attn_output = torch.einsum("ts,sbd->tbd", attn_weights, V)  # (T,1,H)

        # T개의 hidden을 평균 내거나, 첫 번째 항만 가져올 수도 있음
        # 여기선 간단히 모든 T개를 mean
        # => (T,1,H) -> (T,H)
        attn_output = attn_output.squeeze(1)  # (T,H)
        out = attn_output.mean(dim=0)         # (H,) 최종
        return out
