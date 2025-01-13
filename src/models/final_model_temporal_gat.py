# final_model_temporal_gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.models.roi_vit_extractor import ROIViTExtractor

##############################################################################
# 1) 새로 추가할 TemporalGraphGAT (단일 시공간 그래프용 GAT)
##############################################################################
class TemporalGraphGAT(nn.Module):
    """
    단일 시공간(temporal) 그래프에 대해:
      - 2-layer GATConv
      - mean pooling -> 최종 임베딩
    """
    def __init__(self, in_dim=768, gat_hidden=128, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, gat_hidden, heads=heads, concat=True)
        out_dim = gat_hidden * heads
        self.gat2 = GATConv(out_dim, gat_hidden, heads=1, concat=False)
        self.final_fc = nn.Linear(gat_hidden, in_dim)

    def forward(self, x, edge_index):
        # x: (num_nodes, in_dim)
        # edge_index: (2, num_edges)
        # 1) GATConv 1
        x = self.gat1(x, edge_index)    # => (num_nodes, gat_hidden * heads)
        x = F.elu(x)
        # 2) GATConv 2
        x = self.gat2(x, edge_index)    # => (num_nodes, gat_hidden)
        x = F.elu(x)
        # 3) mean pooling => single 임베딩
        x = x.mean(dim=0)               # => shape (gat_hidden,)
        x = self.final_fc(x)            # => (in_dim,)
        return x


##############################################################################
# 2) 시공간 그래프 모델: 모든 프레임×ROI를 하나의 그래프로 묶어 처리
##############################################################################
class FullTemporalGraphModel(nn.Module):
    """
    기존 FullPipelineModel와 달리, 모든 프레임×ROI 노드를 하나의 그래프로 묶어
    GATConv(시공간 엣지)를 한 번에 수행하는 방식.

    가정:
      - roi_extractor(frame_t, boxes_t): (N, hidden_dim) 반환
         (N개 ROI, 마지막 idx가 whole_face라고 가정)
      - 한 배치에서 (B,T,3,H,W), (B,T,N,4) 형태
      - 각 이미지(배치=1개 단위)에 대해 T*N개의 노드를 구성

    절차:
      1) (T회 반복) ROI 임베딩 구함 -> node_feat_list (길이 T)
      2) node_feats = cat(...) => shape: (T*N, hidden_dim)
      3) edge_index 구성
         - same_frame_fc_edges: 프레임 내 모든 ROI fully connected
         - adjacent_frame_edges: 인접 프레임 동일 ROI index 연결
      4) TemporalGraphGAT에 통과
      5) 최종 fc(=classifier) -> (num_classes)
    """
    def __init__(
        self,
        model_name="ViT-B/32",
        device="cuda",
        image_size=224,
        patch_size=32,
        hidden_dim=768,
        num_classes=2
    ):
        super().__init__()
        # (1) ROI 추출 (Overlap ratio + patch)
        self.roi_extractor = ROIViTExtractor(
            model_name=model_name,
            device=device,
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim
        )
        # (2) 한 번에 시공간 그래프 GAT
        self.temporal_gat = TemporalGraphGAT(
            in_dim=hidden_dim,
            gat_hidden=128,
            heads=4
        )
        # (3) 최종 classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.device = device
        self.hidden_dim = hidden_dim

    def _build_temporal_edges(self, T: int, N: int):
        """
        프레임 내 fully-connected + 인접 프레임간 동일 ROI 연결
        => return edge_index (shape: (2, num_edges))

        노드 번호 매핑:
          t번째 프레임의 i번째 ROI => 노드 index = t*N + i
          i \in [0..N-1]
        """
        src_list, dst_list = [], []

        # 1) 프레임 내 fully-connected
        for t in range(T):
            base = t * N
            for i in range(N):
                for j in range(N):
                    src_list.append(base + i)
                    dst_list.append(base + j)

        # 2) 인접 프레임 동일 ROI (양방향)
        for t in range(T - 1):
            base_t = t * N
            base_n = (t + 1) * N
            for r in range(N):
                # t프레임 r번 <-> t+1프레임 r번
                src_list.append(base_t + r)
                dst_list.append(base_n + r)
                src_list.append(base_n + r)
                dst_list.append(base_t + r)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        return edge_index

    def forward(self, frames, bboxes):
        """
        Args:
          frames: (B, T, 3, H, W)
          bboxes: (B, T, N, 4)
        Return:
          logits: (B, num_classes)
        """
        B, T, _, _, _ = frames.shape   # (B,T,3,H,W)
        _, _, N, _ = bboxes.shape      # (B,T,N,4)

        out_logits = []
        for b_idx in range(B):
            # (1) 한 영상(b_idx)에 대해 T개 프레임 => 각각 ROI 추출
            node_feat_list = []
            for t_idx in range(T):
                frame_t = frames[b_idx, t_idx]  # shape: (3,H,W)
                bbox_t  = bboxes[b_idx, t_idx]  # shape: (N,4)
                # (a) ROI 임베딩 => (N, hidden_dim)
                node_feats_t = self.roi_extractor(frame_t, bbox_t)
                node_feat_list.append(node_feats_t)

            # (2) 노드 연결
            # node_feats: (T*N, hidden_dim)
            node_feats = torch.cat(node_feat_list, dim=0).to(self.device)

            # (3) 엣지 구성
            edge_index = self._build_temporal_edges(T, N).to(self.device)

            # (4) 시공간 GAT
            graph_emb = self.temporal_gat(node_feats, edge_index)  # (hidden_dim)

            # (5) 최종 분류
            logit = self.classifier(graph_emb)  # (num_classes)
            out_logits.append(logit)

        # (B, num_classes)
        out_logits = torch.stack(out_logits, dim=0)
        return out_logits
