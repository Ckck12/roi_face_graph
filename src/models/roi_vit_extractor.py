# src/models/roi_vit_extractor.py
import torch
import torch.nn as nn
from src.models.vit_backbone import SimpleViTBackbone
from src.models.roi_aggregation import compute_overlap_ratio


class ROIViTExtractor(nn.Module):
    """
    한 프레임 (C,H,W)을 ViT 백본에 통과 -> patch_tokens
    bboxes => Mi = Overlap/Area -> weighted sum
    """
    def __init__(self, image_size=224, patch_size=16, hidden_dim=768):
        super().__init__()
        self.vit = SimpleViTBackbone(image_size, patch_size, hidden_dim)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

    def _get_patch_coords(self):
        coords = []
        num_side = self.image_size // self.patch_size
        for r in range(num_side):
            for c in range(num_side):
                x1 = c*self.patch_size
                y1 = r*self.patch_size
                x2 = x1 + self.patch_size
                y2 = y1 + self.patch_size
                coords.append([x1,y1,x2,y2])
        return coords

    def forward(self, frame, bboxes):
        """
        frame: (C,H,W) 한 프레임
        bboxes: (N,4) => 왼눈, 오른눈, 코, 입, 머리, 전체 등
        return: (N+1, hidden_dim) => node_embs
               첫 번째 노드는 CLS, 그 뒤 N개는 ROI
        """
        # vit
        frame_in = frame.unsqueeze(0)  # (1,C,H,W)
        out_vit = self.vit(frame_in)   # (1,1+num_patches,hidden_dim)
        cls_token = out_vit[:,0,:]     # (1,hidden_dim)
        patch_tokens = out_vit[:,1:,:] # (1,num_patches,hidden_dim)

        # patch coords
        patch_coords = self._get_patch_coords()

        # node 0 => CLS
        node_embs = [cls_token.squeeze(0)]  # list of Tensors, shape( hidden_dim )

        # node 1..N => ROI
        patch_tokens_2d = patch_tokens.squeeze(0)  # (num_patches, hidden_dim)

        for bbox in bboxes:
            M_list = []
            for pc in patch_coords:
                ratio = compute_overlap_ratio(bbox, pc)
                M_list.append(ratio)
            M_t = torch.tensor(M_list, dtype=torch.float, device=patch_tokens_2d.device)
            weighted = patch_tokens_2d * M_t.unsqueeze(-1)  # (num_patches, hidden_dim)
            roi_feat = weighted.mean(dim=0)                 # (hidden_dim)
            node_embs.append(roi_feat)

        return torch.stack(node_embs, dim=0)  # (N+1, hidden_dim)
