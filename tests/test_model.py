# tests/test_model.py

import pytest
import torch
from src.models.final_model import FullPipelineModel

def test_full_pipeline_model():
    model = FullPipelineModel(image_size=224, patch_size=16, hidden_dim=128, num_classes=2)
    frames = torch.randn(2,32,3,224,224)  # B=2, T=32
    bboxes = torch.zeros(2,32,6,4)       # 6개 ROI
    outputs = model(frames, bboxes)
    assert outputs.shape == (2,2)
