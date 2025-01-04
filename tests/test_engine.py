# tests/test_engine.py

import pytest
import torch
from src.engine.trainer import train_one_epoch
from src.engine.evaluator import evaluate

def test_train_and_eval():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(100,2)
        def forward(self, frames, bboxes):
            B = frames.size(0)
            x = torch.randn(B,100)
            return self.fc(x)

    class FakeLoader:
        def __init__(self):
            self.data = [
                {
                    "frames": torch.randn(2,32,3,224,224),
                    "bboxes": torch.zeros(2,32,6,4),
                    "label": torch.tensor([0,1])
                }
            ]
        def __len__(self):
            return 1
        def __iter__(self):
            yield self.data[0]

    model = FakeModel()
    loader = FakeLoader()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cpu")

    train_loss, train_acc, train_auc = train_one_epoch(model, loader, criterion, optimizer, device)
    val_loss, val_acc, val_auc = evaluate(model, loader, criterion, device, mode="val")

    assert train_loss >= 0.0
    assert val_loss >= 0.0
