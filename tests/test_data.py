# tests/test_data.py

import os
import pytest
from src.data.dataset import FFPlusDataset

@pytest.mark.parametrize("dataset_type", ["train","val"])
def test_ffplus_dataset(dataset_type):
    csv_file = "./ff++_c23.csv"
    shape_predictor_path = "./shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(csv_file):
        pytest.skip("[건너뜀] ff++_c23.csv 파일이 없음.")
    if not os.path.exists(shape_predictor_path):
        pytest.skip("[건너뜀] shape_predictor_68_face_landmarks.dat 파일이 없음.")

    dataset = FFPlusDataset(
        csv_path=csv_file,
        shape_predictor_path=shape_predictor_path,
        dataset_type=dataset_type
    )
    if len(dataset) == 0:
        pytest.skip("[건너뜀] 해당 dataset_type에 대한 데이터가 없음.")

    sample = dataset[0]
    assert "frames" in sample
    assert "bboxes" in sample
    assert "label" in sample
