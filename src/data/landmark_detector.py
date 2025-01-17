# src/data/landmark_detector.py
import dlib
import cv2
import numpy as np

class Landmark68Detector:
    """
    목적:
    - dlib 기반 얼굴 검출 후 68개 랜드마크 좌표를 예측
    데이터 shape:
    - 입력: (H, W, 3) RGB 이미지
    - 출력: (68, 2) 랜드마크 좌표
    """

    def __init__(self, shape_predictor_path: str):
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(shape_predictor_path)
        except RuntimeError as e:
            raise FileNotFoundError(f"[에러] 랜드마크 모델 파일을 찾을 수 없습니다: {shape_predictor_path}") from e

    def detect_landmarks(self, image_array):
        """
        Args:
            image_array: (H, W, 3) RGB
        Returns:
            (68, 2) or None
        """
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray, 1)
        if len(faces) == 0:
            # 얼굴 없음
            return None

        # 첫 번째 얼굴만 처리
        face = faces[0]
        shape = self.predictor(gray, face)
        coords = []
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            coords.append((x,y))
        coords = np.array(coords, dtype=np.float32)
        return coords
