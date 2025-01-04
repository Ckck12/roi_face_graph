# src/data/landmark_detector.py
import dlib
import cv2
import numpy as np

class Landmark68Detector:
    """
    dlib의 frontal_face_detector + shape_predictor_68_face_landmarks.dat를 이용해
    68개 랜드마크를 검출한다.
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
            image_array: (H,W,3) numpy (RGB)
        Returns:
            list of (68,2) => 68개 (x,y)
            한 프레임에 여러 얼굴이 있으면, 첫 번째 얼굴만 사용
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
