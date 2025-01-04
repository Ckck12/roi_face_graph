# src/data/frame_extractor.py
import cv2
import numpy as np
from PIL import Image

def extract_32_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"[에러] 비디오 파일을 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("[에러] 비디오에 유효한 프레임이 없습니다.")

    # 32프레임 균등 간격 추출
    interval = total_frames / 32.0
    indices = [int(round(i * interval)) for i in range(32)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            # 실패하면 검은 화면 대체
            frame = np.zeros((224,224,3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        frames.append(img)

    cap.release()
    return frames
