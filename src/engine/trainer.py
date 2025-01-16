# src/engine/trainer.py

import os
import torch
import numpy as np
from tqdm import tqdm
from src.utils.metrics import compute_metrics  # 메트릭 계산 함수를 외부 모듈에서 임포트
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch=0,
    wandb=None
):
    # 모델을 학습 모드로 전환하여 BatchNorm, Dropout 등이 학습 단계로 동작하도록 설정
    model.train()  # 모델이 학습 모드로 동작하도록 설정, BatchNorm과 Dropout 등이 학습용으로 변경
    # 손실값 관리를 위한 누적 변수
    running_loss = 0.0  # 학습 루프 동안 발생하는 손실값을 누적해 평균 계산
    # 실제 라벨 및 예측 확률을 저장할 리스트
    y_true_list = []
    y_pred_probs_list = []

    # tqdm을 활용해 학습 과정을 시각적으로 모니터링
    for batch in tqdm(dataloader, desc=f"Train Epoch {epoch+1}"):
        # 배치에서 이미지와 박스, 레이블을 추출하여 GPU/CPU에 로드
        frames = batch["frames"].to(device)   # (B,32,C,H,W)
        bboxes = batch["bboxes"].to(device)   # (B,32,6,4)
        labels = batch["label"].to(device)    # (B,)

        # 역전파 단계를 위해 gradient를 0으로 초기화
        optimizer.zero_grad()
        # 모델 예측 수행
        outputs = model(frames, bboxes)  # 모델의 forward 실행, 프레임과 bbox 정보를 함께 입력
        # 손실 함수 계산
        loss = criterion(outputs, labels)
        # 역전파(gradient 계산)
        loss.backward()
        # 옵티마이저로 파라미터 업데이트
        optimizer.step()

        # 배치별 손실값을 합산 (배치 손실 * 배치 크기)
        running_loss += loss.item() * frames.size(0)
        # 예측 확률 중 두 번째 클래스 확률을 별도로 분리
        probs = torch.softmax(outputs, dim=1)[:,1].detach().cpu().numpy()  # 예측 확률(양성 클래스)만 추출
        # 실제 값과 예측 확률을 리스트에 저장
        y_true_list.extend(labels.cpu().numpy())
        y_pred_probs_list.extend(probs)

    # 배치 단위 평균 손실값 (전체 손실 / 전체 샘플 수)
    avg_loss = running_loss / len(dataloader.dataset)
    # 정확도와 AUC 계산
    acc, auc = compute_metrics(np.array(y_true_list), np.array(y_pred_probs_list))  # 정확도 및 AUC 계산

    # wandb 로깅이 설정된 경우, 각 지표를 로깅
    if wandb is not None:
        wandb.log({"train_loss": avg_loss, "train_acc": acc, "train_auc": auc}, step=epoch)

    return avg_loss, acc, auc  # 학습 손실, 정확도, AUC를 반환


