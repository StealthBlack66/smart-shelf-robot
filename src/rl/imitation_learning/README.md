# Diffusion Policy 모방학습

펜 잡기 전문가 궤적을 이용한 Diffusion Policy 학습

## 폴더 구조

```
imitation_learning/
├── convert_to_zarr.py    # NPZ → Zarr 변환
├── train_diffusion.py    # 학습 스크립트
├── Dockerfile            # Docker 이미지
├── docker-compose.yaml   # Docker Compose 설정
└── requirements.txt      # Python 의존성
```

---

## 학습용 노트북에서 설정

### 1. 프로젝트 클론 (최초 1회)

```bash
cd ~
git clone https://github.com/KERNEL3-2/CoWriteBotRL.git
```

### 2. 데이터 복사

이 노트북(MoveIt 노트북)에서 생성한 궤적 데이터를 학습 노트북으로 복사:

```bash
# MoveIt 노트북에서 (이 노트북)
cd ~/CoWriteBotRL
scp -r data/ <학습노트북_사용자>@<학습노트북_IP>:~/CoWriteBotRL/
```

또는 USB/NAS로 복사:
```
복사할 폴더: ~/CoWriteBotRL/data/pen_grasp.zarr
```

### 3. Docker 빌드 및 실행

```bash
# 학습 노트북에서
cd ~/CoWriteBotRL/pen_grasp_rl/imitation_learning

# Docker 이미지 빌드 (최초 1회, 약 5분)
docker compose build

# 컨테이너 시작
docker compose up -d

# 컨테이너 진입
docker compose exec diffusion bash
```

### 4. 학습 실행 (컨테이너 내부)

```bash
cd /workspace/imitation_learning

# 학습 시작
python train_diffusion.py \
    --data_path /workspace/data/pen_grasp.zarr \
    --output_dir /workspace/checkpoints/diffusion_policy \
    --num_epochs 500 \
    --batch_size 64
```

### 5. TensorBoard 모니터링

```bash
# 새 터미널에서 컨테이너 진입
docker compose exec diffusion bash

# TensorBoard 실행
tensorboard --logdir=/workspace/checkpoints --bind_all

# 브라우저: http://localhost:6007
```

### 6. 컨테이너 종료

```bash
exit  # 컨테이너에서 나가기
docker compose down  # 컨테이너 종료
```

---

## 일상적인 사용

### 매일 작업 시작

```bash
cd ~/CoWriteBotRL/pen_grasp_rl/imitation_learning
docker compose up -d
docker compose exec diffusion bash

# 컨테이너 내부
cd /workspace/imitation_learning
python train_diffusion.py --data_path /workspace/data/pen_grasp.zarr --output_dir /workspace/checkpoints/diffusion_policy
```

### 작업 종료

```bash
exit
docker compose down
```

---

## 학습 결과 확인

체크포인트는 호스트의 `~/CoWriteBotRL/checkpoints/` 에 저장됩니다.

```bash
# 호스트에서 확인
ls ~/CoWriteBotRL/checkpoints/diffusion_policy/
```

---

## 학습 파라미터 조정

```bash
python train_diffusion.py \
    --data_path /workspace/data/pen_grasp.zarr \
    --output_dir /workspace/checkpoints/diffusion_policy \
    --num_epochs 500 \      # 에폭 수
    --batch_size 64 \       # 배치 크기 (VRAM에 따라 조정)
    --lr 1e-4 \             # 학습률
    --horizon 16 \          # 예측 horizon
    --n_obs_steps 2         # 관측 스텝 수
```

### GPU 메모리별 권장 batch_size

| GPU VRAM | batch_size |
|----------|------------|
| 8GB | 32 |
| 12GB | 64 |
| 16GB | 128 |
| 24GB | 256 |

---

## 문제 해결

### GPU 인식 안됨
```bash
# Docker에서 GPU 확인
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### 권한 문제
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 데이터 경로 오류
```bash
# 컨테이너 내부에서 데이터 확인
ls -la /workspace/data/
```
