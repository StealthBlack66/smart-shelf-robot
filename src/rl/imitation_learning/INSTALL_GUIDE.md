# Diffusion Policy 모방학습 - Docker 사용 가이드

## 개요

이 가이드는 MoveIt2로 생성한 전문가 궤적을 사용해 Diffusion Policy를 학습하는 방법을 설명합니다.
Isaac Lab Docker보다 훨씬 가볍고 빠르게 설정할 수 있습니다.

### Isaac Lab Docker와 비교

| 항목 | Isaac Lab Docker | Diffusion Policy Docker |
|------|-----------------|------------------------|
| 이미지 크기 | ~30GB | ~8GB |
| 빌드 시간 | ~30분 | ~5분 |
| 용도 | RL 학습 (시뮬레이션) | 모방학습 (데이터 기반) |
| NGC 로그인 | 필요 | 불필요 |

---

## 1. 사전 요구사항

| 항목 | 요구사항 |
|------|----------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA (RTX 3070 이상 권장) |
| NVIDIA Driver | 535 이상 |
| Docker | 26.0.0 이상 |
| Docker Compose | 2.25.0 이상 |

> **참고**: Isaac Lab Docker를 이미 설정했다면 Docker와 NVIDIA Container Toolkit이 이미 설치되어 있습니다.

---

## 2. 최초 설정 (1회만)

### 2.1 Docker 및 NVIDIA 설정 확인

Isaac Lab을 사용 중이라면 이미 설정되어 있습니다. 확인만 해주세요:

```bash
# Docker 확인
docker --version

# GPU 확인
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### 2.2 프로젝트 클론 (최초 1회)

```bash
cd ~
git clone https://github.com/KERNEL3-2/CoWriteBotRL.git
```

이미 클론했다면:
```bash
cd ~/CoWriteBotRL
git pull
```

---

## 3. Docker 빌드 및 실행

### 3.1 Docker 이미지 빌드 (최초 1회, ~5분)

```bash
cd ~/CoWriteBotRL/pen_grasp_rl/imitation_learning
docker compose build
```

### 3.2 컨테이너 시작

```bash
docker compose up -d
```

### 3.3 컨테이너 진입

```bash
docker compose exec diffusion bash
```

> **참고**: 컨테이너 내부에서는 `/workspace/` 경로를 사용합니다.

---

## 4. 학습 실행

### 4.1 학습 시작 (컨테이너 내부)

```bash
cd /workspace/imitation_learning

# 기본 학습 (500 에폭)
python train_diffusion.py \
    --data_path /workspace/data/pen_grasp.zarr \
    --output_dir /workspace/checkpoints/diffusion_policy \
    --num_epochs 500 \
    --batch_size 64
```

### 4.2 GPU 메모리별 권장 설정

| GPU VRAM | batch_size | 예상 학습 시간 |
|----------|------------|---------------|
| 8GB | 32 | ~2시간 |
| 12GB | 64 | ~1.5시간 |
| 16GB | 128 | ~1시간 |
| 24GB | 256 | ~40분 |

```bash
# 8GB GPU 예시
python train_diffusion.py \
    --data_path /workspace/data/pen_grasp.zarr \
    --output_dir /workspace/checkpoints/diffusion_policy \
    --batch_size 32

# 24GB GPU 예시
python train_diffusion.py \
    --data_path /workspace/data/pen_grasp.zarr \
    --output_dir /workspace/checkpoints/diffusion_policy \
    --batch_size 256
```

### 4.3 학습 파라미터 전체 옵션

```bash
python train_diffusion.py \
    --data_path /workspace/data/pen_grasp.zarr \  # 데이터 경로
    --output_dir /workspace/checkpoints/diffusion_policy \  # 체크포인트 저장
    --num_epochs 500 \      # 에폭 수
    --batch_size 64 \       # 배치 크기
    --lr 1e-4 \             # 학습률
    --horizon 16 \          # Action 예측 horizon
    --n_obs_steps 2 \       # 관측 스텝 수
    --save_freq 50          # 체크포인트 저장 주기
```

---

## 5. 학습 모니터링

### 5.1 TensorBoard (새 터미널에서)

```bash
# 호스트에서 새 터미널 열기
cd ~/CoWriteBotRL/pen_grasp_rl/imitation_learning
docker compose exec diffusion bash

# 컨테이너 내부에서
tensorboard --logdir=/workspace/checkpoints --bind_all
```

브라우저에서 `http://localhost:6007` 접속

### 5.2 학습 로그 확인

학습 중 터미널에 출력되는 정보:
```
Epoch 1/500: train_loss=0.0523, val_loss=0.0498
Epoch 2/500: train_loss=0.0412, val_loss=0.0389
  → Best model saved (val_loss=0.0389)
...
```

---

## 6. 컨테이너 관리

### 6.1 컨테이너 종료

```bash
# 컨테이너에서 나가기
exit

# 컨테이너 중지
cd ~/CoWriteBotRL/pen_grasp_rl/imitation_learning
docker compose down
```

### 6.2 컨테이너 재시작

```bash
cd ~/CoWriteBotRL/pen_grasp_rl/imitation_learning
docker compose up -d
docker compose exec diffusion bash
```

---

## 7. 일상적인 사용

### 매일 작업 시작

```bash
cd ~/CoWriteBotRL/pen_grasp_rl/imitation_learning
docker compose up -d
docker compose exec diffusion bash

# 컨테이너 내부
cd /workspace/imitation_learning
python train_diffusion.py \
    --data_path /workspace/data/pen_grasp.zarr \
    --output_dir /workspace/checkpoints/diffusion_policy
```

### 작업 종료

```bash
exit
docker compose down
```

---

## 8. 학습 결과 확인

### 8.1 체크포인트 위치

체크포인트는 호스트의 `~/CoWriteBotRL/checkpoints/` 에 저장됩니다:

```bash
# 호스트에서 확인
ls ~/CoWriteBotRL/checkpoints/diffusion_policy/

# 예상 출력:
# best_model.pt          # 가장 좋은 모델
# final_model.pt         # 마지막 에폭 모델
# checkpoint_epoch_50.pt # 중간 체크포인트
# checkpoint_epoch_100.pt
# config.json            # 학습 설정
```

### 8.2 모델 파일 구조

```python
# best_model.pt 내용
{
    'model': ...,      # 모델 가중치
    'ema': ...,        # EMA 가중치
    'config': {
        'state_dim': 25,
        'action_dim': 6,
        'horizon': 16,
        'n_obs_steps': 2,
    }
}
```

---

## 9. 학습된 모델 사용

### 9.1 추론 예시 (Python)

```python
import torch
from train_diffusion import DiffusionPolicy

# 모델 로드
device = 'cuda'
policy = DiffusionPolicy(
    state_dim=25,
    action_dim=6,
    horizon=16,
    n_obs_steps=2,
    device=device
)
policy.load('/workspace/checkpoints/diffusion_policy/best_model.pt')

# 추론
obs = torch.randn(2, 25).to(device)  # (n_obs_steps, state_dim)
actions = policy.predict(obs)  # (1, horizon, action_dim)
print(f"Predicted actions shape: {actions.shape}")
```

---

## 10. 문제 해결

### GPU 인식 안됨

```bash
# Docker에서 GPU 확인
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# 안되면 NVIDIA Container Toolkit 재설치
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 권한 문제

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 데이터 경로 오류

```bash
# 컨테이너 내부에서 데이터 확인
ls -la /workspace/data/pen_grasp.zarr/
```

### Out of Memory (OOM)

batch_size를 줄이세요:
```bash
python train_diffusion.py \
    --data_path /workspace/data/pen_grasp.zarr \
    --output_dir /workspace/checkpoints/diffusion_policy \
    --batch_size 32  # 64 → 32로 줄임
```

### 컨테이너 상태 확인

```bash
docker ps -a
docker compose logs
```

---

## 11. 요약: 새 PC 설정 체크리스트

- [ ] Docker 설치 확인
- [ ] NVIDIA Container Toolkit 설치 확인
- [ ] 프로젝트 클론 (`git clone https://github.com/KERNEL3-2/CoWriteBotRL.git`)
- [ ] Docker 이미지 빌드 (`docker compose build`)
- [ ] 컨테이너 시작 (`docker compose up -d`)
- [ ] 학습 실행 (`python train_diffusion.py ...`)

---

## 12. 데이터 정보

현재 학습 데이터:
- **에피소드 수**: 935개
- **총 타임스텝**: 55,278개
- **State 차원**: 25 (joint_pos[6] + joint_vel[6] + ee_pose[7] + pen_pos[3] + pen_axis[3])
- **Action 차원**: 6 (joint positions)
- **데이터 크기**: 5.7MB

---

## 13. 다음 단계

학습이 완료되면:

1. **시뮬레이션 테스트**: Isaac Lab에서 학습된 정책 테스트
2. **Sim2Real**: 실제 로봇에 정책 배포
3. **Fine-tuning**: 필요시 실제 로봇 데이터로 추가 학습
