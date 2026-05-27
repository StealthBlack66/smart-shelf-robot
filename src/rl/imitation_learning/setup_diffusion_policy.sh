#!/bin/bash
# Diffusion Policy 환경 설정 스크립트

set -e

echo "=============================================="
echo "Diffusion Policy 환경 설정"
echo "=============================================="

# 1. Conda 환경 생성
echo ""
echo "[1/4] Conda 환경 확인..."
if conda env list | grep -q "diffusion_policy"; then
    echo "  diffusion_policy 환경이 이미 존재합니다."
    echo "  활성화: conda activate diffusion_policy"
else
    echo "  새 환경을 생성합니다..."
    echo ""
    echo "  다음 명령어를 순서대로 실행하세요:"
    echo ""
    echo "  # Diffusion Policy 레포 클론"
    echo "  cd ~/CoWriteBotRL"
    echo "  git clone https://github.com/real-stanford/diffusion_policy.git"
    echo ""
    echo "  # Conda 환경 생성"
    echo "  cd diffusion_policy"
    echo "  conda env create -f conda_environment.yaml"
    echo "  conda activate robodiff"
    echo ""
fi

# 2. 데이터 디렉토리 생성
echo ""
echo "[2/4] 데이터 디렉토리 생성..."
mkdir -p ~/CoWriteBotRL/data
echo "  생성됨: ~/CoWriteBotRL/data"

# 3. 데이터 변환
echo ""
echo "[3/4] 데이터 변환 명령어:"
echo ""
echo "  # NPZ → Zarr 변환"
echo "  cd ~/CoWriteBotRL/pen_grasp_rl/imitation_learning"
echo "  python convert_to_zarr.py \\"
echo "      --input_dir ~/generated_trajectories \\"
echo "      --output_path ~/CoWriteBotRL/data/pen_grasp.zarr"
echo ""

# 4. 학습 실행
echo ""
echo "[4/4] 학습 실행 명령어:"
echo ""
echo "  # 환경변수 설정"
echo "  export DATA_DIR=~/CoWriteBotRL/data"
echo ""
echo "  # 학습 실행"
echo "  cd ~/CoWriteBotRL/diffusion_policy"
echo "  python train.py \\"
echo "      --config-dir=../pen_grasp_rl/imitation_learning/config \\"
echo "      --config-name=pen_grasp_lowdim"
echo ""

echo "=============================================="
echo "설정 완료!"
echo "=============================================="
