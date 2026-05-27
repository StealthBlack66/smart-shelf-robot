# Diffusion Policy 학습용 Docker 이미지
# PyTorch + CUDA 기반 경량 이미지

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --no-cache-dir \
    zarr \
    diffusers \
    tqdm \
    wandb \
    scipy \
    tensorboard

# 작업 디렉토리 설정
WORKDIR /workspace

# 기본 명령어
CMD ["/bin/bash"]
