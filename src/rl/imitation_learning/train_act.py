#!/usr/bin/env python3
"""
ACT (Action Chunking with Transformers) 학습

사용법:
    python train_act.py --epochs 500 --chunk_size 10

학습 후:
    - 체크포인트: pen_grasp_rl/checkpoints/act/
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py

# 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)

from pen_grasp_rl.imitation_learning.act_model import ACT

# 데이터 경로
HDF5_PATH = os.path.join(PROJECT_DIR, "data/pen_grasp_robomimic.hdf5")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "pen_grasp_rl/checkpoints/act")


class ACTDataset(Dataset):
    """
    ACT 학습용 데이터셋

    각 샘플: (obs, action_chunk)
    - obs: 현재 관측값
    - action_chunk: 다음 chunk_size 개의 action
    """
    def __init__(self, hdf5_path: str, demo_keys: list, chunk_size: int = 10):
        self.hdf5_path = hdf5_path
        self.chunk_size = chunk_size
        self.data = []

        with h5py.File(hdf5_path, 'r') as f:
            for key in demo_keys:
                demo = f[f'data/{key}']

                # 관측값 로드
                obs = np.concatenate([
                    demo['obs/joint_pos'][:],
                    demo['obs/joint_vel'][:],
                    demo['obs/ee_pos'][:],
                    demo['obs/ee_quat'][:],
                    demo['obs/pen_pos'][:],
                    demo['obs/pen_axis'][:],
                ], axis=1)  # (T, 25)

                actions = demo['actions'][:]  # (T, 6)
                T = len(obs)

                # Action chunking: 각 timestep에서 다음 chunk_size개 action
                for t in range(T - chunk_size):
                    obs_t = obs[t]
                    action_chunk = actions[t:t + chunk_size]  # (chunk_size, 6)
                    self.data.append((obs_t, action_chunk))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs, action_chunk = self.data[idx]
        return torch.FloatTensor(obs), torch.FloatTensor(action_chunk)


def train(args):
    """ACT 학습"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 데이터 로드
    print(f"\n데이터 로드: {HDF5_PATH}")
    with h5py.File(HDF5_PATH, 'r') as f:
        train_keys = [k.decode() for k in f['mask/train'][:]]
        valid_keys = [k.decode() for k in f['mask/valid'][:]]

    print(f"Train demos: {len(train_keys)}")
    print(f"Valid demos: {len(valid_keys)}")

    train_dataset = ACTDataset(HDF5_PATH, train_keys, chunk_size=args.chunk_size)
    valid_dataset = ACTDataset(HDF5_PATH, valid_keys, chunk_size=args.chunk_size)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 모델 생성
    model = ACT(
        obs_dim=25,
        action_dim=6,
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
    ).to(device)

    print(f"\n모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # 출력 디렉토리
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 학습 루프
    best_valid_loss = float('inf')

    print("\n" + "=" * 60)
    print("ACT 학습 시작")
    print("=" * 60)
    print(f"Chunk size: {args.chunk_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"KL weight: {args.kl_weight}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # === Train ===
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0

        for obs, actions in train_loader:
            obs = obs.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            loss, info = model.compute_loss(obs, actions)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += info['loss']
            train_recon += info['recon_loss']
            train_kl += info['kl_loss']

        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)

        # === Validation ===
        model.eval()
        valid_loss = 0
        valid_recon = 0
        valid_kl = 0

        with torch.no_grad():
            for obs, actions in valid_loader:
                obs = obs.to(device)
                actions = actions.to(device)

                _, info = model.compute_loss(obs, actions)
                valid_loss += info['loss']
                valid_recon += info['recon_loss']
                valid_kl += info['kl_loss']

        valid_loss /= len(valid_loader)
        valid_recon /= len(valid_loader)
        valid_kl /= len(valid_loader)

        scheduler.step()

        # 출력
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Train: {train_loss:.6f} (recon: {train_recon:.6f}, kl: {train_kl:.6f}) | "
                  f"Valid: {valid_loss:.6f} (recon: {valid_recon:.6f}, kl: {valid_kl:.6f}) | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Best 저장
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'config': {
                    'obs_dim': 25,
                    'action_dim': 6,
                    'chunk_size': args.chunk_size,
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'num_layers': args.num_layers,
                    'latent_dim': args.latent_dim,
                    'kl_weight': args.kl_weight,
                },
            }, os.path.join(ckpt_dir, "best_model.pth"))
            print(f"  -> Best model saved (valid_loss: {valid_loss:.6f})")

        # 주기적 저장
        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'config': {
                    'obs_dim': 25,
                    'action_dim': 6,
                    'chunk_size': args.chunk_size,
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'num_layers': args.num_layers,
                    'latent_dim': args.latent_dim,
                    'kl_weight': args.kl_weight,
                },
            }, os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth"))

    # 최종 저장
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': valid_loss,
        'config': {
            'obs_dim': 25,
            'action_dim': 6,
            'chunk_size': args.chunk_size,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'latent_dim': args.latent_dim,
            'kl_weight': args.kl_weight,
        },
    }, os.path.join(ckpt_dir, "final_model.pth"))

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"Best validation loss: {best_valid_loss:.6f}")
    print(f"체크포인트: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description='ACT 학습')

    # 학습 설정
    parser.add_argument('--epochs', type=int, default=500, help='학습 에폭')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 사이즈')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')

    # ACT 모델 설정
    parser.add_argument('--chunk_size', type=int, default=10, help='Action chunk 크기')
    parser.add_argument('--d_model', type=int, default=256, help='Transformer dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer layers')
    parser.add_argument('--latent_dim', type=int, default=32, help='CVAE latent dimension')
    parser.add_argument('--kl_weight', type=float, default=10.0, help='KL loss weight')

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
