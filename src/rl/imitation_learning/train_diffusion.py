#!/usr/bin/env python3
"""
Diffusion Policy 학습 스크립트 (Standalone)

간단한 Diffusion Policy 구현으로, real-stanford/diffusion_policy 레포 없이
독립적으로 실행 가능합니다.

사용법:
    python train_diffusion.py \
        --data_path ~/CoWriteBotRL/data/pen_grasp.zarr \
        --output_dir ~/CoWriteBotRL/checkpoints/diffusion_policy

필요 패키지:
    pip install torch zarr diffusers tqdm wandb
"""

import os
import argparse
import numpy as np
import zarr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm import tqdm
import json
from datetime import datetime


class TrajectoryDataset(Dataset):
    """Zarr 기반 궤적 데이터셋"""

    def __init__(self, zarr_path, horizon=16, n_obs_steps=2, pad_before=1, pad_after=7):
        self.zarr_path = zarr_path
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Zarr 데이터 로드
        root = zarr.open(zarr_path, mode='r')
        self.states = np.array(root['data/state'])
        self.actions = np.array(root['data/action'])
        self.episode_ends = np.array(root['meta/episode_ends'])

        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]

        # 유효한 시작 인덱스 계산
        self._compute_valid_indices()

        print(f"Dataset loaded: {len(self)} samples")
        print(f"  State dim: {self.state_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Episodes: {len(self.episode_ends)}")

    def _compute_valid_indices(self):
        """유효한 샘플링 인덱스 계산"""
        self.valid_indices = []

        episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

        for ep_idx, (start, end) in enumerate(zip(episode_starts, self.episode_ends)):
            ep_len = end - start
            # 충분한 길이가 있는 인덱스만 포함
            for i in range(ep_len - self.horizon + 1):
                global_idx = start + i
                self.valid_indices.append(global_idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        # Observation (n_obs_steps)
        obs_start = max(0, start_idx - self.n_obs_steps + 1)
        obs = self.states[obs_start:start_idx + 1]

        # 패딩 (앞쪽)
        if len(obs) < self.n_obs_steps:
            pad = np.tile(obs[0:1], (self.n_obs_steps - len(obs), 1))
            obs = np.concatenate([pad, obs], axis=0)

        # Action sequence (horizon)
        actions = self.actions[start_idx:start_idx + self.horizon]

        # 패딩 (뒤쪽)
        if len(actions) < self.horizon:
            pad = np.tile(actions[-1:], (self.horizon - len(actions), 1))
            actions = np.concatenate([actions, pad], axis=0)

        return {
            'obs': torch.FloatTensor(obs),  # (n_obs_steps, state_dim)
            'action': torch.FloatTensor(actions),  # (horizon, action_dim)
        }


class ConditionalUNet1D(nn.Module):
    """간단한 1D U-Net (Diffusion Policy용)"""

    def __init__(self, input_dim, cond_dim, hidden_dim=256):
        super().__init__()

        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # Time + Condition embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Encoder blocks (모두 같은 차원)
        self.encoder1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Decoder blocks (skip connection으로 차원 2배)
        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, timestep, cond):
        """
        Args:
            x: (B, T, input_dim) - noisy action sequence
            timestep: (B,) - diffusion timestep
            cond: (B, cond_dim) - condition (flattened observation)

        Returns:
            (B, T, input_dim) - predicted noise
        """
        B, T, D = x.shape

        # Embeddings
        t_emb = self.time_embed(timestep.float().unsqueeze(-1) / 1000)  # (B, hidden_dim)
        c_emb = self.cond_embed(cond)  # (B, hidden_dim)
        emb = t_emb + c_emb  # (B, hidden_dim)

        # Reshape for sequence processing
        x = x.reshape(B * T, D)
        emb = emb.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        # Input projection
        h = self.input_proj(x)  # (B*T, hidden_dim)

        # Encoder
        h1 = self.encoder1(h) + emb
        h2 = self.encoder2(h1) + emb
        h3 = self.encoder3(h2) + emb

        # Decoder with skip connections
        d1 = self.decoder1(torch.cat([h3, h2], dim=-1)) + emb
        d2 = self.decoder2(torch.cat([d1, h1], dim=-1)) + emb

        # Output
        out = self.output_proj(d2)
        out = out.reshape(B, T, D)

        return out


class DiffusionPolicy:
    """Diffusion Policy 학습/추론 클래스"""

    # Doosan E0509 관절 제한 (라디안)
    # pen_workspace.py의 joint_limits_deg를 라디안으로 변환
    JOINT_LIMITS_RAD = {
        "joint_1": (-2 * np.pi, 2 * np.pi),      # ±360°
        "joint_2": (-1.6581, 1.6581),            # ±95°
        "joint_3": (-2.3562, 2.3562),            # ±135°
        "joint_4": (-2 * np.pi, 2 * np.pi),      # ±360°
        "joint_5": (-2.3562, 2.3562),            # ±135°
        "joint_6": (-2 * np.pi, 2 * np.pi),      # ±360°
    }

    def __init__(
        self,
        state_dim,
        action_dim,
        horizon=16,
        n_obs_steps=2,
        num_train_timesteps=100,
        num_inference_steps=16,
        device='cuda',
        clip_actions=True
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.device = device
        self.clip_actions = clip_actions

        # 관절 제한 배열 생성 (추론 시 클리핑용)
        self.joint_min = np.array([self.JOINT_LIMITS_RAD[f"joint_{i+1}"][0] for i in range(6)])
        self.joint_max = np.array([self.JOINT_LIMITS_RAD[f"joint_{i+1}"][1] for i in range(6)])

        # Condition dim = flattened observations
        self.cond_dim = state_dim * n_obs_steps

        # Model
        self.model = ConditionalUNet1D(
            input_dim=action_dim,
            cond_dim=self.cond_dim,
            hidden_dim=256
        ).to(device)

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.num_inference_steps = num_inference_steps

        # EMA
        self.ema = EMAModel(self.model.parameters(), decay=0.9999)

    def compute_loss(self, batch):
        """학습 loss 계산"""
        obs = batch['obs'].to(self.device)  # (B, n_obs_steps, state_dim)
        action = batch['action'].to(self.device)  # (B, horizon, action_dim)

        B = obs.shape[0]

        # Flatten observation for conditioning
        cond = obs.reshape(B, -1)  # (B, n_obs_steps * state_dim)

        # Sample noise
        noise = torch.randn_like(action)

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        # Add noise to actions
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)

        # Predict noise
        pred_noise = self.model(noisy_action, timesteps, cond)

        # MSE loss
        loss = nn.functional.mse_loss(pred_noise, noise)

        return loss

    @torch.no_grad()
    def predict(self, obs):
        """
        주어진 관측으로부터 액션 시퀀스 예측

        Args:
            obs: (B, n_obs_steps, state_dim) or (n_obs_steps, state_dim)

        Returns:
            actions: (B, horizon, action_dim)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        obs = obs.to(self.device)
        B = obs.shape[0]

        # Flatten observation
        cond = obs.reshape(B, -1)

        # Start from random noise
        action = torch.randn((B, self.horizon, self.action_dim), device=self.device)

        # Set inference timesteps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Denoising loop
        for t in self.noise_scheduler.timesteps:
            timestep = torch.full((B,), t, device=self.device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(action, timestep, cond)

            # Denoise step
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        # 관절 제한 클리핑
        if self.clip_actions:
            joint_min_tensor = torch.tensor(self.joint_min, device=self.device, dtype=action.dtype)
            joint_max_tensor = torch.tensor(self.joint_max, device=self.device, dtype=action.dtype)
            action = torch.clamp(action, joint_min_tensor, joint_max_tensor)

        return action

    def check_joint_limits(self, actions):
        """
        액션이 관절 제한 내에 있는지 확인

        Args:
            actions: (B, horizon, action_dim) or (horizon, action_dim)

        Returns:
            is_valid: bool
            violations: 위반된 관절 정보
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        if actions.ndim == 2:
            actions = actions[np.newaxis, ...]

        violations = []
        for b in range(actions.shape[0]):
            for t in range(actions.shape[1]):
                for j in range(actions.shape[2]):
                    val = actions[b, t, j]
                    if val < self.joint_min[j] or val > self.joint_max[j]:
                        violations.append({
                            'batch': b,
                            'timestep': t,
                            'joint': j + 1,
                            'value': val,
                            'min': self.joint_min[j],
                            'max': self.joint_max[j]
                        })

        return len(violations) == 0, violations

    def save(self, path):
        """모델 저장"""
        torch.save({
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'horizon': self.horizon,
                'n_obs_steps': self.n_obs_steps,
                'clip_actions': self.clip_actions,
                'joint_min': self.joint_min.tolist(),
                'joint_max': self.joint_max.tolist(),
            }
        }, path)

    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.ema.load_state_dict(checkpoint['ema'])

        # 관절 제한 설정 로드 (있으면)
        if 'config' in checkpoint:
            config = checkpoint['config']
            if 'clip_actions' in config:
                self.clip_actions = config['clip_actions']
            if 'joint_min' in config:
                self.joint_min = np.array(config['joint_min'])
            if 'joint_max' in config:
                self.joint_max = np.array(config['joint_max'])


def train(args):
    """학습 메인 함수"""

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 데이터셋 로드
    dataset = TrajectoryDataset(
        args.data_path,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps
    )

    # Train/Val 분리
    val_size = int(len(dataset) * 0.02)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Policy 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = DiffusionPolicy(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        device=device
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(train_loader)
    )

    # 학습 설정 저장
    config = {
        'data_path': args.data_path,
        'state_dim': dataset.state_dim,
        'action_dim': dataset.action_dim,
        'horizon': args.horizon,
        'n_obs_steps': args.n_obs_steps,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("Diffusion Policy 학습 시작")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"State dim: {dataset.state_dim}")
    print(f"Action dim: {dataset.action_dim}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print("=" * 60 + "\n")

    # 학습 루프
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        # Train
        policy.model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            loss = policy.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # EMA update
            policy.ema.step(policy.model.parameters())

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = np.mean(train_losses)

        # Validation
        policy.model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                loss = policy.compute_loss(batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # 체크포인트 저장
        if (epoch + 1) % args.save_freq == 0:
            policy.save(os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        # Best 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            policy.save(os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  → Best model saved (val_loss={best_val_loss:.4f})")

    # 최종 모델 저장
    policy.save(os.path.join(args.output_dir, 'final_model.pt'))
    print(f"\n학습 완료! 모델 저장: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Diffusion Policy 학습')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Zarr 데이터 경로')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/diffusion_policy',
                        help='출력 디렉토리')
    parser.add_argument('--horizon', type=int, default=16,
                        help='Action horizon')
    parser.add_argument('--n_obs_steps', type=int, default=2,
                        help='Observation steps')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Checkpoint save frequency')
    args = parser.parse_args()

    args.data_path = os.path.expanduser(args.data_path)
    args.output_dir = os.path.expanduser(args.output_dir)

    train(args)


if __name__ == '__main__':
    main()
