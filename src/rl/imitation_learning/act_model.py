#!/usr/bin/env python3
"""
ACT (Action Chunking with Transformers) 모델

Stanford ALOHA 프로젝트 기반 구현
- Transformer encoder-decoder
- CVAE for multimodal action distribution
- Action chunking (여러 스텝 동시 예측)

Reference: https://github.com/tonyzhaozh/act
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal position encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class ACTEncoder(nn.Module):
    """
    ACT Encoder: 관측값을 latent space로 인코딩

    CVAE의 encoder 역할 - 학습 시에만 사용 (ground truth action 필요)
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        # 관측값 projection
        self.obs_proj = nn.Linear(obs_dim, d_model)

        # Action projection (학습 시 ground truth action 사용)
        self.action_proj = nn.Linear(action_dim, d_model)

        # CLS token for latent
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Position encoding
        self.pos_enc = SinusoidalPositionEncoding(d_model, max_len=chunk_size + 2)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Latent projection (mu, logvar)
        self.latent_proj = nn.Linear(d_model, latent_dim * 2)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_dim) - 현재 관측값
            actions: (batch, chunk_size, action_dim) - ground truth action sequence

        Returns:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        batch_size = obs.shape[0]

        # Project obs and actions
        obs_embed = self.obs_proj(obs).unsqueeze(1)  # (batch, 1, d_model)
        action_embed = self.action_proj(actions)  # (batch, chunk_size, d_model)

        # CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)

        # Concatenate: [CLS, obs, actions]
        tokens = torch.cat([cls_tokens, obs_embed, action_embed], dim=1)  # (batch, 2+chunk_size, d_model)

        # Add position encoding
        tokens = self.pos_enc(tokens)

        # Transformer
        encoded = self.transformer(tokens)

        # Get CLS token output
        cls_output = encoded[:, 0, :]  # (batch, d_model)

        # Project to latent
        latent_params = self.latent_proj(cls_output)
        mu, logvar = latent_params.chunk(2, dim=-1)

        return mu, logvar


class ACTDecoder(nn.Module):
    """
    ACT Decoder: latent + obs로부터 action sequence 생성

    CVAE의 decoder 역할
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Observation projection
        self.obs_proj = nn.Linear(obs_dim, d_model)

        # Latent projection
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # Action query tokens (learnable)
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        nn.init.xavier_uniform_(self.action_queries)

        # Position encoding
        self.pos_enc = SinusoidalPositionEncoding(d_model, max_len=chunk_size + 2)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Action output projection
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim)
            latent: (batch, latent_dim)

        Returns:
            actions: (batch, chunk_size, action_dim)
        """
        batch_size = obs.shape[0]

        # Project obs and latent
        obs_embed = self.obs_proj(obs).unsqueeze(1)  # (batch, 1, d_model)
        latent_embed = self.latent_proj(latent).unsqueeze(1)  # (batch, 1, d_model)

        # Memory: [latent, obs]
        memory = torch.cat([latent_embed, obs_embed], dim=1)  # (batch, 2, d_model)

        # Action queries
        queries = self.action_queries.expand(batch_size, -1, -1)  # (batch, chunk_size, d_model)
        queries = self.pos_enc(queries)

        # Transformer decoder
        decoded = self.transformer(queries, memory)  # (batch, chunk_size, d_model)

        # Project to actions
        actions = self.action_head(decoded)  # (batch, chunk_size, action_dim)

        return actions


class ACT(nn.Module):
    """
    ACT (Action Chunking with Transformers)

    CVAE 구조:
    - Encoder: obs + gt_actions → latent (학습 시에만 사용)
    - Decoder: obs + latent → predicted_actions

    추론 시에는 latent를 prior (N(0,1))에서 샘플링하거나 zero로 설정
    """
    def __init__(
        self,
        obs_dim: int = 25,
        action_dim: int = 6,
        chunk_size: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # Encoder (CVAE encoder - 학습 시에만 사용)
        self.encoder = ACTEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            latent_dim=latent_dim,
        )

        # Decoder (CVAE decoder)
        self.decoder = ACTDecoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            latent_dim=latent_dim,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        use_encoder: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            obs: (batch, obs_dim)
            actions: (batch, chunk_size, action_dim) - ground truth (학습 시에만)
            use_encoder: encoder 사용 여부 (compute_loss에서는 항상 True)

        Returns:
            pred_actions: (batch, chunk_size, action_dim)
            mu: (batch, latent_dim) or None
            logvar: (batch, latent_dim) or None
        """
        if use_encoder and actions is not None:
            # Encoder 사용 (학습/검증 loss 계산 시)
            mu, logvar = self.encoder(obs, actions)
            latent = self.reparameterize(mu, logvar)
        else:
            # 추론 모드: latent = 0 (prior mean)
            batch_size = obs.shape[0]
            latent = torch.zeros(batch_size, self.latent_dim, device=obs.device)
            mu, logvar = None, None

        # Decode
        pred_actions = self.decoder(obs, latent)

        return pred_actions, mu, logvar

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute CVAE loss = reconstruction + KL divergence

        Args:
            obs: (batch, obs_dim)
            actions: (batch, chunk_size, action_dim)

        Returns:
            loss: scalar
            info: dict with loss components
        """
        pred_actions, mu, logvar = self.forward(obs, actions)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(pred_actions, actions)

        # KL divergence loss
        # KL(q(z|x,a) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + self.kl_weight * kl_loss

        info = {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
        }

        return loss, info

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        추론 시 action 예측 (첫 번째 action만 반환)

        Args:
            obs: (batch, obs_dim) or (obs_dim,)

        Returns:
            action: (batch, action_dim) or (action_dim,)
        """
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)

        self.eval()
        pred_actions, _, _ = self.forward(obs, actions=None, use_encoder=False)
        action = pred_actions[:, 0, :]  # 첫 번째 action

        if single:
            action = action.squeeze(0)

        return action

    @torch.no_grad()
    def get_action_chunk(self, obs: torch.Tensor) -> torch.Tensor:
        """
        추론 시 전체 action chunk 반환

        Args:
            obs: (batch, obs_dim) or (obs_dim,)

        Returns:
            actions: (batch, chunk_size, action_dim) or (chunk_size, action_dim)
        """
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)

        self.eval()
        pred_actions, _, _ = self.forward(obs, actions=None, use_encoder=False)

        if single:
            pred_actions = pred_actions.squeeze(0)

        return pred_actions


class TemporalEnsemble:
    """
    Temporal Ensemble for smoother action execution

    여러 시점에서 예측한 action chunk를 가중 평균하여 사용
    """
    def __init__(self, chunk_size: int, action_dim: int, device: str = 'cuda'):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.device = device

        # Action buffer: 각 timestep에서 예측한 action chunk 저장
        self.action_buffer = []
        self.weights = self._compute_weights()

    def _compute_weights(self) -> torch.Tensor:
        """Exponential weights for temporal ensemble"""
        # 최근 예측에 더 높은 가중치
        weights = torch.exp(-0.01 * torch.arange(self.chunk_size, dtype=torch.float32))
        weights = weights / weights.sum()
        return weights.to(self.device)

    def reset(self):
        """Reset buffer"""
        self.action_buffer = []

    def update(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        새로운 action chunk 추가하고 ensemble action 반환

        Args:
            action_chunk: (chunk_size, action_dim)

        Returns:
            action: (action_dim,) - 현재 timestep의 ensemble action
        """
        self.action_buffer.append(action_chunk)

        # 오래된 예측 제거 (chunk_size 개만 유지)
        if len(self.action_buffer) > self.chunk_size:
            self.action_buffer.pop(0)

        # Ensemble: 각 예측에서 현재 timestep에 해당하는 action 추출
        actions = []
        weights = []

        for i, chunk in enumerate(self.action_buffer):
            # i번째 예측에서 현재 timestep은 (len(buffer) - 1 - i)번째 action
            idx = len(self.action_buffer) - 1 - i
            if idx < self.chunk_size:
                actions.append(chunk[idx])
                weights.append(self.weights[i])

        if not actions:
            return action_chunk[0]

        actions = torch.stack(actions, dim=0)  # (n, action_dim)
        weights = torch.stack(weights, dim=0)  # (n,)
        weights = weights / weights.sum()

        # Weighted average
        ensemble_action = (actions * weights.unsqueeze(1)).sum(dim=0)

        return ensemble_action


if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = ACT(
        obs_dim=25,
        action_dim=6,
        chunk_size=10,
        d_model=256,
        nhead=8,
        latent_dim=32,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward
    batch_size = 4
    obs = torch.randn(batch_size, 25, device=device)
    actions = torch.randn(batch_size, 10, 6, device=device)

    # Training mode
    model.train()
    loss, info = model.compute_loss(obs, actions)
    print(f"Training loss: {loss.item():.4f}")
    print(f"Info: {info}")

    # Inference mode
    model.eval()
    pred_action = model.get_action(obs)
    print(f"Predicted action shape: {pred_action.shape}")

    pred_chunk = model.get_action_chunk(obs)
    print(f"Predicted chunk shape: {pred_chunk.shape}")

    print("\nACT model test passed!")
