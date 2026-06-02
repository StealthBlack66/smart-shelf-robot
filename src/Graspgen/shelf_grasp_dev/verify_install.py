#!/usr/bin/env python3
"""설치 검증: PointNet++ 백본으로 GraspGen 추론 확인 (spconv/PTV3 없이)"""

import sys
import torch
import numpy as np
import trimesh
from pathlib import Path
from omegaconf import DictConfig

print(f"[1/5] torch {torch.__version__}, CUDA {torch.version.cuda}, GPU={torch.cuda.get_device_name(0)}")

from grasp_gen.models.grasp_gen import GraspGen
from grasp_gen.dataset.dataset import collate
print("[2/5] GraspGen import OK")

# --- 박스 메쉬 → 점구름 ---
assets_dir = Path(__file__).parent.parent / "graspgen_ws/GraspGen/assets/objects"
box_path = assets_dir / "box.obj"
assert box_path.exists(), f"박스 메쉬 없음: {box_path}"
mesh = trimesh.load(str(box_path))
mesh.apply_translation(-mesh.center_mass)
points, _ = trimesh.sample.sample_surface(mesh, 2000)
point_cloud = torch.tensor(points, dtype=torch.float32)
print(f"[3/5] 점구름 로드 완료 shape={point_cloud.shape}")

# --- 모델 설정 (PointNet++ 전용) ---
device = torch.device("cuda")
backbone = "pointnet"

gen_cfg = DictConfig({
    "num_embed_dim": 256, "num_obs_dim": 512, "diffusion_embed_dim": 512,
    "image_size": 256, "num_diffusion_iters": 10, "num_diffusion_iters_eval": 10,
    "obs_backbone": backbone, "compositional_schedular": False,
    "loss_pointmatching": True, "loss_l1_pos": False, "loss_l1_rot": False,
    "grasp_repr": "r3_6d", "kappa": -1.0, "clip_sample": True,
    "beta_schedule": "squaredcos_cap_v2", "attention": "cat", "grid_size": 0.02,
    "gripper_name": "robotiq_2f_140", "pose_repr": "mlp",
    "num_grasps_per_object": 100, "checkpoint_object_encoder_pretrained": None,
    "ptv3": DictConfig({"grid_size": 0.02}),
})
disc_cfg = DictConfig({
    "num_obs_dim": 512, "obs_backbone": backbone, "grasp_repr": "r3_6d",
    "grid_size": 0.01, "sample_embed_dim": 512, "pose_repr": "mlp",
    "topk_ratio": 0.40, "checkpoint_object_encoder_pretrained": None,
    "kappa": 3.30, "gripper_name": "robotiq_2f_140",
    "ptv3": DictConfig({"grid_size": 0.01}),
})

model = GraspGen(gen_cfg, disc_cfg).to(device)
model.eval()
print(f"[4/5] 모델 초기화 완료 (backbone={backbone}, 랜덤 가중치)")

# --- 배치 준비 → 추론 ---
pc = point_cloud.to(device)
pc_centered = pc - pc.mean(dim=0)
data = {
    "task": "pick",
    "inputs": torch.cat([pc_centered, torch.zeros_like(pc_centered)], dim=-1).float(),
    "points": pc_centered,
}
batch = collate([data])
batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

with torch.inference_mode():
    outputs, _, _ = model.infer(batch)

grasps = outputs["grasps_pred"][0]
print(f"[5/5] 추론 완료: 파지 수={len(grasps)}")
assert len(grasps) == 100, f"예상 100개, 실제 {len(grasps)}개"
print("\n✅ 설치 검증 성공! (PointNet++ backbone, SM 12.0, cu128)")
