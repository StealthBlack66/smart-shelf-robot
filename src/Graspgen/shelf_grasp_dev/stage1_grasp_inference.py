#!/usr/bin/env python3
"""
Stage 1: GraspGen 샘플 점구름 추론 + 파지 결과 저장
- 백본: PointNet++ (Blackwell/SM 12.0 전용)
- 체크포인트: graspgen_robotiq_2f_140 (HF adithyamurali/GraspGenModels)
- 입력: sample_data/meshes/box.obj, bowl.obj, cylinder.obj
- 출력: configs_draft/grasp_results_<mesh>_<timestamp>.npz
"""

import sys
import time
import numpy as np
import torch
import trimesh
from pathlib import Path

GRASPGEN_DIR = Path.home() / "graspgen_ws" / "GraspGen"
CKPT_YML    = Path.home() / "graspgen_ws" / "checkpoints" / "graspgen_robotiq_2f_140.yml"
SAMPLE_DIR  = Path.home() / "graspgen_ws" / "sample_data" / "meshes"
OUT_DIR     = Path.home() / "shelf_grasp_dev" / "configs_draft"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(GRASPGEN_DIR))

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.dataset.dataset_utils import sample_points

print("=" * 60)
print("Stage 1 GraspGen 추론 시작")
print(f"  GPU : {torch.cuda.get_device_name(0)}")
print(f"  torch: {torch.__version__}  CUDA: {torch.version.cuda}")
print("=" * 60)

# --- 모델 로드 ---
print("\n[GATE 3-1] 체크포인트 로드 중...")
cfg = load_grasp_cfg(str(CKPT_YML))
print(f"  백본: {cfg.diffusion.obs_backbone}")
print(f"  그리퍼: {cfg.data.gripper_name}")

sampler = GraspGenSampler(cfg)
print("  모델 로드 완료.")

# --- 메쉬별 추론 ---
meshes = ["box.obj", "bowl.obj", "cylinder.obj"]

for mesh_name in meshes:
    mesh_path = SAMPLE_DIR / mesh_name
    if not mesh_path.exists():
        print(f"\n[SKIP] {mesh_name} 없음")
        continue

    print(f"\n[GATE 3-2] 추론: {mesh_name}")
    mesh = trimesh.load(str(mesh_path))
    mesh.apply_translation(-mesh.center_mass)

    # 점구름 샘플링
    points, _ = trimesh.sample.sample_surface(mesh, 2048)
    pc = torch.tensor(points, dtype=torch.float32)
    print(f"  점구름 shape: {pc.shape}")

    t0 = time.time()
    grasps, scores = GraspGenSampler.run_inference(
        object_pc=pc.numpy(),
        grasp_sampler=sampler,
        grasp_threshold=-1.0,
        num_grasps=200,
        topk_num_grasps=50,
        remove_outliers=False,
    )
    elapsed = time.time() - t0

    grasps_np = grasps.cpu().numpy() if isinstance(grasps, torch.Tensor) else grasps
    scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores

    print(f"  파지 수: {len(grasps_np)}, 추론 시간: {elapsed:.2f}s")
    print(f"  점수 범위: [{scores_np.min():.3f}, {scores_np.max():.3f}]")

    # 결과 저장
    stem = mesh_path.stem
    ts = int(time.time())
    out_path = OUT_DIR / f"grasp_results_{stem}_{ts}.npz"
    np.savez(str(out_path),
             grasps=grasps_np,
             scores=scores_np,
             point_cloud=points,
             mesh=str(mesh_path),
             backbone=cfg.diffusion.obs_backbone,
             gripper=cfg.data.gripper_name)
    print(f"  저장 완료: {out_path}")

    # 상위 5개 파지 출력 (approach 벡터 = Z열)
    print("  [상위 5 파지 자세 (4x4 행렬)]")
    top_idx = np.argsort(scores_np)[::-1][:5]
    for i, idx in enumerate(top_idx):
        g = grasps_np[idx]
        pos = g[:3, 3] if g.shape == (4, 4) else g[:3]
        print(f"    #{i+1} score={scores_np[idx]:.4f}  pos(m)={pos}")

print("\n" + "=" * 60)
print("Stage 1 완료. 결과 위치:", OUT_DIR)
print("=" * 60)
