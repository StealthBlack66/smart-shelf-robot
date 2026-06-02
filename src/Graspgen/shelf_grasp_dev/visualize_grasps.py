#!/usr/bin/env python3
"""
Stage 1 시각화: 저장된 파지 결과를 viser 웹 뷰어로 표시
브라우저에서 http://localhost:8080 열면 됩니다.
"""

import sys
import time
import argparse
import numpy as np
import trimesh
from pathlib import Path

GRASPGEN_DIR = Path.home() / "graspgen_ws" / "GraspGen"
DRAFT_DIR    = Path.home() / "shelf_grasp_dev" / "configs_draft"
MESH_DIR     = Path.home() / "graspgen_ws" / "sample_data" / "meshes"

sys.path.insert(0, str(GRASPGEN_DIR))

from grasp_gen.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
)

def load_result(npz_path: Path):
    d = np.load(str(npz_path), allow_pickle=True)
    return d["grasps"], d["scores"], d["point_cloud"]

def visualize_one(vis, mesh_name: str, npz_path: Path, offset_x: float = 0.0):
    grasps, scores, pc = load_result(npz_path)

    # 메쉬
    mesh_path = MESH_DIR / mesh_name
    if mesh_path.exists():
        mesh = trimesh.load(str(mesh_path))
        mesh.apply_translation(-mesh.center_mass)
        verts = np.array(mesh.vertices) + [offset_x, 0, 0]
        visualize_mesh(vis, f"{mesh_name}/mesh",
                       trimesh.Trimesh(verts, mesh.faces),
                       color=[180, 180, 200])

    # 점구름
    pc_offset = pc + np.array([offset_x, 0, 0])
    visualize_pointcloud(vis, f"{mesh_name}/pc", pc_offset,
                         color=[100, 200, 100], point_size=0.004)

    # 상위 20 파지 (점수 → 색상 그라데이션)
    top_k = min(20, len(grasps))
    idx_sorted = np.argsort(scores)[::-1][:top_k]
    for rank, i in enumerate(idx_sorted):
        g = grasps[i].copy()
        g[:3, 3] += [offset_x, 0, 0]
        color = get_color_from_score(float(scores[i]))
        visualize_grasp(vis, f"{mesh_name}/grasp_{rank:02d}", g,
                        color=color, gripper_name="robotiq_2f_140", linewidth=2.0)

    print(f"  [{mesh_name}] 파지 {top_k}개 표시, 점수 [{scores[idx_sorted].min():.3f}, {scores[idx_sorted].max():.3f}]")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print("=" * 60)
    print(f"GraspGen 시각화 서버 시작: http://localhost:{args.port}")
    print("브라우저에서 위 주소를 열어주세요. Ctrl+C로 종료.")
    print("=" * 60)

    vis = create_visualizer(port=args.port)

    npz_files = sorted(DRAFT_DIR.glob("grasp_results_*.npz"))
    if not npz_files:
        print("configs_draft/ 에 결과 파일이 없습니다.")
        return

    offsets = {"box": -0.6, "bowl": 0.0, "cylinder": 0.6}

    for npz in npz_files:
        stem = npz.stem  # grasp_results_box_...
        obj_name = stem.split("_")[2]   # box / bowl / cylinder
        mesh_file = f"{obj_name}.obj"
        offset = offsets.get(obj_name, 0.0)
        print(f"\n[{obj_name}] 로드 중...")
        try:
            visualize_one(vis, mesh_file, npz, offset_x=offset)
        except Exception as e:
            print(f"  오류: {e}")

    print("\n✅ 시각화 준비 완료.")
    print("   브라우저: http://localhost:8080")
    print("   (파란색→빨간색 그라데이션 = 낮은→높은 점수)")
    print("   Ctrl+C 로 종료\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("종료합니다.")

if __name__ == "__main__":
    main()
