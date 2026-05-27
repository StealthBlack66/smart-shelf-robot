#!/usr/bin/env python3
"""
Zarr 데이터를 robomimic HDF5 형식으로 변환

robomimic HDF5 형식:
- data/
    - demo_0/
        - obs/
            - joint_pos (T, 6)
            - joint_vel (T, 6)
            - ee_pos (T, 3)
            - ee_quat (T, 4)
            - pen_pos (T, 3)
            - pen_axis (T, 3)
        - actions (T, 6)
        - rewards (T,)
        - dones (T,)
    - demo_1/
        ...
- mask/
    - train (indices)
    - valid (indices)

사용법:
    python convert_zarr_to_hdf5.py
"""

import os
import h5py
import zarr
import numpy as np
from tqdm import tqdm

# 경로 설정 (스크립트 위치 기준 상대 경로)
# 스크립트: pen_grasp_rl/imitation_learning/convert_zarr_to_hdf5.py
# 프로젝트: CoWriteBotRL/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # imitation_learning/
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # CoWriteBotRL/
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# 명령행 인자로 경로 지정 가능
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None, help='입력 zarr 경로')
parser.add_argument('--output', type=str, default=None, help='출력 hdf5 경로')
_args, _ = parser.parse_known_args()

INPUT_PATH = _args.input or os.path.join(DATA_DIR, "pen_grasp.zarr")
OUTPUT_PATH = _args.output or os.path.join(DATA_DIR, "pen_grasp_robomimic.hdf5")

# State 구조 (25차원)
# joint_pos(6) + joint_vel(6) + ee_pose(7) + pen_pos(3) + pen_axis(3)
STATE_SLICES = {
    "joint_pos": (0, 6),      # 관절 위치
    "joint_vel": (6, 12),     # 관절 속도
    "ee_pos": (12, 15),       # EE 위치
    "ee_quat": (15, 19),      # EE 쿼터니언
    "pen_pos": (19, 22),      # 펜 위치
    "pen_axis": (22, 25),     # 펜 축 방향
}

# 검증 비율
VALID_RATIO = 0.1


def convert():
    print("=" * 60)
    print("Zarr → robomimic HDF5 변환")
    print("=" * 60)

    # 입력 데이터 로드
    print(f"\n입력: {INPUT_PATH}")
    root_in = zarr.open(INPUT_PATH, mode='r')

    states = np.array(root_in['data/state'])
    actions = np.array(root_in['data/action'])
    episode_ends = np.array(root_in['meta/episode_ends'])

    num_episodes = len(episode_ends)
    print(f"총 에피소드: {num_episodes}")
    print(f"총 타임스텝: {len(states)}")

    # 출력 HDF5 생성
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    print(f"\n출력: {OUTPUT_PATH}")

    with h5py.File(OUTPUT_PATH, 'w') as f:
        # data 그룹 생성
        data_grp = f.create_group('data')

        # 에피소드별로 변환
        prev_end = 0
        demo_lengths = []

        for i in tqdm(range(num_episodes), desc="에피소드 변환"):
            end = episode_ends[i]

            # 에피소드 데이터 추출
            ep_states = states[prev_end:end]
            ep_actions = actions[prev_end:end]
            ep_len = len(ep_states)
            demo_lengths.append(ep_len)

            # demo 그룹 생성
            demo_grp = data_grp.create_group(f'demo_{i}')

            # obs 그룹 생성
            obs_grp = demo_grp.create_group('obs')

            # state를 개별 관측값으로 분리
            for obs_name, (start, end_idx) in STATE_SLICES.items():
                obs_grp.create_dataset(
                    obs_name,
                    data=ep_states[:, start:end_idx].astype(np.float32)
                )

            # actions
            demo_grp.create_dataset('actions', data=ep_actions.astype(np.float32))

            # rewards (더미 - robomimic에서 필요하지만 BC에서는 사용 안 함)
            demo_grp.create_dataset('rewards', data=np.zeros(ep_len, dtype=np.float32))

            # dones
            dones = np.zeros(ep_len, dtype=np.float32)
            dones[-1] = 1.0  # 마지막 스텝에서 done
            demo_grp.create_dataset('dones', data=dones)

            # 속성 추가
            demo_grp.attrs['num_samples'] = ep_len

            prev_end = end

        # 전체 속성
        data_grp.attrs['total'] = num_episodes
        data_grp.attrs['env_args'] = '{}'  # 환경 인자 (필요시 수정)

        # train/valid 마스크 생성
        mask_grp = f.create_group('mask')

        num_valid = int(num_episodes * VALID_RATIO)
        num_train = num_episodes - num_valid

        # 랜덤 셔플
        indices = np.arange(num_episodes)
        np.random.seed(42)
        np.random.shuffle(indices)

        train_indices = sorted(indices[:num_train])
        valid_indices = sorted(indices[num_train:])

        # 마스크 저장 (문자열 형식)
        train_demos = [f'demo_{i}' for i in train_indices]
        valid_demos = [f'demo_{i}' for i in valid_indices]

        mask_grp.create_dataset('train', data=np.array(train_demos, dtype='S'))
        mask_grp.create_dataset('valid', data=np.array(valid_demos, dtype='S'))

        # 메타데이터
        f.attrs['env_name'] = 'PenGrasp'
        f.attrs['env_type'] = 2  # Isaac Lab

    # 결과 출력
    print("\n" + "=" * 60)
    print("변환 완료!")
    print("=" * 60)
    print(f"총 에피소드: {num_episodes}")
    print(f"  - Train: {num_train}")
    print(f"  - Valid: {num_valid}")
    print(f"\n에피소드 길이 통계:")
    print(f"  - 평균: {np.mean(demo_lengths):.1f}")
    print(f"  - 최소: {np.min(demo_lengths)}")
    print(f"  - 최대: {np.max(demo_lengths)}")

    # 검증
    print("\n=== HDF5 구조 검증 ===")
    with h5py.File(OUTPUT_PATH, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        print(f"data keys: {list(f['data'].keys())[:5]}...")
        demo0 = f['data/demo_0']
        print(f"demo_0 keys: {list(demo0.keys())}")
        print(f"demo_0/obs keys: {list(demo0['obs'].keys())}")
        for k, v in demo0['obs'].items():
            print(f"  {k}: {v.shape}")
        print(f"demo_0/actions: {demo0['actions'].shape}")


if __name__ == "__main__":
    convert()
