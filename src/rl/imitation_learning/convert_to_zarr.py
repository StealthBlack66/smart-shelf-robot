#!/usr/bin/env python3
"""
NPZ 궤적 데이터를 Diffusion Policy용 Zarr 형식으로 변환

Diffusion Policy 데이터 형식:
- state: (N, state_dim) - 로봇 상태 (joint positions + velocities + ee pose)
- action: (N, action_dim) - 로봇 액션 (joint positions)
- episode_ends: (num_episodes,) - 각 에피소드 끝 인덱스

사용법:
    python convert_to_zarr.py \
        --input_dir ~/generated_trajectories \
        --output_path ~/CoWriteBotRL/data/pen_grasp.zarr
"""

import os
import argparse
import numpy as np
import zarr
from glob import glob
from tqdm import tqdm


def load_trajectories(input_dir):
    """NPZ 궤적 파일들 로드"""
    traj_files = sorted(glob(os.path.join(input_dir, 'traj_*.npz')))

    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {input_dir}")

    print(f"Found {len(traj_files)} trajectory files")

    trajectories = []
    for traj_file in tqdm(traj_files, desc="Loading trajectories"):
        data = np.load(traj_file, allow_pickle=True)
        trajectories.append({
            'joint_positions': data['joint_positions'],
            'joint_velocities': data['joint_velocities'],
            'timestamps': data['timestamps'],
            'ee_poses_world': data['ee_poses_world'],
            'pen_position': data['pen_position'],
            'pen_axis': data['pen_axis'],
        })

    return trajectories


def create_state_action(trajectories, state_type='full', action_type='absolute'):
    """
    State와 Action 배열 생성

    Args:
        trajectories: 궤적 리스트
        state_type: 'full' (joint+vel+ee), 'joint_only', 'joint_vel'
        action_type: 'absolute' (절대 관절 위치), 'delta' (관절 위치 변화량)

    Returns:
        states, actions, episode_ends
    """
    all_states = []
    all_actions = []
    episode_ends = []

    current_idx = 0

    for traj in tqdm(trajectories, desc="Processing trajectories"):
        joint_pos = traj['joint_positions']  # (T, 6)
        joint_vel = traj['joint_velocities']  # (T, 6)
        ee_pose = traj['ee_poses_world']  # (T, 7) - x,y,z,qx,qy,qz,qw
        pen_pos = traj['pen_position']  # (3,)
        pen_axis = traj['pen_axis']  # (3,)

        T = len(joint_pos)

        # State 구성
        if state_type == 'full':
            # joint_pos(6) + joint_vel(6) + ee_pos(3) + ee_quat(4) + pen_pos(3) + pen_axis(3) = 25
            pen_pos_repeated = np.tile(pen_pos, (T, 1))
            pen_axis_repeated = np.tile(pen_axis, (T, 1))
            states = np.concatenate([
                joint_pos,           # 6
                joint_vel,           # 6
                ee_pose[:, :3],      # 3 (position)
                ee_pose[:, 3:],      # 4 (quaternion)
                pen_pos_repeated,    # 3
                pen_axis_repeated,   # 3
            ], axis=1)
        elif state_type == 'joint_vel':
            # joint_pos(6) + joint_vel(6) = 12
            states = np.concatenate([joint_pos, joint_vel], axis=1)
        elif state_type == 'joint_only':
            # joint_pos(6) only
            states = joint_pos
        else:
            raise ValueError(f"Unknown state_type: {state_type}")

        # Action 구성
        # BC에서 action[t]는 "다음에 가야 할 위치"를 의미
        # 따라서 action[t] = joint_pos[t+1]
        if action_type == 'absolute':
            # 다음 타임스텝의 관절 위치를 액션으로 사용
            # action[t] = joint_pos[t+1], 마지막은 현재 위치 유지
            actions = np.zeros_like(joint_pos)
            actions[:-1] = joint_pos[1:]  # action[t] = joint_pos[t+1]
            actions[-1] = joint_pos[-1]   # 마지막은 현재 위치 유지
        elif action_type == 'delta':
            # 관절 위치 변화량을 액션으로 사용
            actions = np.zeros_like(joint_pos)
            actions[1:] = joint_pos[1:] - joint_pos[:-1]
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        all_states.append(states)
        all_actions.append(actions)

        current_idx += T
        episode_ends.append(current_idx)

    # 모든 데이터 연결
    all_states = np.concatenate(all_states, axis=0).astype(np.float32)
    all_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    return all_states, all_actions, episode_ends


def save_to_zarr(states, actions, episode_ends, output_path, metadata=None):
    """Zarr 형식으로 저장"""

    # 기존 파일 삭제
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)

    # Zarr 저장소 생성
    root = zarr.open(output_path, mode='w')

    # 데이터 저장 (압축 사용)
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)

    root.create_dataset(
        'data/state',
        data=states,
        chunks=(1000, states.shape[1]),
        compressor=compressor
    )

    root.create_dataset(
        'data/action',
        data=actions,
        chunks=(1000, actions.shape[1]),
        compressor=compressor
    )

    root.create_dataset(
        'meta/episode_ends',
        data=episode_ends,
        compressor=compressor
    )

    # 메타데이터 저장
    if metadata:
        root.attrs['metadata'] = metadata

    root.attrs['state_dim'] = states.shape[1]
    root.attrs['action_dim'] = actions.shape[1]
    root.attrs['num_episodes'] = len(episode_ends)
    root.attrs['total_timesteps'] = len(states)

    print(f"\nSaved to: {output_path}")
    print(f"  State shape: {states.shape}")
    print(f"  Action shape: {actions.shape}")
    print(f"  Episodes: {len(episode_ends)}")
    print(f"  Total timesteps: {len(states)}")

    return root


def main():
    parser = argparse.ArgumentParser(description='NPZ to Zarr 변환 (Diffusion Policy용)')
    parser.add_argument('--input_dir', type=str, default='~/generated_trajectories',
                        help='NPZ 궤적 파일들이 있는 폴더')
    parser.add_argument('--output_path', type=str, default='~/CoWriteBotRL/data/pen_grasp.zarr',
                        help='출력 Zarr 파일 경로')
    parser.add_argument('--state_type', type=str, default='full',
                        choices=['full', 'joint_vel', 'joint_only'],
                        help='State 구성 방식')
    parser.add_argument('--action_type', type=str, default='absolute',
                        choices=['absolute', 'delta'],
                        help='Action 구성 방식')
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_path = os.path.expanduser(args.output_path)

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 60)
    print("NPZ → Zarr 변환 (Diffusion Policy용)")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_path}")
    print(f"State type: {args.state_type}")
    print(f"Action type: {args.action_type}")
    print("=" * 60)

    # 궤적 로드
    trajectories = load_trajectories(input_dir)

    # State/Action 생성
    states, actions, episode_ends = create_state_action(
        trajectories,
        state_type=args.state_type,
        action_type=args.action_type
    )

    # 메타데이터
    metadata = {
        'state_type': args.state_type,
        'action_type': args.action_type,
        'source': input_dir,
        'robot': 'Doosan E0509 + Gripper',
        'task': 'Pen Grasp',
    }

    # Zarr로 저장
    save_to_zarr(states, actions, episode_ends, output_path, metadata)

    print("\n변환 완료!")


if __name__ == '__main__':
    main()
