#!/usr/bin/env python3
"""
robomimic을 사용한 Behavioral Cloning (BC) 학습

사용법:
    # 기본 학습
    python train_bc.py

    # 에폭 수 변경
    python train_bc.py --epochs 1000

    # 배치 사이즈 변경
    python train_bc.py --batch_size 128

설치 (robomimic이 없는 경우):
    pip install robomimic

학습 후:
    - 체크포인트: pen_grasp_rl/checkpoints/robomimic_bc/
    - TensorBoard: tensorboard --logdir pen_grasp_rl/checkpoints/robomimic_bc/
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)

# 데이터 경로
HDF5_PATH = os.path.join(PROJECT_DIR, "data/pen_grasp_robomimic.hdf5")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "robomimic_config/bc_pen_grasp.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "pen_grasp_rl/checkpoints/robomimic_bc")


def check_robomimic():
    """robomimic 설치 확인"""
    try:
        import robomimic
        print(f"robomimic version: {robomimic.__version__}")
        return True
    except ImportError:
        print("=" * 60)
        print("robomimic이 설치되어 있지 않습니다!")
        print("설치 방법:")
        print("  pip install robomimic")
        print("=" * 60)
        return False


def train_with_robomimic(args):
    """robomimic을 사용한 BC 학습"""
    import robomimic.utils.train_utils as TrainUtils
    import robomimic.utils.torch_utils as TorchUtils
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.file_utils as FileUtils
    from robomimic.config import config_factory
    from robomimic.algo import algo_factory
    from robomimic.utils.log_utils import DataLogger

    # 설정 로드
    print(f"\n설정 파일: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        ext_cfg = json.load(f)

    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)

    # 명령행 인자로 덮어쓰기
    if args.epochs:
        config.train.num_epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.algo.optim_params.policy.learning_rate.initial = args.lr

    config.train.data = HDF5_PATH
    config.train.output_dir = OUTPUT_DIR

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_dir = os.path.join(OUTPUT_DIR, "logs")
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 시드 설정
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n" + "=" * 60)
    print("Behavioral Cloning (BC) 학습")
    print("=" * 60)
    print(f"데이터: {HDF5_PATH}")
    print(f"에폭: {config.train.num_epochs}")
    print(f"배치 사이즈: {config.train.batch_size}")
    print(f"학습률: {config.algo.optim_params.policy.learning_rate.initial}")
    print(f"출력: {OUTPUT_DIR}")
    print("=" * 60)

    # 관측 설정 초기화
    ObsUtils.initialize_obs_utils_with_config(config)

    # 데이터셋 메타데이터 로드
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )

    print(f"\n관측 shape: {shape_meta['all_shapes']}")
    print(f"액션 차원: {shape_meta['ac_dim']}")

    # 모델 생성
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    print(f"Device: {device}")

    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    print(f"\n모델 구조:")
    print(model)

    # 데이터 로더 생성
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"]
    )

    train_sampler = trainset.get_dataset_sampler()
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    if config.experiment.validate and validset is not None:
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=min(config.train.num_data_workers, 1),
            drop_last=True,
        )
    else:
        valid_loader = None

    print(f"\n학습 데이터: {len(trainset)} 샘플")
    if validset:
        print(f"검증 데이터: {len(validset)} 샘플")

    # 로거 설정
    data_logger = DataLogger(
        log_dir,
        config=config,
        log_tb=config.experiment.logging.log_tb
    )

    # 설정 저장
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 학습 루프
    best_valid_loss = float('inf')
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    print("\n학습 시작...")
    for epoch in range(1, config.train.num_epochs + 1):
        # 학습
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps
        )
        model.on_epoch_end(epoch)

        # 로깅
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
            else:
                data_logger.record(f"Train/{k}", v, epoch)

        # 검증
        if valid_loader is not None and config.experiment.validate:
            with torch.no_grad():
                valid_log = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps
                )

            for k, v in valid_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)

            valid_loss = valid_log.get("Loss", float('inf'))
        else:
            valid_loss = step_log.get("Loss", float('inf'))

        # 출력
        train_loss = step_log.get("Loss", 0)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{config.train.num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Valid Loss: {valid_loss:.6f}")

        # 체크포인트 저장
        should_save = False
        ckpt_name = f"model_epoch_{epoch}"

        # 주기적 저장
        if config.experiment.save.every_n_epochs and epoch % config.experiment.save.every_n_epochs == 0:
            should_save = True

        # Best validation 저장
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if config.experiment.save.on_best_validation:
                should_save = True
                ckpt_name = f"best_model_epoch_{epoch}"

        # 마지막 에폭 저장
        if epoch == config.train.num_epochs:
            should_save = True
            ckpt_name = "final_model"

        if should_save:
            env_meta = {"env_name": "PenGrasp", "type": 2}
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, f"{ckpt_name}.pth"),
            )

    data_logger.close()

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"Best validation loss: {best_valid_loss:.6f}")
    print(f"체크포인트: {ckpt_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")


def train_simple_bc(args):
    """robomimic 없이 간단한 BC 학습 (fallback)"""
    import h5py
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split

    print("\n" + "=" * 60)
    print("Simple BC 학습 (robomimic 없음)")
    print("=" * 60)

    # 데이터셋 클래스
    class BCDataset(Dataset):
        def __init__(self, hdf5_path, demo_keys):
            self.hdf5_path = hdf5_path
            self.demo_keys = demo_keys
            self.data = []

            with h5py.File(hdf5_path, 'r') as f:
                for key in demo_keys:
                    demo = f[f'data/{key}']
                    obs = np.concatenate([
                        demo['obs/joint_pos'][:],
                        demo['obs/joint_vel'][:],
                        demo['obs/ee_pos'][:],
                        demo['obs/ee_quat'][:],
                        demo['obs/pen_pos'][:],
                        demo['obs/pen_axis'][:],
                    ], axis=1)
                    actions = demo['actions'][:]

                    for i in range(len(obs)):
                        self.data.append((obs[i], actions[i]))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            obs, action = self.data[idx]
            return torch.FloatTensor(obs), torch.FloatTensor(action)

    # Policy 네트워크
    class BCPolicy(nn.Module):
        def __init__(self, obs_dim=25, action_dim=6, hidden_dims=[512, 512, 256]):
            super().__init__()
            layers = []
            in_dim = obs_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ])
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, action_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, obs):
            return self.net(obs)

    # 데이터 로드
    with h5py.File(HDF5_PATH, 'r') as f:
        train_keys = [k.decode() for k in f['mask/train'][:]]
        valid_keys = [k.decode() for k in f['mask/valid'][:]]

    print(f"데이터: {HDF5_PATH}")
    print(f"Train demos: {len(train_keys)}")
    print(f"Valid demos: {len(valid_keys)}")

    train_dataset = BCDataset(HDF5_PATH, train_keys)
    valid_dataset = BCDataset(HDF5_PATH, valid_keys)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    # 모델, 옵티마이저, 손실함수
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BCPolicy().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    print(f"\n모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 출력 디렉토리
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 학습 루프
    best_valid_loss = float('inf')

    print(f"\n학습 시작... (epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr})")
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for obs, action in train_loader:
            obs, action = obs.to(device), action.to(device)

            optimizer.zero_grad()
            pred_action = model(obs)
            loss = criterion(pred_action, action)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Valid
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for obs, action in valid_loader:
                obs, action = obs.to(device), action.to(device)
                pred_action = model(obs)
                loss = criterion(pred_action, action)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)

        scheduler.step()

        # 출력
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Valid Loss: {valid_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Best 저장
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, os.path.join(ckpt_dir, "best_model.pth"))

        # 주기적 저장
        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth"))

    # 최종 저장
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': valid_loss,
    }, os.path.join(ckpt_dir, "final_model.pth"))

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"Best validation loss: {best_valid_loss:.6f}")
    print(f"체크포인트: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description='BC 학습')
    parser.add_argument('--epochs', type=int, default=500, help='학습 에폭')
    parser.add_argument('--batch_size', type=int, default=256, help='배치 사이즈')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--simple', action='store_true', help='robomimic 없이 간단한 BC 사용')
    args = parser.parse_args()

    if args.simple or not check_robomimic():
        print("\nSimple BC 모드로 학습합니다...")
        train_simple_bc(args)
    else:
        train_with_robomimic(args)


if __name__ == "__main__":
    main()
