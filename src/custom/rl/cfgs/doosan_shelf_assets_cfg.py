# -*- coding: utf-8 -*-
# 2.3 두산 E0509 및 스마트 가판대 USD 에셋 임포트 명세
# ~/smart-shelf-robot/src/custom/rl/cfgs/doosan_shelf_assets_cfg.py 

import os
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg

# 1. 두산 E0509 협동로봇 에셋 기하학적 강체 컴포넌트 정의 
DOOSAN_E0509_CFG = ArticulationCfg( 
    prim_path="{ENV_REGEX_EXPR}/robot", 
    spawn=sim_utils.UsdFileCfg( 
        usd_path=os.path.expanduser("~/smart-shelf-robot/src/custom/rl/assets/doosan_e0509.usd"), 
        rigid_props=sim_utils.RigidBodyPropertiesCfg( 
            disable_gravity=False, 
            retain_accelerations=False, 
            linear_damping=0.0, 
            angular_damping=0.0, 
            max_linear_velocity=1000.0, 
            max_angular_velocity=10.0, 
        ), 
        articulation_props=sim_utils.ArticulationRootPropertiesCfg( 
            enable_self_collisions=True,  
            solver_position_iteration_count=8,  
            solver_velocity_iteration_count=2 
        ), 
    ), 
    init_state=ArticulationCfg.InitialStateCfg( 
        joint_pos={ 
            "joint1": 0.0, 
            "joint2": 0.0, 
            "joint3": 1.57,  # 가판대 진열 접근을 위한 기본 홈 포즈(Home Pose) 설정 
            "joint4": 0.0, 
            "joint5": 1.57, 
            "joint6": 0.0, 
        }, 
        pos=(0.0, 0.0, 0.0), 
        rot=(1.0, 0.0, 0.0, 0.0), 
    ), 
) 

# 2. 편의점 스마트 진열대 가판대 환경 구조화 정의 
SMART_SHELF_CFG = AssetBaseCfg( 
    prim_path="{ENV_REGEX_EXPR}/smart_shelf", 
    spawn=sim_utils.UsdFileCfg( 
        usd_path=os.path.expanduser("~/smart-shelf-robot/src/custom/rl/assets/convenience_shelf.usd"), 
        # 물품 파지 충격력에 가판대가 물리적으로 튕겨나가지 않도록 강제 고정 고정체(Static Object) 처리 
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True) 
    ) 
) 
