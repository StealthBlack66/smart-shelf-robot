# -*- coding: utf-8 -*-
# 2.4 Hand-in-Eye 카메라 가상 링크 연동 및 관측 공간 정의
# ~/smart-shelf-robot/src/custom/rl/cfgs/sensory_cfg.py 

from omni.isaac.lab.sensors import CameraCfg 

HAND_IN_EYE_CAMERA_CFG = CameraCfg( 
    prim_path="{ENV_REGEX_EXPR}/robot/link_6/realsense_camera", 
    update_period=0.033,  # 데이터 레코더 주기 사양과 완벽 동기화되는 30Hz 관측 루프 개통 
    data_types=["rgb", "distance_to_image_plane"],  # 상위 VLA 상황판단 인지 및 파지용 깊이 맵 동시 수집 
    width=640, 
    height=480, 
    # 실물 Hand-in-Eye 캘리브레이션 매트릭스 결과물 ($^{Flange}T_{Camera}$) 오프셋 주입 
    pos=(0.05, 0.0, 0.03),  # 물리적 6축 Flange 표면 중심 기준 RealSense D435 하드웨어 오프셋 (m 단위) 
    rot=(0.5, -0.5, 0.5, -0.5),  # 카메라 렌즈 광학 전방 축(Z-forward) 매칭 정렬용 단위 쿼터니언 고정 
) 