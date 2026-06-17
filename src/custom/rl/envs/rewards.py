# -*- coding: utf-8 -*-
# ~/smart-shelf-robot/src/custom/rl/envs/rewards.py 

import torch 

def compute_shelf_rewards( 
    product_pos: torch.Tensor,  
    target_pos: torch.Tensor,  
    product_rot: torch.Tensor,  
    w_1: float = 2.0,  
    w_2: float = 1.0 
) -> torch.Tensor: 
    """ 
    Tensor 기반 병렬 환경 대응 보상 연산 엔진 (Isaac Lab 가속) 
    """ 
    # 1. 타겟 매대 격자점까지의 거리 패널티 계산 
    dist = torch.norm(product_pos - target_pos, dim=-1) 
    r_dist = -torch.square(dist) 
        
    # 2. 물품의 상방 회전 벡터 추출 (Quaternion에서 Z축 방향 벡터 계산) [cite: 240, 241]
    # product_rot layout: [qw, qx, qy, qz] 
    qw, qx, qy, qz = product_rot[:, 0], product_rot[:, 1], product_rot[:, 2], product_rot[:, 3] 
        
    # 회전 행렬의 3번째 열 벡터(물체의 로컬 Z축) 연산 
    z_bx = 2 * (qx * qz + qw * qy) 
    z_by = 2 * (qy * qz - qw * qx) 
    z_bz = qw**2 - qx**2 - qy**2 + qz**2 
        
    # 세계 좌표계의 수직 축인 [0, 0, 1]과의 내적값은 결국 z_bz와 동일함 [cite: 241, 242]
    r_align = z_bz 
        
    # 3. 통합 가속 보상 텐서 일괄 반환 
    return (w_1 * r_dist) + (w_2 * r_align) 