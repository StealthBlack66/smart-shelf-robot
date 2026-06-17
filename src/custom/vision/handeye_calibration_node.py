#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 4.1 실물 눈-손 캘리브레이션 매트릭스 주입 및 동차 변환 오프셋 정렬

import rclpy
from rclpy.node import Node
import numpy as np
import cv2

class HandEyeCalibrationNode(Node):
    def __init__(self):
        super().__init__('handeye_calibration_node')
        self.get_logger().info("📐 [CHAPTER 4] 실물 아루코 마커 기반 캘리브레이션 모듈 시동.")
        
        # 1. 체스보드/아루코 마커 기반 실측 데이터 매트릭스 정의
        # 캘리브레이션 융합 연산을 거쳐 확보한 4x4 동차 변환 행렬 ($^{Flange}T_{Camera}$)
        self.T_flange_camera = np.array([
            [ 0.0000, -1.0000,  0.0000,  0.0500],  # X축 오프셋 50mm 사상
            [ 1.0000,  0.0000,  0.0000,  0.0000],  # Y축 오프셋
            [ 0.0000,  0.0000,  1.0000,  0.0300],  # Z축 오프셋 30mm 사상
            [ 0.0000,  0.0000,  0.0000,  1.0000]
        ])
        
    def transform_camera_to_base(self, T_base_flange, P_camera_target):
        """
        수식 반영: P_base = T_base_flange * T_flange_camera * P_camera
        """
        # 2. 호스트 캘리브레이션 적용 및 가판대 타깃 좌표 최종 정렬
        T_base_camera = np.dot(T_base_flange, self.T_flange_camera)
        P_base_target = np.dot(T_base_camera, P_camera_target)
        return P_base_target

if __name__ == '__main__':
    rclpy.init()
    node = HandEyeCalibrationNode()
    # 좌표 변환 수식 무결성 검증을 위한 더미 데이터 테스트 실행
    T_bf_dummy = np.eye(4)
    P_c_dummy = np.array([0.0, 0.0, 0.200, 1.0]) # 카메라 정면 200mm 지점 물체
    res = node.transform_camera_to_base(T_bf_dummy, P_c_dummy)
    print(f"🎯 변환 계산 완료 -> 실제 로봇 베이스 기준 타깃 기하 위치: {res[:3]}")
    rclpy.shutdown()