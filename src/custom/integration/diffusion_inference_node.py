#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 3.4 하위 제어단 LeRobot Diffusion Policy 추론 파이프라인 구성

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
import numpy as np

class DiffusionInferenceNode(Node):
    def __init__(self, alpha=0.2):
        super().__init__('diffusion_inference_node')
        self.alpha = alpha  # EMA 평활화 가중치 파라미터 고정
        
        # 1장 규격 사양 7차원 상대 변위 벡터 초기화
        # [ΔX, ΔY, ΔZ, Δqx, Δqy, Δqz, Gripper_stroke]
        self.previous_action = np.zeros(7) 
        
        # 1. 30Hz 주기 타이머 개통을 통해 하드웨어 제어 주파수 가속 보장
        self.timer = self.create_timer(0.033, self.inference_loop)
        
        # 2. 로봇 제어 신호 퍼블리셔 및 이미지 버퍼 바인딩
        self.action_pub = self.create_publisher(Twist, '/dsr01/servol_cmd', 10)
        self.get_logger().info("🔒 [CHAPTER 3] LeRobot Diffusion Policy 30Hz 실시간 추론 코어 정상 가동.")

    def inference_loop(self):
        # 3. 50단계 가우시안 노이즈 제거(Denoising Process) 에뮬레이션
        # 실제 모델 탑재 시에는 ONNX Runtime / TensorRT 인프라가 작동하는 구간
        raw_model_output = self.simulate_denoising_process()
        
        # 4. 실물 하드웨어 보호를 위한 지수이동평균(EMA) 로우패스 필터 주입
        smoothed_action = self.apply_ema_filter(raw_model_output)
        
        # 5. 두산 제어 API 및 cuRobo 인터록을 위한 ROS 2 메시지 퍼블리시 전개
        cmd_msg = Twist()
        cmd_msg.linear.x = smoothed_action[0]
        cmd_msg.linear.y = smoothed_action[1]
        cmd_msg.linear.z = smoothed_action[2]
        cmd_msg.angular.x = smoothed_action[3]
        cmd_msg.angular.y = smoothed_action[4]
        cmd_msg.angular.z = smoothed_action[5]
        
        self.action_pub.publish(cmd_msg)
        self.previous_action = smoothed_action

    def simulate_denoising_process(self):
        # LeRobot 가판대 조작(LIBERO) 가중치 기반 예측 변위 임의 생성 (테스트 가드)
        # 말단 공간 기하 제어 오차를 수정하기 위한 30Hz 노이즈 감소 트랙
        return np.array([5.0, -2.0, 1.5, 0.01, -0.01, 0.0, 0.8])

    def apply_ema_filter(self, current_action):
        # 수식 기반 물리 제어 필터링 구현: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
        return self.alpha * current_action + (1.0 - self.alpha) * self.previous_action

if __name__ == '__main__':
    rclpy.init()
    node = DiffusionInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()