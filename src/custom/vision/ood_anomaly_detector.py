#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 5.2 모방학습 분포 외 상황(Out-of-Distribution) 인지 실패 조치 매뉴얼
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np

class OodAnomalyDetector(Node):
    def __init__(self):
        super().__init__('ood_anomaly_detector')
        
        # 1. 실물 카메라 영상 이미지 토픽 리스너 바인딩
        self.img_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.process_image_anomaly, 10
        )
        
        # 2. 로봇 제어권 강제 정지 명령을 위한 퍼블리셔 개통
        self.safety_cmd_pub = self.create_publisher(String, '/dsr01/safety_lock_cmd', 10)
        self.confidence_threshold = 0.80  # 신뢰도 세이프티 가드 라인 80% 확정
        
        self.get_logger().info("🔒 [CHAPTER 5] VLA 잠재 공간 기반 OOD 이상치 실시간 탐지기 가동.")

    def process_image_anomaly(self, msg):
        # 3. VLA 임베딩 거리를 모사한 잠재 마할라노비스 거리 연산 에뮬레이션
        # 실제 모델 배포단에서는 OpenVLA의 Hidden State Feature Tensor를 받아 연산 처리
        simulated_confidence = self.calculate_latent_distribution_score(msg)
        
        # 4. 80% 미만 신상품 혹은 이상 환경 감지 시 세이프티 액션 인가
        if simulated_confidence < self.confidence_threshold:
            self.get_logger().error(f"🚨 [OOD 감지] 미지의 신상품 혹은 미학습 배치 발견! 신뢰도: {simulated_confidence:.4f}")
            lock_msg = String()
            lock_msg.data = "PAUSE_SYSTEM_FOR_SAFETY"
            self.safety_cmd_pub.publish(lock_msg)

    def calculate_latent_distribution_score(self, img_msg):
        # 훈련 데이터셋 분포 내부를 검사하는 가상 스코어 필터 (테스트 코드용 정상 판정)
        # 실제 배포 시에는 PyTorch 인코더 가중치 연산 결과가 입력됨
        return float(np.random.uniform(0.85, 0.99))

if __name__ == '__main__':
    rclpy.init()
    node = OodAnomalyDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()