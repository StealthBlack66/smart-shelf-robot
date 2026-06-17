#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 4.2 하위 Diffusion Policy의 실물 로봇 servol_rt API 서보 제어 연동 브릿지

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
# 두산 공식 DART platform Humble 라이브러리 인터페이스 바인딩 가정
from dsr_msgs.msg import RobotAction 

class DoosanRealHardwareBridge(Node):
    def __init__(self):
        super().__init__('doosan_real_hardware_bridge')
        
        # 1. 30Hz 디퓨전 액션 토픽 수집 리스너 개통
        self.action_sub = self.create_subscription(
            Twist, '/dsr01/servol_cmd', self.hardware_routing_callback, 10
        )
        
        # 2. 두산 공식 실물 컨트롤러 서보 API 엔드포인트 개통
        self.dsr_servo_pub = self.create_publisher(RobotAction, '/dsr01/servo_vector_cmd', 10)
        self.get_logger().info("🚀 [CHAPTER 4] 두산 실물 로봇 servol_rt 실시간 통신 브릿지 개통 완료 (IP: 110.120.1.39).")

    def hardware_routing_callback(self, msg):
        # 3. 모델이 출력한 TCP 밀리미터 변위를 두산 컨트롤러 입력 스펙으로 직렬화 변환
        dsr_cmd = RobotAction()
        
        # 상대 변위 벡터 값을 실물 로봇 드라이버 자료형 구조체로 사상
        dsr_cmd.data = [
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ]
        
        # 4. servol_rt 커널 통로로 초저지연 다운스트림 패킷 투입
        self.dsr_servo_pub.publish(dsr_cmd)

if __name__ == '__main__':
    rclpy.init()
    node = DoosanRealHardwareBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
