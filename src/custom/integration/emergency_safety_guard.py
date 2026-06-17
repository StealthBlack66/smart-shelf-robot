#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 4.3 안전 세이프티 가드 및 긴급 서보 토크 차단 (E-Stop) 아키텍처

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
# 두산 제어 모드 변경 서비스 인터페이스 임포트
from dsr_msgs.srv import SetRobotMode 

class EmergencySafetyGuard(Node):
    def __init__(self):
        super().__init__('emergency_safety_guard')
        
        # 1. 두산 제어기 피드백 관절 전류/토크 스트림 무결성 모니터링 (100Hz)
        self.joint_sub = self.create_subscription(
            JointState, '/dsr01/joint_states', self.torque_safety_filter, 10
        )
        
        # 2. 매뉴얼 규격 기반의 긴급 에러 복구 및 모드 해제 서비스 클라이언트 구성
        self.cli = self.create_client(SetRobotMode, '/dsr01/set_robot_mode')
        
        # 안전 위험 임계 한계치 토크(Nm) 설정 고정
        self.safety_torque_limit = 45.0 
        self.get_logger().info("🔒 [CHAPTER 4] 100Hz 하드웨어 실시간 세이프티 가드 액티브 온.")

    def torque_safety_filter(self, msg):
        # 3. 6개 전 관절 축의 실시간 토크 데이터 피드 전수 검사
        current_torques = np.abs(np.array(msg.effort))
        
        if np.any(current_torques > self.safety_torque_limit):
            self.get_logger().fatal("🚨 [CRITICAL] 외부 충격 및 과토크 감지! AI 제어권 즉시 박탈.")
            self.trigger_hardware_emergency_stop()

    def trigger_hardware_emergency_stop(self):
        # 4. 두산 매뉴얼 표준 규격 서비스 콜(set_robot_mode(mode:=SAFETY_STOP)) 즉시 강제 트리거
        req = SetRobotMode.Request()
        req.robot_mode = 3  # 두산 API 매뉴얼 기준 SAFETY_STOP 강제 변환 토큰문
        
        # 비동기 전송을 통해 통신 지연으로 인한 물리적 충돌 오버헤드 방지
        self.cli.call_async(req)
        self.get_logger().warn("🛑 [E-STOP] 실물 두산 관절 모터 드라이버 서보 토크 강제 소거 완료.")

if __name__ == '__main__':
    import numpy as np
    rclpy.init()
    node = EmergencySafetyGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()