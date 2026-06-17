#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 5.1 로봇 하드웨어 기동 에러 및 실시간 API 피드백 트러블슈팅

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, Bool
import os
import subprocess
import re

class DrNetworkMonitor(Node):
    def __init__(self):
        super().__init__('dr_network_monitor')
        
        # 두산 로봇 제어기 표준 내부 IP 설정
        self.robot_ip = "110.120.1.39" 
        self.latency_threshold_ms = 2.0
        self.consecutive_error_limit = 3
        self.error_count = 0
        
        # 1. 네트워크 세이프티 상태 퍼블리셔 개통
        self.safety_status_pub = self.create_publisher(Bool, '/dsr01/network_safety_ok', 10)
        
        # 2. 100Hz 초고속 실시간 실효 지연 감시 타이머 개통 (10ms 주기)
        self.timer = self.create_timer(0.01, self.check_network_health)
        self.get_logger().info(f"📊 [CHAPTER 5] 두산 실시간 servol_rt 네트워크 모니터 활성화 (대상: {self.robot_ip})")

    def check_network_health(self):
        status_msg = Bool()
        status_msg.data = True
        
        try:
            # 리눅스 시스템 ping 명령어를 통해 1회의 초고속 rtt 계측 실행
            cmd = ["ping", "-c", "1", "-W", "1", self.robot_ip]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                self.error_count += 1
                self.get_logger().warn("⚠️ [통신 경고] 두산 로봇 제어기 패킷 유실 발생.")
            else:
                # 정규식을 이용한 시간(ms) 매칭 및 파싱
                match = re.search(r"time=([\d\.]+)\s+ms", result.stdout)
                if match:
                    latency = float(match.group(1))
                    
                    # 2ms 세이프티 임계치 제약 조건 검사
                    if latency > self.latency_threshold_ms:
                        self.error_count += 1
                        self.get_logger().warn(f"⏳ [지연 경고] 실시간 Latency 임계치 초과: {latency} ms")
                    else:
                        self.error_count = max(0, self.error_count - 1)
                        
        except Exception as e:
            self.error_count += 1
            self.get_logger().error(f"❌ 네트워크 모니터 커널 예외 발생: {str(e)}")

        # 3. 3회 연속 실패 시 하위 제어단 제약 조건 차단 트리거 전개
        if self.error_count >= self.consecutive_error_limit:
            self.get_logger().fatal("🚨 [네트워크 비상] servol_rt 실시간 패킷 완전 오염 감지. 안전 감속 모드 강제 트리거.")
            status_msg.data = False
            self.error_count = self.consecutive_error_limit # 카운터 오버플로우 방지

        self.safety_status_pub.publish(status_msg)

if __name__ == '__main__':
    rclpy.init()
    node = DrNetworkMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
