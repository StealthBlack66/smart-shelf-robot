#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 3.3 비동기 프로세스 통신 (ROS 2 Action Server) 인터페이스 구성

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
# 가상의 프로젝트 전용 액션 인터페이스 가정
from custom_interfaces.action import ShelfManipulate 

class VlaBridgeNode(Node):
    def __init__(self):
        super().__init__('vla_bridge_node')
        self._latest_image = None
        
        # 1. 시뮬레이터/실물 Hand-in-Eye 이미지 구독 개통
        self.img_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10
        )
        
        # 2. 하위 제어단을 위한 ROS 2 Action Server 활성화
        self._action_server = ActionServer(
            self,
            ShelfManipulate,
            'shelf_manipulate',
            self.execute_callback
        )
        self.get_logger().info("🏗️ [CHAPTER 3] VLA-Diffusion 비동기 제어 브릿지 액션 서버 가동 완료.")

    def image_callback(self, msg):
        self._latest_image = msg

    async def execute_callback(self, goal_handle):
        self.get_logger().info(f"📥 고수준 자연어 명령 접수: '{goal_handle.request.instruction}'")
        
        feedback_msg = ShelfManipulate.Feedback()
        result = ShelfManipulate.Result()

        if self._latest_image is None:
            self.get_logger().warn("⚠️ 입력 카메라 프레임이 인입되지 않았습니다.")
            goal_handle.abort()
            result.success = False
            return result

        # 3. OpenVLA / pi0 추론 시뮬레이션 (1~2Hz 비동기 루프 가동 가정)
        self.get_logger().info("🧠 OpenVLA 4-bit LoRA 인지 엔진 역추론 중...")
        
        # 가상의 가판대 빈 공간 매대 타깃 포즈 설정 (매뉴얼 규격 매핑)
        target_waypoint = PoseStamped()
        target_waypoint.header.stamp = self.get_clock().now().to_msg()
        target_waypoint.header.frame_id = "shelf_base"
        target_waypoint.pose.position.x = 0.450  # 450mm
        target_waypoint.pose.position.y = 0.120  # 120mm
        target_waypoint.pose.position.z = 0.300  # 300mm
        target_waypoint.pose.orientation.w = 1.0

        # 4. 하위 제어기(Diffusion Policy)가 연산을 마칠 때까지 30Hz 주기로 피드백 스트리밍
        for step in range(1, 101):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("❌ 상위 명령 취소 트리거 수신.")
                result.success = False
                return result

            feedback_msg.current_status = f"하위 궤적 Denoising 매핑 중... 진행률: {step}%"
            feedback_msg.macro_target = target_waypoint
            goal_handle.publish_feedback(feedback_msg)
            
            # 주파수 간극 처리를 위한 고속 논블로킹 슬립 (30Hz 대응)
            time.sleep(0.033)

        goal_handle.succeed()
        result.success = True
        self.get_logger().info("🏁 [SUCCESS] 가판대 정렬 상위 플래닝 시퀀스 무결 종결.")
        return result

if __name__ == '__main__':
    import time
    rclpy.init()
    node = VlaBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
