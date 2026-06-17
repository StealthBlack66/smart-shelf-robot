#!/usr/bin/env python3
# 1.4 Conda 런타임 종속형 LeRobot 포맷 데이터 레코딩 핵심 스크립트
# -*- coding: utf-8 -*-
# ~/smart-shelf-robot/src/custom/rl/data/doosan_lerobot_dataset_builder.py 

import os
import json
import time
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

class DoosanLeRobotRecorder(Node): 
    def __init__(self, dataset_name="doosan_shelf_v1", output_dir="~/smart-shelf-robot/src/custom/rl/data"): 
        super().__init__('doosan_lerobot_recorder')
        self.bridge = CvBridge()
        
        # Git 추적 마스터 가드 폴더 하위 경로 매핑 
        self.output_dir = os.path.expanduser(os.path.join(output_dir, dataset_name)) 
        self.episodes_dir = os.path.join(self.output_dir, "episodes") 
        os.makedirs(self.episodes_dir, exist_ok=True) 
        
        # LeRobot v3.0 표준 자율 제어 메타데이터 선언 
        self.metadata = { 
            "fps": 30, 
            "robot_type": "doosan_e0509", 
            "camera_type": "realsense_d435_hand_in_eye", 
            "action_space": ["delta_x", "delta_y", "delta_z", "delta_qx", "delta_qy", "delta_qz", "gripper_stroke"] 
        } 

        # 글로벌 캐싱 버퍼 개통 
        self.latest_frame = None 
        self.latest_joints = None 
        self.current_episode_data = [] 
        self.video_writer = None 
        self.episode_idx = 0 
        
        # ROS 2 Humble 토픽 구독 개통 
        self.img_sub = self.create_subscription(Image, '/camera/color/image_raw', self.img_callback, 10) 
        self.joint_sub = self.create_subscription(JointState, '/dsr01/joint_states', self.joint_callback, 10) 
        
        self.get_logger().info("🔒 [Conda 격리환경 1] 두산 ROS 2 안전 데이터 레코더가 정상 활성화되었습니다.")

    def img_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def joint_callback(self, msg):
        # 두산 매뉴얼 표준 사양 피드 백본 접수 (J1 ~ J6 관절 상태값 수신)
        self.latest_joints = list(msg.position)

    def start_episode(self, episode_idx): 
        self.get_logger().info(f"▶ [에피소드 {episode_idx}] 녹화 기동. 물품 진열 시연을 시작하세요...") 
        self.episode_idx = episode_idx 
        self.current_episode_data = [] 
        
        video_path = os.path.join(self.episodes_dir, f"episode_{self.episode_idx:04d}.mp4") 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480)) 

    def record_step(self, timestamp, gripper_val): # dsr_tcp_pose는 구독 버퍼 데이터 활용
        if self.latest_frame is None or self.latest_joints is None: 
            return 
            
        # 1. Hand-in-Eye 실시간 비디오 프레임 압축 직렬화 
        if self.video_writer is not None: 
            self.video_writer.write(self.latest_frame) 

        # 2. LeRobot 규격 표준 컬럼 사상 (30Hz 제어 동기화) 
        step_log = { 
            "timestamp": timestamp, 
            "observation.joint_pos": self.latest_joints[:3],  # 로봇 말단 TCP 위치 
            "observation.joint_rot": self.latest_joints[3:],  # 로봇 말단 회전 쿼터니언 
            "observation.gripper": gripper_val, 
            "action.delta_pos": self.latest_joints[:3], 
            "action.gripper": gripper_val 
        } 
        self.current_episode_data.append(step_log) 

    def stop_episode(self, episode_idx): 
        if self.video_writer is not None: 
            self.video_writer.release() 
            
        if not self.current_episode_data: 
            return 

        # Apache Parquet 기반 고속 시계열 이진 직렬화 파일 마감 
        df = pd.DataFrame(self.current_episode_data) 
        table = pa.Table.from_pandas(df) 
        parquet_path = os.path.join(self.episodes_dir, f"episode_{episode_idx:04d}.parquet") 
        pq.write_table(table, parquet_path) 
        self.get_logger().info(f"■ [Episode {episode_idx}] 저장 완료 -> {parquet_path}") 

    def finalize_dataset(self): 
        with open(os.path.join(self.output_dir, "info.json"), "w") as f: 
            json.dump(self.metadata, f, indent=4) 
        self.get_logger().info(f"🎉 모든 시퀀스가 종료되었습니다. 데이터셋 성공 패키징: {self.output_dir}")

def main(args=None): 
    rclpy.init(args=args) 
    node = DoosanLeRobotRecorder() 
    node.start_episode(0) 
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) 
    dummy_pose = np.array([450.0, 120.0, 300.0, 0.0, 0.0, 0.0, 1.0]) 
    
    start_time = time.time() 
    try: 
        for t in range(90): 
            cv2.putText(dummy_frame, f"Frame {t}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
            node.latest_frame = dummy_frame 
            node.latest_joints = dummy_pose.tolist() 
            node.record_step(time.time(), 0.8) 
            time.sleep(1/30.0) 
    except KeyboardInterrupt: 
        pass 
        
    node.stop_episode(0) 
    node.finalize_dataset() 
    rclpy.shutdown() 

if __name__ == '__main__': 
    main() 