#!/usr/bin/env python3
"""
Object Tracking Node (Grounding DINO + RealSense Depth)

텍스트 프롬프트로 물체를 제로샷 인식하고, depth로 3D 위치를 계산하여
cuRobo planner에 목표를 전송합니다.

Usage:
    ros2 run e0509_gripper_description object_tracking_node.py \
        --ros-args -p prompt:="red block"

Keys:
    s: 선택된 물체 위치로 로봇 이동
    p: 선택된 물체 pick (접근 → 열기 → 하강 → 닫기 → 들기)
    1-9: 감지된 물체 중 선택
    q: 종료
"""

import os
import numpy as np
import cv2
import pyrealsense2 as rs
import torch
import warnings
warnings.filterwarnings("ignore")

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor
from scipy.spatial.transform import Rotation as R
import json
import threading

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from dsr_msgs2.srv import MoveJoint

from groundingdino.util.inference import load_model, load_image, predict
# from scipy.interpolate import RBFInterpolator
from ultralytics import YOLO

class ObjectTrackingNode(Node):
    def __init__(self):
        super().__init__("object_tracking_node")

        # Parameters
        self.declare_parameter("prompt", "yellow paper",
            ParameterDescriptor(description="Text prompt for object detection"))
        self.declare_parameter("calibration_path",
            os.path.expanduser("~/Downloads/hand_eye_calibration/calibration_data/eye_in_hand_result.npz"),
            ParameterDescriptor(description="EIH calibration file path"))
        self.declare_parameter("box_threshold", 0.3,
            ParameterDescriptor(description="Detection confidence threshold"))
        self.declare_parameter("detection_interval", 5,
            ParameterDescriptor(description="Run detection every N frames"))

        self.prompt = self.get_parameter("prompt").value
        calib_path = self.get_parameter("calibration_path").value
        self.box_threshold = self.get_parameter("box_threshold").value
        self.detection_interval = self.get_parameter("detection_interval").value

        # Load EIH calibration (T_cam2gripper — 고정값)
        self.get_logger().info(f"Loading EIH calibration: {calib_path}")
        calib = np.load(calib_path)
        self.T_cam2gripper = calib['T_cam2gripper'].copy()
        self.get_logger().info(f"T_cam2gripper loaded")

        # FK — eih_fk_publisher.py 토픽 구독
        self._T_cam2base = None
        self._T_cam2base_lock = threading.Lock()
        self._fk_ok_count = 0
        self.create_subscription(
            Float64MultiArray, '/eih/T_cam2base', self._fk_cb, 10)

        # Joint state 구독 (카메라 flip 보정용)
        self._current_joints = None
        self.create_subscription(JointState, '/dsr01/joint_states', self._joint_state_cb, 10)

        # MoveJoint 클라이언트
        self._cli_movej = self.create_client(MoveJoint, '/dsr01/motion/move_joint')

        self.error_map_rbf = None
        # if os.path.exists(error_map_path):
        #     try:
        #         em = np.load(error_map_path)
        #         self.error_map_rbf = RBFInterpolator(
        #             em['xy_measured'], em['errors'], kernel='thin_plate_spline')
        #         self.get_logger().info(f"Error map 로드 완료: {error_map_path}")
        #     except Exception as e:
        #         self.get_logger().warn(f"Error map 로드 실패: {e}")
        # else:
        #     self.get_logger().warn(f"Error map 없음 (보정 없이 동작)")
    
        # Load Grounding DINO
        self.get_logger().info("Loading Grounding DINO...")
        self.gdino = load_model(
            os.path.expanduser("~/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
            os.path.expanduser("~/models/groundingdino_swint_ogc.pth"),
        )
        self.get_logger().info("Grounding DINO loaded!")

        # Load YOLO
        self.get_logger().info("Loading YOLO...")

        self.yolo = YOLO(
            os.path.expanduser(
                "~/pen_detecting.v4i.yolov8/runs/segment/train/weights/best.pt"
            )
        )

        self.get_logger().info("YOLO loaded!")

        # RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.get_logger().info("RealSense started")

        # Publishers
        self.target_pub = self.create_publisher(PoseStamped, "/dsr01/curobo/target_pose", 10)
        self.pick_pub = self.create_publisher(PoseStamped, "/dsr01/curobo/pick_pose", 10)
        # self.place_pub = self.create_publisher(PoseStamped, "/dsr01/curobo/place_pose", 10)
        # self.place_target = None
        self.shelf_pub = self.create_publisher(String, "/dsr01/curobo/shelf_target", 10)
        # GraspGen 6-DOF 자세 전용 토픽 (orientation을 그대로 쓰는 범용 파지 경로)
        self.grasp_pose_pub = self.create_publisher(PoseStamped, "/dsr01/curobo/grasp_pose", 10)
        self.obstacles_pub = self.create_publisher(String, "/dsr01/curobo/obstacles", 10)
        self.marker_pub = self.create_publisher(Marker, "/graspgen/preview_marker", 10)

        # GraspGen 미리보기 대기 중인 grasp pose ('s'로 확인 후 실행, 'r'로 취소)
        self.pending_grasp_pose = None  # (pos_base[3], quat_xyzw[4]) or None

        # Shelf mode state: 'o' → floor(1-3) → slot(1-3)
        self.shelf_mode = False
        self.shelf_floor = 0

        # State
        self.detections = []
        self.yolo_masks = []  # YOLO 세그멘트 마스크 저장# list of (phrase, pos_base, logit, bbox, grasp_angle_rad)
        self.selected_idx = 0
        self.frame_count = 0
        self.locked = False
        self.locked_target = None  # (phrase, pos_base, grasp_angle_rad)

        # camera_loop는 main()에서 메인 스레드로 직접 실행 (타이머 불사용)

        self.get_logger().info("=" * 50)
        self.get_logger().info("  Object Tracking Node Ready")
        self.get_logger().info(f"  Prompt: {self.prompt}")
        self.get_logger().info("  Keys: 's'=move, 'p'=pick, 1-9=select, 'q'=quit")
        self.get_logger().info("=" * 50)

    def _fk_cb(self, msg: Float64MultiArray):
        """eih_fk_publisher에서 발행한 T_cam2base 수신."""
        T = np.array(msg.data).reshape(4, 4)
        with self._T_cam2base_lock:
            self._T_cam2base = T
        self._fk_ok_count += 1

    def _joint_state_cb(self, msg: JointState):
        joint_map = {name: pos for name, pos in zip(msg.name, msg.position)}
        joints = [joint_map.get(f'joint_{i}', 0.0) for i in range(1, 7)]
        if len(joints) == 6:
            self._current_joints = joints

    def flip_joint6(self):
        """현재 joint 6에 180° 더해서 카메라 뒤집기 보정."""
        if self._current_joints is None:
            self.get_logger().warn("Joint state 없음 — 잠시 후 다시 시도하세요")
            return
        import math
        joints_deg = [math.degrees(j) for j in self._current_joints]
        joints_deg[5] += 180.0  # joint 6 (index 5) += 180°
        req = MoveJoint.Request()
        req.pos = joints_deg
        req.vel = 30.0
        req.acc = 30.0
        req.time = 0.0
        req.radius = 0.0
        req.mode = 0
        req.blend_type = 0
        req.sync_type = 1
        self._cli_movej.call_async(req)
        self.get_logger().info(f"Joint 6 flip: {joints_deg[5]-180.0:.1f}° → {joints_deg[5]:.1f}°")
        if self._fk_ok_count == 1:
            t = T[:3, 3]
            self.get_logger().info(
                f"[FK] 첫 수신: cam2base t=({t[0]*1000:.1f},{t[1]*1000:.1f},{t[2]*1000:.1f})mm")

    def _update_fk(self):
        pass  # 토픽 구독으로 대체됨

    def camera_loop(self):
        self._update_fk()
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        if not cf:
            return

        image = np.asanyarray(cf.get_data())
        display = image.copy()

        # YOLO 마스크 그리기
        for pts in getattr(self, 'yolo_masks', []):
            if len(pts) > 0:
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        self.frame_count += 1

        # Store depth frame for angle computation
        self._last_depth_frame = df

        # Run detection periodically (locked 여부 무관 — 위치 계속 갱신)
        if self.frame_count % self.detection_interval == 0:
            self.run_detection(image, df)
            # 잠긴 타겟 위치 갱신 (FK가 나중에 준비돼도 즉시 반영)
            if self.locked and self.locked_target and self.selected_idx < len(self.detections):
                det = self.detections[self.selected_idx]
                if det[1] is not None:
                    phrase, _, angle = self.locked_target
                    self.locked_target = (phrase, det[1], angle)
                    self._last_valid_pos = det[1]   # 마지막 유효 위치 저장

        # Draw detections
        h, w = image.shape[:2]
        for i, det in enumerate(self.detections):
            phrase, pos_base, logit, bbox, grasp_angle = det
            x1 = int((bbox[0] - bbox[2] / 2) * w)
            y1 = int((bbox[1] - bbox[3] / 2) * h)
            x2 = int((bbox[0] + bbox[2] / 2) * w)
            y2 = int((bbox[1] + bbox[3] / 2) * h)

            # Selected object: green, others: gray
            if i == self.selected_idx:
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (150, 150, 150)
                thickness = 1

            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            label = f"[{i+1}] {phrase} ({logit:.2f}) {np.degrees(det[4]):.0f}deg"
            cv2.putText(display, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            if pos_base is not None:
                coord = f"({pos_base[0]*100:.1f},{pos_base[1]*100:.1f},{pos_base[2]*100:.1f})cm"
                cv2.putText(display, coord, (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Status bar
        if self.locked and self.locked_target:
            phrase, pos, _angle = self.locked_target

            cv2.putText(display, f"LOCKED: {phrase} (press 'r' to unlock)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if pos is not None:
                cv2.putText(display, f"Target: X={pos[0]*1000:.1f} Y={pos[1]*1000:.1f} Z={pos[2]*1000:.1f}",
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        elif self.detections:
            sel = self.detections[self.selected_idx]
            cv2.putText(display, f"Selected: [{self.selected_idx+1}] {sel[0]}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # GraspGen 미리보기 대기 오버레이
        if self.pending_grasp_pose is not None:
            pos_b, _quat = self.pending_grasp_pose
            cv2.putText(display,
                        f"GRASP PREVIEW: X={pos_b[0]*1000:.0f} Y={pos_b[1]*1000:.0f} Z={pos_b[2]*1000:.0f}mm "
                        f"— RViz 화살표 확인 후 's'=실행 'r'=취소",
                        (10, h // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # 서랍 모드 오버레이
        if self.shelf_mode:
            if self.shelf_floor == 0:
                shelf_msg = "SHELF MODE: 층 선택 (1/2/3) | r=취소"
            else:
                shelf_msg = f"SHELF MODE: {self.shelf_floor}층 | 칸 선택 (1=왼쪽 2=가운데 3=오른쪽) | r=취소"
            cv2.putText(display, shelf_msg, (10, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

        with self._T_cam2base_lock:
            fk_ready = self._T_cam2base is not None
        fk_color = (0, 255, 0) if fk_ready else (0, 0, 255)
        fk_text = "FK READY" if fk_ready else "FK 연결 중..."
        cv2.putText(display, fk_text, (w - 160, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fk_color, 2)
        cv2.putText(display, f"Prompt: {self.prompt}",
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, "'g'=graspgen 's'=실행 'p'=pick 'f'=카메라flip 'o'=서랍 1-9=lock 'r'=취소 'q'=quit",
                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        cv2.imshow("Object Tracking", display)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('s'):
            # GraspGen 미리보기가 떠 있으면 확인 후 실행, 없으면 일반 pre-grasp 이동
            if not self.confirm_grasp_preview():
                self.send_move()
        elif k == ord('g'):
            self.send_graspgen()
        elif k == ord('p'):
            self.send_pick()
        elif k == ord('o'):
            # 서랍 모드 진입: o → 층(1-3) → 칸(1-3)
            self.shelf_mode = True
            self.shelf_floor = 0
            self.get_logger().info("서랍 모드: 층 선택 (1=1층, 2=2층, 3=3층)")
        elif self.shelf_mode and ord('1') <= k <= ord('3'):
            num = k - ord('0')
            if self.shelf_floor == 0:
                self.shelf_floor = num
                self.get_logger().info(f"{num}층 선택됨. 칸 선택 (1=왼쪽, 2=가운데, 3=오른쪽)")
            else:
                msg = String()
                msg.data = f"{self.shelf_floor},{num}"
                self.shelf_pub.publish(msg)
                self.get_logger().info(f"서랍 이동: {self.shelf_floor}층 {num}번 칸")
                self.shelf_mode = False
                self.shelf_floor = 0
        elif not self.shelf_mode and ord('1') <= k <= ord('9'):
            idx = k - ord('1')
            if idx < len(self.detections):
                self.selected_idx = idx
                det = self.detections[idx]
                self.locked = True
                self.locked_target = (det[0], det[1], det[4])
                self.get_logger().info(f"LOCKED: [{idx+1}] {det[0]} (grasp angle: {np.degrees(det[4]):.1f}°)")
                self.publish_obstacles()
        elif k == ord('f'):
            self.flip_joint6()
        elif k == ord('r'):
            self.shelf_mode = False
            self.shelf_floor = 0
            self.locked = False
            self.locked_target = None
            had_preview = self.clear_grasp_preview()
            if had_preview:
                self.get_logger().info("GraspGen 미리보기 취소됨")
            self.get_logger().info("UNLOCKED: detection resumed")
        # elif k == ord('t'):  # 기존 place_target 지정 (비활성화)
        #     ...
        # elif k == ord('o'):  # 기존 send_place (비활성화)
        #     self.send_place()

        elif k == 82 or k == ord('w'):  # Up arrow or 'w': rotate +5deg
            if self.locked and self.locked_target:
                phrase, pos, angle = self.locked_target
                angle += np.radians(5)
                self.locked_target = (phrase, pos, angle)
                self.get_logger().info(f"Angle adjusted: {np.degrees(angle):.1f}°")
        elif k == 84 or k == ord('x'):  # Down arrow or 'x': rotate -5deg
            if self.locked and self.locked_target:
                phrase, pos, angle = self.locked_target
                angle -= np.radians(5)
                self.locked_target = (phrase, pos, angle)
                self.get_logger().info(f"Angle adjusted: {np.degrees(angle):.1f}°")
        elif k == ord('q'):
            raise SystemExit

    def _get_depth(self, depth_frame, cx, cy, w, h, bbox_w=0, bbox_h=0):
        """중심에서 점점 넓어지는 영역으로 depth 샘플링. 금속 캔 표면 반사 대응."""
        # 1단계: 중심 31x31
        depths = []
        for du in range(-15, 16, 2):
            for dv in range(-15, 16, 2):
                pu, pv = cx + du, cy + dv
                if 0 <= pu < w and 0 <= pv < h:
                    d = depth_frame.get_distance(pu, pv)
                    if 0.1 < d < 3.0:
                        depths.append(d)
        if len(depths) >= 3:
            return float(np.median(depths))

        # 2단계: bbox 전체 영역으로 확장
        half_w = max(int(bbox_w * w / 2), 30)
        half_h = max(int(bbox_h * h / 2), 30)
        for du in range(-half_w, half_w + 1, 4):
            for dv in range(-half_h, half_h + 1, 4):
                pu, pv = cx + du, cy + dv
                if 0 <= pu < w and 0 <= pv < h:
                    d = depth_frame.get_distance(pu, pv)
                    if 0.1 < d < 3.0:
                        depths.append(d)
        if len(depths) >= 3:
            return float(np.median(depths))
        return 0.0

    def run_detection(self, image, depth_frame):
        """Run Grounding DINO detection and compute 3D positions."""
        cv2.imwrite("/tmp/objtrack.jpg", image)
        src, tensor = load_image("/tmp/objtrack.jpg")
        boxes, logits, phrases = predict(
            self.gdino, tensor, self.prompt,
            box_threshold=self.box_threshold, text_threshold=0.2)

        h, w = image.shape[:2]
        self.detections = []
        self.yolo_masks = []  # YOLO 세그멘트 마스크 저장

        # -------------------------
        # YOLO Detection
        # -------------------------
        yolo_results = self.yolo(image, conf=0.1, verbose=False)

        for result in yolo_results:
            if result.masks is not None:          
                for mask in result.masks.xy:      
                    self.yolo_masks.append(       
                        np.array(mask, dtype=np.int32)) 

            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for idx, (box_xyxy, conf, cls_id) in enumerate(zip(
                boxes_xyxy,
                confs,
                classes
            )):
                phrase = self.yolo.names[int(cls_id)]

                x1, y1, x2, y2 = box_xyxy

                # Use mask centroid for more accurate grasping position
                if result.masks is not None and idx < len(result.masks.xy):
                    mask_pts = np.array(result.masks.xy[idx], dtype=np.int32)
                    if len(mask_pts) > 0:
                        cx = int(np.mean(mask_pts[:, 0]))
                        cy = int(np.mean(mask_pts[:, 1]))
                    else:
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                else:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                bx = cx / w
                by = cy / h

                bbox = np.array([bx, by, bw, bh])

                pos_base = None

                depth_m = self._get_depth(depth_frame, cx, cy, w, h, bw, bh)

                if depth_m > 0.1:

                    intr = depth_frame.profile \
                        .as_video_stream_profile() \
                        .intrinsics

                    pt3d = rs.rs2_deproject_pixel_to_point(
                        intr,
                        [cx, cy],
                        depth_m
                    )

                    tc = np.array(pt3d)

                    with self._T_cam2base_lock:
                        T_c2b = self._T_cam2base
                    if T_c2b is not None:
                        tc_h = np.append(tc, 1.0)
                        pos_base = (T_c2b @ tc_h)[:3]
                    else:
                        pos_base = None

                grasp_angle_rad = self.compute_grasp_angle(
                    image,
                    bbox,
                    h,
                    w
                )

                self.detections.append(
                    (
                        phrase,
                        pos_base,
                        float(conf),
                        bbox,
                        grasp_angle_rad
                    )
                )

        for box, logit, phrase in zip(boxes, logits, phrases):
            cx = int(box[0] * w)
            cy = int(box[1] * h)

            # Depth-based 3D position
            pos_base = None
            # 중심 근처 작은 영역만 사용
            cx_box = int(box[0] * w)
            cy_box = int(box[1] * h)
            bw_g = float(box[2]); bh_g = float(box[3])
            depth_m = self._get_depth(depth_frame, cx_box, cy_box, w, h, bw_g, bh_g)
            if depth_m > 0.1:
                intr = depth_frame.profile.as_video_stream_profile().intrinsics
                pt3d = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_m)
                tc = np.array(pt3d)
                if self._T_cam2base is not None:
                    tc_h = np.append(tc, 1.0)
                    pos_base = (self._T_cam2base @ tc_h)[:3]
                else:
                    pos_base = None

            grasp_angle_rad = self.compute_grasp_angle(image, box.cpu().numpy(), h, w)

            # Keep last valid angle if current is 0 (depth failed)
            cache_key = f"{phrase}_{int(box[0]*100)}_{int(box[1]*100)}"
            if abs(grasp_angle_rad) > 0.01:
                if not hasattr(self, '_angle_cache'):
                    self._angle_cache = {}
                self._angle_cache[cache_key] = grasp_angle_rad
            elif hasattr(self, '_angle_cache') and cache_key in self._angle_cache:
                grasp_angle_rad = self._angle_cache[cache_key]

            self.detections.append((phrase, pos_base, float(logit), box.cpu().numpy(), grasp_angle_rad))

        # Keep selection in range
        if self.selected_idx >= len(self.detections):
            self.selected_idx = 0

        # Obstacles are published once when locking a target

    def compute_grasp_angle(self, image, box, h, w):
        """Compute grasp angle using 3D PCA on depth point cloud within bbox."""
        if not hasattr(self, '_last_depth_frame') or self._last_depth_frame is None:
            return 0.0

        df = self._last_depth_frame
        intr = df.profile.as_video_stream_profile().intrinsics

        x1 = max(0, int((box[0] - box[2] / 2) * w))
        y1 = max(0, int((box[1] - box[3] / 2) * h))
        x2 = min(w, int((box[0] + box[2] / 2) * w))
        y2 = min(h, int((box[1] + box[3] / 2) * h))

        if x2 - x1 < 10 or y2 - y1 < 10:
            return 0.0

        # Collect 3D points within bbox
        # Sample every 2 pixels for speed
        points_robot = []
        # 중심 depth 없으면 bbox 내 최초 유효값 사용
        center_depth = df.get_distance(int(box[0] * w), int(box[1] * h))
        if center_depth < 0.1:
            for py in range(y1, y2, 4):
                for px in range(x1, x2, 4):
                    d = df.get_distance(px, py)
                    if d > 0.1:
                        center_depth = d
                        break
                if center_depth >= 0.1:
                    break
        if center_depth < 0.1:
            return 0.0

        for py in range(y1, y2, 2):
            for px in range(x1, x2, 2):
                d = df.get_distance(px, py)
                if d > 0.1 and abs(d - center_depth) < 0.05:
                    pt3d = rs.rs2_deproject_pixel_to_point(intr, [px, py], d)
                    with self._T_cam2base_lock:
                        T_c2b = self._T_cam2base
                    if T_c2b is not None:
                        tc_h = np.append(np.array(pt3d), 1.0)
                        pt_robot = (T_c2b @ tc_h)[:3]
                        points_robot.append(pt_robot)

        if len(points_robot) < 20:
            return 0.0

        points = np.array(points_robot)

        # PCA on XY plane (ignore Z for table-top objects)
        xy = points[:, :2]
        center = xy.mean(axis=0)
        centered = xy - center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Major axis (largest eigenvalue)
        major = eigenvectors[:, -1]
        robot_angle = np.arctan2(major[1], major[0])

        # Rotate 90 degrees so gripper opens perpendicular to major axis
        angle = robot_angle + np.pi / 2
        # 그리퍼는 180도 대칭 → ±90도 범위로 정규화
        # 180도 돌아서 집는 것과 그냥 집는 것이 동일하므로
        while angle > np.pi / 2:
            angle -= np.pi
        while angle < -np.pi / 2:
            angle += np.pi
        return angle

    @staticmethod
    def make_down_quaternion(grasp_angle_rad):
        """Create quaternion for gripper pointing down + rotated by angle around Z axis.
        grasp_angle_rad: rotation around vertical (Z) axis in robot frame.
        Returns: [x, y, z, w] quaternion
        """
        # Base quaternion: gripper pointing down
        # q_base = (w=0, x=0.7071, y=0.7071, z=0)
        # Rotation around Z: q_z = (w=cos(a/2), x=0, y=0, z=sin(a/2))
        # Combined: q_z * q_base

        ca = np.cos(grasp_angle_rad / 2)
        sa = np.sin(grasp_angle_rad / 2)

        # q_z (wxyz)
        qz_w, qz_x, qz_y, qz_z = ca, 0.0, 0.0, sa

        # q_base (wxyz)
        qb_w, qb_x, qb_y, qb_z = 0.0, 0.7071, 0.7071, 0.0

        # Quaternion multiplication q_z * q_base
        w = qz_w*qb_w - qz_x*qb_x - qz_y*qb_y - qz_z*qb_z
        x = qz_w*qb_x + qz_x*qb_w + qz_y*qb_z - qz_z*qb_y
        y = qz_w*qb_y - qz_x*qb_z + qz_y*qb_w + qz_z*qb_x
        z = qz_w*qb_z + qz_x*qb_y - qz_y*qb_x + qz_z*qb_w

        return [float(x), float(y), float(z), float(w)]

    @staticmethod
    def make_side_quaternion(approach_angle_rad):
        """Gripper pointing horizontally toward the can (side grasp).
        approach_angle_rad: atan2(can_y, can_x) — direction from robot to can.
        Ry(90°) tilts tool Z from up to +X, then Rz(approach) rotates in XY plane.
        """
        # q_tilt: 90° around Y → tool Z points along +X
        qt_w, qt_x, qt_y, qt_z = 0.7071, 0.0, 0.7071, 0.0
        # q_z: rotate around world Z by approach_angle
        ca = np.cos(approach_angle_rad / 2)
        sa = np.sin(approach_angle_rad / 2)
        qz_w, qz_x, qz_y, qz_z = ca, 0.0, 0.0, sa
        # Combined: q_z * q_tilt
        w = qz_w*qt_w - qz_x*qt_x - qz_y*qt_y - qz_z*qt_z
        x = qz_w*qt_x + qz_x*qt_w + qz_y*qt_z - qz_z*qt_y
        y = qz_w*qt_y - qz_x*qt_z + qz_y*qt_w + qz_z*qt_x
        z = qz_w*qt_z + qz_x*qt_y - qz_y*qt_x + qz_z*qt_w
        return [float(x), float(y), float(z), float(w)]

    def publish_obstacles(self):
        """Publish detected objects as obstacles to cuRobo (except target)."""
        obstacles = []
        locked_phrase = self.locked_target[0] if self.locked and self.locked_target else None

        for i, (phrase, pos_base, logit, bbox, _angle) in enumerate(self.detections):
            if pos_base is None:
                continue
            # Skip the locked target (we want to reach it, not avoid it)
            if self.locked and i == self.selected_idx:
                continue

            # Estimate object size from bbox (rough approximation)
            h_img, w_img = 480, 640
            obj_w = float(bbox[2]) * w_img * 0.001  # rough width in meters
            obj_h = float(bbox[3]) * h_img * 0.001  # rough height in meters
            obj_size = max(obj_w, obj_h, 0.03)  # minimum 3cm

            obstacles.append({
                "name": f"{phrase}_{i}",
                "pos": [float(pos_base[0]), float(pos_base[1]), float(pos_base[2])],
                "dims": [obj_size, obj_size, obj_size]
            })

        msg = String()
        msg.data = json.dumps(obstacles)
        self.obstacles_pub.publish(msg)

    def _make_pose_msg(self, pos, grasp_angle_rad=0.0):  # grasp_angle_rad unused (side grasp)
        approach_angle = np.arctan2(pos[1], pos[0])
        quat = self.make_side_quaternion(approach_angle)
        msg = PoseStamped()
        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        return msg

    def _get_target(self):
        """Get currently selected/locked target. Returns (phrase, pos, angle)."""
        if self.locked and self.locked_target:
            phrase, pos, angle = self.locked_target
            if pos is None and hasattr(self, '_last_valid_pos'):
                pos = self._last_valid_pos
            return phrase, pos, angle
        if self.detections and self.selected_idx < len(self.detections):
            det = self.detections[self.selected_idx]
            return det[0], det[1], det[4]
        return None, None, 0.0

    def send_graspgen(self):
        """g key: 현재 depth에서 포인트클라우드 추출 → GraspGen 추론 → RViz 미리보기 표시.
        실제 로봇 실행은 미리보기 화살표를 확인한 뒤 's'로 한다 (안전 확인 단계)."""
        import zmq, threading

        phrase, pos, angle = self._get_target()
        if pos is None:
            self.get_logger().warn("No valid object selected")
            return
        if not hasattr(self, '_last_depth_frame') or self._last_depth_frame is None:
            self.get_logger().warn("Depth frame 없음")
            return

        df = self._last_depth_frame
        intr = df.profile.as_video_stream_profile().intrinsics
        h_img, w_img = 480, 640

        # 잠긴 타겟의 영역에서 포인트클라우드 추출
        if not (self.locked and self.locked_target and self.selected_idx < len(self.detections)):
            self.get_logger().warn("Lock 먼저 하세요 (1키)")
            return

        bbox = self.detections[self.selected_idx][3]

        x1 = max(0, int((bbox[0] - bbox[2]/2) * w_img) - 10)
        y1 = max(0, int((bbox[1] - bbox[3]/2) * h_img) - 10)
        x2 = min(w_img, int((bbox[0] + bbox[2]/2) * w_img) + 10)
        y2 = min(h_img, int((bbox[1] + bbox[3]/2) * h_img) + 10)

        # YOLO 세그멘테이션 폴리곤이 있으면 물체 표면만 정확히 샘플링
        # (사각 bbox에는 배경/테이블이 섞여 GraspGen 품질이 떨어짐 — 마스크로 표면만 추출)
        seg_mask = None
        if self.selected_idx < len(self.yolo_masks) and len(self.yolo_masks[self.selected_idx]) > 0:
            seg_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(seg_mask, [self.yolo_masks[self.selected_idx]], 255)
            self.get_logger().info("GraspGen: YOLO 세그멘트 마스크로 포인트클라우드 추출")
        else:
            self.get_logger().info("GraspGen: 세그멘트 마스크 없음 — bbox 영역으로 추출")

        # 1차 샘플링 (마스크/bbox 내부 + depth 유효 범위)
        candidates = []  # (px, py, depth)
        for py in range(y1, y2, 2):
            for px in range(x1, x2, 2):
                if seg_mask is not None and seg_mask[py, px] == 0:
                    continue
                d = df.get_distance(px, py)
                if 0.05 < d < 2.0:
                    candidates.append((px, py, d))

        if len(candidates) < 50:
            self.get_logger().warn(f"포인트 부족: {len(candidates)}개")
            return

        # depth 이상치 제거: 중앙값 기준 ±7cm 밖은 배경/반사로 간주하고 제외
        median_d = float(np.median([d for _, _, d in candidates]))
        pts = [
            rs.rs2_deproject_pixel_to_point(intr, [px, py], d)
            for px, py, d in candidates if abs(d - median_d) < 0.07
        ]

        if len(pts) < 50:
            self.get_logger().warn(f"포인트 부족(필터 후): {len(pts)}개")
            return

        pc = np.array(pts, dtype=np.float32)
        pc -= pc.mean(axis=0)  # 중심 정규화 (GraspGen 요구사항)

        self.get_logger().info(f"GraspGen 요청: {len(pc)}개 포인트 전송 중...")

        def run():
            try:
                from grasp_gen.serving.zmq_client import GraspGenClient
                with GraspGenClient(host='localhost', port=5556) as client:
                    grasps, confs = client.infer(pc, num_grasps=50, topk_num_grasps=30)
                if len(grasps) == 0:
                    self.get_logger().warn("GraspGen: 파지 자세 없음")
                    return
                self.get_logger().info(
                    f"GraspGen: {len(grasps)}개 파지, conf={confs[0]:.3f}")

                with self._T_cam2base_lock:
                    T_c2b = self._T_cam2base
                if T_c2b is None:
                    self.get_logger().error("FK 없음")
                    return

                # 포인트클라우드 중심(캔 위치)을 더해서 실제 위치 복원
                pc_center = np.array(pts, dtype=np.float32).mean(axis=0)
                T_center = np.eye(4)
                T_center[:3, 3] = pc_center

                # 수평 필터(|Z|<0.4) → 로봇→캔 수직 방향(CCW 90°)으로 접근하는 grasp 선택
                # 이전 성공 패턴: 캔이 +X쪽에 있을 때 approach가 +Y (수직 방향)
                can_pos_world = (T_c2b @ T_center)[:3, 3]
                can_xy = can_pos_world[:2]
                can_xy_norm = can_xy / (np.linalg.norm(can_xy) + 1e-6)
                # 로봇→캔 방향을 CCW 90° 회전 → 선호 접근 방향
                preferred = np.array([-can_xy_norm[1], can_xy_norm[0]])

                HORIZ_THRESH = 0.4
                candidates = []
                for i, g in enumerate(grasps):
                    T_g = g.copy(); T_g[3, 3] = 1.0
                    approach_world = (T_c2b @ T_g)[:3, 2]
                    if abs(approach_world[2]) < HORIZ_THRESH:
                        ap_xy = approach_world[:2]
                        ap_xy_norm = ap_xy / (np.linalg.norm(ap_xy) + 1e-6)
                        alignment = float(np.dot(ap_xy_norm, preferred))
                        candidates.append((alignment, i))

                if candidates:
                    candidates.sort(reverse=True)
                    best_idx = candidates[0][1]
                    self.get_logger().info(
                        f"GraspGen 수평후보 {len(candidates)}개 중 선택: [{best_idx}] alignment={candidates[0][0]:.3f} preferred=({preferred[0]:.2f},{preferred[1]:.2f})")
                else:
                    best_idx = 0
                    best_z = float('inf')
                    for i, g in enumerate(grasps):
                        T_g = g.copy(); T_g[3, 3] = 1.0
                        z = abs((T_c2b @ T_g)[:3, 2][2])
                        if z < best_z:
                            best_z = z; best_idx = i
                    self.get_logger().warn(f"수평 grasp 없음 — |Z| 최소 fallback [{best_idx}]")

                T_grasp_cam = grasps[best_idx].copy()
                T_grasp_cam[3, 3] = 1.0
                T_grasp_base = T_c2b @ T_center @ T_grasp_cam

                approach_sel = T_grasp_base[:3, 2]
                self.get_logger().info(
                    f"GraspGen approach=({approach_sel[0]:.2f},{approach_sel[1]:.2f},{approach_sel[2]:.2f})")

                pos_b = T_grasp_base[:3, 3]
                quat = R.from_matrix(T_grasp_base[:3, :3]).as_quat()  # [x,y,z,w]

                # 로봇을 움직이지 않고 RViz에 미리보기만 표시 — 's'로 확인 후 실행
                self.pending_grasp_pose = (pos_b.copy(), quat.copy())
                self.publish_grasp_marker(pos_b, quat, action=Marker.ADD)
                self.get_logger().info(
                    f"GraspGen 미리보기: X={pos_b[0]*1000:.1f} Y={pos_b[1]*1000:.1f} Z={pos_b[2]*1000:.1f}mm "
                    f"(conf={confs[0]:.3f}) — RViz 화살표 확인 후 's'=실행 'r'=취소")
            except Exception as e:
                self.get_logger().error(f"GraspGen 오류: {e}")

        threading.Thread(target=run, daemon=True).start()

    def publish_grasp_marker(self, pos, quat_xyzw, action=Marker.ADD):
        """GraspGen best grasp pose를 RViz ARROW 마커로 표시.
        GraspGen 컨벤션: 그리퍼 frame의 +Z축이 접근(approach) 방향이므로,
        화살표는 '접근 시작점 → grasp 지점' 방향(= world상의 +Z 회전축)으로 그린다."""
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "graspgen_preview"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = action

        if action == Marker.ADD:
            rot = R.from_quat(quat_xyzw).as_matrix()
            approach_axis = rot[:, 2]  # grasp frame의 +Z (접근 방향), world 좌표
            length = 0.08
            start = np.asarray(pos) - approach_axis * length
            end = np.asarray(pos)
            marker.points = [
                Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
                Point(x=float(end[0]),   y=float(end[1]),   z=float(end[2])),
            ]
            marker.scale.x = 0.012  # shaft 지름
            marker.scale.y = 0.025  # head 지름
            marker.scale.z = 0.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.2
            marker.color.a = 0.9
            marker.lifetime.sec = 0  # 0 = 명시적으로 DELETE할 때까지 유지

        self.marker_pub.publish(marker)

    def confirm_grasp_preview(self):
        """'s' key: 미리보기 중인 GraspGen 자세를 cuRobo로 전송해 실행.
        대기 중인 미리보기가 없으면 False를 반환 (호출 측에서 일반 이동으로 분기)."""
        if self.pending_grasp_pose is None:
            return False

        pos_b, quat = self.pending_grasp_pose
        msg = PoseStamped()
        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pos_b[0])
        msg.pose.position.y = float(pos_b[1])
        msg.pose.position.z = float(pos_b[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        # /dsr01/curobo/grasp_pose 로 전송 — orientation을 그대로 보존하는 범용 파지 경로
        # (target_pose 토픽은 down_quat을 강제하는 "캔 집기" 전용 콜백이라 사용하지 않음)
        self.grasp_pose_pub.publish(msg)
        self.get_logger().info(
            f"GraspGen 확인 → 실행: X={pos_b[0]*1000:.1f} Y={pos_b[1]*1000:.1f} Z={pos_b[2]*1000:.1f}mm")

        self.clear_grasp_preview()
        return True

    def clear_grasp_preview(self):
        """대기 중인 GraspGen 미리보기를 취소하고 RViz 마커를 지운다."""
        if self.pending_grasp_pose is None:
            return False
        self.pending_grasp_pose = None
        self.publish_grasp_marker(None, None, action=Marker.DELETE)
        return True

    def send_move(self):
        """s key: 캔 위치 전송 → planner가 pre-grasp (X,Z 맞추고 Y 18cm 뒤 대기)."""
        phrase, pos, angle = self._get_target()
        if pos is None:
            self.get_logger().warn("No valid object selected")
            return
        msg = self._make_pose_msg(pos, angle)  # 실제 캔 위치 전송 (planner가 오프셋 계산)
        self.target_pub.publish(msg)
        self.get_logger().info(f"PRE-GRASP {phrase}: X={pos[0]*1000:.1f} Y={pos[1]*1000:.1f} Z={pos[2]*1000:.1f}mm")
        # self.locked = False
        # self.locked_target = None
        # self.get_logger().info("Lock released - ready for next detection")

    def send_pick(self):
        """Pick selected object (full sequence)."""
        phrase, pos, angle = self._get_target()
        if pos is None:
            self.get_logger().warn("No valid object selected")
            return
        msg = self._make_pose_msg(pos, angle)
        self.pick_pub.publish(msg)

    # def send_place(self):  # 기존 place (비활성화 — 서랍 모드로 대체)
    #     if self.place_target is None:
    #         return
    #     phrase, pos, angle = self.place_target
    #     if pos is None:
    #         return
    #     msg = self._make_pose_msg(pos, angle)
    #     self.place_pub.publish(msg)
    #     self.place_target = None

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackingNode()

    # ROS 콜백(TF, 퍼블리셔 등)은 백그라운드 스레드에서 spin
    # OpenCV는 메인 스레드에서만 실행해야 안전
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            node.camera_loop()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
