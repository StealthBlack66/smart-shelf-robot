#!/usr/bin/env python3
"""
cuRobo Motion Planner Node for Doosan E0509

Subscribes to target pose, plans collision-free trajectory using cuRobo,
and executes via Doosan MoveSplineJoint service.

Usage:
    ros2 run e0509_gripper_description curobo_planner_node.py

Test:
    ros2 topic pub --once /dsr01/curobo/target_pose geometry_msgs/msg/PoseStamped \
        "{header: {frame_id: 'base_link'}, pose: {position: {x: 0.3, y: 0.2, z: 0.3}, \
        orientation: {x: 0.0, y: 0.7071, z: 0.0, w: 0.7071}}}"
"""

import os
import math
import time
import torch
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from dsr_msgs2.srv import MoveSplineJoint, MoveJoint, MoveLine, GetCurrentPosx, MoveStop
# 그리퍼 = dsr_gripper_tcp 브리지 (open=set_position(0), 파지=safe_grasp 액션)
from rclpy.action import ActionClient
from dsr_gripper_tcp_interfaces.srv import SetPosition
from dsr_gripper_tcp_interfaces.action import SafeGrasp

from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState as CuroboJointState, RobotConfig
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.geom.types import WorldConfig, Cuboid
from std_msgs.msg import String
import json


class CuroboPlanner(Node):
    # E0509 joint order as expected by cuRobo and Doosan services
    JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    def __init__(self):
        super().__init__("curobo_planner_node")

        # Separate callback group for service calls to avoid deadlock
        self.service_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()

        self.get_logger().info("Initializing cuRobo planner...")

        # Current joint state
        self.current_joints = None

        # Config path — smart-shelf-robot/config/curobo (패키지 루트 기준)
        # __file__ = src/motion/curobo_planner_node.py → 3단계 상위가 패키지 루트
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "curobo"
        )
        # Fallback: 설치형 패키지(ament share)
        if not os.path.exists(config_dir):
            from ament_index_python.packages import get_package_share_directory
            config_dir = os.path.join(
                get_package_share_directory("smart_shelf_robot"),
                "config", "curobo"
            )

        self.get_logger().info(f"Config dir: {config_dir}")

        # Initialize cuRobo
        tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

        robot_cfg = RobotConfig.from_basic(
            urdf_path=os.path.join(config_dir, "e0509_gripper.urdf"),
            base_link="base_link",
            ee_link="gripper_rh_p12_rn_base",
            tensor_args=tensor_args,
        )

        # World: table as obstacle (adjustable later)
        world_cfg = WorldConfig(
            cuboid=[
                Cuboid(name="table", pose=[0.0, 0.0, -0.02, 1, 0, 0, 0], dims=[1.2, 1.2, 0.04]),
            ]
        )

        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            tensor_args=tensor_args,
            num_trajopt_seeds=4,
            num_graph_seeds=4,
            collision_cache={"obb": 30, "mesh": 10},
        )
        self.motion_gen = MotionGen(motion_gen_cfg)
        self.motion_gen.warmup(warmup_js_trajopt=False)
        self.get_logger().info("cuRobo MotionGen warmed up!")

        # ROS2 interfaces
        self.joint_sub = self.create_subscription(
            JointState, "/dsr01/joint_states",
            self.joint_state_cb, 10)

        self.target_sub = self.create_subscription(
            PoseStamped, "/dsr01/curobo/target_pose",
            self.target_pose_cb, 10)

        # Pick pose subscription
        self.pick_sub = self.create_subscription(
            PoseStamped, "/dsr01/curobo/pick_pose",
            self.pick_pose_cb, 10)

        # Obstacles update subscription (JSON format)
        self.obstacles_sub = self.create_subscription(
            String, "/dsr01/curobo/obstacles",
            self.obstacles_cb, 10)

        self.tensor_args = tensor_args
        self.robot_cfg = robot_cfg

        # Doosan services (use separate callback group)
        self.cli_spline = self.create_client(
            MoveSplineJoint, "/dsr01/motion/move_spline_joint",
            callback_group=self.service_cb_group)
        self.cli_movej = self.create_client(
            MoveJoint, "/dsr01/motion/move_joint",
            callback_group=self.service_cb_group)
        self.cli_movel = self.create_client(
            MoveLine, "/dsr01/motion/move_line",
            callback_group=self.service_cb_group)
        # 이동 직후 motion 상태 클리어용 (그래야 flange serial 그리퍼 명령이 닿음)
        self.cli_stop = self.create_client(
            MoveStop, "/dsr01/motion/move_stop",
            callback_group=self.service_cb_group)
        # 현재 TCP 자세 읽기 (lift 시 자세 유지용 — 캔이 서있게 회전 없이 Z만 상승)
        self.cli_posx = self.create_client(
            GetCurrentPosx, "/dsr01/aux_control/get_current_posx",
            callback_group=self.service_cb_group)
        # 그리퍼(브리지): 열기=set_position(0) 서비스, 파지=safe_grasp 액션(전류기반)
        self.cli_gripper_open = self.create_client(
            SetPosition, "/gripper_service/set_position",
            callback_group=self.service_cb_group)
        self.act_safe_grasp = ActionClient(
            self, SafeGrasp, "/gripper_service/safe_grasp",
            callback_group=self.service_cb_group)
        # 파지 파라미터(물성별 max_current 는 추후 클래스 연동; 기본값 tunable)
        self.declare_parameter('grasp_target_position', 700)   # 0~700, 700=완전닫힘
        self.declare_parameter('grasp_max_current', 600)       # mA (bottle 수준 기본)
        self.declare_parameter('grasp_current_delta', 30)      # 파지검출 전류증분
        self.declare_parameter('grasp_open_position', 0)       # 열기 위치

        self.get_logger().info("========================================")
        self.get_logger().info("cuRobo Planner Ready!")
        self.get_logger().info("  Subscribe: /dsr01/curobo/target_pose (move)")
        self.get_logger().info("  Subscribe: /dsr01/curobo/pick_pose (pick)")
        self.get_logger().info("  Subscribe: /dsr01/curobo/obstacles (world update)")
        self.get_logger().info("========================================")

    def obstacles_cb(self, msg: String):
        """Update cuRobo world with detected obstacles.
        JSON format: [{"name": "obj1", "pos": [x,y,z], "dims": [w,h,d]}, ...]
        """
        try:
            obstacles_data = json.loads(msg.data)
            cuboids = [
                Cuboid(
                    name="table",
                    pose=[0.0, 0.0, -0.02, 1, 0, 0, 0],
                    dims=[1.2, 1.2, 0.04]
                )
            ]

            for obj in obstacles_data:
                cuboids.append(Cuboid(
                    name=obj["name"],
                    pose=[obj["pos"][0], obj["pos"][1], obj["pos"][2], 1, 0, 0, 0],
                    dims=obj.get("dims", [0.05, 0.05, 0.05])
                ))

            world_cfg = WorldConfig(cuboid=cuboids)
            self.motion_gen.update_world(world_cfg)
            self.get_logger().info(f"World updated: {len(cuboids)} obstacles (table + {len(obstacles_data)} objects)")

        except Exception as e:
            self.get_logger().error(f"Failed to update obstacles: {e}")

    def joint_state_cb(self, msg: JointState):
        """Store current joint positions in correct order."""
        joint_map = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                joint_map[name] = msg.position[i]

        joints = []
        for name in self.JOINT_NAMES:
            if name in joint_map:
                joints.append(joint_map[name])
            else:
                return  # Missing joint data

        self.current_joints = joints

    def target_pose_cb(self, msg: PoseStamped):
        """Receive target pose, plan trajectory, and execute."""
        if self.current_joints is None:
            self.get_logger().warn("No joint state received yet")
            return

        pos = msg.pose.position
        ori = msg.pose.orientation
        self.get_logger().info(
            f"Target received: pos=[{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}] "
            f"ori=[{ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}, {ori.w:.3f}]")

        # 🛑 바닥 안전 가드 — TCP 목표가 바닥 하한(TCP_Z_MIN) 아래면 클램프.
        # (pick_pose_cb 에는 MIN_Z 가드가 있었으나 's'(target_pose_cb)엔 없어서
        #  GraspGen 파지점 z=-14mm 를 그대로 보내 바닥 충돌함 — 동일 가드 추가)
        safe_z = pos.z
        if safe_z < self.TCP_Z_MIN:
            self.get_logger().warn(
                f"🛑 [FLOOR GUARD] target z={pos.z*1000:.1f}mm < "
                f"{self.TCP_Z_MIN*1000:.0f}mm → {self.TCP_Z_MIN*1000:.0f}mm 로 클램프 "
                f"(바닥 충돌 방지)")
            safe_z = self.TCP_Z_MIN

        # Plan
        traj = self.plan(
            self.current_joints,
            [pos.x, pos.y, safe_z],
            [ori.w, ori.x, ori.y, ori.z],  # cuRobo uses wxyz
        )

        if traj is not None:
            self.execute_spline(traj)   # 충돌회피 궤적 전체 추종 (장애물 피해 이동)
        else:
            self.get_logger().error("Planning failed!")

    # 🛑 TCP 목표 Z 하한 (m, base 기준). 작업 바닥 z≈-30mm 위로 여유.
    # 's'(target_pose_cb)/'p'(pick) 모두 이 아래로는 못 내려감 → 바닥 충돌 방지.
    TCP_Z_MIN = 0.02
    # 모션 속도 스케일 — 검증 중 0.5(절반), 검증 끝나면 CUROBO_VEL_SCALE=1.0 로 복구.
    VEL_SCALE = float(os.environ.get('CUROBO_VEL_SCALE', '0.5'))

    # joint_4(인덱스 3) 하한 (rad). 이 아래면 손목이 바닥에 박힘 → 양수 해 우선.
    # 0.0=양수만. 너무 엄격하면 약간 음수 허용(-0.2 등) 으로 완화 가능.
    J4_MIN_RAD = 0.0
    # yaw 등가 재시도 각도 (대칭 그리퍼: 180°는 진짜 대칭, ±90°는 사각물체 등가)
    YAW_RETRY = (np.pi, np.pi / 2, -np.pi / 2)

    def _traj_j4_ok(self, positions):
        """trajectory(rad) 의 joint_4(idx3) 가 전 구간 J4_MIN_RAD 이상인가."""
        return float(np.min(positions[:, 3])) >= self.J4_MIN_RAD

    def _quat_yaw(self, quat_wxyz, dyaw):
        """tool 접근축(local Z) 기준 dyaw 회전한 quat(wxyz). 대칭 그리퍼 등가 파지."""
        w, x, y, z = quat_wxyz
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
        c, s = np.cos(dyaw), np.sin(dyaw)
        Rn = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        t = Rn[0, 0] + Rn[1, 1] + Rn[2, 2]
        if t > 0:
            S = np.sqrt(t+1.0)*2
            q = [0.25*S, (Rn[2, 1]-Rn[1, 2])/S, (Rn[0, 2]-Rn[2, 0])/S, (Rn[1, 0]-Rn[0, 1])/S]
        elif Rn[0, 0] > Rn[1, 1] and Rn[0, 0] > Rn[2, 2]:
            S = np.sqrt(1+Rn[0, 0]-Rn[1, 1]-Rn[2, 2])*2
            q = [(Rn[2, 1]-Rn[1, 2])/S, 0.25*S, (Rn[0, 1]+Rn[1, 0])/S, (Rn[0, 2]+Rn[2, 0])/S]
        elif Rn[1, 1] > Rn[2, 2]:
            S = np.sqrt(1+Rn[1, 1]-Rn[0, 0]-Rn[2, 2])*2
            q = [(Rn[0, 2]-Rn[2, 0])/S, (Rn[0, 1]+Rn[1, 0])/S, 0.25*S, (Rn[1, 2]+Rn[2, 1])/S]
        else:
            S = np.sqrt(1+Rn[2, 2]-Rn[0, 0]-Rn[1, 1])*2
            q = [(Rn[1, 0]-Rn[0, 1])/S, (Rn[0, 2]+Rn[2, 0])/S, (Rn[1, 2]+Rn[2, 1])/S, 0.25*S]
        n = float(np.linalg.norm(q))
        return [v/n for v in q]

    def _quat_to_rotm(self, quat_wxyz):
        w, x, y, z = quat_wxyz
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

    def _rotm_to_quat_wxyz(self, Rn):
        t = Rn[0, 0] + Rn[1, 1] + Rn[2, 2]
        if t > 0:
            S = np.sqrt(t+1.0)*2
            q = [0.25*S, (Rn[2, 1]-Rn[1, 2])/S, (Rn[0, 2]-Rn[2, 0])/S, (Rn[1, 0]-Rn[0, 1])/S]
        elif Rn[0, 0] > Rn[1, 1] and Rn[0, 0] > Rn[2, 2]:
            S = np.sqrt(1+Rn[0, 0]-Rn[1, 1]-Rn[2, 2])*2
            q = [(Rn[2, 1]-Rn[1, 2])/S, 0.25*S, (Rn[0, 1]+Rn[1, 0])/S, (Rn[0, 2]+Rn[2, 0])/S]
        elif Rn[1, 1] > Rn[2, 2]:
            S = np.sqrt(1+Rn[1, 1]-Rn[0, 0]-Rn[2, 2])*2
            q = [(Rn[0, 2]-Rn[2, 0])/S, (Rn[0, 1]+Rn[1, 0])/S, 0.25*S, (Rn[1, 2]+Rn[2, 1])/S]
        else:
            S = np.sqrt(1+Rn[2, 2]-Rn[0, 0]-Rn[1, 1])*2
            q = [(Rn[1, 0]-Rn[0, 1])/S, (Rn[0, 2]+Rn[2, 0])/S, (Rn[1, 2]+Rn[2, 1])/S, 0.25*S]
        n = float(np.linalg.norm(q))
        return [v/n for v in q]

    def _quat_tilt(self, quat_wxyz, deg):
        """approach(grasp Z)를 world 아래(-Z)로 deg 기울임 + 핑거축 수평 유지. wxyz."""
        R = self._quat_to_rotm(quat_wxyz)
        a = R[:, 2]
        down = np.array([0.0, 0.0, -1.0])
        axis = np.cross(a, down)
        na = np.linalg.norm(axis)
        if na < 1e-6:
            return list(quat_wxyz)
        axis = axis / na
        th = np.radians(deg)
        a2 = (a*np.cos(th) + np.cross(axis, a)*np.sin(th)
              + axis*np.dot(axis, a)*(1-np.cos(th)))
        a2 = a2 / np.linalg.norm(a2)
        up = np.array([0.0, 0.0, 1.0])
        if abs(a2[2]) > 0.97:
            x = np.array([1.0, 0.0, 0.0])
        else:
            x = np.cross(up, a2); x = x / np.linalg.norm(x)
        y = np.cross(a2, x); y = y / np.linalg.norm(y)
        return self._rotm_to_quat_wxyz(np.column_stack([x, y, a2]))

    def _plan_once(self, start_state, target_pos, quat_wxyz):
        """단일 plan_single → 성공 시 positions(rad), 실패 시 None."""
        target_pose = Pose(
            position=torch.tensor([target_pos], device="cuda:0", dtype=torch.float32),
            quaternion=torch.tensor([quat_wxyz], device="cuda:0", dtype=torch.float32),
        )
        result = self.motion_gen.plan_single(start_state, target_pose)
        if result.success.item():
            return result.get_interpolated_plan().position.cpu().numpy()
        return None

    def plan(self, start_joints, target_pos, target_quat_wxyz,
             prefer_positive_j4=True, allow_yaw_retry=True):
        """cuRobo 경로계획 — joint_4 양수 우선(바닥 충돌 회피) + yaw 등가 재시도 + fallback.
        prefer_positive_j4: joint_4 양수 해 우선. allow_yaw_retry: 대칭 yaw 회전 재시도 허용."""
        t0 = time.time()
        # 수평(옆면) 그랩이면 j4-양수/yaw회전 끔 — 바닥 박을 일 없고, yaw 180°가
        # 손목(joint_6)을 빙글 돌리는 원인. 접근축(grasp Z) 의 수직성분으로 판정.
        w, x, y, z = target_quat_wxyz
        approach_z = 1.0 - 2.0 * (x * x + y * y)   # R[2,2] = grasp Z 의 base z 성분
        if abs(approach_z) < 0.5:                  # 수평-ish (옆면 그랩)
            # j4-양수 강제는 끔(원래 자세 그대로 = 불필요한 yaw 180° spin 방지).
            # 단 원래 자세가 도달 불가일 때만 yaw 재시도로 도달 가능한 해 탐색(폴백).
            prefer_positive_j4 = False
            allow_yaw_retry = True
            self.get_logger().info(
                f"수평 그랩 감지(az={approach_z:+.2f}) → j4강제 OFF(spin방지), "
                f"yaw는 도달실패시만 폴백")
        start_state = CuroboJointState.from_position(
            position=torch.tensor([start_joints], device="cuda:0", dtype=torch.float32),
            joint_names=self.JOINT_NAMES,
        )

        # 1차: 원본 target
        base = self._plan_once(start_state, target_pos, target_quat_wxyz)
        if base is not None and (not prefer_positive_j4 or self._traj_j4_ok(base)):
            self.get_logger().info(
                f"Planning SUCCESS: {(time.time()-t0)*1000:.1f}ms, {base.shape[0]} pts "
                f"(j4_min={np.degrees(base[:,3].min()):.1f}°)")
            return base

        # joint_4 음수(또는 1차 실패) → yaw 등가 재시도로 양수 해 탐색
        if allow_yaw_retry and prefer_positive_j4:
            for dyaw in self.YAW_RETRY:
                cand = self._plan_once(
                    start_state, target_pos, self._quat_yaw(target_quat_wxyz, dyaw))
                if cand is not None and self._traj_j4_ok(cand):
                    self.get_logger().info(
                        f"Planning SUCCESS (joint_4 양수 — yaw {np.degrees(dyaw):+.0f}° 등가): "
                        f"{(time.time()-t0)*1000:.1f}ms, j4_min={np.degrees(cand[:,3].min()):.1f}°")
                    return cand

        # fallback: 원래 해 있으면 사용 (경로 자체를 못 찾는 것 방지)
        if base is not None:
            self.get_logger().warn(
                f"joint_4 양수 해 없음 — 원래 해 사용(j4_min={np.degrees(base[:,3].min()):.1f}°). "
                f"바닥 충돌 주의 (J4_MIN 완화 또는 target z 상향 고려)")
            return base

        # 1차도 실패 → yaw 등가로 '아무 해'라도 탐색
        if allow_yaw_retry:
            for dyaw in self.YAW_RETRY:
                cand = self._plan_once(
                    start_state, target_pos, self._quat_yaw(target_quat_wxyz, dyaw))
                if cand is not None:
                    self.get_logger().warn(
                        f"원본 실패 — yaw {np.degrees(dyaw):+.0f}° 등가 해 사용 "
                        f"(j4_min={np.degrees(cand[:,3].min()):.1f}°)")
                    return cand

        # 기울임 폴백: 수평으론 도달 불가 → approach 를 아래(top-down)로 점진 기울여
        # 도달 가능한 해 탐색. (수평 우선이지만 안 되면 약간 기울여서라도 잡음)
        if os.environ.get('CUROBO_TILT_FALLBACK', '0') != '0':   # 기본 OFF (수평만)
            for _tilt in (20, 35, 50, 65, 80):
                _tq = self._quat_tilt(target_quat_wxyz, _tilt)
                cand = self._plan_once(start_state, target_pos, _tq)
                if cand is None and allow_yaw_retry:
                    for dyaw in self.YAW_RETRY:
                        cand = self._plan_once(
                            start_state, target_pos, self._quat_yaw(_tq, dyaw))
                        if cand is not None:
                            break
                if cand is not None:
                    self.get_logger().warn(
                        f"기울임 폴백 {_tilt}° 도달 (수평 불가 → 약간 기울여 잡음): "
                        f"j4_min={np.degrees(cand[:,3].min()):.1f}°")
                    return cand

        self.get_logger().error(
            f"Planning FAILED (수평+yaw+기울임 모두): {(time.time()-t0)*1000:.1f}ms")
        return None

    def execute_spline(self, traj_rad):
        """Execute trajectory via MoveSplineJoint (expects degrees)."""
        if not self.cli_spline.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveSplineJoint service not available")
            return

        # Convert rad to deg, subsample to max 100 points
        traj_deg = np.rad2deg(traj_rad)
        n_points = traj_deg.shape[0]

        # Subsample if more than 100 points
        if n_points > 100:
            indices = np.linspace(0, n_points - 1, 100, dtype=int)
            traj_deg = traj_deg[indices]
            n_points = 100

        # Build request
        req = MoveSplineJoint.Request()
        req.pos_cnt = n_points

        for i in range(n_points):
            point = Float64MultiArray()
            point.data = traj_deg[i].tolist()
            req.pos.append(point)

        req.vel = [30.0 * self.VEL_SCALE] * 6   # deg/sec
        req.acc = [60.0] * 6   # deg/sec^2
        req.time = 0.0
        req.mode = 0    # ABSOLUTE
        req.sync_type = 0  # SYNC

        self.get_logger().info(f"Executing spline trajectory ({n_points} points)...")
        self.get_logger().info(f"  Start (deg): {[f'{v:.2f}' for v in traj_deg[0]]}")
        self.get_logger().info(f"  End   (deg): {[f'{v:.2f}' for v in traj_deg[-1]]}")
        self.get_logger().info(f"  Current joints (rad): {[f'{v:.4f}' for v in self.current_joints]}")

        future = self.cli_spline.call_async(req)
        # spin_until_future_complete 는 콜백 내 호출 시 deadlock 위험 → while 대기
        t0 = time.time()
        while not future.done() and (time.time() - t0) < 30.0:
            time.sleep(0.05)
        if future.done() and future.result() and future.result().success:
            self.get_logger().info("Trajectory execution complete!")
        else:
            self.get_logger().error("Trajectory execution failed!")

    def execute_movej(self, traj_rad):
        """Execute only the final pose via MoveJoint (for testing)."""
        if not self.cli_movej.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveJoint service not available")
            return

        # Take the last point of trajectory, convert to degrees
        final_joints_deg = np.rad2deg(traj_rad[-1]).tolist()

        self.get_logger().info(f"Executing MoveJoint to: {[f'{v:.2f}' for v in final_joints_deg]}")

        req = MoveJoint.Request()
        req.pos = final_joints_deg
        req.vel = 30.0 * self.VEL_SCALE
        req.acc = 30.0 * self.VEL_SCALE
        req.time = 0.0
        req.radius = 0.0
        req.mode = 0      # ABSOLUTE
        req.blend_type = 0
        req.sync_type = 0  # SYNC — 응답=모션 완료. 도착 후 그리퍼 닫기 위해(닫힘 중 회전 방지)

        future = self.cli_movej.call_async(req)

        # Wait for response without spin_until_future_complete (avoids deadlock)
        timeout = 30.0
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            time.sleep(0.1)

        if future.done() and future.result() and future.result().success:
            self.get_logger().info("MoveJoint execution complete!")
        elif not future.done():
            self.get_logger().error("MoveJoint timed out!")
        else:
            self.get_logger().error("MoveJoint execution failed!")

    def pick_pose_cb(self, msg: PoseStamped):
        """Pick from current position: open → descend (cuRobo) → close → lift (cuRobo)."""
        if self.current_joints is None:
            self.get_logger().warn("No joint state received yet")
            return

        self.get_logger().info("=== PICK: open → descend → grasp → lift ===")

        pos = msg.pose.position
        ori = msg.pose.orientation
        GRASP_HEIGHT = 0.0      # object_tracking_node 가 접근축(grasp Z)으로 12cm 오프셋 적용 → 여기선 0 (중복 방지)
        LIFT_HEIGHT = 0.15      # lift 15cm after grasp
        MIN_Z = self.TCP_Z_MIN  # 🛑 바닥 하한 (클래스 상수와 통일)

        target_z = max(pos.z, MIN_Z)
        if pos.z < MIN_Z:
            self.get_logger().warn(
                f"🛑 [FLOOR GUARD] pick z={pos.z*1000:.1f}mm < "
                f"{MIN_Z*1000:.0f}mm → 클램프 (바닥 충돌 방지)")
        grasp_z = target_z + GRASP_HEIGHT

        # ===== Step 1: Open gripper =====
        self.get_logger().info("=== PICK Step 1/4: Open gripper (set_position 0) ===")
        self.gripper_open()
        time.sleep(1.5)

        # ===== Step 2: Descend to grasp height (cuRobo, avoids obstacles) =====
        self.get_logger().info(f"=== PICK Step 2/4: Descend to Z={grasp_z*1000:.1f}mm (cuRobo) ===")
        traj = self.plan(
            self.current_joints,
            [pos.x, pos.y, grasp_z],
            [ori.w, ori.x, ori.y, ori.z],
        )
        if traj is not None:
            self.execute_spline(traj)   # 충돌회피 궤적 전체 추종 (장애물 피해 하강)
            time.sleep(2.0)
        else:
            self.get_logger().error("Pick failed: descend planning failed")
            return

        # ===== Step 3: Close gripper =====
        # 이동 직후엔 컨트롤러가 motion 상태라 flange serial(그리퍼)이 안 닿음 →
        # move_stop 으로 motion 상태 클리어 후 close (실측: stop 없으면 그리퍼 안닫힘).
        self.get_logger().info("=== PICK Step 3/4: move_stop → Close gripper ===")
        try:
            if self.cli_stop.wait_for_service(timeout_sec=2.0):
                sr = MoveStop.Request(); sr.stop_mode = 1   # DR_QSTOP
                f = self.cli_stop.call_async(sr)
                t0 = time.time()
                while not f.done() and time.time() - t0 < 3.0:
                    time.sleep(0.05)
        except Exception as e:
            self.get_logger().warn(f"move_stop 실패(무시): {e}")
        time.sleep(1.0)   # motion 상태 클리어 대기
        self.gripper_grasp()   # safe_grasp 액션(전류기반 파지)
        time.sleep(1.5)

        # ===== Step 4: Lift up — 현재 자세 유지하고 Z만 상승 (캔이 서있게, 회전 없음) =====
        self.get_logger().info("=== PICK Step 4/4: Lift (자세 유지 Z상승) ===")
        self.lift_straight_up(LIFT_HEIGHT, vel=50.0)
        time.sleep(1.5)

        self.get_logger().info("=== PICK COMPLETE ===")

    def lift_straight_up(self, dz_m, vel=50.0):
        """현재 TCP 자세(rx/ry/rz) 그대로 두고 Z 만 dz_m(m) 만큼 수직 상승.
        → 잡은 물체(캔)가 회전 없이 그대로 서서 올라감. (top-down 강제 X)"""
        # 1) 현재 posx 읽기 [x,y,z(mm), rx,ry,rz(deg)]
        if not self.cli_posx.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("GetCurrentPosx 없음 — top-down move_linear fallback")
            return self.move_linear(0, 0, dz_m, vel=vel)  # fallback (자세 top-down)
        fut = self.cli_posx.call_async(GetCurrentPosx.Request())
        start = time.time()
        while not fut.done() and (time.time() - start) < 3.0:
            time.sleep(0.05)
        if not (fut.done() and fut.result()):
            self.get_logger().error("GetCurrentPosx 실패 — lift 취소")
            return
        cur = list(fut.result().task_pos_info[0].data)   # [x,y,z,rx,ry,rz] mm/deg
        # 2) Z 만 +dz, 나머지(x,y, rx,ry,rz) 그대로 → move_line
        req = MoveLine.Request()
        req.pos = [cur[0], cur[1], cur[2] + dz_m * 1000.0, cur[3], cur[4], cur[5]]
        _v = vel * self.VEL_SCALE
        req.vel = [_v, 30.0 * self.VEL_SCALE]; req.acc = [_v, 30.0 * self.VEL_SCALE]
        req.time = 0.0
        req.ref = 0; req.mode = 0; req.blend_type = 0; req.sync_type = 1
        f2 = self.cli_movel.call_async(req)
        s2 = time.time()
        while not f2.done() and (time.time() - s2) < 15.0:
            time.sleep(0.1)
        ok = f2.done() and f2.result() and f2.result().success
        self.get_logger().info(
            f"Lift {'완료' if ok else '실패'}: Z {cur[2]:.0f}→{cur[2]+dz_m*1000:.0f}mm "
            f"(자세 rx={cur[3]:.0f} ry={cur[4]:.0f} rz={cur[5]:.0f}° 유지)")

    def move_linear(self, x, y, z, vel=100.0):
        """Move TCP linearly (movel) in mm, pointing down."""
        if not self.cli_movel.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveLine service not available")
            return

        req = MoveLine.Request()
        req.pos = [x * 1000, y * 1000, z * 1000, 0.0, 180.0, 0.0]  # mm, deg
        req.vel = [vel * self.VEL_SCALE, 30.0 * self.VEL_SCALE]
        req.acc = [vel * self.VEL_SCALE, 30.0 * self.VEL_SCALE]
        req.time = 0.0
        req.ref = 0       # base frame
        req.mode = 0       # absolute
        req.blend_type = 0
        req.sync_type = 1  # async

        future = self.cli_movel.call_async(req)
        timeout = 15.0
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            time.sleep(0.1)

        if future.done() and future.result() and future.result().success:
            self.get_logger().info(f"MoveLine complete: Z={z*1000:.1f}mm")
        else:
            self.get_logger().error("MoveLine failed!")

    def gripper_open(self, position=None, timeout_sec=3.0):
        """그리퍼 열기 — 브리지 /gripper_service/set_position (기본 position=0=완전개방)."""
        if position is None:
            position = int(self.get_parameter('grasp_open_position').value)
        if not self.cli_gripper_open.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("set_position service not available"); return False
        req = SetPosition.Request()
        req.position = int(position)
        req.timeout_sec = float(timeout_sec)
        future = self.cli_gripper_open.call_async(req)
        start = time.time()
        while not future.done() and (time.time() - start) < timeout_sec + 1.0:
            time.sleep(0.05)
        res = future.result() if future.done() else None
        self.get_logger().info(
            f"gripper_open(set_position {position}): "
            f"{'success' if res and res.success else 'fail/timeout'}")
        return bool(res and res.success)

    def gripper_grasp(self):
        """파지 — 브리지 /gripper_service/safe_grasp 액션(전류기반, 물성별 max_current)."""
        tp = int(self.get_parameter('grasp_target_position').value)
        mc = int(self.get_parameter('grasp_max_current').value)
        cd = int(self.get_parameter('grasp_current_delta').value)
        if not self.act_safe_grasp.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("safe_grasp action server not available"); return False
        goal = SafeGrasp.Goal()
        goal.target_position = tp
        goal.max_current = mc
        goal.current_delta_threshold = cd
        goal.timeout_sec = 5.0
        self.get_logger().info(
            f"safe_grasp goal: pos={tp} max_current={mc}mA delta={cd}")
        send_future = self.act_safe_grasp.send_goal_async(goal)
        start = time.time()
        while not send_future.done() and (time.time() - start) < 4.0:
            time.sleep(0.05)
        gh = send_future.result() if send_future.done() else None
        if gh is None or not gh.accepted:
            self.get_logger().warn("safe_grasp goal rejected/timeout"); return False
        res_future = gh.get_result_async()
        start = time.time()
        while not res_future.done() and (time.time() - start) < 8.0:
            time.sleep(0.05)
        wrapped = res_future.result() if res_future.done() else None
        res = wrapped.result if wrapped else None
        if res:
            self.get_logger().info(
                f"safe_grasp result: success={res.success} "
                f"grasp_detected={res.grasp_detected} "
                f"final_current={res.final_current}mA")
        else:
            self.get_logger().warn("safe_grasp result timeout")
        return bool(res and res.success)

    def call_trigger(self, client):
        """Call a Trigger service (구버전 — 미사용. 그리퍼는 gripper_open/grasp 사용)."""
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("Trigger service not available")
            return

        req = Trigger.Request()
        future = client.call_async(req)
        timeout = 10.0
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            time.sleep(0.1)

        if future.done() and future.result():
            self.get_logger().info(f"Trigger: {future.result().message}")


def main():
    rclpy.init()
    node = CuroboPlanner()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
