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
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from dsr_msgs2.srv import MoveSplineJoint, MoveJoint, MoveLine

from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState as CuroboJointState, RobotConfig
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig, Cuboid
from std_msgs.msg import String
import json


class CuroboPlanner(Node):
    # E0509 joint order as expected by cuRobo and Doosan services
    JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    # GraspGen 범용 파지 경로 파라미터 (orientation을 그대로 사용 — down_quat 강제 없음)
    GRASP_STANDOFF_DIST = 0.08   # grasp pose에서 접근축 반대 방향으로 후퇴하는 거리 (m)
    GRASP_LIFT_HEIGHT   = 0.15   # 파지 후 들어올리는 높이, world Z 기준 (m)
    GRASP_VEL_SLOW      = 15.0   # 첫 실전 실행 안전을 위한 저속 (deg/s, deg/s^2)

    def __init__(self):
        super().__init__("curobo_planner_node")

        # Separate callback group for service calls to avoid deadlock
        self.service_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()

        self.get_logger().info("Initializing cuRobo planner...")

        # Current joint state
        self.current_joints = None

        # Config path
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "curobo"
        )
        # Fallback for installed package
        if not os.path.exists(config_dir):
            from ament_index_python.packages import get_package_share_directory
            config_dir = os.path.join(
                get_package_share_directory("e0509_gripper_description"),
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

        # GraspGen 6-DOF 파지 자세 구독: orientation을 그대로 사용하는 범용 파지 경로
        # (target_pose_cb/pick_pose_cb는 down_quat을 강제하는 "캔 집기" 전용 경로)
        self.grasp_pose_sub = self.create_subscription(
            PoseStamped, "/dsr01/curobo/grasp_pose",
            self.grasp_pose_cb, 10)

        # Obstacles update subscription (JSON format)
        self.obstacles_sub = self.create_subscription(
            String, "/dsr01/curobo/obstacles",
            self.obstacles_cb, 10)

        # Shelf target subscription: "floor,slot" (e.g. "1,2")
        self.shelf_sub = self.create_subscription(
            String, "/dsr01/curobo/shelf_target",
            self.shelf_target_cb, 10,
            callback_group=self.service_cb_group)

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
        self.cli_gripper_open = self.create_client(
            Trigger, "/dsr01/gripper/open",
            callback_group=self.service_cb_group)
        self.cli_gripper_close = self.create_client(
            Trigger, "/dsr01/gripper/close",
            callback_group=self.service_cb_group)

        self.get_logger().info("========================================")
        self.get_logger().info("cuRobo Planner Ready!")
        self.get_logger().info("  Subscribe: /dsr01/curobo/target_pose (move)")
        self.get_logger().info("  Subscribe: /dsr01/curobo/pick_pose (pick)")
        self.get_logger().info("  Subscribe: /dsr01/curobo/grasp_pose (GraspGen 범용 파지)")
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
        """s key: 그리퍼 열기 → cuRobo 캔 위 안전 높이 → 수직 하강."""
        if self.current_joints is None:
            self.get_logger().warn("No joint state received yet")
            return

        pos = msg.pose.position

        # 비정상 좌표 차단
        dist_xy = (pos.x**2 + pos.y**2) ** 0.5
        if dist_xy > 0.85 or dist_xy < 0.05 or pos.z < -0.25:
            self.get_logger().error(
                f"Position rejected: X={pos.x*1000:.0f} Y={pos.y*1000:.0f} Z={pos.z*1000:.0f}mm")
            return

        SAFE_HEIGHT   = 0.35
        MIN_Z         = 0.05
        GRASP_Z_LIFT  = 0.095  # 캔 중간 높이 보정
        GRASP_Y_OFFSET = 0.02  # Y 보정 (+= 왼쪽)
        grasp_z = max(pos.z, MIN_Z) + GRASP_Z_LIFT
        grasp_x = pos.x
        grasp_y = pos.y + GRASP_Y_OFFSET

        self.get_logger().info(
            f"=== PRE-GRASP: can=({pos.x*1000:.0f},{pos.y*1000:.0f},{pos.z*1000:.0f}mm) "
            f"→ target=({grasp_x*1000:.0f},{grasp_y*1000:.0f},{grasp_z*1000:.0f}mm) ==="
        )

        # Step 1: 그리퍼 열기
        self.call_trigger(self.cli_gripper_open)

        # Step 2: cuRobo → 캔 바로 위 안전 높이 (그리퍼 아래 방향)
        down_quat = [0.0, 0.7071, 0.7071, 0.0]
        traj = self.plan(self.current_joints, [grasp_x, grasp_y, SAFE_HEIGHT], down_quat)
        if traj is not None:
            self.execute_movej(traj)
            time.sleep(2.0)
        else:
            self.get_logger().error("Pre-grasp planning failed")
            return

        # Step 3: 수직 하강 → 캔 높이
        self.move_linear(grasp_x, grasp_y, grasp_z)
        time.sleep(1.5)

        self.get_logger().info("=== PRE-GRASP READY — press [p] to grab ===")

    def plan(self, start_joints, target_pos, target_quat_wxyz):
        """Plan trajectory using cuRobo."""
        t0 = time.time()

        start_state = CuroboJointState.from_position(
            position=torch.tensor([start_joints], device="cuda:0", dtype=torch.float32),
            joint_names=self.JOINT_NAMES,
        )

        target_pose = Pose(
            position=torch.tensor([target_pos], device="cuda:0", dtype=torch.float32),
            quaternion=torch.tensor([target_quat_wxyz], device="cuda:0", dtype=torch.float32),
        )

        result = self.motion_gen.plan_single(
            start_state,
            target_pose,
            MotionGenPlanConfig(
                max_attempts=60,  #높이면 성공률 올라가고 속도는 줄어듬.
                enable_graph=False, #true면 복잡한 경로도 찾지만 느리다.
                enable_opt=True, #false면 빠르지만 경로가 거칠다.
                use_start_state_as_retract=True,
            )
        )
        plan_time = (time.time() - t0) * 1000

        if result.success.item():
            traj = result.get_interpolated_plan()
            positions = traj.position.cpu().numpy()
            self.get_logger().info(
                f"Planning SUCCESS: {plan_time:.1f}ms, {positions.shape[0]} points")
            return positions
        else:
            self.get_logger().error(f"Planning FAILED: {plan_time:.1f}ms")
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

        req.vel = [30.0] * 6   # deg/sec
        req.acc = [60.0] * 6   # deg/sec^2
        req.time = 0.0
        req.mode = 0    # ABSOLUTE
        req.sync_type = 0  # SYNC

        self.get_logger().info(f"Executing spline trajectory ({n_points} points)...")
        self.get_logger().info(f"  Start (deg): {[f'{v:.2f}' for v in traj_deg[0]]}")
        self.get_logger().info(f"  End   (deg): {[f'{v:.2f}' for v in traj_deg[-1]]}")
        self.get_logger().info(f"  Current joints (rad): {[f'{v:.4f}' for v in self.current_joints]}")

        future = self.cli_spline.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        if future.result() and future.result().success:
            self.get_logger().info("Trajectory execution complete!")
        else:
            self.get_logger().error("Trajectory execution failed!")

    def execute_movej(self, traj_rad, vel=30.0, acc=30.0):
        """Execute only the final pose via MoveJoint (for testing)."""
        if not self.cli_movej.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveJoint service not available")
            return

        # Take the last point of trajectory, convert to degrees
        final_joints_deg = np.rad2deg(traj_rad[-1]).tolist()

        # 비정상 관절각 검사 (cuRobo가 degenerate 해를 반환하는 경우 차단)
        if any(abs(j) > 200.0 for j in final_joints_deg):
            self.get_logger().error(
                f"MoveJoint ABORTED: abnormal joint angles {[f'{v:.1f}' for v in final_joints_deg]}"
            )
            return

        self.get_logger().info(f"Executing MoveJoint to: {[f'{v:.2f}' for v in final_joints_deg]}")

        req = MoveJoint.Request()
        req.pos = final_joints_deg
        req.vel = vel
        req.acc = acc
        req.time = 0.0
        req.radius = 0.0
        req.mode = 0      # ABSOLUTE
        req.blend_type = 0
        req.sync_type = 1  # ASYNC (don't block robot controller)

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
            result = future.result()
            self.get_logger().error(f"MoveJoint execution failed! result={result}")

    def pick_pose_cb(self, msg: PoseStamped):
        """p key: 그리퍼 닫기 → 수직 들기. (s로 pre-grasp 완료 후 실행)"""
        pos = msg.pose.position
        MIN_Z       = 0.05
        LIFT_HEIGHT = 0.25

        grasp_z = max(pos.z, MIN_Z)

        self.get_logger().info(
            f"=== GRAB: ({pos.x*1000:.0f},{pos.y*1000:.0f},{grasp_z*1000:.0f}mm) ==="
        )

        # Step 1: 그리퍼 닫기
        self.call_trigger(self.cli_gripper_close)
        time.sleep(1.5)

        # Step 2: cuRobo로 들기 (movel 서비스 대신)
        if self.current_joints is None:
            self.get_logger().error("관절 정보 없음 — 들기 실패")
            return
        down_quat = [0.0, 0.7071, 0.7071, 0.0]
        lift_z = grasp_z + LIFT_HEIGHT
        traj = self.plan(self.current_joints, [pos.x, pos.y, lift_z], down_quat)
        if traj is not None:
            self.execute_movej(traj)
            time.sleep(2.0)
        else:
            self.get_logger().error("들기 계획 실패")

        self.get_logger().info("=== GRAB COMPLETE ===")

    def grasp_pose_cb(self, msg: PoseStamped):
        """GraspGen 6-DOF 파지 자세 구독 콜백 — 접근(approach)까지만 수행.

        target_pose_cb/pick_pose_cb는 down_quat을 강제하는 "캔 집기" 전용 경로라
        GraspGen이 계산한 임의의 orientation을 그대로 살릴 수 없다. 이 콜백은
        orientation을 절대 덮어쓰지 않고, GraspGen 컨벤션(그리퍼 frame +Z축 = 접근
        방향)에 따라 standoff(후퇴) → grasp pose 접근까지만 수행한다.

        파지(닫기)·들기는 여기서 바로 하지 않고 grasp_pick_cb에서 별도로 확인 후
        실행한다 — 이동 중간에 그리퍼가 닫혀버리는 문제를 막고, 도착한 자세를
        눈으로 확인한 뒤 사용자가 명시적으로 "잡기"를 확정할 수 있게 하기 위함.
        """
        if self.current_joints is None:
            self.get_logger().warn("No joint state received yet")
            return

        pos = msg.pose.position
        quat_xyzw = [msg.pose.orientation.x, msg.pose.orientation.y,
                     msg.pose.orientation.z, msg.pose.orientation.w]

        # 비정상 좌표 차단 (target_pose_cb와 동일 기준)
        dist_xy = (pos.x**2 + pos.y**2) ** 0.5
        if dist_xy > 0.85 or dist_xy < 0.05 or pos.z < -0.25:
            self.get_logger().error(
                f"Grasp pose rejected: X={pos.x*1000:.0f} Y={pos.y*1000:.0f} Z={pos.z*1000:.0f}mm")
            return

        grasp_pos = np.array([pos.x, pos.y, pos.z])

        # GraspGen 컨벤션: 그리퍼 frame의 +Z축 = 접근(approach) 방향
        approach_axis = Rotation.from_quat(quat_xyzw).as_matrix()[:, 2]
        standoff_pos = grasp_pos - approach_axis * self.GRASP_STANDOFF_DIST

        # cuRobo Pose는 quaternion을 wxyz 순서로 받음
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        # grasp_pick_cb에서 동일 orientation/위치로 파지+들기를 수행하도록 저장
        self._last_grasp_pose = (grasp_pos.copy(), quat_wxyz)

        self.get_logger().info(
            f"=== GRASP APPROACH(GraspGen): target=({grasp_pos[0]*1000:.0f},{grasp_pos[1]*1000:.0f},{grasp_pos[2]*1000:.0f}mm) "
            f"standoff=({standoff_pos[0]*1000:.0f},{standoff_pos[1]*1000:.0f},{standoff_pos[2]*1000:.0f}mm) "
            f"orientation 유지, 저속 {self.GRASP_VEL_SLOW:.0f}deg/s ==="
        )

        # Step 1: 그리퍼 열기
        self.call_trigger(self.cli_gripper_open)

        # Step 2: standoff pose로 이동 (orientation 유지)
        traj = self.plan(self.current_joints, standoff_pos.tolist(), quat_wxyz)
        if traj is None:
            self.get_logger().error("GRASP 실패: standoff 경로 계획 실패")
            self._last_grasp_pose = None
            return
        self.execute_movej(traj, vel=self.GRASP_VEL_SLOW, acc=self.GRASP_VEL_SLOW)
        time.sleep(2.0)

        # Step 3: 접근축을 따라 grasp pose까지 이동 (orientation 유지) — 여기서 정지
        traj = self.plan(self.current_joints, grasp_pos.tolist(), quat_wxyz)
        if traj is None:
            self.get_logger().error("GRASP 실패: 접근 경로 계획 실패")
            self._last_grasp_pose = None
            return
        self.execute_movej(traj, vel=self.GRASP_VEL_SLOW, acc=self.GRASP_VEL_SLOW)
        time.sleep(1.5)

        self.get_logger().info("=== GRASP READY — 자세 확인 후 [p] 로 파지+들기 ===")

    def grasp_pick_cb(self, msg: Empty):
        """GraspGen 파지 확정 콜백 — 그리퍼 닫기 + 들기.

        grasp_pose_cb로 접근까지 완료된 뒤, 도착한 자세를 눈으로 확인하고
        사용자가 'p'로 명시적으로 확정했을 때만 호출된다 (안전 확인 단계).
        """
        if self._last_grasp_pose is None:
            self.get_logger().warn("GraspGen 접근 정보 없음 — 먼저 grasp_pose로 접근하세요")
            return
        if self.current_joints is None:
            self.get_logger().warn("No joint state received yet")
            return

        grasp_pos, quat_wxyz = self._last_grasp_pose

        self.get_logger().info(
            f"=== GRASP PICK(GraspGen): ({grasp_pos[0]*1000:.0f},{grasp_pos[1]*1000:.0f},{grasp_pos[2]*1000:.0f}mm) "
            f"닫기 + {self.GRASP_LIFT_HEIGHT*1000:.0f}mm 들기 ==="
        )

        # Step 1: 그리퍼 닫기
        self.call_trigger(self.cli_gripper_close)
        time.sleep(1.5)

        # Step 2: 들어올리기 — orientation은 유지한 채 world Z로만 상승
        # (잡은 자세를 바꾸면 충돌·낙하 위험이 있어 자세를 다시 계산하지 않음)
        lift_pos = grasp_pos.copy()
        lift_pos[2] += self.GRASP_LIFT_HEIGHT
        traj = self.plan(self.current_joints, lift_pos.tolist(), quat_wxyz)
        if traj is None:
            self.get_logger().error("GRASP: 들기 경로 계획 실패 (물체는 잡은 상태로 유지됨)")
            return
        self.execute_movej(traj, vel=self.GRASP_VEL_SLOW, acc=self.GRASP_VEL_SLOW)
        time.sleep(2.0)

        self._last_grasp_pose = None
        self.get_logger().info("=== GRASP(GraspGen) COMPLETE ===")

    def move_linear_side(self, x, y, z, approach_deg, vel=100.0):
        """Move TCP linearly with gripper pointing horizontally at approach_deg."""
        if not self.cli_movel.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveLine service not available")
            return

        req = MoveLine.Request()
        # [rx, ry, rz]: Ry=90° tilts tool from up to horizontal, Rz=approach_deg rotates in XY
        req.pos = [x * 1000, y * 1000, z * 1000, 0.0, 90.0, approach_deg]
        req.vel = [vel, 30.0]
        req.acc = [vel, 30.0]
        req.time = 0.0
        req.ref = 0
        req.mode = 0
        req.blend_type = 0
        req.sync_type = 1

        future = self.cli_movel.call_async(req)
        timeout = 15.0
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            time.sleep(0.1)

        if future.done() and future.result() and future.result().success:
            self.get_logger().info(f"MoveLine (side) complete")
        else:
            self.get_logger().error("MoveLine (side) failed!")

    def move_linear(self, x, y, z, vel=100.0):
        """Move TCP linearly (movel) pointing down."""
        if not self.cli_movel.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveLine service not available")
            return

        req = MoveLine.Request()
        req.pos = [x * 1000, y * 1000, z * 1000, 0.0, 180.0, 0.0]  # mm, deg
        req.vel = [vel, 30.0]
        req.acc = [vel, 30.0]
        req.time = 0.0
        req.ref = 0
        req.mode = 0
        req.blend_type = 0
        req.sync_type = 1

        future = self.cli_movel.call_async(req)
        timeout = 15.0
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            time.sleep(0.1)

        if future.done() and future.result() and future.result().success:
            self.get_logger().info(f"MoveLine complete: Z={z*1000:.1f}mm")
        else:
            self.get_logger().error("MoveLine failed!")

    # ──────────────────────────────────────────────
    # 서랍 배치 (Shelf placement)
    # ──────────────────────────────────────────────
    # 기준점: 로봇 베이스 기준 x=0.60, y=-0.335, z=0.04 (바깥 1층 바닥 중앙)
    # 외부 크기: 폭 33.2cm(Y), 길이 24.5cm(X), 높이 16cm  /  층 간 간격 1.5cm
    SHELF_X       = 0.60    # 서랍 바깥면 X
    SHELF_Y_CTR   = -0.335  # 서랍 Y 중심
    SHELF_Z1      = 0.04    # 1층 바닥 Z
    SHELF_W       = 0.332   # 폭 (Y방향)
    SHELF_H       = 0.16    # 층 높이
    SHELF_GAP     = 0.015   # 층간 간격
    SHELF_SLIDE   = 0.10    # 서랍 안으로 들어가는 깊이
    SHELF_APPR    = 0.18    # 바깥면 앞 pre-approach 거리
    SHELF_SAFE_Z  = 0.65    # cuRobo 경유 안전 높이

    def shelf_target_cb(self, msg: String):
        """Receive '층,칸' string and place object in shelf."""
        try:
            parts = msg.data.strip().split(',')
            floor = int(parts[0])
            slot  = int(parts[1])
            if not (1 <= floor <= 3 and 1 <= slot <= 3):
                self.get_logger().error(f"서랍 범위 초과: {floor}층 {slot}칸")
                return
            self.place_at(floor, slot)
        except Exception as e:
            self.get_logger().error(f"shelf_target_cb 오류: {e}")

    def place_at(self, floor: int, slot: int):

        """Place held object at shelf[floor][slot]. floor/slot: 1-indexed."""
        if self.current_joints is None:
            self.get_logger().warn("관절 정보 없음")
            return

        # Y 위치: 3칸 균등 배분
        y = self.SHELF_Y_CTR - self.SHELF_W / 2 + self.SHELF_W / 3 * (slot - 0.5)
        # Z 위치: 각 층 중간 높이
        z = self.SHELF_Z1 + (floor - 1) * (self.SHELF_H + self.SHELF_GAP) + self.SHELF_H / 2
        # X 위치
        x_pre   = self.SHELF_X - self.SHELF_APPR   # 바깥면 앞
        x_place = self.SHELF_X + self.SHELF_SLIDE   # 서랍 안

        approach_deg = math.degrees(math.atan2(y, self.SHELF_X))

        self.get_logger().info(
            f"=== PLACE: {floor}층 {slot}칸 → ({x_place:.3f}, {y:.3f}, {z:.3f})m ==="
        )

        # Step 1: cuRobo → 안전 높이 (down orientation)
        down_quat = [0.0, 0.7071, 0.7071, 0.0]
        traj = self.plan(self.current_joints, [x_pre, y, self.SHELF_SAFE_Z], down_quat)
        if traj is not None:
            self.execute_movej(traj)
            time.sleep(2.0)
        else:
            self.get_logger().error("PLACE 실패: 안전 높이 경로 계획 실패")
            return

        # Step 2: movel → 수평 자세로 서랍 앞 높이로 하강
        self.move_linear_side(x_pre, y, z, approach_deg)
        time.sleep(1.5)

        # Step 3: movel → 서랍 안으로 슬라이드인
        self.move_linear_side(x_place, y, z, approach_deg)
        time.sleep(1.5)

        # Step 4: 그리퍼 열어서 물건 내려놓기
        self.call_trigger(self.cli_gripper_open)
        time.sleep(1.0)

        # Step 5: movel → 서랍 밖으로 후퇴
        self.move_linear_side(x_pre, y, z, approach_deg)
        time.sleep(1.5)

        self.get_logger().info("=== PLACE COMPLETE ===")

    def call_trigger(self, client):
        """Call a Trigger service (gripper open/close)."""
        if not client.service_is_ready():
            self.get_logger().warn("Gripper service not ready — skipping")
            return

        req = Trigger.Request()
        future = client.call_async(req)
        timeout = 2.0  # 그리퍼 시리얼 불량 시 오래 기다리지 않음
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            time.sleep(0.05)

        if future.done() and future.result():
            self.get_logger().info(f"Trigger: {future.result().message}")
        else:
            self.get_logger().warn("Gripper trigger timed out — continuing")


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
