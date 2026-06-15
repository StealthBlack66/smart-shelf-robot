#!/usr/bin/env python3
"""
ArmControllerNode — cuRobo 경로계획 + Doosan 실행 통합 노드.

Pipeline A (서비스 기반 — MainControllerNode 연동):
  /move_to_shelf_view, /move_to_product_view, /move_to_pick,
  /move_to_place, /move_to_home  (Trigger 서비스)

Pipeline B (토픽 기반 — WebcamSegNode 직접 연동):
  /dsr01/curobo/target_pose, /dsr01/curobo/pick_pose,
  /dsr01/curobo/obstacles, /dsr01/curobo/grasp_class
"""

import os
import sys
import time
import json

# CUDA_VISIBLE_DEVICES="" 이면 GPU 숨겨짐 → 임포트 전 해제
if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
    del os.environ["CUDA_VISIBLE_DEVICES"]

import yaml
import torch
import numpy as np
from scipy.spatial.transform import Rotation

if not torch.cuda.is_available():
    print("[ERROR] CUDA를 사용할 수 없습니다. cuRobo는 GPU가 필요합니다.")
    print("  1. nvidia-smi 로 GPU 확인")
    print("  2. sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm 후 재시도")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState

from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState as CuroboJointState, RobotConfig
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig, Cuboid

from dsr_msgs2.srv import MoveSplineJoint, MoveJoint, MoveLine, GetCurrentPosx, MoveStop
from dsr_gripper_tcp_interfaces.srv import SetPosition
from dsr_gripper_tcp_interfaces.action import SafeGrasp


class ArmControllerNode(Node):
    JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    # Named joint targets (degrees)
    HOME_JOINTS_DEG         = [-6.73, 8.12, 104.62, 80.22, 93.13, -23.49]
    SHELF_VIEW_JOINTS_DEG   = [-6.73, 8.12, 104.62, 80.22, 93.13, -23.49]
    PRODUCT_VIEW_JOINTS_DEG = [0.0, -36.0, 56.0, 5.0, 110.0, 0.0]

    NAMED_TARGETS_DEG = {
        'home':         HOME_JOINTS_DEG,
        'shelf_view':   SHELF_VIEW_JOINTS_DEG,
        'product_view': PRODUCT_VIEW_JOINTS_DEG,
    }

    # Place 시퀀스 좌표 (posx = [x,y,z mm, a,b,c ZYZ deg])
    SLOT0_POINT_J_DEG  = [27.67, 5.82, 95.72, 90.93, 61.98, -10.64]
    SLOT0_L_POSX       = [357.21, 511.97, 487.17, 89.98, 94.59, 90.0]
    SLOT0_DOWN_L_POSX  = [357.230, 498.590, 468.640, 89.98, 94.59, 90.0]

    SLOT1_POINT_J_DEG  = [21.17, 23.36, 72.79, 87.48, 68.49, -4.8]
    SLOT1_L_POSX       = [483.77, 449.08, 493.21, 89.99, 94.59, 90.01]

    SLOT2_POINT_J_DEG  = [22.17, 23.13, 103.31, 99.79, 69.5, -37.32]
    SLOT2_L_POSX       = [413.61, 511.97, 310.56, 90.00, 90.58, 90.0]
    SLOT2_DOWN_L_POSX  = [413.61, 511.97, 290.15, 103.00, 90.58, 90.0]

    PLACE_TARGETS = {
        'bottle': {
            'point_j':         SLOT0_POINT_J_DEG,
            'entry_posx':      SLOT0_L_POSX,
            'grasp_down_posx': SLOT0_DOWN_L_POSX,
        },
        'snack_bag': {
            'point_j':         SLOT1_POINT_J_DEG,
            'entry_posx':      SLOT1_L_POSX,
            'grasp_down_posx': None,
        },
        'can': {
            'point_j':         SLOT2_POINT_J_DEG,
            'entry_posx':      SLOT2_L_POSX,
            'grasp_down_posx': SLOT2_DOWN_L_POSX,
        },
    }

    # 안전/속도 상수
    TCP_Z_MIN  = 0.02   # m, base 기준 TCP 최저 허용 Z (바닥 충돌 방지)
    VEL_SCALE  = float(os.environ.get('CUROBO_VEL_SCALE', '0.5'))
    J4_MIN_RAD = 0.0    # joint_4 하한 (rad) — 이 아래면 손목이 바닥 방향
    YAW_RETRY  = (np.pi, np.pi / 2, -np.pi / 2)

    def __init__(self):
        super().__init__('arm_controller_node')

        self.declare_parameter('place_target', 'can')
        self.declare_parameter('grasp_target_position', 700)
        self.declare_parameter('grasp_max_current',     600)
        self.declare_parameter('grasp_current_delta',   30)
        self.declare_parameter('grasp_open_position',   0)

        self.service_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()

        # 상태
        self.current_joints = None
        self.object_pose    = None
        self.grasp_class    = None

        # cuRobo 초기화 (GPU 1회만)
        self.get_logger().info("cuRobo 초기화 중...")
        config_dir = self._find_config_dir()
        self.tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

        robot_cfg = RobotConfig.from_basic(
            urdf_path=os.path.join(config_dir, "e0509_gripper.urdf"),
            base_link="base_link",
            ee_link="gripper_rh_p12_rn_base",
            tensor_args=self.tensor_args,
        )
        world_cfg = WorldConfig(
            cuboid=[Cuboid(name="table", pose=[0.0, 0.0, -0.02, 1, 0, 0, 0],
                           dims=[1.2, 1.2, 0.04])]
        )
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg, world_cfg, self.tensor_args,
            num_trajopt_seeds=4, num_graph_seeds=4,
            collision_cache={"obb": 30, "mesh": 10},
        )
        self.motion_gen = MotionGen(motion_gen_cfg)
        self.motion_gen.warmup(warmup_js_trajopt=False)
        self.get_logger().info("cuRobo 준비 완료!")

        self.grasp_force_params = self._load_grasp_force_params(config_dir)

        # Subscribers
        self.create_subscription(JointState, '/dsr01/joint_states',
                                 self._joint_state_cb, 10)
        self.create_subscription(PoseStamped, '/object_pose',
                                 self._object_pose_cb, 10)
        # Pipeline B 토픽
        self.create_subscription(PoseStamped, '/dsr01/curobo/target_pose',
                                 self._target_pose_cb, 10)
        self.create_subscription(PoseStamped, '/dsr01/curobo/pick_pose',
                                 self._pick_pose_cb, 10)
        self.create_subscription(String, '/dsr01/curobo/obstacles',
                                 self._obstacles_cb, 10)
        self.create_subscription(String, '/dsr01/curobo/grasp_class',
                                 self._grasp_class_cb, 10)

        # Pipeline A 서비스 서버
        for name, cb in [
            ('/move_to_shelf_view',   self._srv_shelf_view),
            ('/move_to_product_view', self._srv_product_view),
            ('/move_to_pick',         self._srv_pick),
            ('/move_to_place',        self._srv_place),
            ('/move_to_home',         self._srv_home),
        ]:
            self.create_service(Trigger, name, cb,
                                callback_group=self.service_cb_group)

        # Doosan 서비스/액션 클라이언트
        def _cli(srv_type, topic):
            return self.create_client(srv_type, topic,
                                      callback_group=self.service_cb_group)

        self.cli_spline = _cli(MoveSplineJoint, '/dsr01/motion/move_spline_joint')
        self.cli_movej  = _cli(MoveJoint,       '/dsr01/motion/move_joint')
        self.cli_movel  = _cli(MoveLine,        '/dsr01/motion/move_line')
        self.cli_stop   = _cli(MoveStop,        '/dsr01/motion/move_stop')
        self.cli_posx   = _cli(GetCurrentPosx,  '/dsr01/aux_control/get_current_posx')
        self.cli_gripper_open = _cli(SetPosition, '/gripper_service/set_position')
        self.act_safe_grasp   = ActionClient(self, SafeGrasp,
                                             '/gripper_service/safe_grasp',
                                             callback_group=self.service_cb_group)

        self.get_logger().info("========================================")
        self.get_logger().info("ArmControllerNode 준비 완료 (Pipeline A + B)")
        self.get_logger().info("  Pipeline A: /move_to_* 서비스")
        self.get_logger().info("  Pipeline B: /dsr01/curobo/* 토픽")
        self.get_logger().info("========================================")

    # ── 초기화 헬퍼 ───────────────────────────────────────────

    def _find_config_dir(self):
        local = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "curobo")
        if os.path.exists(local):
            return local
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory("e0509_gripper_description"),
            "config", "curobo")

    def _load_grasp_force_params(self, config_dir):
        path = os.path.join(os.path.dirname(config_dir), "grasp_force_params.yaml")
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            gf = data.get('grasp_force', {})
            self.get_logger().info(f"파지힘 로드: {list(gf.keys())}")
            return gf
        except Exception as e:
            self.get_logger().warn(f"grasp_force_params.yaml 로드 실패({e}) → 기본값 사용")
            return {}

    # ── Subscriber 콜백 ───────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        joint_map = {n: msg.position[i]
                     for i, n in enumerate(msg.name) if i < len(msg.position)}
        joints = [joint_map.get(n) for n in self.JOINT_NAMES]
        if None not in joints:
            self.current_joints = joints

    def _object_pose_cb(self, msg: PoseStamped):
        self.object_pose = msg

    def _grasp_class_cb(self, msg: String):
        self.grasp_class = msg.data.strip() or None
        self.get_logger().info(f"[grasp_class] {self.grasp_class}")

    def _obstacles_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            cuboids = [Cuboid(name="table", pose=[0.0, 0.0, -0.02, 1, 0, 0, 0],
                              dims=[1.2, 1.2, 0.04])]
            for obj in data:
                cuboids.append(Cuboid(
                    name=obj["name"],
                    pose=[obj["pos"][0], obj["pos"][1], obj["pos"][2], 1, 0, 0, 0],
                    dims=obj.get("dims", [0.05, 0.05, 0.05])))
            self.motion_gen.update_world(WorldConfig(cuboid=cuboids))
            self.get_logger().info(
                f"장애물 업데이트: table + {len(data)}개")
        except Exception as e:
            self.get_logger().error(f"장애물 업데이트 실패: {e}")

    # Pipeline B 토픽 콜백
    def _target_pose_cb(self, msg: PoseStamped):
        if self.current_joints is None:
            self.get_logger().warn("joint_states 미수신")
            return
        pos, ori = msg.pose.position, msg.pose.orientation
        safe_z = pos.z
        if safe_z < self.TCP_Z_MIN:
            self.get_logger().warn(
                f"[FLOOR GUARD] z={pos.z*1000:.1f}mm → {self.TCP_Z_MIN*1000:.0f}mm 클램프")
            safe_z = self.TCP_Z_MIN
        traj = self.plan(self.current_joints, [pos.x, pos.y, safe_z],
                         [ori.w, ori.x, ori.y, ori.z])
        if traj is not None:
            self.execute_spline(traj)
        else:
            self.get_logger().error("경로계획 실패")

    def _pick_pose_cb(self, msg: PoseStamped):
        if self.current_joints is None:
            self.get_logger().warn("joint_states 미수신")
            return
        self._do_pick_sequence(msg)

    # ── Pipeline A 서비스 핸들러 ──────────────────────────────

    def _srv_shelf_view(self, request, response):
        ok = self._move_to_named_target('shelf_view')
        response.success = ok
        response.message = "완료" if ok else "실패"
        return response

    def _srv_product_view(self, request, response):
        ok = self._move_to_named_target('product_view')
        response.success = ok
        response.message = "완료" if ok else "실패"
        return response

    def _srv_pick(self, request, response):
        if self.object_pose is None:
            response.success = False
            response.message = "/object_pose 미수신"
            return response
        if self.current_joints is None:
            response.success = False
            response.message = "joint_states 미수신"
            return response
        self._do_pick_sequence(self.object_pose)
        response.success = True
        response.message = "Pick 완료"
        return response

    def _srv_place(self, request, response):
        ok = self._move_to_place()
        place_target = self.get_parameter('place_target').get_parameter_value().string_value
        response.success = ok
        response.message = f"Place {'완료' if ok else '실패'} ({place_target})"
        return response

    def _srv_home(self, request, response):
        ok = self._move_to_named_target('home')
        response.success = ok
        response.message = "완료" if ok else "실패"
        return response

    # ── Pick 시퀀스 ──────────────────────────────────────────

    def _do_pick_sequence(self, pose: PoseStamped):
        """open → cuRobo 하강 → move_stop → safe_grasp → lift"""
        pos, ori = pose.pose.position, pose.pose.orientation
        LIFT_HEIGHT = 0.15  # m
        grasp_z = pos.z
        if pos.z < self.TCP_Z_MIN:
            self.get_logger().warn(
                f"[FLOOR GUARD] pick z={pos.z*1000:.1f}mm → {self.TCP_Z_MIN*1000:.0f}mm 클램프")
            grasp_z = self.TCP_Z_MIN

        self.get_logger().info("=== PICK Step 1/4: 그리퍼 열기 ===")
        self.gripper_open()
        time.sleep(1.5)

        self.get_logger().info(f"=== PICK Step 2/4: 하강 Z={grasp_z*1000:.1f}mm (cuRobo) ===")
        traj = self.plan(self.current_joints, [pos.x, pos.y, grasp_z],
                         [ori.w, ori.x, ori.y, ori.z])
        if traj is None:
            self.get_logger().error("Pick 실패: 경로계획 불가")
            return
        self.execute_spline(traj)
        time.sleep(2.0)

        # 이동 직후 컨트롤러 motion 상태 클리어 → 그래야 flange serial 그리퍼가 닿음
        self.get_logger().info("=== PICK Step 3/4: move_stop → 그리퍼 닫기 ===")
        try:
            if self.cli_stop.wait_for_service(timeout_sec=2.0):
                sr = MoveStop.Request()
                sr.stop_mode = 1  # DR_QSTOP
                f = self.cli_stop.call_async(sr)
                t0 = time.time()
                while not f.done() and time.time() - t0 < 3.0:
                    time.sleep(0.05)
        except Exception as e:
            self.get_logger().warn(f"move_stop 실패(무시): {e}")
        time.sleep(1.0)
        self.gripper_grasp()
        time.sleep(1.5)

        self.get_logger().info("=== PICK Step 4/4: 수직 상승 (자세 유지) ===")
        self.lift_straight_up(LIFT_HEIGHT, vel=50.0)
        time.sleep(1.5)
        self.get_logger().info("=== PICK 완료 ===")

    # ── 고수준 모션 ──────────────────────────────────────────

    def _move_to_named_target(self, name: str) -> bool:
        if name not in self.NAMED_TARGETS_DEG:
            self.get_logger().error(f"알 수 없는 named target: {name}")
            return False
        if self.current_joints is None:
            self.get_logger().error("joint_states 미수신")
            return False
        target_deg = self.NAMED_TARGETS_DEG[name]
        traj = self._plan_js(self.current_joints, target_deg)
        if traj is None:
            return False
        self.get_logger().info(f"MoveJoint → '{name}' {target_deg}")
        return self._execute_movej(target_deg)

    def _move_to_place(self) -> bool:
        """접근 joint → 진입 pose → 수직 하강(선택) → 그리퍼 열기 → 복귀
        cuRobo 검증 없이 직접 실행 (place 좌표는 사전 검증된 하드코딩값)."""
        place_target = self.get_parameter('place_target').get_parameter_value().string_value
        if place_target not in self.PLACE_TARGETS:
            self.get_logger().error(
                f"알 수 없는 place_target: '{place_target}' "
                f"(가능: {list(self.PLACE_TARGETS.keys())})")
            return False
        target = self.PLACE_TARGETS[place_target]
        self.get_logger().info(f"_move_to_place: '{place_target}'")
        time.sleep(10.0)

        if not self._execute_movej(target['point_j']):
            return False
        if not self._execute_movel(*self._posx_to_xyzquat(target['entry_posx'])):
            return False
        if target['grasp_down_posx'] is not None:
            if not self._execute_movel(*self._posx_to_xyzquat(target['grasp_down_posx'])):
                return False
        self.gripper_open(0)
        time.sleep(1.5)
        if not self._execute_movej(target['point_j']):
            return False
        return True

    # ── cuRobo 경로계획 ───────────────────────────────────────

    def _plan_once(self, start_state, target_pos, quat_wxyz):
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
        """cuRobo 경로계획 — j4 양수 우선(바닥 충돌 회피) + yaw 등가 재시도 + fallback"""
        t0 = time.time()
        w, x, y, z = target_quat_wxyz
        approach_z = 1.0 - 2.0 * (x * x + y * y)  # grasp Z 의 base z 성분
        if abs(approach_z) < 0.5:
            # 수평 그랩: j4 강제 끔(spin 방지), yaw 는 도달실패 시만 폴백
            prefer_positive_j4 = False
            allow_yaw_retry = True
            self.get_logger().info(
                f"수평 그랩(az={approach_z:+.2f}) → j4강제 OFF")

        start_state = CuroboJointState.from_position(
            position=torch.tensor([start_joints], device="cuda:0", dtype=torch.float32),
            joint_names=self.JOINT_NAMES,
        )

        # 1차: 원본 target
        base = self._plan_once(start_state, target_pos, target_quat_wxyz)
        if base is not None and (not prefer_positive_j4 or self._traj_j4_ok(base)):
            self.get_logger().info(
                f"계획 OK: {(time.time()-t0)*1000:.1f}ms, {base.shape[0]}pts "
                f"(j4_min={np.degrees(base[:,3].min()):.1f}°)")
            return base

        # yaw 등가 재시도 (j4 양수 해 탐색)
        if allow_yaw_retry and prefer_positive_j4:
            for dyaw in self.YAW_RETRY:
                cand = self._plan_once(start_state, target_pos,
                                       self._quat_yaw(target_quat_wxyz, dyaw))
                if cand is not None and self._traj_j4_ok(cand):
                    self.get_logger().info(
                        f"계획 OK (yaw {np.degrees(dyaw):+.0f}° 등가): "
                        f"{(time.time()-t0)*1000:.1f}ms, "
                        f"j4_min={np.degrees(cand[:,3].min()):.1f}°")
                    return cand

        # fallback: 원래 해 있으면 사용
        if base is not None:
            self.get_logger().warn(
                f"j4 양수 해 없음 — 원래 해 사용 "
                f"(j4_min={np.degrees(base[:,3].min()):.1f}°, 바닥 충돌 주의)")
            return base

        # 1차도 실패 → yaw 등가로 아무 해라도
        if allow_yaw_retry:
            for dyaw in self.YAW_RETRY:
                cand = self._plan_once(start_state, target_pos,
                                       self._quat_yaw(target_quat_wxyz, dyaw))
                if cand is not None:
                    self.get_logger().warn(
                        f"원본 실패 — yaw {np.degrees(dyaw):+.0f}° 해 사용 "
                        f"(j4_min={np.degrees(cand[:,3].min()):.1f}°)")
                    return cand

        # 기울임 폴백 (환경변수로 활성, 기본 OFF)
        if os.environ.get('CUROBO_TILT_FALLBACK', '0') != '0':
            for tilt in (20, 35, 50, 65, 80):
                tq = self._quat_tilt(target_quat_wxyz, tilt)
                cand = self._plan_once(start_state, target_pos, tq)
                if cand is None and allow_yaw_retry:
                    for dyaw in self.YAW_RETRY:
                        cand = self._plan_once(start_state, target_pos,
                                               self._quat_yaw(tq, dyaw))
                        if cand is not None:
                            break
                if cand is not None:
                    self.get_logger().warn(f"기울임 폴백 {tilt}° 성공")
                    return cand

        self.get_logger().error(
            f"계획 실패 (수평+yaw+기울임 모두): {(time.time()-t0)*1000:.1f}ms")
        return None

    def _plan_js(self, start_joints, target_joints_deg):
        """joint space 경로계획 (named target 이동용)"""
        start_state = CuroboJointState.from_position(
            position=torch.tensor([start_joints], device="cuda:0", dtype=torch.float32),
            joint_names=self.JOINT_NAMES,
        )
        goal_state = CuroboJointState.from_position(
            position=torch.deg2rad(
                torch.tensor([target_joints_deg], device="cuda:0", dtype=torch.float32)),
            joint_names=self.JOINT_NAMES,
        )
        result = self.motion_gen.plan_single_js(
            start_state, goal_state,
            MotionGenPlanConfig(max_attempts=60, enable_graph=False, enable_opt=True,
                                use_start_state_as_retract=True),
        )
        if result.success.item():
            positions = result.get_interpolated_plan().position.cpu().numpy()
            self.get_logger().info(f"JS 계획 OK: {positions.shape[0]}pts")
            return positions
        self.get_logger().error("JS 경로계획 실패")
        return None

    def _traj_j4_ok(self, positions):
        return float(np.min(positions[:, 3])) >= self.J4_MIN_RAD

    def _quat_yaw(self, quat_wxyz, dyaw):
        """tool Z축 기준 dyaw 회전 (대칭 그리퍼 등가 파지)"""
        Rn = self._quat_to_rotm(quat_wxyz) @ np.array([
            [np.cos(dyaw), -np.sin(dyaw), 0],
            [np.sin(dyaw),  np.cos(dyaw), 0],
            [0,             0,            1]])
        return self._rotm_to_quat_wxyz(Rn)

    def _quat_tilt(self, quat_wxyz, deg):
        """접근축(grasp Z)을 world -Z 방향으로 deg 기울임"""
        R = self._quat_to_rotm(quat_wxyz)
        a = R[:, 2]
        down = np.array([0.0, 0.0, -1.0])
        axis = np.cross(a, down)
        na = np.linalg.norm(axis)
        if na < 1e-6:
            return list(quat_wxyz)
        axis /= na
        th = np.radians(deg)
        a2 = (a * np.cos(th) + np.cross(axis, a) * np.sin(th)
              + axis * np.dot(axis, a) * (1 - np.cos(th)))
        a2 /= np.linalg.norm(a2)
        up = np.array([0.0, 0.0, 1.0])
        x_ax = np.cross(up, a2) if abs(a2[2]) <= 0.97 else np.array([1.0, 0.0, 0.0])
        x_ax /= np.linalg.norm(x_ax)
        y_ax = np.cross(a2, x_ax)
        y_ax /= np.linalg.norm(y_ax)
        return self._rotm_to_quat_wxyz(np.column_stack([x_ax, y_ax, a2]))

    def _quat_to_rotm(self, quat_wxyz):
        w, x, y, z = quat_wxyz
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

    def _rotm_to_quat_wxyz(self, Rn):
        t = Rn[0, 0] + Rn[1, 1] + Rn[2, 2]
        if t > 0:
            S = np.sqrt(t + 1.0) * 2
            q = [0.25*S, (Rn[2,1]-Rn[1,2])/S, (Rn[0,2]-Rn[2,0])/S, (Rn[1,0]-Rn[0,1])/S]
        elif Rn[0,0] > Rn[1,1] and Rn[0,0] > Rn[2,2]:
            S = np.sqrt(1 + Rn[0,0] - Rn[1,1] - Rn[2,2]) * 2
            q = [(Rn[2,1]-Rn[1,2])/S, 0.25*S, (Rn[0,1]+Rn[1,0])/S, (Rn[0,2]+Rn[2,0])/S]
        elif Rn[1,1] > Rn[2,2]:
            S = np.sqrt(1 + Rn[1,1] - Rn[0,0] - Rn[2,2]) * 2
            q = [(Rn[0,2]-Rn[2,0])/S, (Rn[0,1]+Rn[1,0])/S, 0.25*S, (Rn[1,2]+Rn[2,1])/S]
        else:
            S = np.sqrt(1 + Rn[2,2] - Rn[0,0] - Rn[1,1]) * 2
            q = [(Rn[1,0]-Rn[0,1])/S, (Rn[0,2]+Rn[2,0])/S, (Rn[1,2]+Rn[2,1])/S, 0.25*S]
        n = float(np.linalg.norm(q))
        return [v / n for v in q]

    # ── Doosan 실행 ───────────────────────────────────────────

    def execute_spline(self, traj_rad):
        if not self.cli_spline.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveSplineJoint 서비스 없음")
            return
        traj_deg = np.rad2deg(traj_rad)
        n = traj_deg.shape[0]
        if n > 100:
            traj_deg = traj_deg[np.linspace(0, n - 1, 100, dtype=int)]
            n = 100

        req = MoveSplineJoint.Request()
        req.pos_cnt = n
        for row in traj_deg:
            pt = Float64MultiArray()
            pt.data = row.tolist()
            req.pos.append(pt)
        req.vel = [30.0 * self.VEL_SCALE] * 6
        req.acc = [60.0] * 6
        req.time = 0.0; req.mode = 0; req.sync_type = 0

        self.get_logger().info(
            f"Spline 실행 ({n}pts) "
            f"start={[f'{v:.1f}' for v in traj_deg[0]]} "
            f"end={[f'{v:.1f}' for v in traj_deg[-1]]}")
        future = self.cli_spline.call_async(req)
        t0 = time.time()
        while not future.done() and (time.time() - t0) < 30.0:
            time.sleep(0.05)
        ok = future.done() and future.result() and future.result().success
        self.get_logger().info("Spline " + ("완료" if ok else "실패"))

    def _execute_movej(self, joints_deg, vel: float = 30.0, acc: float = 30.0) -> bool:
        if not self.cli_movej.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveJoint 서비스 없음")
            return False
        req = MoveJoint.Request()
        req.pos = joints_deg; req.vel = vel; req.acc = acc
        req.time = 0.0; req.radius = 0.0; req.mode = 0
        req.blend_type = 0; req.sync_type = 0
        self.get_logger().info(f"MoveJoint → {joints_deg}")
        return self._wait_for_motion(self.cli_movej.call_async(req), "MoveJoint")

    def _execute_movel(self, x, y, z, qx, qy, qz, qw, vel: float = 100.0) -> bool:
        """목표 포즈(m, 쿼터니언) → ZYZ 오일러각 변환 후 MoveLine 실행"""
        if not self.cli_movel.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveLine 서비스 없음")
            return False
        a, b, c = Rotation.from_quat([qx, qy, qz, qw]).as_euler('ZYZ', degrees=True)
        req = MoveLine.Request()
        req.pos = [x * 1000.0, y * 1000.0, z * 1000.0, a, b, c]
        req.vel = [vel, 30.0]; req.acc = [vel, 30.0]
        req.time = 0.0; req.ref = 0; req.mode = 0
        req.blend_type = 0; req.sync_type = 0
        self.get_logger().info(
            f"MoveLine → ({x*1000:.1f},{y*1000:.1f},{z*1000:.1f})mm "
            f"rot=({a:.1f},{b:.1f},{c:.1f})°")
        return self._wait_for_motion(self.cli_movel.call_async(req), "MoveLine")

    def _wait_for_motion(self, future, label: str, timeout: float = 30.0) -> bool:
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            time.sleep(0.1)
        if future.done() and future.result() and future.result().success:
            self.get_logger().info(f"{label} 완료")
            return True
        self.get_logger().error(
            f"{label} {'타임아웃' if not future.done() else '실패'}")
        return False

    def lift_straight_up(self, dz_m, vel=50.0):
        """현재 TCP 자세(rx/ry/rz) 유지 + Z만 dz_m(m) 상승 — 물체가 회전 없이 올라감"""
        if not self.cli_posx.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("GetCurrentPosx 없음 — top-down fallback")
            return self.move_linear(0, 0, dz_m, vel=vel)
        fut = self.cli_posx.call_async(GetCurrentPosx.Request())
        t0 = time.time()
        while not fut.done() and (time.time() - t0) < 3.0:
            time.sleep(0.05)
        if not (fut.done() and fut.result()):
            self.get_logger().error("GetCurrentPosx 실패 — lift 취소")
            return
        cur = list(fut.result().task_pos_info[0].data)  # [x,y,z,rx,ry,rz] mm/deg
        req = MoveLine.Request()
        req.pos = [cur[0], cur[1], cur[2] + dz_m * 1000.0, cur[3], cur[4], cur[5]]
        _v = vel * self.VEL_SCALE
        req.vel = [_v, 30.0 * self.VEL_SCALE]
        req.acc = [_v, 30.0 * self.VEL_SCALE]
        req.time = 0.0; req.ref = 0; req.mode = 0
        req.blend_type = 0; req.sync_type = 1
        f2 = self.cli_movel.call_async(req)
        s2 = time.time()
        while not f2.done() and (time.time() - s2) < 15.0:
            time.sleep(0.1)
        ok = f2.done() and f2.result() and f2.result().success
        self.get_logger().info(
            f"Lift {'완료' if ok else '실패'}: "
            f"Z {cur[2]:.0f}→{cur[2]+dz_m*1000:.0f}mm "
            f"(rx={cur[3]:.0f} ry={cur[4]:.0f} rz={cur[5]:.0f}° 유지)")

    def move_linear(self, x, y, z, vel=100.0):
        """TCP를 (x,y,z)[m] top-down 자세로 직선 이동 (lift fallback용)"""
        if not self.cli_movel.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveLine 서비스 없음")
            return
        req = MoveLine.Request()
        req.pos = [x * 1000, y * 1000, z * 1000, 0.0, 180.0, 0.0]
        req.vel = [vel * self.VEL_SCALE, 30.0 * self.VEL_SCALE]
        req.acc = [vel * self.VEL_SCALE, 30.0 * self.VEL_SCALE]
        req.time = 0.0; req.ref = 0; req.mode = 0
        req.blend_type = 0; req.sync_type = 1
        future = self.cli_movel.call_async(req)
        t0 = time.time()
        while not future.done() and (time.time() - t0) < 15.0:
            time.sleep(0.1)

    # ── 그리퍼 ───────────────────────────────────────────────

    def gripper_open(self, position=None, timeout_sec=3.0):
        if position is None:
            position = int(self.get_parameter('grasp_open_position').value)
        if not self.cli_gripper_open.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("set_position 서비스 없음")
            return False
        req = SetPosition.Request()
        req.position = int(position)
        req.timeout_sec = float(timeout_sec)
        future = self.cli_gripper_open.call_async(req)
        t0 = time.time()
        while not future.done() and (time.time() - t0) < timeout_sec + 1.0:
            time.sleep(0.05)
        res = future.result() if future.done() else None
        ok = bool(res and res.success)
        self.get_logger().info(
            f"gripper_open(set_position {position}): {'성공' if ok else '실패/타임아웃'}")
        return ok

    def gripper_grasp(self):
        """물성별 safe_grasp 액션 (grasp_force_params.yaml → fallback 파라미터)"""
        cls = self.grasp_class
        gf = self.grasp_force_params.get(cls) if cls else None
        if gf:
            tp = int(gf.get('goal_position', 700))
            mc = int(gf.get('max_current', 600))
            cd = int(gf.get('current_delta_threshold', 30))
            self.get_logger().info(
                f"물성별 파지: class={cls} max_current={mc}mA delta={cd}")
        else:
            tp = int(self.get_parameter('grasp_target_position').value)
            mc = int(self.get_parameter('grasp_max_current').value)
            cd = int(self.get_parameter('grasp_current_delta').value)
            if cls:
                self.get_logger().warn(
                    f"class '{cls}' 파지힘 정의 없음 → 기본 {mc}mA "
                    f"(정의: {list(self.grasp_force_params.keys())})")

        if not self.act_safe_grasp.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("safe_grasp 액션 서버 없음")
            return False

        goal = SafeGrasp.Goal()
        goal.target_position = tp
        goal.max_current = mc
        goal.current_delta_threshold = cd
        goal.timeout_sec = 5.0
        self.get_logger().info(
            f"safe_grasp goal: pos={tp} max_current={mc}mA delta={cd}")

        send_future = self.act_safe_grasp.send_goal_async(goal)
        t0 = time.time()
        while not send_future.done() and (time.time() - t0) < 4.0:
            time.sleep(0.05)
        gh = send_future.result() if send_future.done() else None
        if gh is None or not gh.accepted:
            self.get_logger().warn("safe_grasp 목표 거부/타임아웃")
            return False

        res_future = gh.get_result_async()
        t0 = time.time()
        while not res_future.done() and (time.time() - t0) < 8.0:
            time.sleep(0.05)
        wrapped = res_future.result() if res_future.done() else None
        res = wrapped.result if wrapped else None
        if res:
            self.get_logger().info(
                f"safe_grasp 결과: success={res.success} "
                f"grasp_detected={res.grasp_detected} "
                f"final_current={res.final_current}mA")
        else:
            self.get_logger().warn("safe_grasp 결과 타임아웃")
        return bool(res and res.success)

    # ── 유틸리티 ─────────────────────────────────────────────

    @staticmethod
    def _posx_to_xyzquat(posx):
        """Doosan posx[mm, ZYZ deg] → (x,y,z,qx,qy,qz,qw)[m]"""
        x, y, z, a, b, c = posx
        qx, qy, qz, qw = Rotation.from_euler('ZYZ', [a, b, c], degrees=True).as_quat()
        return x / 1000.0, y / 1000.0, z / 1000.0, qx, qy, qz, qw

    @staticmethod
    def _posx_to_pose_stamped(posx, frame_id='base_link') -> PoseStamped:
        x, y, z, qx, qy, qz, qw = ArmControllerNode._posx_to_xyzquat(posx)
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x; pose.pose.position.y = y; pose.pose.position.z = z
        pose.pose.orientation.x = qx; pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz; pose.pose.orientation.w = qw
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = ArmControllerNode()
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


if __name__ == '__main__':
    main()
