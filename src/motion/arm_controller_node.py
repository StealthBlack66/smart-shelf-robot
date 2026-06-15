#!/usr/bin/env python3
import os
import sys
import time

# CUDA_VISIBLE_DEVICES=""(빈 문자열)이면 GPU가 숨겨지므로 임포트 전에 해제
if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
    del os.environ["CUDA_VISIBLE_DEVICES"]

import yaml
import torch
from scipy.spatial.transform import Rotation

if not torch.cuda.is_available():
    print("[ERROR] CUDA를 사용할 수 없습니다. cuRobo는 GPU가 필요합니다.")
    print("  확인 사항:")
    print("  1. nvidia-smi 명령 실행 후 GPU가 보이는지 확인")
    print("  2. sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm 실행 후 재시도")
    print("  3. 위가 안 되면 시스템 재부팅")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState

from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState as CuroboJointState, RobotConfig
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig, Cuboid
from dsr_msgs2.srv import MoveJoint, MoveLine
from dsr_gripper_tcp_interfaces.srv import SetPosition
from dsr_gripper_tcp_interfaces.action import SafeGrasp


class ArmControllerNode(Node):
    JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    # 명세서 기준 named joint targets (degrees) — 실제 값으로 조정 필요
    HOME_JOINTS_DEG         = [409.280, 198.1, 359.840, 86.22, 103.79, 83.7]
    SHELF_VIEW_JOINTS_DEG   = [-6.73, 8.12, 104.62, 80.22, 93.13, -23.49]   # Global_shelf_view(수정 필요)
    PRODUCT_VIEW_JOINTS_DEG = [-1.99, -4.79, 96.32, 5.09, 84.03, -2.56]     # Global_product_view(수정 필요)

    NAMED_TARGETS_DEG = {
        'home':         HOME_JOINTS_DEG,
        'shelf_view':   SHELF_VIEW_JOINTS_DEG,
        'product_view': PRODUCT_VIEW_JOINTS_DEG,
    }

    # Place 시퀀스 좌표 (slot0 → bottle 놓기)
    SLOT0_POINT_J_DEG = [27.67, 5.82, 95.72, 90.93, 61.98, -10.64]       # posj (deg)
    SLOT0_L_POSX      = [357.21, 511.97, 487.17, 89.98, 94.59, 90.0]     # posx (mm, ZYZ deg)
    SLOT0_DOWN_L_POSX = [357.230, 498.590, 468.640, 89.98, 94.59, 90.0]  # posx (mm, ZYZ deg) — 수직 하강

    # Place 시퀀스 좌표 (slot1 → snack_bag 놓기, 수직 하강 없음)
    SLOT1_POINT_J_DEG = [21.17, 23.36, 72.79, 87.48, 68.49, -4.8]    # posj (deg)
    SLOT1_L_POSX      = [483.77, 449.08, 493.21, 89.99, 94.59, 90.01]  # posx (mm, ZYZ deg)

    # Place 시퀀스 좌표 (slot2 → can 놓기)
    SLOT2_POINT_J_DEG = [22.17, 23.13, 103.31, 99.79, 69.5, -37.32]   # posj (deg)
    SLOT2_L_POSX      = [413.61, 511.97, 310.56, 90.00, 90.58, 90.0]   # posx (mm, ZYZ deg) — z +30mm
    SLOT2_DOWN_L_POSX = [413.61, 511.97, 290.15, 103.00, 90.58, 90.0]   # posx (mm, ZYZ deg) — 수직 하강, z +30mm

    # /move_to_place 시퀀스에서 'place_target' 파라미터로 선택할 슬롯 정보
    PLACE_TARGETS = {
        'bottle': {
            'point_j':        SLOT0_POINT_J_DEG,
            'entry_posx':     SLOT0_L_POSX,
            'grasp_down_posx': SLOT0_DOWN_L_POSX,
        },
        'snack_bag': {
            'point_j':        SLOT1_POINT_J_DEG,
            'entry_posx':     SLOT1_L_POSX,
            'grasp_down_posx': None,  # 수직 하강 없음
        },
        'can': {
            'point_j':        SLOT2_POINT_J_DEG,
            'entry_posx':     SLOT2_L_POSX,
            'grasp_down_posx': SLOT2_DOWN_L_POSX,
        },
    }

    def __init__(self):
        super().__init__('arm_controller_node')

        # /move_to_place에서 놓을 물체 종류: 'can' | 'bottle' | 'snack_bag'
        self.declare_parameter('place_target', 'can')

        self.service_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()

        # Subscribers
        self.sub_object_pose = self.create_subscription(
            PoseStamped, '/object_pose', self.object_pose_callback, 10)
        self.sub_joint_state = self.create_subscription(
            JointState, '/dsr01/joint_states', self._joint_state_cb, 10)

        # Publishers
        self.pub_joint_command = self.create_publisher(JointState, '/joint_command', 10)

        # Services (명세서 "motion" 역할)
        self.srv_move_to_shelf_view = self.create_service(
            Trigger, '/move_to_shelf_view', self.move_to_shelf_view_callback,
            callback_group=self.service_cb_group)
        self.srv_move_to_product_view = self.create_service(
            Trigger, '/move_to_product_view', self.move_to_product_view_callback,
            callback_group=self.service_cb_group)
        self.srv_move_to_place = self.create_service(
            Trigger, '/move_to_place', self.move_to_place_callback,
            callback_group=self.service_cb_group)
        self.srv_move_to_home = self.create_service(
            Trigger, '/move_to_home', self.move_to_home_callback,
            callback_group=self.service_cb_group)

        # NOTE: 명세서상 /move_to_pick의 서버는 "simulation"(GraspGen+cuRobo)이지만,
        # 시뮬 파이프라인 구현 전이라 비전이 발행하는 /object_pose로 임시 테스트한다.
        self.srv_move_to_pick = self.create_service(
            Trigger, '/move_to_pick', self.move_to_pick_callback,
            callback_group=self.service_cb_group)

        # Doosan motion clients
        self.cli_movej = self.create_client(
            MoveJoint, '/dsr01/motion/move_joint',
            callback_group=self.service_cb_group)
        self.cli_movel = self.create_client(
            MoveLine, '/dsr01/motion/move_line',
            callback_group=self.service_cb_group)
        self.cli_gripper_set_position = self.create_client(
            SetPosition, '/gripper_service/set_position',
            callback_group=self.service_cb_group)
        self.cli_safe_grasp = ActionClient(
            self, SafeGrasp, '/gripper_service/safe_grasp',
            callback_group=self.service_cb_group)

        self.object_pose = None
        self.current_joints = None
        self.motion_gen = None
        self.tensor_args = None
        self.grasp_force_params = self._load_grasp_force_params()

        self._init_curobo()

    # ──────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────

    def _load_grasp_force_params(self) -> dict:
        """config/grasp_force_params.yaml에서 물체별(snack_bag/bottle/can) 파지힘 설정을 로드."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "grasp_force_params.yaml"
        )
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['grasp_force']

    def _init_curobo(self):
        self.get_logger().info("Initializing cuRobo...")

        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "curobo"
        )
        if not os.path.exists(config_dir):
            from ament_index_python.packages import get_package_share_directory
            config_dir = os.path.join(
                get_package_share_directory("e0509_gripper_description"),
                "config", "curobo"
            )

        self.tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

        robot_cfg = RobotConfig.from_basic(
            urdf_path=os.path.join(config_dir, "e0509_gripper.urdf"),
            base_link="base_link",
            ee_link="gripper_rh_p12_rn_base",
            tensor_args=self.tensor_args,
        )

        world_cfg = WorldConfig(
            cuboid=[
                Cuboid(name="table", pose=[0.0, 0.0, -0.02, 1, 0, 0, 0], dims=[1.2, 1.2, 0.04]),
            ]
        )

        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            self.tensor_args,
            num_trajopt_seeds=4,
            num_graph_seeds=4,
            collision_cache={"obb": 30, "mesh": 10},
        )
        self.motion_gen = MotionGen(motion_gen_cfg)
        self.motion_gen.warmup(warmup_js_trajopt=False)
        self.get_logger().info("cuRobo ready!")

    # ──────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        joint_map = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                joint_map[name] = msg.position[i]

        joints = []
        for name in self.JOINT_NAMES:
            if name not in joint_map:
                return
            joints.append(joint_map[name])

        self.current_joints = joints

    def object_pose_callback(self, msg: PoseStamped):
        self.object_pose = msg

    # ──────────────────────────────────────────────
    # Service handlers
    # ──────────────────────────────────────────────

    def move_to_shelf_view_callback(self, request, response):
        success = self._move_to_named_target('shelf_view')
        response.success = success
        response.message = "Shelf view position reached" if success else "Move to shelf view failed"
        return response

    def move_to_product_view_callback(self, request, response):
        success = self._move_to_named_target('product_view')
        response.success = success
        response.message = "Product view position reached" if success else "Move to product view failed"
        return response

    def move_to_pick_callback(self, request, response):
        # NOTE: 명세서상 본 서비스는 "simulation"(GraspGen+cuRobo) 담당이지만,
        # 시뮬 파이프라인 구현 전이므로 비전이 발행하는 /object_pose로 임시 테스트한다.
        if self.object_pose is None:
            response.success = False
            response.message = "No /object_pose received yet"
            return response
        if self.current_joints is None:
            response.success = False
            response.message = "Joint state not received"
            return response

        if not self._move_to_pose(self.object_pose):
            response.success = False
            response.message = "cuRobo planning failed"
            return response

        grasp_success = self.call_safe_grasp()
        response.success = grasp_success
        response.message = "Pick complete (grasped)" if grasp_success else "Pick position reached but grasp failed"
        return response

    def move_to_place_callback(self, request, response):
        place_target = self.get_parameter('place_target').get_parameter_value().string_value
        success = self._move_to_place()
        response.success = success
        response.message = (
            f"Place sequence complete ({place_target})" if success
            else f"Place sequence failed ({place_target})")
        return response

    def move_to_home_callback(self, request, response):
        success = self._move_to_named_target('home')
        response.success = success
        response.message = "Home reached" if success else "Move to home failed"
        return response

    # ──────────────────────────────────────────────
    # Motion primitives
    # ──────────────────────────────────────────────

    def _move_to_pose(self, pose: PoseStamped) -> bool:
        """cuRobo로 충돌없는 경로인지 검증한 뒤, MoveLine으로 목표 포즈까지 직선 이동."""
        if self.current_joints is None:
            self.get_logger().error("No joint state received")
            return False

        p = pose.pose.position
        q = pose.pose.orientation

        # 1) cuRobo로 충돌없이 도달 가능한 경로인지 사전 검증
        traj = self._plan(self.current_joints, [p.x, p.y, p.z], [q.w, q.x, q.y, q.z])
        if traj is None:
            return False

        # 2) 실제 이동은 MoveLine으로 직선 실행
        return self._execute_movel(p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    def _move_to_pose_direct(self, pose: PoseStamped, vel: float = 100.0) -> bool:
        """cuRobo 충돌 검증 없이, MoveLine으로 목표 포즈까지 바로 직선 이동."""
        p = pose.pose.position
        q = pose.pose.orientation
        return self._execute_movel(p.x, p.y, p.z, q.x, q.y, q.z, q.w, vel)

    def _move_to_named_target(self, name: str) -> bool:
        """명세서에 정의된 named position(joint space)으로, cuRobo 충돌 검증 후 이동."""
        if name not in self.NAMED_TARGETS_DEG:
            self.get_logger().error(f"Unknown named target: {name}")
            return False

        if self.current_joints is None:
            self.get_logger().error("No joint state received")
            return False

        target_deg = self.NAMED_TARGETS_DEG[name]

        # cuRobo로 충돌없이 도달 가능한 경로인지 사전 검증
        traj = self._plan_js(self.current_joints, target_deg)
        if traj is None:
            return False

        self.get_logger().info(f"MoveJoint → '{name}' {target_deg}")
        return self._execute_movej(target_deg)

    def _move_to_place(self) -> bool:
        """Place 시퀀스: 접근 joint pos → 진입 pose → 수직 하강 → 그리퍼 열기 → 접근 joint pos로 복귀.
        'place_target' 파라미터(can/bottle/snack_bag)로 놓을 슬롯을 선택한다.
        cuRobo 검증 없이 직접 실행 (movej: _execute_movej, movel: _move_to_pose_direct)."""
        place_target = self.get_parameter('place_target').get_parameter_value().string_value
        if place_target not in self.PLACE_TARGETS:
            self.get_logger().error(
                f"Unknown place_target: '{place_target}' "
                f"(expected one of {list(self.PLACE_TARGETS.keys())})")
            return False
        target = self.PLACE_TARGETS[place_target]

        self.get_logger().info(f"_move_to_place: place_target='{place_target}'")
        time.sleep(10.0) # 시퀀스 시작 전 잠시 대기 (필요시 제거) — cuRobo로 시뮬 검증할 때는 이 부분 제거하고 충분히 멀리서 시작하도록 조정할 것
        # 1) movej: 접근 joint position
        if not self._execute_movej(target['point_j']):
            return False

        # 2) movel: 슬롯 진입 pose
        if not self._move_to_pose_direct(self._posx_to_pose_stamped(target['entry_posx'])):
            return False

        # 3) movel: 수직 하강 (해당 슬롯에 하강 포인트가 없으면 스킵)
        if target['grasp_down_posx'] is not None:
            if not self._move_to_pose_direct(self._posx_to_pose_stamped(target['grasp_down_posx'])):
                return False

        # 4) 그리퍼 열기 (물체 놓기)
        self.call_set_position(0)
        time.sleep(1.5)

        # 5) movej: 접근 joint position으로 복귀
        if not self._execute_movej(target['point_j']):
            return False

        return True

    @staticmethod
    def _posx_to_pose_stamped(posx, frame_id: str = "base_link") -> PoseStamped:
        """Doosan posx(x,y,z,a,b,c) [mm, ZYZ deg] → PoseStamped (m, quaternion)."""
        x, y, z, a, b, c = posx
        qx, qy, qz, qw = Rotation.from_euler('ZYZ', [a, b, c], degrees=True).as_quat()

        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x / 1000.0
        pose.pose.position.y = y / 1000.0
        pose.pose.position.z = z / 1000.0
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    # ──────────────────────────────────────────────
    # cuRobo planning
    # ──────────────────────────────────────────────

    def _plan(self, start_joints, target_pos, target_quat_wxyz):
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
                max_attempts=60,
                enable_graph=False,
                enable_opt=True,
                use_start_state_as_retract=True,
            )
        )
        if result.success.item():
            traj = result.get_interpolated_plan()
            positions = traj.position.cpu().numpy()
            self.get_logger().info(f"Planning OK: {positions.shape[0]} waypoints")
            return positions
        else:
            self.get_logger().error("cuRobo planning failed")
            return None

    def _plan_js(self, start_joints, target_joints_deg):
        start_state = CuroboJointState.from_position(
            position=torch.tensor([start_joints], device="cuda:0", dtype=torch.float32),
            joint_names=self.JOINT_NAMES,
        )
        goal_state = CuroboJointState.from_position(
            position=torch.deg2rad(torch.tensor([target_joints_deg], device="cuda:0", dtype=torch.float32)),
            joint_names=self.JOINT_NAMES,
        )
        result = self.motion_gen.plan_single_js(
            start_state,
            goal_state,
            MotionGenPlanConfig(
                max_attempts=60,
                enable_graph=False,
                enable_opt=True,
                use_start_state_as_retract=True,
            )
        )
        if result.success.item():
            traj = result.get_interpolated_plan()
            positions = traj.position.cpu().numpy()
            self.get_logger().info(f"Planning OK: {positions.shape[0]} waypoints")
            return positions
        else:
            self.get_logger().error("cuRobo planning failed")
            return None

    # ──────────────────────────────────────────────
    # Execution
    # ──────────────────────────────────────────────

    def _execute_movel(self, x, y, z, qx, qy, qz, qw, vel: float = 100.0) -> bool:
        """목표 포즈(m, 쿼터니언)까지 MoveLine으로 직선 이동. Doosan은 ZYZ 오일러각(deg) 사용."""
        if not self.cli_movel.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveLine service not available")
            return False

        a, b, c = Rotation.from_quat([qx, qy, qz, qw]).as_euler('ZYZ', degrees=True)

        req = MoveLine.Request()
        req.pos = [x * 1000.0, y * 1000.0, z * 1000.0, a, b, c]  # mm, deg (ZYZ)
        req.vel = [vel, 30.0]
        req.acc = [vel, 30.0]
        req.time = 0.0
        req.ref = 0
        req.mode = 0       # ABSOLUTE
        req.blend_type = 0
        req.sync_type = 0  # SYNC

        self.get_logger().info(
            f"MoveLine → pos=({x*1000:.1f},{y*1000:.1f},{z*1000:.1f})mm "
            f"rot=({a:.1f},{b:.1f},{c:.1f})deg")
        future = self.cli_movel.call_async(req)
        return self._wait_for_motion(future, "MoveLine")

    def _execute_movej(self, joints_deg, vel: float = 30.0, acc: float = 30.0) -> bool:
        """목표 joint 각도(deg)까지 MoveJoint로 이동."""
        if not self.cli_movej.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveJoint service not available")
            return False

        req = MoveJoint.Request()
        req.pos = joints_deg
        req.vel = vel
        req.acc = acc
        req.time = 0.0
        req.radius = 0.0
        req.mode = 0       # ABSOLUTE
        req.blend_type = 0
        req.sync_type = 0  # SYNC

        self.get_logger().info(f"MoveJoint → {joints_deg}")
        future = self.cli_movej.call_async(req)
        return self._wait_for_motion(future, "MoveJoint")

    def _wait_for_motion(self, future, label: str, timeout: float = 30.0) -> bool:
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            time.sleep(0.1)

        if future.done() and future.result() and future.result().success:
            self.get_logger().info(f"{label} complete")
            return True
        elif not future.done():
            self.get_logger().error(f"{label} timed out")
        else:
            self.get_logger().error(f"{label} failed")
        return False

    def call_set_position(self, position: int, timeout_sec: float = 5.0):
        """Call /gripper_service/set_position (dsr_gripper_tcp)."""
        if not self.cli_gripper_set_position.service_is_ready():
            self.get_logger().warn("Gripper set_position service not ready — skipping")
            return

        req = SetPosition.Request()
        req.position = position
        req.timeout_sec = timeout_sec
        future = self.cli_gripper_set_position.call_async(req)
        start = time.time()
        while not future.done() and (time.time() - start) < (timeout_sec + 1.0):
            time.sleep(0.05)

        if future.done() and future.result():
            self.get_logger().info(f"set_position({position}): {future.result().message}")
        else:
            self.get_logger().warn("Gripper set_position timed out — continuing")

    def call_safe_grasp(self, object_type: str = None) -> bool:
        """grasp_force_params.yaml의 물체별 설정으로 /gripper_service/safe_grasp 액션 호출.
        object_type 미지정 시 'place_target' 파라미터 값(can/bottle/snack_bag)을 사용한다.
        timeout_sec은 8.0으로 고정."""
        if object_type is None:
            object_type = self.get_parameter('place_target').get_parameter_value().string_value

        params = self.grasp_force_params.get(object_type)
        if params is None:
            self.get_logger().error(
                f"Unknown object type for grasp_force: '{object_type}' "
                f"(expected one of {list(self.grasp_force_params.keys())})")
            return False

        if not self.cli_safe_grasp.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("SafeGrasp action server not available")
            return False

        goal_msg = SafeGrasp.Goal()
        goal_msg.target_position = int(params['goal_position'])
        goal_msg.max_current = int(params['max_current'])
        goal_msg.current_delta_threshold = int(params['current_delta_threshold'])
        goal_msg.timeout_sec = 8.0  # 고정

        self.get_logger().info(
            f"SafeGrasp → object='{object_type}' "
            f"target_position={goal_msg.target_position} "
            f"max_current={goal_msg.max_current} "
            f"current_delta_threshold={goal_msg.current_delta_threshold} "
            f"timeout_sec={goal_msg.timeout_sec}")

        send_goal_future = self.cli_safe_grasp.send_goal_async(goal_msg)
        start = time.time()
        while not send_goal_future.done() and (time.time() - start) < 5.0:
            time.sleep(0.05)

        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("SafeGrasp goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        start = time.time()
        timeout = goal_msg.timeout_sec + 5.0
        while not result_future.done() and (time.time() - start) < timeout:
            time.sleep(0.05)

        if not result_future.done():
            self.get_logger().error("SafeGrasp timed out")
            return False

        result = result_future.result().result
        self.get_logger().info(
            f"SafeGrasp result: success={result.success} message='{result.message}'")
        return result.success


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
