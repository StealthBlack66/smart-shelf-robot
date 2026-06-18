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
from geometry_msgs.msg import PoseStamped, PoseArray
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

    # Named targets / place targets — config/place_targets.yaml 에서 로드 (_load_place_targets)

    # 안전/속도 상수
    TCP_Z_MIN  = 0.02   # m, base 기준 TCP 최저 허용 Z (바닥 충돌 방지)
    VEL_SCALE  = float(os.environ.get('CUROBO_VEL_SCALE', '0.5'))
    J4_MIN_RAD = 0.0    # joint_4 하한 (rad) — 이 아래면 손목이 바닥 방향
    YAW_RETRY  = (np.pi, np.pi / 2, -np.pi / 2)

    def __init__(self):
        super().__init__('arm_controller_node')

        self.declare_parameter('place_target', 'can')
        self.declare_parameter('grasp_target_position', 700)
        self.declare_parameter('grasp_max_current',     400)
        self.declare_parameter('grasp_current_delta',   300)
        self.declare_parameter('grasp_open_position',   0)

        self.service_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()

        # 상태
        self.current_joints = None

        self.grasp_class    = None
        # [goalset 실험] webcam_seg가 보낸 GraspGen 후보 EE pose 묶음 + 수신시각
        self.grasp_candidates = []   # [(pos[3], quat_xyzw[4]), ...]
        self._cand_t = 0.0
        # plan_grasp 최종진입 시 손가락 충돌 면제할 그리퍼 링크
        # 파지 최종 진입 시 충돌면제할 그리퍼 링크 (핑거팁 r2/l2 포함 — 물체 접촉 허용)
        self.gripper_coll_links = [
            'gripper_rh_p12_rn_base', 'gripper_rh_p12_rn_r1', 'gripper_rh_p12_rn_l1',
            'gripper_rh_p12_rn_r2', 'gripper_rh_p12_rn_l2', 'attached_object']
        self.PREGRASP_STANDOFF = float(os.environ.get('CUROBO_PREGRASP_STANDOFF', '0.06'))

        # cuRobo 초기화 (GPU 1회만)
        self.get_logger().info("cuRobo 초기화 중...")
        config_dir = self._find_config_dir()
        self.tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

        # yml(from_dict) 로드 → 충돌구체(sim44, Isaac Sim 메시핏) + attached_object 적재.
        # → plan_grasp(disable_collision_links) goalset 사용 가능 + 자기충돌·장애물 회피.
        # ※ stage7 검증 설정 그대로: lock r1만, base_link(책상마운트) 충돌검사 제외.
        #   base_link 안 빼면 정상자세에서도 base 구체가 팔과 헛충돌 → 모든 plan 실패(실증).
        _cfg = yaml.safe_load(open(os.path.join(config_dir, "e0509_gripper.yml"), encoding='utf-8'))
        _kin = _cfg["robot_cfg"]["kinematics"]
        # ★관절한계 URDF (시뮬팀 리스크① — 손목 플립 방지). joint_4 ±180° / joint_5 0~135°.
        #   cuRobo BoundCost는 URDF를 init에서 clone → 런타임 텐서 수정 무효, 반드시 URDF로 박아야 함.
        #   안 박으면 후보가 멀쩡해도 실기 팔이 플립(joint_5 음수=손목 뒤집힘).
        _src_urdf = os.path.join(config_dir, _kin["urdf_path"])
        _kin["urdf_path"] = self._make_jlim_urdf(
            _src_urdf, "/tmp/e0509_gripper_jlim.urdf",
            {"joint_4": (-3.141592653589793, 3.141592653589793),   # ±180°
             "joint_5": (0.0, 2.356194490192345)})                 # 0~135°
        _kin["asset_root_path"] = config_dir
        _sph = yaml.safe_load(open(os.path.join(config_dir, _kin["collision_spheres"]), encoding='utf-8'))
        _kin["collision_spheres"] = _sph["collision_spheres"]
        _kin["lock_joints"] = {"gripper_rh_r1": 0.0}   # mimic(r2/l1/l2)는 따라옴
        _kin["collision_link_names"] = [l for l in _kin["collision_link_names"] if l != "base_link"]
        if isinstance(_kin.get("self_collision_ignore"), dict):
            _kin["self_collision_ignore"].pop("base_link", None)
            for _k in _kin["self_collision_ignore"]:
                _kin["self_collision_ignore"][_k] = [x for x in _kin["self_collision_ignore"][_k] if x != "base_link"]
        if isinstance(_kin.get("self_collision_buffer"), dict):
            _kin["self_collision_buffer"].pop("base_link", None)
        # 핑거팁(r2/l2 distal 핑거)도 충돌검사 포함 — sim44 에 구체 있음, mimic 이지만 FK OK.
        #   안 넣으면 핑거 끝마디가 충돌검사 안 됨(접근 중 끝이 장애물에 닿아도 모름).
        for _t in ('gripper_rh_p12_rn_r2', 'gripper_rh_p12_rn_l2'):
            if _t not in _kin["collision_link_names"]:
                _kin["collision_link_names"].append(_t)
        # 그리퍼 클러스터(손목~핑거)는 서로 가까운 강체 → 상호 자기충돌 면제(헛충돌 방지)
        _gcluster = ['link_5', 'link_6', 'gripper_rh_p12_rn_base',
                     'gripper_rh_p12_rn_r1', 'gripper_rh_p12_rn_l1',
                     'gripper_rh_p12_rn_r2', 'gripper_rh_p12_rn_l2', 'attached_object']
        for _a in _gcluster:
            _ign = set(_kin["self_collision_ignore"].get(_a, []))
            _ign.update(x for x in _gcluster if x != _a)
            _kin["self_collision_ignore"][_a] = list(_ign)
        robot_cfg = RobotConfig.from_dict(_cfg["robot_cfg"], self.tensor_args)
        world_cfg = WorldConfig(
            cuboid=[Cuboid(name="table", pose=[0.0, 0.0, -0.02, 1, 0, 0, 0],
                           dims=[1.2, 1.2, 0.04])]
        )
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg, world_cfg, self.tensor_args,
            num_trajopt_seeds=4, num_graph_seeds=4,
            collision_cache={"obb": 30, "mesh": 10},
            use_cuda_graph=False,   # ★goalset(plan_grasp)↔single(plan) 혼용 시 'changing goal
                                    #   type, cuda graph reset' 에러 회피 (가이드 4절 처방).
        )
        self.motion_gen = MotionGen(motion_gen_cfg)
        self.motion_gen.warmup(warmup_js_trajopt=False)
        self.get_logger().info("cuRobo 준비 완료!")

        self.grasp_force_params = self._load_grasp_force_params(config_dir)
        self._load_place_targets(config_dir)

        # Subscribers
        self.create_subscription(JointState, '/dsr01/joint_states',
                                 self._joint_state_cb, 10)
        # Pipeline B 토픽
        # pick/target 콜백은 내부에서 서비스(spline/movel/posx)를 동기 대기하므로
        # reentrant 그룹에 둬야 콜백 실행 중에도 서비스 응답 future 가 처리됨
        # (기본 그룹이면 spline/posx future 가 30s/3s 타임아웃 → false 실패).
        self.create_subscription(PoseStamped, '/dsr01/curobo/pick_pose',
                                 self._pick_pose_cb, 10,
                                 callback_group=self.service_cb_group)
        self.create_subscription(String, '/dsr01/curobo/obstacles',
                                 self._obstacles_cb, 10)
        self.create_subscription(String, '/dsr01/curobo/grasp_class',
                                 self._grasp_class_cb, 10)
        # [goalset 실험] GraspGen 후보 묶음 구독
        self.create_subscription(PoseArray, '/dsr01/curobo/grasp_candidates',
                                 self._grasp_candidates_cb, 10)

        # Pipeline A 서비스 서버
        for name, cb in [
            ('/move_to_shelf_view',   self._srv_shelf_view),
            ('/move_to_product_view', self._srv_product_view),
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

    def _make_jlim_urdf(self, src_urdf, dst_urdf, overrides):
        """원본 URDF를 안 건드리고 지정 관절 position 한계만 바꾼 복사본 생성 (시뮬팀 stage7 동일).
        cuRobo BoundCost는 URDF를 init에서 clone → 런타임 텐서 수정 무효, 반드시 URDF로 줘야 반영됨.
        overrides={joint:(lower,upper)}. 손목 플립 방지(joint_5 음수 금지)용."""
        import xml.etree.ElementTree as ET
        tree = ET.parse(src_urdf); root = tree.getroot(); hit = []
        for j in root.findall("joint"):
            if j.get("name") in overrides:
                lim = j.find("limit")
                if lim is None:
                    continue
                lo, up = overrides[j.get("name")]
                lim.set("lower", repr(float(lo))); lim.set("upper", repr(float(up)))
                hit.append(j.get("name"))
        tree.write(dst_urdf, encoding="utf-8", xml_declaration=True)
        self.get_logger().info(f"[관절한계URDF] {dst_urdf} — 수정 관절 {hit} (손목플립 방지)")
        return dst_urdf

    def _find_config_dir(self):
        # __file__ = src/motion/curobo_planner_node.py → 3단계 위가 패키지 루트
        # (src/config 가 아니라 패키지루트/config). 그래야 grasp_force_params.yaml 도 로드됨.
        local = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "curobo")
        if os.path.exists(local):
            return local
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory("e0509_gripper_description"),
            "config", "curobo")

    def _load_place_targets(self, config_dir):
        path = os.path.join(os.path.dirname(config_dir), "place_targets.yaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        self.NAMED_TARGETS_DEG = {k: list(v) for k, v in data.get('named_targets', {}).items()}
        self.PLACE_TARGETS     = {k: {ik: (list(iv) if iv is not None else None)
                                       for ik, iv in v.items()}
                                   for k, v in data.get('place_targets', {}).items()}
        self.get_logger().info(
            f"place_targets 로드: named={list(self.NAMED_TARGETS_DEG)}, "
            f"place={list(self.PLACE_TARGETS)}")

    def _load_grasp_force_params(self, config_dir):
        path = os.path.join(os.path.dirname(config_dir), "grasp_force_params.yaml")
        try:
            with open(path, encoding='utf-8') as f:
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

    def _grasp_class_cb(self, msg: String):
        self.grasp_class = msg.data.strip() or None
        if self.grasp_class and self.grasp_class in self.PLACE_TARGETS:
            self.get_logger().info(
                f"[grasp_class] {self.grasp_class} → place 준비됨")
        else:
            self.get_logger().warn(
                f"[grasp_class] '{self.grasp_class}' — PLACE_TARGETS에 없음 "
                f"(가능: {list(self.PLACE_TARGETS.keys())})")

    def _grasp_candidates_cb(self, msg: PoseArray):
        """[goalset 실험] webcam_seg가 보낸 GraspGen 후보 EE pose 묶음 저장."""
        cands = []
        for p in msg.poses:
            cands.append((
                [p.position.x, p.position.y, p.position.z],
                [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]))
        self.grasp_candidates = cands
        self._cand_t = time.time()
        self.get_logger().info(f"[grasp_candidates] 후보 {len(cands)}개 수신")

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
            #self.get_logger().info(
                #f"장애물 업데이트: table + {len(data)}개")
        except Exception as e:
            self.get_logger().error(f"장애물 업데이트 실패: {e}")

    # Pipeline B 토픽 콜백
    def _pick_pose_cb(self, msg: PoseStamped):
        if self.current_joints is None:
            self.get_logger().warn("joint_states 미수신")
            return
        # 기본: standoff-approach 단일경로 (파지점-STANDOFF 뒤 프리그래스프 → 축방향 직선 전진).
        # goalset(plan_grasp) 실험경로는 CUROBO_USE_GOALSET=1 일 때만 (현재 44-구체 모델
        # 링크명 불일치로 plan_grasp 예외 → 비활성. 고치면 재활성).
        if (os.environ.get('CUROBO_USE_GOALSET', '0') != '0'
                and self.grasp_candidates and (time.time() - self._cand_t) < 60.0):
            if self._do_pick_goalset():
                return
            self.get_logger().warn("plan_grasp 실패 → standoff 단일경로 fallback")
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

    def _srv_place(self, request, response):
        used_target = (self.grasp_class
                       if self.grasp_class and self.grasp_class in self.PLACE_TARGETS
                       else self.get_parameter('place_target').get_parameter_value().string_value)
        ok = self._move_to_place()
        response.success = ok
        response.message = f"Place {'완료' if ok else '실패'} ({used_target})"
        return response

    def _srv_home(self, request, response):
        ok = self._move_to_named_target('home')
        response.success = ok
        response.message = "완료" if ok else "실패"
        return response

    # ── Pick 시퀀스 ──────────────────────────────────────────

    def _do_pick_sequence(self, pose: PoseStamped):
        """open → (cuRobo) 접근축 STANDOFF 뒤 프리그래스프 → (MoveLine 직선) 축방향 전진
        → move_stop → safe_grasp → lift. 캔에 비스듬히 내려박지 않고 파지축으로 곧게 진입."""
        pos, ori = pose.pose.position, pose.pose.orientation
        quat_wxyz = [ori.w, ori.x, ori.y, ori.z]
        LIFT_HEIGHT = 0.15  # m
        STANDOFF = float(os.environ.get('CUROBO_PICK_STANDOFF', '0.06'))  # m, 축방향 후퇴거리 (s와 통일)

        grasp = np.array([pos.x, pos.y, pos.z], dtype=float)
        if grasp[2] < self.TCP_Z_MIN:
            self.get_logger().warn(
                f"[FLOOR GUARD] pick z={grasp[2]*1000:.1f}mm → {self.TCP_Z_MIN*1000:.0f}mm 클램프")
            grasp[2] = self.TCP_Z_MIN
        # 접근축 = 그리퍼 Z (orientation 3번째 열). 프리그래스프 = 파지점 - STANDOFF·approach.
        approach = self._quat_to_rotm(quat_wxyz)[:, 2]
        approach = approach / (np.linalg.norm(approach) + 1e-9)
        pre = grasp - STANDOFF * approach

        self.get_logger().info("=== PICK Step 1/5: 그리퍼 열기 ===")
        self.gripper_open()
        time.sleep(1.5)

        self.get_logger().info(
            f"=== PICK Step 2/5: 프리그래스프 이동 (파지점-{STANDOFF*100:.0f}cm·approach, cuRobo) "
            f"→ ({pre[0]*1000:.0f},{pre[1]*1000:.0f},{pre[2]*1000:.0f})mm ===")
        traj = self.plan(self.current_joints, pre.tolist(), quat_wxyz)
        if traj is None:
            self.get_logger().error("Pick 실패: 프리그래스프 경로계획 불가")
            return
        if not self.execute_spline(traj):
            self.get_logger().error("Pick 실패: 프리그래스프 실행 실패")
            return
        spline_vel_scale = 1.5 if self.grasp_class == 'snack_bag' else 1.0
        self.execute_spline(traj, vel_scale=spline_vel_scale)
        time.sleep(2.0)
        # 프리그래스프에서 실제 풀린 자세(원본/180°플립) — 전진도 같은 자세로 (손목 연속)
        q = getattr(self, '_last_plan_quat_wxyz', quat_wxyz)  # [w,x,y,z]

        # Step 3: 같은 자세로 grasp 지점까지 cuRobo joint 경로 전진.
        #   MoveLine(오일러 ZYZ b=90° 짐벌락 → 직진중 손목 90° 튐) 회피.
        #   현재(프리그래스프) joint 에서 seed → 손목 연속, allow_yaw_retry=False(재플립 금지).
        self.get_logger().info(
            f"=== PICK Step 3/5: 축방향 전진 {STANDOFF*100:.0f}cm (cuRobo joint, 손목연속) "
            f"→ ({grasp[0]*1000:.0f},{grasp[1]*1000:.0f},{grasp[2]*1000:.0f})mm ===")
        traj2 = self.plan(self.current_joints, grasp.tolist(), q,
                          allow_yaw_retry=False)
        if traj2 is None:
            self.get_logger().error("Pick 실패: 전진 경로계획 불가")
            return
        if not self.execute_spline(traj2):
            self.get_logger().error("Pick 실패: 전진 실행 실패")
            return
        time.sleep(1.5)

        # 이동 직후 컨트롤러 motion 상태 클리어 → 그래야 flange serial 그리퍼가 닿음
        self.get_logger().info("=== PICK Step 4/5: move_stop → 그리퍼 닫기 ===")
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

        self.get_logger().info("=== PICK Step 5/5: 수직 상승 (자세 유지) ===")
        self.lift_straight_up(LIFT_HEIGHT, vel=50.0)
        time.sleep(1.5)
        self.get_logger().info("=== PICK 완료 ===")

    def _do_pick_goalset(self):
        """[goalset 실험] open → plan_grasp(후보 묶음, 무충돌·도달 best 선택) →
        2단계 진입(approach→grasp) → move_stop → safe_grasp → lift."""
        cands = self.grasp_candidates
        N = len(cands)
        if N == 0:
            return False
        _pl = np.zeros((1, N, 3), dtype=np.float32)
        _ql = np.zeros((1, N, 4), dtype=np.float32)
        for i, (p, q) in enumerate(cands):
            _pl[0, i] = p
            _ql[0, i] = [q[3], q[0], q[1], q[2]]   # xyzw → wxyz
        gposes = Pose(position=self.tensor_args.to_device(_pl),
                      quaternion=self.tensor_args.to_device(_ql))
        start_state = CuroboJointState.from_position(
            position=torch.tensor([self.current_joints], device="cuda:0", dtype=torch.float32),
            joint_names=self.JOINT_NAMES)

        self.get_logger().info(f"=== PICK(goalset) 1/4: 그리퍼 열기 (후보 {N}개) ===")
        self.gripper_open()
        time.sleep(1.5)

        self.get_logger().info(f"=== PICK(goalset) 2/4: plan_grasp({N}후보, 2단계 진입) ===")
        try:
            gres = self.motion_gen.plan_grasp(
                start_state, gposes, MotionGenPlanConfig(max_attempts=4),
                grasp_approach_offset=Pose.from_list(
                    [0, 0, -self.PREGRASP_STANDOFF, 1, 0, 0, 0]),
                disable_collision_links=list(self.gripper_coll_links),
                plan_grasp_to_retract=False)
        except Exception as e:
            self.get_logger().error(f"plan_grasp 예외: {e}")
            return False
        if not bool(gres.success.item()):
            self.get_logger().warn("plan_grasp: 도달가능·무충돌 후보 없음")
            return False
        gi = int(gres.goalset_index.item())
        self.get_logger().info(f"[goalset] 선택 후보 #{gi+1}/{N}")
        try:
            self.get_logger().info(
                f"  → 2a) 프리그래스프 접근 (파지점 {self.PREGRASP_STANDOFF*100:.0f}cm 뒤로)")
            cmd_a = gres.approach_result.get_interpolated_plan().position.cpu().numpy()
            self.execute_spline(cmd_a)
            time.sleep(2.0)
            self.get_logger().info(
                f"  → 2b) 축방향 {self.PREGRASP_STANDOFF*100:.0f}cm 쭉 전진해서 파지점 도달")
            cmd_g = gres.grasp_result.get_interpolated_plan().position.cpu().numpy()
            self.execute_spline(cmd_g)
            time.sleep(1.5)
        except Exception as e:
            self.get_logger().error(f"goalset 궤적 실행 실패: {e}")
            return False

        self.get_logger().info("=== PICK(goalset) 3/4: move_stop → safe_grasp ===")
        try:
            if self.cli_stop.wait_for_service(timeout_sec=2.0):
                sr = MoveStop.Request(); sr.stop_mode = 1
                f = self.cli_stop.call_async(sr)
                t0 = time.time()
                while not f.done() and time.time() - t0 < 3.0:
                    time.sleep(0.05)
        except Exception as e:
            self.get_logger().warn(f"move_stop 실패(무시): {e}")
        time.sleep(1.0)
        self.gripper_grasp()
        time.sleep(1.5)

        self.get_logger().info("=== PICK(goalset) 4/4: 수직 상승 ===")
        self.lift_straight_up(0.15, vel=50.0)
        time.sleep(1.5)
        self.get_logger().info("=== PICK(goalset) 완료 ===")
        return True

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
        # self.grasp_class 우선 사용 (grasp_class 토픽에서 직접 갱신됨)
        # 없으면 파라미터 fallback
        param_target = self.get_parameter('place_target').get_parameter_value().string_value
        place_target = (self.grasp_class
                        if self.grasp_class and self.grasp_class in self.PLACE_TARGETS
                        else param_target)
        if place_target not in self.PLACE_TARGETS:
            self.get_logger().error(
                f"알 수 없는 place_target: '{place_target}' "
                f"(가능: {list(self.PLACE_TARGETS.keys())})")
            return False
        target = self.PLACE_TARGETS[place_target]
        self.get_logger().info(f"_move_to_place: '{place_target}'")
        # time.sleep(10.0)

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

        # 1차: 원본 + 180°플립(평행조 그리퍼는 같은 파지) 둘 다 풀어서
        #      손목(J6)이 적게 도는 쪽 선택 → 불필요한 한바퀴 회전 방지.
        sj6 = float(start_joints[5])
        flip_quat = self._quat_yaw(target_quat_wxyz, np.pi)
        base = self._plan_once(start_state, target_pos, target_quat_wxyz)
        flip = self._plan_once(start_state, target_pos, flip_quat)
        # 선택한 해의 orientation 저장 (전진 단계가 그 자세 그대로 명령 — 손목 안 돌게)
        self._last_plan_quat_wxyz = list(target_quat_wxyz)
        cands = []
        for nm, tr, q in (('원본', base, target_quat_wxyz), ('180°flip', flip, flip_quat)):
            if tr is not None and (not prefer_positive_j4 or self._traj_j4_ok(tr)):
                cands.append((nm, tr, q))
        if cands:
            nm, best, q = min(cands, key=lambda c: self._wrist_travel(c[1], sj6))
            self._last_plan_quat_wxyz = list(q)
            self.get_logger().info(
                f"계획 OK [{nm}]: {(time.time()-t0)*1000:.1f}ms, {best.shape[0]}pts "
                f"(손목 J6 {np.degrees(self._wrist_travel(best, sj6)):.0f}° 회전, "
                f"j4_min={np.degrees(best[:,3].min()):.1f}°)")
            return best

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

    def _wrist_travel(self, positions, start_j6):
        """trajectory 가 J6(손목)를 start 에서 얼마나 멀리 도는지(rad). 작을수록 좋음.
        한바퀴(>180°) 도는 해 vs 등가 해 비교용."""
        j6 = positions[:, 5]
        return float(max(abs(float(j6.max()) - start_j6),
                         abs(float(j6.min()) - start_j6)))

    def _advance_along(self, approach, dist_m, vel=60.0):
        """현재 TCP 자세(rx/ry/rz) 그대로 유지하며 approach 방향으로 dist_m 직선 전진.
        orientation 을 다시 명령하지 않으므로 손목이 돌지 않음 (플립 해도 그대로 진입)."""
        if not self.cli_posx.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("GetCurrentPosx 없음 — 전진 취소"); return False
        fut = self.cli_posx.call_async(GetCurrentPosx.Request())
        t0 = time.time()
        while not fut.done() and (time.time() - t0) < 3.0:
            time.sleep(0.05)
        if not (fut.done() and fut.result()):
            self.get_logger().error("GetCurrentPosx 실패 — 전진 취소"); return False
        cur = list(fut.result().task_pos_info[0].data)  # [x,y,z,rx,ry,rz] mm/deg
        a = np.asarray(approach, dtype=float)
        a = a / (np.linalg.norm(a) + 1e-9) * dist_m * 1000.0  # mm
        req = MoveLine.Request()
        req.pos = [cur[0]+a[0], cur[1]+a[1], cur[2]+a[2], cur[3], cur[4], cur[5]]
        _v = vel * self.VEL_SCALE
        req.vel = [_v, 30.0 * self.VEL_SCALE]; req.acc = [_v, 30.0 * self.VEL_SCALE]
        req.time = 0.0; req.ref = 0; req.mode = 0
        req.blend_type = 0; req.sync_type = 1
        self.get_logger().info(
            f"전진(자세유지): ({cur[0]:.0f},{cur[1]:.0f},{cur[2]:.0f}) → "
            f"({cur[0]+a[0]:.0f},{cur[1]+a[1]:.0f},{cur[2]+a[2]:.0f})mm "
            f"rz={cur[5]:.0f}° 유지")
        return self._wait_for_motion(self.cli_movel.call_async(req), "MoveLine전진")

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

    def execute_spline(self, traj_rad, vel_scale: float = 1.0):
        """[단발 movej] cuRobo 궤적 끝점까지 movej 한 번으로 이동 = 홈 이동과 동일 방식
        → 한 개의 매끄러운 사다리꼴 속도프로파일이라 끊김 없음. (movesj=hang, 다점
        movej=서로 덮어써 끊김 → 단발 movej 가 가장 부드럽고 확실. 2026-06-17)"""
        if not self.cli_movej.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("MoveJoint 서비스 없음")
            return False
        traj_deg = np.rad2deg(traj_rad)
        target_deg = [float(v) for v in traj_deg[-1]]
        target_rad = np.asarray(traj_rad[-1], dtype=float)
        vel_deg = float(os.environ.get('CUROBO_SPLINE_VEL', '70')) * self.VEL_SCALE * float(vel_scale)
        acc_deg = float(os.environ.get('CUROBO_SPLINE_ACC', '150')) * self.VEL_SCALE
        req = MoveJoint.Request()
        req.pos = target_deg; req.vel = vel_deg; req.acc = acc_deg
        req.time = 0.0; req.radius = 0.0; req.mode = 0
        req.blend_type = 0; req.sync_type = 0
        self.get_logger().info(
            f"모션(단발 movej) → end={[f'{v:.1f}' for v in target_deg]} vel={vel_deg:.0f}°/s")
        self.cli_movej.call_async(req)
        # 실제 관절 도달 대기 (movej sync_type=0 은 접수 즉시 반환 → joint_states 로 판정)
        t0 = time.time(); reached = False; last_log = 0.0
        while time.time() - t0 < 40.0:
            cj = self.current_joints
            if cj is not None:
                err = float(np.max(np.abs(np.asarray(cj, dtype=float) - target_rad)))
                if err < np.radians(3.5):
                    reached = True; break
                if time.time() - t0 - last_log > 5.0:
                    last_log = time.time() - t0
                    self.get_logger().info(
                        f"  모션 진행중... 도달오차 {np.degrees(err):.1f}° ({last_log:.0f}s)")
            time.sleep(0.1)
        cj = self.current_joints
        self.get_logger().info(
            "모션 " + ("완료" if reached else "실패(미도달)") +
            (f" (도달오차 {np.degrees(float(np.max(np.abs(np.asarray(cj,dtype=float)-target_rad)))):.1f}°)"
             if cj is not None else ""))
        return reached

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
