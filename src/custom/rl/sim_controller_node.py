import numpy as np
import zmq

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
from std_srvs.srv import Trigger

try:
    import sensor_msgs_py.point_cloud2 as pc2
except ImportError:
    pc2 = None


ZMQ_GRASPGEN_ADDRESS = 'tcp://localhost:5556'

# GraspGen 서버가 반환하는 파지 후보 한 건의 바이트 크기
# [x, y, z, qx, qy, qz, qw, score] = float32 x 8
GRASPGEN_CANDIDATE_DIM = 8


class SimControllerNode(Node):
    def __init__(self):
        super().__init__('sim_controller_node')

        # ZMQ — graspgen conda 환경의 GraspGen 서버에 연결
        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.REQ)
        self._zmq_sock.connect(ZMQ_GRASPGEN_ADDRESS)
        self.get_logger().info(f'GraspGen ZMQ 서버 연결: {ZMQ_GRASPGEN_ADDRESS}')

        # Subscribers
        self.sub_pointcloud = self.create_subscription(
            PointCloud2, '/object_pointcloud', self._pointcloud_cb, 10)

        # Publishers
        self.pub_joint_command = self.create_publisher(JointState, '/joint_command', 10)

        # Service server — integration 노드가 호출
        self.srv_move_to_pick = self.create_service(
            Trigger, '/move_to_pick', self._move_to_pick_cb)

        self.latest_pointcloud = None
        self._cuRobo = None
        self._init_curobo()

        self.get_logger().info('Sim controller node started')

    # ── 초기화 ────────────────────────────────────────────────

    def _init_curobo(self):
        # TODO: cuRobo 초기화
        #   from curobo.types.robot import RobotConfig
        #   from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
        #   config = MotionGenConfig.load_from_robot_config(
        #       robot_cfg='doosan_e0509.yml',
        #       world_cfg='collision_world.yml',
        #   )
        #   self._cuRobo = MotionGen(config)
        #   self._cuRobo.warmup()
        pass

    # ── 콜백 ──────────────────────────────────────────────────

    def _pointcloud_cb(self, msg: PointCloud2):
        self.latest_pointcloud = msg

    def _move_to_pick_cb(self, request, response):
        if self.latest_pointcloud is None:
            response.success = False
            response.message = 'pointcloud 미수신 (/object_pointcloud)'
            return response

        # 1. GraspGen으로 파지 후보 생성
        grasp_pose = self._call_graspgen(self.latest_pointcloud)
        if grasp_pose is None:
            response.success = False
            response.message = 'GraspGen 파지 후보 생성 실패'
            return response

        # 2. cuRobo 궤적 계획
        trajectory = self._plan_trajectory(grasp_pose)
        if trajectory is None:
            response.success = False
            response.message = 'cuRobo 궤적 계획 실패'
            return response

        # 3. 궤적 실행
        self._execute_trajectory(trajectory)

        response.success = True
        response.message = 'pick 완료'
        return response

    # ── GraspGen ──────────────────────────────────────────────

    def _call_graspgen(self, cloud_msg: PointCloud2):
        # TODO: 포인트클라우드 → numpy 변환 후 ZMQ 서버에 전송, 파지 후보 수신
        #
        # [전송 형식] float32 numpy array, shape (N, 3), x/y/z
        #   points = np.array(list(pc2.read_points(cloud_msg, field_names=('x','y','z'),
        #                                           skip_nans=True)), dtype=np.float32)
        #   self._zmq_sock.send(points.tobytes())
        #
        # [수신 형식] float32 numpy array, shape (K, 8)
        #   각 행: [x, y, z, qx, qy, qz, qw, score]
        #   raw = self._zmq_sock.recv()
        #   candidates = np.frombuffer(raw, dtype=np.float32).reshape(-1, GRASPGEN_CANDIDATE_DIM)
        #
        # [선택] score(마지막 열) 기준 최고 후보 반환
        #   best = candidates[candidates[:, -1].argmax()]  # shape (8,)
        #   return best[:7]  # [x, y, z, qx, qy, qz, qw]
        return None

    # ── cuRobo ────────────────────────────────────────────────

    def _plan_trajectory(self, grasp_pose: np.ndarray):
        # TODO: cuRobo로 파지 자세까지의 충돌 회피 궤적 계획
        #   grasp_pose: float32 array [x, y, z, qx, qy, qz, qw] (base_link 기준)
        #
        #   from curobo.types.math import Pose
        #   from curobo.types.state import JointState as CuJointState
        #   goal_pose = Pose(position=grasp_pose[:3], quaternion=grasp_pose[3:])
        #   result = self._cuRobo.plan_single(start_state, goal_pose, plan_config)
        #   if result.success:
        #       return result.get_interpolated_plan()  # list[JointState]
        #   return None
        return None

    def _execute_trajectory(self, trajectory):
        # TODO: 궤적 각 포인트를 /joint_command로 발행
        #   for waypoint in trajectory:
        #       msg = JointState()
        #       msg.header.stamp = self.get_clock().now().to_msg()
        #       msg.name = ['joint1','joint2','joint3','joint4','joint5','joint6']
        #       msg.position = waypoint.position.tolist()
        #       self.pub_joint_command.publish(msg)
        #       # 필요시 rate sleep
        pass


def main(args=None):
    rclpy.init(args=args)
    node = SimControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
