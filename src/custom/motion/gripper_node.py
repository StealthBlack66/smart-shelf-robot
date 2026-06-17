import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState


# 물성별 파지힘 파라미터 (단위: N)
GRASP_FORCE_TABLE = {
    'bread':   {'force': 10.0,  'max_force': 15.0},   # 식빵: 소프트바디, 상한 제어
    'snack':   {'force': 20.0,  'max_force': 30.0},   # 과자봉지: 끝단 파지
    'bottle':  {'force': 40.0,  'max_force': 60.0},   # PET병: 원통형, 적정 파지힘
    'can':     {'force': 80.0,  'max_force': 100.0},  # 캔: 금속, 최대 파지힘
}


class GripperNode(Node):
    def __init__(self):
        super().__init__('gripper_node')

        # Subscribers
        self.sub_object_class = self.create_subscription(
            Float32, '/object_class', self.object_class_callback, 10)
        self.sub_grasp_force = self.create_subscription(
            Float32, '/grasp_force', self.grasp_force_callback, 10)

        # Publishers
        self.pub_gripper_command = self.create_publisher(
            JointState, '/gripper_command', 10)

        # Services
        self.srv_open = self.create_service(
            Trigger, '/gripper/open', self.open_callback)
        self.srv_close = self.create_service(
            Trigger, '/gripper/close', self.close_callback)
        self.srv_grasp = self.create_service(
            Trigger, '/gripper/grasp', self.grasp_callback)

        self.current_force = 0.0
        self.current_class = None

    def object_class_callback(self, msg):
        # TODO: 물체 클래스 수신 후 파지힘 자동 설정
        # self.current_class = msg.data
        # params = GRASP_FORCE_TABLE.get(self.current_class)
        # if params:
        #     self.current_force = params['force']
        pass

    def grasp_force_callback(self, msg):
        self.current_force = msg.data

    def open_callback(self, request, response):
        # TODO: 그리퍼 열기
        # self._send_gripper_command(position=0.0, effort=0.0)
        response.success = False
        response.message = 'TODO: implement'
        return response

    def close_callback(self, request, response):
        # TODO: 그리퍼 닫기 (최대 파지힘)
        # self._send_gripper_command(position=1.101, effort=self.current_force)
        response.success = False
        response.message = 'TODO: implement'
        return response

    def grasp_callback(self, request, response):
        # TODO: 물성 기반 파지힘으로 파지
        # params = GRASP_FORCE_TABLE.get(self.current_class)
        # self._send_gripper_command(
        #     position=1.101,
        #     effort=min(self.current_force, params['max_force'])
        # )
        response.success = False
        response.message = 'TODO: implement'
        return response

    def _send_gripper_command(self, position, effort):
        # TODO: /gripper_command 토픽으로 명령 전송
        # msg = JointState()
        # msg.name = ['rh_r1']
        # msg.position = [position]
        # msg.effort = [effort]
        # self.pub_gripper_command.publish(msg)
        pass


def main(args=None):
    rclpy.init(args=args)
    node = GripperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
