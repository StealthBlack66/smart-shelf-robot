import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState


class ArmControllerNode(Node):
    def __init__(self):
        super().__init__('arm_controller_node')

        # Subscribers
        self.sub_target_pose = self.create_subscription(
            PoseStamped, '/place_target', self.place_target_callback, 10)

        # Publishers
        self.pub_joint_command = self.create_publisher(JointState, '/joint_command', 10)

        # Services
        self.srv_move_to_pick = self.create_service(
            Trigger, '/move_to_pick', self.move_to_pick_callback)
        self.srv_move_to_place = self.create_service(
            Trigger, '/move_to_place', self.move_to_place_callback)
        self.srv_move_to_home = self.create_service(
            Trigger, '/move_to_home', self.move_to_home_callback)

        self.target_pose = None
        self.moveit = None
        self._init_moveit()

    def _init_moveit(self):
        # TODO: MoveIt2 인터페이스 초기화
        # from moveit.planning import MoveItPy
        # self.moveit = MoveItPy(node_name='arm_controller')
        pass

    def place_target_callback(self, msg):
        self.target_pose = msg

    def move_to_pick_callback(self, request, response):
        # TODO: 물체 파지 위치로 이동
        # pose = self._get_pick_pose()
        # success = self._move_to_pose(pose)
        # response.success = success
        response.success = False
        response.message = 'TODO: implement'
        return response

    def move_to_place_callback(self, request, response):
        # TODO: 매대 적재 위치로 이동
        # success = self._move_to_pose(self.target_pose)
        # response.success = success
        response.success = False
        response.message = 'TODO: implement'
        return response

    def move_to_home_callback(self, request, response):
        # TODO: 홈 포지션으로 이동
        # success = self._move_to_named_target('home')
        # response.success = success
        response.success = False
        response.message = 'TODO: implement'
        return response

    def _move_to_pose(self, pose):
        # TODO: MoveIt으로 목표 포즈까지 경로 계획 및 실행
        # self.moveit.move_to_pose(pose)
        pass

    def _move_to_named_target(self, name):
        # TODO: 이름으로 정의된 포즈로 이동 (home 등)
        pass


def main(args=None):
    rclpy.init(args=args)
    node = ArmControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
