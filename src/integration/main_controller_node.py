import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger


# 작업 상태 정의
class TaskState:
    IDLE        = 'idle'
    DETECTING   = 'detecting'
    MOVING_PICK = 'moving_to_pick'
    GRASPING    = 'grasping'
    MOVING_PLACE = 'moving_to_place'
    PLACING     = 'placing'
    DONE        = 'done'
    ERROR       = 'error'


class MainControllerNode(Node):
    def __init__(self):
        super().__init__('main_controller_node')

        # Subscribers
        self.sub_object_class = self.create_subscription(
            String, '/object_class', self.object_class_callback, 10)
        self.sub_object_pose = self.create_subscription(
            PoseStamped, '/object_pose', self.object_pose_callback, 10)

        # Publishers
        self.pub_grasp_force = self.create_publisher(Float32, '/grasp_force', 10)
        self.pub_place_target = self.create_publisher(PoseStamped, '/place_target', 10)

        # Service clients
        self.cli_move_to_pick  = self.create_client(Trigger, '/move_to_pick')
        self.cli_move_to_place = self.create_client(Trigger, '/move_to_place')
        self.cli_move_to_home  = self.create_client(Trigger, '/move_to_home')
        self.cli_gripper_open  = self.create_client(Trigger, '/gripper/open')
        self.cli_gripper_grasp = self.create_client(Trigger, '/gripper/grasp')

        # 상태 머신
        self.state = TaskState.IDLE
        self.object_class = None
        self.object_pose = None

        # 메인 루프 타이머 (5Hz)
        self.timer = self.create_timer(0.2, self.state_machine_callback)

        self.get_logger().info('Main controller node started')

    def object_class_callback(self, msg):
        self.object_class = msg.data

    def object_pose_callback(self, msg):
        self.object_pose = msg

    def state_machine_callback(self):
        if self.state == TaskState.IDLE:
            self._on_idle()
        elif self.state == TaskState.DETECTING:
            self._on_detecting()
        elif self.state == TaskState.MOVING_PICK:
            self._on_moving_to_pick()
        elif self.state == TaskState.GRASPING:
            self._on_grasping()
        elif self.state == TaskState.MOVING_PLACE:
            self._on_moving_to_place()
        elif self.state == TaskState.PLACING:
            self._on_placing()
        elif self.state == TaskState.DONE:
            self._on_done()
        elif self.state == TaskState.ERROR:
            self._on_error()

    def _on_idle(self):
        # TODO: 작업 시작 트리거 처리
        # 감지 시작하면 DETECTING으로 전환
        # self.state = TaskState.DETECTING
        pass

    def _on_detecting(self):
        # TODO: 물체 감지 완료 확인
        # object_class, object_pose 수신되면 다음 단계로
        # if self.object_class and self.object_pose:
        #     self._set_grasp_force(self.object_class)
        #     self.state = TaskState.MOVING_PICK
        pass

    def _on_moving_to_pick(self):
        # TODO: 팔이 파지 위치 도달 확인
        # move_to_pick 서비스 호출 후 완료 대기
        pass

    def _on_grasping(self):
        # TODO: 그리퍼 파지 실행 및 완료 확인
        # gripper_grasp 서비스 호출
        pass

    def _on_moving_to_place(self):
        # TODO: 팔이 적재 위치 도달 확인
        # place_target 퍼블리시 후 move_to_place 서비스 호출
        pass

    def _on_placing(self):
        # TODO: 그리퍼 열어서 물체 내려놓기
        # gripper_open 서비스 호출
        pass

    def _on_done(self):
        # TODO: 작업 완료 후 홈으로 복귀
        # move_to_home 서비스 호출 후 IDLE로 전환
        pass

    def _on_error(self):
        # TODO: 에러 처리 및 안전 복귀
        # 그리퍼 열기 → 홈으로 복귀 → IDLE
        pass

    def _set_grasp_force(self, object_class):
        # TODO: 물체 클래스에 따라 파지힘 설정 후 퍼블리시
        # force_map = {'bread': 10.0, 'snack': 20.0, 'bottle': 40.0, 'can': 80.0}
        # msg = Float32()
        # msg.data = force_map.get(object_class, 30.0)
        # self.pub_grasp_force.publish(msg)
        pass


def main(args=None):
    rclpy.init(args=args)
    node = MainControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
