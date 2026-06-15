import concurrent.futures
import yaml
import os

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from control_msgs.action import GripperCommand


# 작업 상태 정의
class TaskState:
    IDLE                 = 'idle'
    MOVE_TO_SHELF_VIEW   = 'move_to_shelf_view'
    SHELF_DETECTING      = 'shelf_detecting'
    MOVE_TO_PRODUCT_VIEW = 'move_to_product_view'
    PRODUCT_DETECTING    = 'product_detecting'
    MOVING_PICK          = 'moving_to_pick'
    GRASPING             = 'grasping'
    MOVING_PLACE         = 'moving_to_place'
    PLACING              = 'placing'
    DONE                 = 'done'
    ERROR                = 'error'


# 물체 클래스별 목표전류 (mA)
CURRENT_MAP = {
    'can':      800.0,
    'bottle':   400.0,
    'snack_bag': 200.0,
}

MAX_CURRENT_MAP = {
    'can':      1000.0,
    'bottle':    600.0,
    'snack_bag': 300.0,
}


class MainControllerNode(Node):
    def __init__(self):
        super().__init__('main_controller_node')

        # config 로드
        config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
        self._place_targets = self._load_yaml(os.path.join(config_dir, 'place_targets.yaml'))

        # Subscribers
        self.sub_object_class = self.create_subscription(
            String, '/object_class', self.object_class_callback, 10)
        self.sub_object_pose = self.create_subscription(
            PoseStamped, '/object_pose', self.object_pose_callback, 10)
        self.sub_empty_slot = self.create_subscription(
            String, '/shelf/empty_slot', self.empty_slot_callback, 10)

        # Publishers
        self.pub_place_target = self.create_publisher(PoseStamped, '/place_target', 10)

        # Service clients
        self.cli_move_to_shelf_view   = self.create_client(Trigger, '/move_to_shelf_view')
        self.cli_move_to_product_view = self.create_client(Trigger, '/move_to_product_view')
        self.cli_move_to_pick         = self.create_client(Trigger, '/move_to_pick')
        self.cli_move_to_place        = self.create_client(Trigger, '/move_to_place')
        self.cli_move_to_home         = self.create_client(Trigger, '/move_to_home')
        self.cli_gripper_open         = self.create_client(Trigger, '/gripper/open')

        # Action client
        self.act_gripper_grasp = ActionClient(self, GripperCommand, '/gripper/grasp')

        # 상태 머신
        self.state        = TaskState.IDLE
        self.object_class = None
        self.object_pose  = None
        self.empty_slot   = None   # "slot_id,class" 예: "0,can"
        self.target_current = 0.0

        # 비동기 호출 추적
        self._pending_future = None
        self._tick_count     = 0

        # 메인 루프 타이머 (5Hz)
        self.timer = self.create_timer(0.2, self.state_machine_callback)
        self.get_logger().info('Main controller node started')

    # ── 콜백 ────────────────────────────────────────────────
    def object_class_callback(self, msg):
        self.object_class = msg.data

    def object_pose_callback(self, msg):
        self.object_pose = msg

    def empty_slot_callback(self, msg):
        self.empty_slot = msg.data  # "0,can"

    # ── 상태머신 ─────────────────────────────────────────────
    def state_machine_callback(self):
        handlers = {
            TaskState.IDLE:                 self._on_idle,
            TaskState.MOVE_TO_SHELF_VIEW:   self._on_move_to_shelf_view,
            TaskState.SHELF_DETECTING:      self._on_shelf_detecting,
            TaskState.MOVE_TO_PRODUCT_VIEW: self._on_move_to_product_view,
            TaskState.PRODUCT_DETECTING:    self._on_product_detecting,
            TaskState.MOVING_PICK:          self._on_moving_to_pick,
            TaskState.GRASPING:             self._on_grasping,
            TaskState.MOVING_PLACE:         self._on_moving_to_place,
            TaskState.PLACING:              self._on_placing,
            TaskState.DONE:                 self._on_done,
            TaskState.ERROR:                self._on_error,
        }
        handlers[self.state]()

    def _on_idle(self):
        self._tick_count += 1
        if self._tick_count >= 10:
            self.get_logger().info('[IDLE] 작업 시작 → MOVE_TO_SHELF_VIEW')
            self._transition(TaskState.MOVE_TO_SHELF_VIEW)

    def _on_move_to_shelf_view(self):
        self._await_service(
            self.cli_move_to_shelf_view, '[MOVE_TO_SHELF_VIEW]',
            on_success=lambda: self._transition(TaskState.SHELF_DETECTING),
            on_fail=lambda: self._transition(TaskState.ERROR),
        )

    def _on_shelf_detecting(self):
        if self.empty_slot is None:
            self._tick_count += 1
            if self._tick_count < 5:
                return  # 비전 노드 데이터 대기
            # 더미: 1초(5틱) 후 가짜 빈 슬롯 주입
            self.empty_slot = '0,can'
            self.get_logger().info('[SHELF_DETECTING] 더미 empty_slot: 0,can')

        if self.empty_slot:
            parts = self.empty_slot.split(',')
            self.object_class = parts[1] if len(parts) == 2 else None
            self.get_logger().info(f'[SHELF_DETECTING] 빈 슬롯: {self.empty_slot} → MOVE_TO_PRODUCT_VIEW')
            self._transition(TaskState.MOVE_TO_PRODUCT_VIEW)
        else:
            self.get_logger().info('[SHELF_DETECTING] 빈 슬롯 없음 → IDLE')
            self._reset_and_idle()

    def _on_move_to_product_view(self):
        self._await_service(
            self.cli_move_to_product_view, '[MOVE_TO_PRODUCT_VIEW]',
            on_success=lambda: self._transition(TaskState.PRODUCT_DETECTING),
            on_fail=lambda: self._transition(TaskState.ERROR),
        )

    def _on_product_detecting(self):
        if self.object_pose is None:
            self._tick_count += 1
            if self._tick_count < 5:
                return  # 비전 노드 데이터 대기
            # 더미: 1초(5틱) 후 가짜 포즈 주입
            pose = PoseStamped()
            pose.header.frame_id = 'base_link'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = 0.5
            pose.pose.position.y = 0.0
            pose.pose.position.z = 0.3
            pose.pose.orientation.w = 1.0
            self.object_pose = pose
            self.get_logger().info('[PRODUCT_DETECTING] 더미 object_pose 설정')

        if self.object_class and self.object_pose:
            self.target_current = CURRENT_MAP.get(self.object_class, 300.0)
            self.get_logger().info(
                f'[PRODUCT_DETECTING] {self.object_class} 감지 완료, '
                f'목표전류: {self.target_current}mA → MOVING_PICK'
            )
            self._transition(TaskState.MOVING_PICK)
        else:
            self.get_logger().warn('[PRODUCT_DETECTING] 상품 미검출 → MOVE_TO_SHELF_VIEW')
            self._transition(TaskState.MOVE_TO_SHELF_VIEW)

    def _on_moving_to_pick(self):
        self._await_service(
            self.cli_move_to_pick, '[MOVING_PICK]',
            on_success=lambda: self._transition(TaskState.GRASPING),
            on_fail=lambda: self._transition(TaskState.ERROR),
        )

    def _on_grasping(self):
        # 액션 클라이언트로 목표전류 전송
        if self._pending_future is None:
            if not self.act_gripper_grasp.server_is_ready():
                self.get_logger().warn('[GRASPING] 액션 서버 미연결 → 더미 성공')
                f = concurrent.futures.Future()
                f.set_result(True)
                self._pending_future = f
            else:
                goal = GripperCommand.Goal()
                goal.command.position = self.target_current
                goal.command.max_effort = MAX_CURRENT_MAP.get(self.object_class, 500.0)
                self._pending_future = self.act_gripper_grasp.send_goal_async(goal)
                self.get_logger().info(f'[GRASPING] 목표전류 {self.target_current}mA 전송')
        elif self._pending_future.done():
            self._pending_future = None
            self.get_logger().info('[GRASPING] 파지 완료 → MOVING_PLACE')
            self._transition(TaskState.MOVING_PLACE)

    def _on_moving_to_place(self):
        if self._pending_future is None:
            self._publish_place_target()

        self._await_service(
            self.cli_move_to_place, '[MOVING_PLACE]',
            on_success=lambda: self._transition(TaskState.PLACING),
            on_fail=lambda: self._transition(TaskState.ERROR),
        )

    def _on_placing(self):
        self._await_service(
            self.cli_gripper_open, '[PLACING]',
            on_success=lambda: self._transition(TaskState.DONE),
            on_fail=lambda: self._transition(TaskState.ERROR),
        )

    def _on_done(self):
        self._await_service(
            self.cli_move_to_home, '[DONE]',
            on_success=self._reset_and_idle,
        )

    def _on_error(self):
        if self._pending_future is not None:
            if not self._pending_future.done():
                return
            self._pending_future = None
            self._tick_count += 1

        if self._tick_count == 0:
            self.get_logger().warn('[ERROR] 복구 시작: 그리퍼 열기')
            self._call_service_async(self.cli_gripper_open, '[ERROR]')
        elif self._tick_count == 1:
            self.get_logger().warn('[ERROR] 홈 복귀')
            self._call_service_async(self.cli_move_to_home, '[ERROR]')
        else:
            self.get_logger().warn('[ERROR] 복구 완료 → IDLE')
            self._reset_and_idle()

    # ── 헬퍼 ─────────────────────────────────────────────────
    def _publish_place_target(self):
        if self._place_targets is None or self.empty_slot is None:
            return
        parts = self.empty_slot.split(',')
        if len(parts) != 2:
            return
        class_name = parts[1]  # e.g. 'can', 'bottle'
        target = self._place_targets.get('place_targets', {}).get(class_name)
        if target is None:
            self.get_logger().warn(f'[MOVING_PLACE] place_targets.yaml에 {class_name} 없음')
            return
        msg = PoseStamped()
        msg.header.frame_id = 'base_link'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = target['position']['x']
        msg.pose.position.y = target['position']['y']
        msg.pose.position.z = target['position']['z']
        msg.pose.orientation.x = target['orientation']['x']
        msg.pose.orientation.y = target['orientation']['y']
        msg.pose.orientation.z = target['orientation']['z']
        msg.pose.orientation.w = target['orientation']['w']
        self.pub_place_target.publish(msg)
        self.get_logger().info(f'[MOVING_PLACE] {class_name} place_target 퍼블리시')

    def _reset_and_idle(self):
        self.object_class = None
        self.object_pose  = None
        self.empty_slot   = None
        self.target_current = 0.0
        self._transition(TaskState.IDLE)

    def _transition(self, new_state):
        self.get_logger().info(f'State: {self.state} → {new_state}')
        self.state = new_state
        self._pending_future = None
        self._tick_count = 0

    def _call_service_async(self, client, tag):
        if not client.service_is_ready():
            self.get_logger().warn(f'{tag} 서비스 미연결 → 더미 성공으로 시뮬레이션')
            f = concurrent.futures.Future()
            resp = Trigger.Response()
            resp.success = True
            resp.message = 'dummy ok'
            f.set_result(resp)
            self._pending_future = f
        else:
            self._pending_future = client.call_async(Trigger.Request())
            self.get_logger().info(f'{tag} 서비스 호출')

    def _await_service(self, client, tag, on_success, on_fail=None):
        if self._pending_future is None:
            self._call_service_async(client, tag)
        elif self._pending_future.done():
            result = self._pending_future.result()
            self._pending_future = None
            if result.success:
                self.get_logger().info(f'{tag} 완료')
                on_success()
            else:
                self.get_logger().error(f'{tag} 실패: {result.message}')
                if on_fail:
                    on_fail()
                else:
                    self._reset_and_idle()

    def _load_yaml(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.get_logger().warn(f'yaml 로드 실패: {path} → {e}')
            return None


def main(args=None):
    rclpy.init(args=args)
    node = MainControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()