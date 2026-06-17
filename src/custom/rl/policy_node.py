import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')

        # Subscribers
        self.sub_object_pose = self.create_subscription(
            PoseStamped, '/object_pose', self.object_pose_callback, 10)
        self.sub_joint_states = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)

        # Publishers
        self.pub_action = self.create_publisher(
            Float32MultiArray, '/policy/action', 10)

        # 추론 타이머 (10Hz)
        self.timer = self.create_timer(0.1, self.inference_callback)

        self.policy = None
        self.object_pose = None
        self.joint_states = None
        self._load_policy()

    def _load_policy(self):
        # TODO: 학습된 policy 로드 (ONNX or PyTorch)
        # import onnxruntime as ort
        # self.policy = ort.InferenceSession('policy.onnx')
        pass

    def object_pose_callback(self, msg):
        self.object_pose = msg

    def joint_states_callback(self, msg):
        self.joint_states = msg

    def inference_callback(self):
        if self.object_pose is None or self.joint_states is None:
            return
        obs = self._get_observation()
        action = self._infer(obs)
        if action is not None:
            self._publish_action(action)

    def _get_observation(self):
        # TODO: 관측값 벡터 구성
        # obs = [
        #     joint_positions (6),
        #     object_pose (7: x,y,z,qx,qy,qz,qw),
        # ]
        return None

    def _infer(self, obs):
        # TODO: policy 추론
        # input = {self.policy.get_inputs()[0].name: obs}
        # action = self.policy.run(None, input)[0]
        # return action
        return None

    def _publish_action(self, action):
        # TODO: 추론된 action 퍼블리시
        # msg = Float32MultiArray()
        # msg.data = action.tolist()
        # self.pub_action.publish(msg)
        pass


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
