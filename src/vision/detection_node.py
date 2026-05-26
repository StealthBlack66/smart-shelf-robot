import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        # Subscribers
        self.sub_image = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)

        # Publishers
        self.pub_object_class = self.create_publisher(String, '/object_class', 10)
        self.pub_object_pose = self.create_publisher(PoseStamped, '/object_pose', 10)

        self.bridge = CvBridge()
        self.model = None
        self._load_model()

    def _load_model(self):
        # TODO: YOLO 모델 로드
        # self.model = YOLO('best.pt')
        pass

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        detections = self._detect(cv_image)
        for det in detections:
            self._publish_class(det)
            self._publish_pose(det)

    def _detect(self, image):
        # TODO: YOLO 추론 후 detection 결과 반환
        # results = self.model(image)
        # return results
        return []

    def _publish_class(self, detection):
        # TODO: 감지된 물체 클래스 퍼블리시
        # msg = String()
        # msg.data = detection.class_name
        # self.pub_object_class.publish(msg)
        pass

    def _publish_pose(self, detection):
        # TODO: 감지된 물체 3D 포즈 퍼블리시
        # msg = PoseStamped()
        # msg.pose.position.x = ...
        # self.pub_object_pose.publish(msg)
        pass


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
