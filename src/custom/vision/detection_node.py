import os
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from ultralytics import YOLO


# YOLO11s-seg fine-tuned, 4-class (bottle/can/snack_bag/bread). webcam_seg_node 와 동일 가중치.
DEFAULT_MODEL = "/home/fastcampus/Downloads/test/로봇강의_예제/_RealProject_0/doosan_ws/" \
                "src/e0509_gripper_description/models/pose_robust_seg.pt"
# 모델 클래스명 → interface_definition.md 클래스명 정규화
CLASS_MAP = {"snack_bag": "snack", "snackbag": "snack"}


class DetectionNode(Node):
    """컬러 이미지 → YOLO-seg 검출 → /object_class(클래스) + /object_pose_2d(픽셀 u,v + 마스크 yaw).

    3D 위치는 depth 가 필요하므로 pose_estimation_node 가 담당. 본 노드는 2D(클래스/픽셀/파지각)만.
    파이프라인: /camera/color/image_raw → 본 노드 → (/object_class, /object_pose_2d)
                → pose_estimation_node → /object_pose(3D, base_link).
    """

    def __init__(self):
        super().__init__('detection_node')

        self.declare_parameter('model_path', DEFAULT_MODEL)
        self.declare_parameter('conf', 0.5)
        self.model_path = self.get_parameter('model_path').value
        self.conf = float(self.get_parameter('conf').value)

        # Subscribers
        self.sub_image = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)

        # Publishers
        self.pub_object_class = self.create_publisher(String, '/object_class', 10)
        # 2D 픽셀 핸드오프(→ pose_estimation_node). position.x/y=픽셀(u,v), orientation=파지 yaw(Z회전).
        self.pub_object_pose_2d = self.create_publisher(PoseStamped, '/object_pose_2d', 10)

        self.bridge = CvBridge()
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"YOLO 모델 없음: {self.model_path}")
            raise FileNotFoundError(self.model_path)
        self.model = YOLO(self.model_path)
        self.get_logger().info(f"YOLO 로드: {self.model_path}, 클래스={self.model.names}")

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        det = self._detect(cv_image)          # 최고신뢰 검출 1개 (없으면 None)
        if det is None:
            return
        self._publish_class(det)
        self._publish_pose(det, msg.header)

    def _detect(self, image):
        """YOLO-seg 추론 → 최고신뢰 검출의 (클래스명, u, v, yaw_rad) 반환."""
        results = self.model(image, conf=self.conf, verbose=False)
        if not results:
            return None
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        confs = r.boxes.conf.cpu().numpy()
        i = int(np.argmax(confs))                       # 최고 신뢰 검출
        cls_id = int(r.boxes.cls[i].item())
        name = self.model.names[cls_id]
        name = CLASS_MAP.get(name, name)
        x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0          # bbox 중심 픽셀
        yaw = self._mask_yaw(r, i)                       # seg 마스크 주축 → 파지각
        return name, float(u), float(v), float(yaw)

    def _mask_yaw(self, result, idx):
        """세그 마스크 주축 각도(라디안). 마스크 없으면 0(=정렬 안함)."""
        try:
            if result.masks is None:
                return 0.0
            poly = result.masks.xy[idx]                  # (N,2) 픽셀 폴리곤
            if poly is None or len(poly) < 5:
                return 0.0
            (_, _), (_, _), angle = cv2.minAreaRect(poly.astype(np.float32))
            return math.radians(angle)                   # 평행그리퍼는 ±90° 대칭이라 부호 무관
        except Exception:
            return 0.0

    def _publish_class(self, det):
        name = det[0]
        msg = String()
        msg.data = name
        self.pub_object_class.publish(msg)

    def _publish_pose(self, det, header):
        """2D 픽셀 + 파지 yaw 를 PoseStamped 로 발행 (position.x/y=u,v, orientation=Z회전 yaw)."""
        _, u, v, yaw = det
        msg = PoseStamped()
        msg.header = header                              # 카메라 프레임/타임스탬프 유지
        msg.pose.position.x = u
        msg.pose.position.y = v
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)     # Z축 회전 쿼터니언
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub_object_pose_2d.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
