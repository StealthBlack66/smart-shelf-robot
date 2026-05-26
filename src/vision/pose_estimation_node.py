import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')

        # Subscribers
        self.sub_depth = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.sub_object_pose_2d = self.create_subscription(
            PoseStamped, '/object_pose_2d', self.pose_2d_callback, 10)

        # Publishers
        self.pub_object_pose_3d = self.create_publisher(PoseStamped, '/object_pose', 10)

        self.depth_image = None

    def depth_callback(self, msg):
        # TODO: depth 이미지 저장
        # self.depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        pass

    def pose_2d_callback(self, msg):
        if self.depth_image is None:
            return
        pose_3d = self._estimate_3d_pose(msg)
        if pose_3d:
            self.pub_object_pose_3d.publish(pose_3d)

    def _estimate_3d_pose(self, pose_2d):
        # TODO: 2D 픽셀 좌표 + depth → 3D 좌표 변환
        # depth = self.depth_image[v, u]
        # x, y, z = self._pixel_to_3d(u, v, depth)
        # 카메라 → 로봇 좌표계 변환 (TF)
        return None

    def _pixel_to_3d(self, u, v, depth):
        # TODO: 카메라 내부 파라미터로 3D 좌표 계산
        # fx, fy, cx, cy = camera_intrinsics
        # x = (u - cx) * depth / fx
        # y = (v - cy) * depth / fy
        # z = depth
        pass


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
