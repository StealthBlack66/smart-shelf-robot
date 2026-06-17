import os
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge


# 핸드아이 캘리브레이션 결과 (T_cam2base 4x4 + camera_matrix). 08_핸드아이_캘리브레이션 산출물.
# object_tracking_node.py 와 동일 파일을 공유한다.
DEFAULT_CALIB = "/home/fastcampus/Downloads/test/로봇강의_예제/_RealProject_0/doosan_ws/" \
                "src/e0509_gripper_description/scripts/calibration_result.npz"


class PoseEstimationNode(Node):
    """2D 픽셀(/object_pose_2d) + depth → 로봇 base_link 기준 6D pose(/object_pose).

    파이프라인: detection(2D) → /object_pose_2d(u,v 픽셀) ─┐
                RealSense depth /camera/depth/image_rect_raw ┼→ 본 노드 → /object_pose(PoseStamped, base_link)
    - 위치: pinhole 역투영(카메라 내부파라미터) → 카메라 광학좌표 3D → T_cam2base 로 base_link 변환.
    - 자세: top-down 파지 쿼터니언(make_down_quaternion) + 입력 2D pose 가 yaw 를 실어오면 반영 → 6D.
    """

    def __init__(self):
        super().__init__('pose_estimation_node')

        # 파라미터
        self.declare_parameter('calib_path', DEFAULT_CALIB)
        self.declare_parameter('depth_scale', 0.001)   # realsense2_camera 16UC1 = mm → m
        self.declare_parameter('base_frame', 'base_link')
        calib_path = self.get_parameter('calib_path').value
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.base_frame = self.get_parameter('base_frame').value

        # 캘리브레이션 로드: 카메라→base 외부파라미터 + 내부파라미터
        self.R_cal = None
        self.t_cal = None
        self.fx = self.fy = self.cx = self.cy = None
        self._load_calibration(calib_path)

        self.bridge = CvBridge()
        self.depth_image = None      # (H,W) uint16, mm

        # Subscribers
        self.sub_depth = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.sub_object_pose_2d = self.create_subscription(
            PoseStamped, '/object_pose_2d', self.pose_2d_callback, 10)
        # camera_info 가 있으면 실시간 해상도에 맞는 내부파라미터로 덮어씀(없으면 calib 값 사용)
        self.sub_cam_info = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', self.camera_info_callback, 10)

        # Publishers
        self.pub_object_pose_3d = self.create_publisher(PoseStamped, '/object_pose', 10)

        self.get_logger().info(
            f"pose_estimation_node 시작: fx={self.fx:.1f} fy={self.fy:.1f} "
            f"cx={self.cx:.1f} cy={self.cy:.1f}, base_frame={self.base_frame}")

    # ------------------------------------------------------------------ calib
    def _load_calibration(self, path):
        if not os.path.exists(path):
            self.get_logger().error(f"캘리브레이션 파일 없음: {path}")
            raise FileNotFoundError(path)
        calib = np.load(path)
        T = calib['T_cam2base']                  # 4x4 (camera optical → base_link)
        self.R_cal = T[:3, :3]
        self.t_cal = T[:3, 3]
        K = calib['camera_matrix']               # 3x3 내부파라미터
        self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
        self.cx, self.cy = float(K[0, 2]), float(K[1, 2])

    # --------------------------------------------------------------- callbacks
    def camera_info_callback(self, msg):
        # 실제 발행 해상도의 내부파라미터로 갱신 (depth 이미지와 정합 보장)
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def depth_callback(self, msg):
        # depth 이미지 저장 (16UC1, mm)
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def pose_2d_callback(self, msg):
        if self.depth_image is None:
            self.get_logger().warn('depth 미수신 — pose 추정 skip', throttle_duration_sec=2.0)
            return
        pose_3d = self._estimate_3d_pose(msg)
        if pose_3d is not None:
            self.pub_object_pose_3d.publish(pose_3d)

    # ------------------------------------------------------------- estimation
    def _estimate_3d_pose(self, pose_2d):
        h, w = self.depth_image.shape[:2]
        u = int(round(pose_2d.pose.position.x))   # 픽셀 열
        v = int(round(pose_2d.pose.position.y))   # 픽셀 행
        if not (0 <= u < w and 0 <= v < h):
            self.get_logger().warn(f'픽셀 범위 밖 (u={u},v={v}) / 이미지 {w}x{h}')
            return None

        depth_m = self._sample_depth(u, v)
        if depth_m is None:
            self.get_logger().warn(f'유효 depth 없음 @({u},{v})', throttle_duration_sec=2.0)
            return None

        # 1) 픽셀+depth → 카메라 광학좌표 3D
        x, y, z = self._pixel_to_3d(u, v, depth_m)
        # 2) 카메라 → 로봇 base_link 변환 (핸드아이 캘리브)
        pos_base = self.R_cal @ np.array([x, y, z]) + self.t_cal

        # 3) 자세: top-down 파지 + (입력이 실어온) yaw → 6D
        grasp_angle = self._extract_yaw(pose_2d.pose.orientation)
        quat = self.make_down_quaternion(grasp_angle)   # [x,y,z,w]

        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.base_frame
        out.pose.position.x = float(pos_base[0])
        out.pose.position.y = float(pos_base[1])
        out.pose.position.z = float(pos_base[2])
        out.pose.orientation.x = quat[0]
        out.pose.orientation.y = quat[1]
        out.pose.orientation.z = quat[2]
        out.pose.orientation.w = quat[3]
        return out

    def _pixel_to_3d(self, u, v, depth):
        # 카메라 내부파라미터로 역투영 (Brown-Conrady, 왜곡 무시 근사 = object_tracking 동일)
        z = depth
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return x, y, z

    # --------------------------------------------------------------- helpers
    def _sample_depth(self, u, v, win=2):
        """(u,v) 주변 작은 창에서 0 이 아닌 depth 의 중앙값 → 노이즈/홀 견고. 단위: m."""
        h, w = self.depth_image.shape[:2]
        u0, u1 = max(0, u - win), min(w, u + win + 1)
        v0, v1 = max(0, v - win), min(h, v + win + 1)
        patch = self.depth_image[v0:v1, u0:u1].astype(np.float32)
        vals = patch[patch > 0]
        if vals.size == 0:
            return None
        depth_m = float(np.median(vals)) * self.depth_scale
        if not (0.1 <= depth_m <= 5.0):   # RealSense 유효 범위 밖이면 버림
            return None
        return depth_m

    @staticmethod
    def _extract_yaw(q):
        """입력 쿼터니언(주로 Z축 회전)에서 yaw 추출. identity 면 0."""
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return float(np.arctan2(siny, cosy))

    @staticmethod
    def make_down_quaternion(grasp_angle_rad):
        """그리퍼가 아래(-Z)를 향하고 Z축으로 grasp_angle 만큼 회전한 쿼터니언 [x,y,z,w].
        object_tracking_node.make_down_quaternion 와 동일 규약."""
        ca, sa = np.cos(grasp_angle_rad / 2), np.sin(grasp_angle_rad / 2)
        qz_w, qz_x, qz_y, qz_z = ca, 0.0, 0.0, sa        # Z축 회전
        qb_w, qb_x, qb_y, qb_z = 0.0, 0.7071, 0.7071, 0.0  # top-down 기본자세
        w = qz_w*qb_w - qz_x*qb_x - qz_y*qb_y - qz_z*qb_z
        x = qz_w*qb_x + qz_x*qb_w + qz_y*qb_z - qz_z*qb_y
        y = qz_w*qb_y - qz_x*qb_z + qz_y*qb_w + qz_z*qb_x
        z = qz_w*qb_z + qz_x*qb_y - qz_y*qb_x + qz_z*qb_w
        return [float(x), float(y), float(z), float(w)]


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
