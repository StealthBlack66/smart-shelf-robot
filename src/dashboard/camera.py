"""카메라 브리지 — vision 노드의 CompressedImage(JPEG) 를 받아 MJPEG 로 중계.

영상은 절대 WebSocket(JSON)에 싣지 않는다. 별도 HTTP multipart 스트림으로 분리해
텔레메트리 그래프가 영상 부하에 영향받지 않게 한다.

vision 노드(webcam_seg_node.py)에 아래 한 줄을 추가하는 것을 전제로 한다:
    self.pub_view = self.create_publisher(CompressedImage, '/dashboard/camera/compressed', 1)
    # 매 프레임: msg.format='jpeg'; msg.data = cv2.imencode('.jpg', vis)[1].tobytes()
토픽이 없으면 카메라 패널은 '대기' 상태로 표시된다.
"""
from __future__ import annotations

import threading
from typing import Optional

from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

try:
    from sensor_msgs.msg import CompressedImage
    _IMG_AVAIL = True
except Exception:                       # pragma: no cover
    CompressedImage = None
    _IMG_AVAIL = False

CAMERA_TOPIC = '/dashboard/camera/compressed'
# 클라이언트가 없으면 빈 프레임 보여줄 1x1 placeholder (검은 점)
_PLACEHOLDER = (b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
                b'\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07'
                b'\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f')


class CameraBridge:
    """최신 JPEG 프레임을 들고 있다가 MJPEG 제너레이터에 넘긴다."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self._lock = threading.Lock()
        self._frame: Optional[bytes] = None
        self._cond = threading.Condition(self._lock)
        self._seq = 0
        self.available = _IMG_AVAIL

        if _IMG_AVAIL:
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
            node.create_subscription(CompressedImage, CAMERA_TOPIC, self._on_image, qos)
            node.get_logger().info(f'카메라 브리지: {CAMERA_TOPIC} 구독')
        else:
            node.get_logger().warn('CompressedImage 미가용 — 카메라 비활성')

    def _on_image(self, msg) -> None:
        data = bytes(msg.data)
        with self._cond:
            self._frame = data
            self._seq += 1
            self._cond.notify_all()

    def has_frame(self) -> bool:
        with self._lock:
            return self._frame is not None

    def mjpeg_generator(self, fps_cap: float = 30.0):
        """multipart/x-mixed-replace 용 프레임 제너레이터.

        새 프레임이 올 때만 보낸다(Condition 대기) → CPU 낭비 없음.
        """
        boundary = b'--frame'
        last_seq = -1
        timeout = max(1.0 / max(fps_cap, 1.0), 1.0 / 60)
        while True:
            with self._cond:
                # 새 프레임이 올 때까지 대기 (최대 1s 마다 깨서 keep-alive)
                self._cond.wait_for(lambda: self._seq != last_seq, timeout=1.0)
                frame = self._frame
                last_seq = self._seq
            if frame is None:
                frame = _PLACEHOLDER
            yield (boundary + b'\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
                   + frame + b'\r\n')
