#!/usr/bin/env python3
"""Eye-in-Hand 변환 발행 (TF 기반 — 느린 get_current_posx 우회).

T_cam2base = TF(base_link → link_6=flange) @ T_cam2gripper.
TF 는 joint_states(RT 루프)에서 빠르게 갱신되므로, 컨트롤러 서비스
(get_current_posx)가 느려도(1.6s) 영향 없이 10Hz 발행.
검증: link_6 위치(0.040,0.014,0.746) == get_current_posx 위치 정확히 일치.
"""
import numpy as np
import time
import sys
import os

# 캘리브 실제 위치(다운로드 루트). 구 경로(hand_eye_calibration/...)는 없어서 수정.
_CALIB_CANDIDATES = [
    "~/Downloads/eye_in_hand_result.npz",
    "~/Downloads/hand_eye_calibration/calibration_data/eye_in_hand_result.npz",
]
CALIB_FILE = next(
    (p for p in _CALIB_CANDIDATES if os.path.exists(os.path.expanduser(p))),
    _CALIB_CANDIDATES[0])

try:
    import rclpy
    from std_msgs.msg import Float64MultiArray
    import tf2_ros
    from tf2_ros import TransformException
    import threading
except ImportError:
    print("ROS2 환경 필요"); sys.exit(1)


def quat_to_rotm(qx, qy, qz, qw):
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]])


def main():
    d = np.load(os.path.expanduser(CALIB_FILE))
    T_cam2gripper = d['T_cam2gripper'].copy()
    print(f"캘리브레이션 로드: {CALIB_FILE}", flush=True)

    rclpy.init()
    node = rclpy.create_node('eih_fk_publisher')
    pub = node.create_publisher(Float64MultiArray, '/eih/T_cam2base', 10)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)

    # TF 수신용 백그라운드 spin
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    base_frame = os.environ.get('EIH_BASE_FRAME', 'base_link')
    flange_frame = os.environ.get('EIH_FLANGE_FRAME', 'link_6')
    print(f"TF {base_frame} → {flange_frame} 기반 발행 시작 (posx 우회)", flush=True)

    first = True
    miss = 0
    rate_hz = 10.0
    while rclpy.ok():
        try:
            tr = tf_buffer.lookup_transform(
                base_frame, flange_frame, rclpy.time.Time())
        except TransformException:
            miss += 1
            if miss in (20, 100):
                print(f"TF 대기중... ({base_frame}→{flange_frame})", flush=True)
            time.sleep(0.1)
            continue
        miss = 0
        t = tr.transform.translation
        q = tr.transform.rotation
        T_g2b = np.eye(4)
        T_g2b[:3, :3] = quat_to_rotm(q.x, q.y, q.z, q.w)
        T_g2b[:3, 3] = [t.x, t.y, t.z]
        T_c2b = T_g2b @ T_cam2gripper

        msg = Float64MultiArray()
        msg.data = T_c2b.flatten().tolist()
        pub.publish(msg)
        if first:
            print(f"[TF] 첫 발행: flange=({t.x*1000:.1f},{t.y*1000:.1f},"
                  f"{t.z*1000:.1f})mm  cam=({T_c2b[0,3]*1000:.1f},"
                  f"{T_c2b[1,3]*1000:.1f},{T_c2b[2,3]*1000:.1f})mm", flush=True)
            first = False
        time.sleep(1.0 / rate_hz)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
