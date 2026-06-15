#!/usr/bin/env python3
"""
EIH FK 퍼블리셔 — GetCurrentPosx/Rotm으로 T_cam2base 계산해서 토픽 발행.
verify_calibration.py와 동일한 방식으로 서비스 호출.
"""
import numpy as np
import time
import sys
import os

CALIB_FILE = os.path.expanduser(
    "~/Downloads/hand_eye_calibration/calibration_data/eye_in_hand_result.npz")

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from dsr_msgs2.srv import GetCurrentPosx, GetCurrentRotm
except ImportError:
    print("ROS2 환경 필요"); sys.exit(1)


def main():
    d = np.load(os.path.expanduser(CALIB_FILE))
    T_cam2gripper = d['T_cam2gripper'].copy()
    print(f"캘리브레이션 로드: {CALIB_FILE}")

    rclpy.init()
    node = rclpy.create_node('eih_fk_publisher')
    cli_posx = node.create_client(GetCurrentPosx, '/dsr01/aux_control/get_current_posx')
    cli_rotm = node.create_client(GetCurrentRotm, '/dsr01/aux_control/get_current_rotm')
    pub = node.create_publisher(Float64MultiArray, '/eih/T_cam2base', 10)

    print("서비스 대기 중...")
    cli_posx.wait_for_service(timeout_sec=15.0)
    cli_rotm.wait_for_service(timeout_sec=15.0)
    print("연결 완료 — FK 퍼블리싱 시작")

    first = True
    while rclpy.ok():
        try:
            f_p = cli_posx.call_async(GetCurrentPosx.Request())
            rclpy.spin_until_future_complete(node, f_p, timeout_sec=1.0)
            res_p = f_p.result()
            if res_p is None or not res_p.success:
                time.sleep(0.1); continue
            pd = res_p.task_pos_info[0].data
            t = np.array([pd[0]/1000.0, pd[1]/1000.0, pd[2]/1000.0])

            f_r = cli_rotm.call_async(GetCurrentRotm.Request())
            rclpy.spin_until_future_complete(node, f_r, timeout_sec=1.0)
            res_r = f_r.result()
            if res_r is None or not res_r.success:
                time.sleep(0.1); continue
            Rot = np.array([list(row.data) for row in res_r.rot_matrix])

            T_g2b = np.eye(4); T_g2b[:3, :3] = Rot; T_g2b[:3, 3] = t
            T_c2b = T_g2b @ T_cam2gripper

            msg = Float64MultiArray()
            msg.data = T_c2b.flatten().tolist()
            pub.publish(msg)

            if first:
                print(f"[FK] 첫 발행: TCP=({t[0]*1000:.1f},{t[1]*1000:.1f},{t[2]*1000:.1f})mm")
                first = False
        except Exception as e:
            print(f"[FK] 오류: {e}")
        time.sleep(0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
