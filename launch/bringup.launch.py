from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # ── Vision ────────────────────────────────────────────
        # 담당: 남상훈 (detection), 이현호 (pose, pointcloud)
        Node(
            package='smart_shelf_robot',
            executable='detection_node',
            name='detection_node',
            output='screen',
        ),
        Node(
            package='smart_shelf_robot',
            executable='pose_estimation_node',
            name='pose_estimation_node',
            output='screen',
        ),
        Node(
            package='smart_shelf_robot',
            executable='pointcloud_node',
            name='pointcloud_node',
            output='screen',
        ),

        # ── Motion ────────────────────────────────────────────
        # 담당: 김민성
        # move_to_shelf_view / move_to_product_view / move_to_place / move_to_home 서비스 서버
        Node(
            package='smart_shelf_robot',
            executable='arm_controller_node',
            name='arm_controller_node',
            output='screen',
        ),
        Node(
            package='smart_shelf_robot',
            executable='gripper_node',
            name='gripper_node',
            output='screen',
        ),

        # ── Simulation (GraspGen + cuRobo) ────────────────────
        # 담당: 남정혁
        # move_to_pick 서비스 서버
        # 주의: isaacsim conda 환경에서 실행 필요
        #   ~/isaacsim/python.sh src/rl/sim_controller_node.py
        # TODO: conda 환경 문제 해결 후 아래 Node() 활성화
        # Node(
        #     package='smart_shelf_robot',
        #     executable='sim_controller_node',
        #     name='sim_controller_node',
        #     output='screen',
        # ),

        # ── Integration ───────────────────────────────────────
        # 담당: 심예영
        # 전체 상태머신 — 다른 모든 노드가 준비된 후 마지막에 시작
        Node(
            package='smart_shelf_robot',
            executable='main_controller_node',
            name='main_controller_node',
            output='screen',
        ),

    ])
