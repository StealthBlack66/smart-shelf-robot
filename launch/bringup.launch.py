#!/usr/bin/env python3
"""smart-shelf-robot 통합 bringup (e0509 bringup 인라인 + 그리퍼 브리지 교체).

e0509_gripper_description/bringup.launch.py 를 통째로 인라인해서:
  - 기존 e0509 gripper_service_node 제거
  - dsr_gripper_tcp gripper_service_node(브리지)로 교체  → 그리퍼 노드 중복 없음(TCP 연결 정상)

빌드 전 실행(파일 경로 직접):
  source /opt/ros/humble/setup.bash
  source ~/doosan_ws/install/setup.bash      # 로봇 드라이버 + 그리퍼 브리지(dsr_gripper_tcp) 통합
  export ROS_DOMAIN_ID=100
  ros2 launch <이 파일 절대경로> mode:=real host:=110.120.1.32 rt_host:=110.120.1.16 \
      controller_host:=<게이트웨이IP>

남은 e0509_gripper_description 의존(URDF/시각화 — 완전제거는 URDF 마이그레이션 필요):
  - urdf/e0509_with_gripper.urdf.xacro  (robot_description)
  - gripper_joint_publisher             (RViz용 그리퍼 관절 발행)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import (Command, LaunchConfiguration,
                                   PathJoinSubstitution, PythonExpression)
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    ARGUMENTS = [
        DeclareLaunchArgument('name',  default_value='dsr01',      description='NAME_SPACE'),
        DeclareLaunchArgument('host',  default_value='110.120.1.32', description='ROBOT_IP'),
        DeclareLaunchArgument('port',  default_value='12345',      description='ROBOT_PORT'),
        DeclareLaunchArgument('mode',  default_value='real',       description='OPERATION MODE'),
        DeclareLaunchArgument('model', default_value='e0509',      description='ROBOT_MODEL'),
        DeclareLaunchArgument('color', default_value='white',      description='ROBOT_COLOR'),
        DeclareLaunchArgument('rt_host', default_value='110.120.1.16', description='ROBOT_RT_IP'),
        DeclareLaunchArgument('rviz',  default_value='true',       description='Launch RViz'),
        # 그리퍼 브리지(dsr_gripper_tcp) TCP 게이트웨이 — 기본값=로봇 host(110.120.1.32)
        # (그리퍼 TCP 게이트웨이가 로봇 컨트롤러에 있으므로 host 를 자동 추종)
        DeclareLaunchArgument('controller_host', default_value=LaunchConfiguration('host'), description='GRIPPER_TCP_HOST (기본=robot host)'),
        DeclareLaunchArgument('tcp_port',        default_value='20002',        description='GRIPPER_TCP_PORT'),
    ]

    # robot_description 은 e0509 xacro 사용 (URDF — 로봇 인프라)
    pkg_path = get_package_share_directory('e0509_gripper_description')
    xacro_file = os.path.join(pkg_path, 'urdf', 'e0509_with_gripper.urdf.xacro')

    mode = LaunchConfiguration('mode')
    rviz = LaunchConfiguration('rviz')
    controller_host = LaunchConfiguration('controller_host')
    tcp_port = LaunchConfiguration('tcp_port')

    robot_description_content = Command([
        'xacro ', xacro_file,
        ' name:=', LaunchConfiguration('name'),
        ' host:=', LaunchConfiguration('host'),
        ' rt_host:=', LaunchConfiguration('rt_host'),
        ' port:=', LaunchConfiguration('port'),
        ' mode:=', LaunchConfiguration('mode'),
        ' model:=', LaunchConfiguration('model'),
        ' color:=', LaunchConfiguration('color'),
        ' update_rate:=100',
    ])

    robot_controllers = [
        PathJoinSubstitution([
            FindPackageShare("dsr_controller2"), "config", "dsr_controller2.yaml",
        ])
    ]

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare("dsr_description2"), "rviz", "default.rviz"
    ])

    # Emulator (virtual 모드 전용)
    run_emulator_node = Node(
        package="dsr_bringup2", executable="run_emulator",
        namespace=LaunchConfiguration('name'),
        parameters=[
            {"name": LaunchConfiguration('name')}, {"rate": 100}, {"standby": 5000},
            {"command": True}, {"host": LaunchConfiguration('host')},
            {"port": LaunchConfiguration('port')}, {"mode": LaunchConfiguration('mode')},
            {"model": LaunchConfiguration('model')}, {"gripper": "none"}, {"mobile": "none"},
            {"rt_host": LaunchConfiguration('rt_host')},
        ],
        condition=IfCondition(PythonExpression(["'", mode, "' == 'virtual'"])),
        output="screen",
    )

    # Controller Manager
    control_node = Node(
        package="controller_manager", executable="ros2_control_node",
        namespace=LaunchConfiguration('name'),
        parameters=[{"robot_description": robot_description_content}] + robot_controllers,
        output="both",
    )

    # Robot State Publisher
    robot_state_pub_node = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', namespace=LaunchConfiguration('name'),
        output='both', parameters=[{'robot_description': robot_description_content}],
    )

    # Gripper Joint State Publisher (RViz 시각화용 — e0509 의존)
    gripper_joint_pub_node = Node(
        package='e0509_gripper_description', executable='gripper_joint_publisher',
        name='gripper_joint_publisher', namespace=LaunchConfiguration('name'),
        output='screen',
    )

    # ★ 그리퍼 서비스 = dsr_gripper_tcp 브리지 (e0509 gripper_service_node 대체)
    #   서비스: /gripper_service/set_position(0=열기), 액션: /gripper_service/safe_grasp(파지)
    #   namespace 안 줌 → 서비스가 /gripper_service/... (curobo/webcam_seg 코드와 일치)
    #   'namespace' 파라미터(dsr01)는 브리지가 호출하는 로봇 서비스(set_robot_mode 등)용
    gripper_service_node = Node(
        package='dsr_gripper_tcp', executable='gripper_service_node',
        name='gripper_service', output='screen',
        parameters=[{
            'controller_host': controller_host,
            'tcp_port': tcp_port,
            'namespace': 'dsr01',
            'initialize_on_start': True,
        }],
    )

    # RViz (조건부)
    rviz_node = Node(
        package="rviz2", executable="rviz2", namespace=LaunchConfiguration('name'),
        name="rviz2", output="log", arguments=["-d", rviz_config_file],
        condition=IfCondition(rviz),
    )

    # Joint State Broadcaster
    joint_state_broadcaster_spawner = Node(
        package="controller_manager", namespace=LaunchConfiguration('name'),
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "controller_manager",
                   "--controller-manager-timeout", "120"],
    )

    # Doosan Controller
    robot_controller_spawner = Node(
        package="controller_manager", namespace=LaunchConfiguration('name'),
        executable="spawner",
        arguments=["dsr_controller2", "-c", "controller_manager",
                   "--controller-manager-timeout", "120"],
    )

    # control_node 를 7초 지연(emulator 초기화 대기, virtual)
    delayed_control_node = TimerAction(period=7.0, actions=[control_node])

    # control_node 시작 후 5초 뒤 joint_state_broadcaster
    delay_jsb_after_control_node = RegisterEventHandler(
        OnProcessStart(target_action=control_node,
                       on_start=[TimerAction(period=5.0,
                                             actions=[joint_state_broadcaster_spawner])]))

    # jsb 준비 후 controller spawn
    delay_controller = RegisterEventHandler(
        OnProcessExit(target_action=joint_state_broadcaster_spawner,
                      on_exit=[robot_controller_spawner]))

    # controller 준비 후 RViz + 그리퍼 브리지(로봇 서비스 필요하므로 controller 뒤에)
    delay_rviz = RegisterEventHandler(
        OnProcessExit(target_action=robot_controller_spawner,
                      on_exit=[rviz_node, gripper_service_node]))

    nodes = [
        run_emulator_node,
        robot_state_pub_node,
        gripper_joint_pub_node,
        delayed_control_node,
        delay_jsb_after_control_node,
        delay_controller,
        delay_rviz,
    ]

    return LaunchDescription(ARGUMENTS + nodes)
