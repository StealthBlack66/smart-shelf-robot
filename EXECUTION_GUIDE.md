# Smart Shelf Robot: 편의점 매대 정리 로봇 실행 가이드 (Execution Guide)

이 문서는 **두산 E0509 협동로봇**과 **로보티즈 RH-P12-RN-A 스마트 그리퍼**를 활용하여 편의점 매대를 자동 정리하는 로봇 시스템의 개발, 데이터 연동, 그리고 통합 테스트를 수행하기 위한 종합 실행 가이드입니다.

본 가이드는 프로젝트의 외부 의존성 패키지(`src/external`)와 자체 구현 노드(`src/custom`), 그리고 글로벌 오픈 데이터셋(MetaGraspNetV2, LeRobot, Bridge, Fractal)의 활용 방안을 유기적으로 연계하여 프로젝트를 완수할 수 있는 표준 운영 가이드를 제공합니다.

---

## 📌 목차
1. [시스템 아키텍처 및 폴더 구조](#1-시스템-아키텍처-및-폴더-구조)
2. [외부 의존성 패키지 검토 (src/external)](#2-외부-의존성-패키지-검토-srcexternal)
3. [오픈 비전 및 매니퓰레이터 데이터셋 활용방안](#3-오픈-비전-및-매니퓰레이터-데이터셋-활용방안)
4. [자체 개발 핵심 노드 검토 (src/custom)](#4-자체-개발-핵심-노드-검토-srccustom)
5. [환경 구축 및 의존 패키지 빌드](#5-환경-구축-및-의존-패키지-빌드)
6. [시뮬레이션 및 에뮬레이터 테스트 (Virtual Mode)](#6-시뮬레이션-및-에뮬레이터-테스트-virtual-mode)
7. [실제 하드웨어 및 디지털 트윈 제어 (Real Mode)](#7-실제-하드웨어-및-디지털-트윈-제어-real-mode)
8. [비전 및 모션 통합 테스트 가이드](#8-비전-및-모션-통합-테스트-가이드)

---

## 1. 시스템 아키텍처 및 폴더 구조

이 프로젝트는 센서 데이터 획득(Vision), 경로 계획 및 액션 제어(Motion), 강화학습 및 모방학습 데이터 구축(RL), 그리고 상태 머신(Integration)으로 구성되어 있습니다.

```
smart-shelf-robot/
├── src/
│   ├── custom/                      # 자체 구현 노드 패키지
│   │   ├── vision/                  # YOLO 감지, 3D 포즈 추정, 캘리브레이션 노드
│   │   ├── motion/                  # 조인트 제어(cuRobo) 및 Modbus 그리퍼 노드
│   │   ├── rl/                      # LeRobot 데이터 레코더, 강화학습 정책 노드
│   │   └── integration/             # 메인 FSM, VLA 브릿지, 비상 안전 가드 노드
│   └── external/                    # 외부 의존성 서브 패키지
│       ├── doosan-robot2/           # 두산 로봇 ROS2 Humble 드라이버 (Flange Serial 지원)
│       ├── RH-P12-RN-A/             # 로보티즈 그리퍼 드라이버 및 Description
│       ├── e0509_gripper_description/ # E0509 + 그리퍼 결합 URDF, Launch, cuRobo 설정
│       └── handeye_calibration_ros2/ # ROS2 기반 Hand-Eye 캘리브레이션 유틸리티
├── config/                          # 그리퍼 파지력 및 가판대 목표 배치 설정 파일
├── docs/                            # 프로젝트 명세서 및 가이드 문서
├── launch/                          # ROS2 Launch 스크립트 모음
└── requirements/                    # 파이썬 가상환경 설치 요구사항
```

---

## 2. 외부 의존성 패키지 검토 (src/external)

프로젝트 빌드 전, `src/external` 내부의 패키지들이 담당하는 역할과 기능 사양을 파악해야 합니다.

1. **`doosan-robot2`**
   * **역할**: 두산 로봇의 ROS2 Humble 공식 드라이버 인터페이스 패키지입니다.
   * **주요 기능**: 가상 시뮬레이션 제어(Virtual Mode용 Docker 에뮬레이터 지원) 및 실제 로봇 하드웨어 통신을 지원합니다.
   * **특징**: 그리퍼와 Modbus RTU 통신을 연동하기 위해 로봇의 **Tool Flange Serial** 통신 서비스를 지원하는 커스텀 포크 버전(Forked repository)으로 셋업되어 있습니다.
2. **`RH-P12-RN-A`**
   * **역할**: 로보티즈 스마트 그리퍼의 하드웨어 드라이버 및 조인트 묘사(Description) 패키지입니다.
3. **`e0509_gripper_description`**
   * **역할**: 두산 E0509 로봇(6축)과 로보티즈 그리퍼(4축)를 물리적으로 결합한 10축 통합 로봇 시스템 설정입니다.
   * **주요 기능**: Gazebo 물리 시뮬레이션 연동, RViz 시각화, **cuRobo 기반 GPU 가속 모션 플래닝**, Modbus RTU 기반 실제 그리퍼 서비스(`gripper_service_node`) 및 디지털 트윈 연동 브릿지를 총괄 제공합니다.
4. **`handeye_calibration_ros2`**
   * **역할**: 카메라의 상대적 위치(Eye-to-Hand 구조)와 로봇 베이스 프레임 간의 정확한 위치 변환 행렬($T_{\text{cam2base}}$)을 산출해 주는 캘리브레이션 유틸리티입니다.

---

## 3. 오픈 비전 및 매니퓰레이터 데이터셋 활용방안

편의점 매대 위 물품(빵, 캔, 음료수, 과자 등)은 물성이 다양하며 겹쳐있는 경우가 많습니다. 본 프로젝트에서는 이 문제를 극복하기 위해 아래와 같이 글로벌 오픈 데이터셋을 연계하여 활용합니다.

### ① 비전 데이터셋: MetaGraspNetV2
* **데이터셋 개요**: 다양한 재질과 형태의 일상 용품이 무작위로 쌓여 있는(cluttered) 환경에 대한 정밀한 3D 바운딩 박스, 물체 분할(Segmentation) 마스크, 그리고 파지점 후보군(Grasp Labels)을 담은 대형 데이터셋입니다.
* **프로젝트 활용**: 
  * `src/custom/vision/detection_node.py` 내의 YOLO 모델 학습 및 튜닝 시 편의점 물품 클래스 학습 데이터로 전이학습(Transfer Learning)을 수행합니다.
  * 물체가 겹쳐 있거나 난반사가 심한 포장지(과자 봉지 등)에 대한 3D 마스크 마찰 영역을 정밀하게 추출하여 파지 실패율을 현격히 줄일 수 있습니다.

### ② 매니퓰레이터 모방학습 데이터셋: LeRobot & Parquet 포맷
* **데이터셋 개요**: Hugging Face의 LeRobot 표준에 따라 이미지 프레임(카메라 피드)과 로봇 조인트 위치/속도/그리퍼 액션을 30Hz 주기로 동기화하여 이진 parquet 파일 및 mp4 동영상으로 저장하는 포맷입니다.
* **프로젝트 활용**:
  * `src/custom/rl/doosan_lerobot_dataset_builder.py` 스크립트를 사용하여 작업자가 원격 제어(Teleoperation)하거나 사전에 조작된 경로로 움직이는 두산 로봇의 시연(Demonstration) 데이터를 축적합니다.
  * 저장된 데이터는 OpenVLA 등 파운데이션 모델에 주입 가능한 포맷으로, 매대 정리 태스크 모방학습(Imitation Learning)의 학습 자산이 됩니다.

### ③ VLA 파인튜닝 데이터셋: Bridge & Fractal
* **데이터셋 개요**: 로봇 팔 조작 태스크 연구를 위해 구축된 대용량 멀티모달 제어 데이터셋(Bridge Dataset v2, Fractal2021 등)입니다. 다양한 환경에서의 RGB 이미지, 조종 궤적, 텍스트 작업 지시어가 결합되어 있습니다.
* **프로젝트 활용**:
  * 사전 학습된 VLA(Vision-Language-Action) 모델의 Few-shot 가이드라인으로 활용하여, 로봇이 "캔 음료를 집어 첫 번째 칸에 진열해줘"와 같은 텍스트 지시어를 입력받아 즉각적인 Pick-and-Place 동작 정책(Policy)을 추론할 수 있게 유도합니다.

---

## 4. 자체 개발 핵심 노드 검토 (src/custom)

1. **`vision`**
   * **`detection_node.py`**: RealSense 카메라 토픽을 받아 YOLO-seg 추론을 실행하고, 검출 대상의 2D 픽셀 좌표(u, v) 및 마스크 주축 분석(PCA)을 활용한 파지각(yaw)을 발행합니다.
   * **`pose_estimation_node.py`**: 2D 픽셀과 깊이(depth) 값을 맵핑하고, 핸드아이 캘리브레이션 변환을 적용하여 로봇 베이스 프레임 기준의 3D 목표 좌표(x, y, z)로 보정해 발행합니다.
2. **`motion`**
   * **`arm_controller_node.py`**: 목표 3D 포즈를 입력받아 조인트 명령을 생성하며, 충돌 회피를 위한 cuRobo 모션 플래너 스크립트와 동기화됩니다.
   * **`gripper_node.py`**: Modbus RTU 또는 ROS2 토픽 기반 그리퍼 스트로크 제어를 중계합니다.
3. **`rl`**
   * **`doosan_lerobot_dataset_builder.py`**: 실시간 로봇 상태 및 비디오 피드를 parquet 포맷으로 레코딩하여 Hugging Face LeRobot 학습 데이터셋을 빌드합니다.
   * **`policy_node.py`**: 학습 완료된 RL/IL 모델 가중치에 기반해 실시간 액션을 추론하여 로봇에 인가합니다.
4. **`integration`**
   * **`main_controller_node.py`**: 전체 시나리오 상태 머신(대기 → 비전 탐색 → cuRobo 경로 계획 → 그리퍼 파지 → 매대 이동 → 하강 및 정렬 → 파지 해제 → 복귀)을 컨트롤합니다.
   * **`emergency_safety_guard.py`**: 하드웨어 충돌 경보, 오동작 이상 상태를 감지하여 즉시 로봇 동작을 홀드(정지)시키는 감시 노드입니다.

---

## 5. 환경 구축 및 의존 패키지 빌드

### ① ROS2 Humble 가상환경 및 ROS2 환경 변수 세팅
```bash
# 콘다 가상환경 활성화
conda activate isaaclab

# ROS2 환경 변수 소싱
source /opt/ros/humble/setup.bash

# 도산 워크스페이스(doosan_ws) 폴더 생성 및 소스 이동
mkdir -p ~/doosan_ws/src
cp -r ~/smart-shelf-robot ~/doosan_ws/src/
cd ~/doosan_ws
```

### ② 의존성 패키지 설치 및 빌드
```bash
# rosdep을 통한 시스템 라이브러리 및 드라이버 의존성 설치
rosdep install --from-paths src --ignore-src -r -y

# colcon 빌드 실행
colcon build --symlink-install
source install/setup.bash
```

---

## 6. 시뮬레이션 및 에뮬레이터 테스트 (Virtual Mode)

실제 로봇 구동 전에 에뮬레이터 Docker 컨테이너와 Gazebo 환경을 활용하여 가상 테스트를 진행합니다.

### ① Virtual 에뮬레이터 기동 (Docker 필수)
```bash
# docker 권한 등록 상태 확인 후 에뮬레이터 시작
cd ~/doosan_ws/src/smart-shelf-robot/src/external/doosan-robot2
chmod +x ./install_emulator.sh
sudo ./install_emulator.sh

# 가상 로봇 에뮬레이터 실행
ros2 launch e0509_gripper_description bringup.launch.py mode:=virtual
```

### ② Gazebo 시뮬레이터와 가상 조인트 구동
```bash
# 터미널 2: Gazebo Fortress 환경 및 로봇 모델 팝업
ros2 launch e0509_gripper_description gazebo.launch.py

# 터미널 3: 조인트 제어 Trajectory 테스트 명령 전송
ros2 topic pub --once /e0509_gripper/joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
  joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6],
  points: [{positions: [0.5, 0.3, 0.3, 0.0, 0.5, 0.0], time_from_start: {sec: 2}}]
}"
```

---

## 7. 실제 하드웨어 및 디지털 트윈 제어 (Real Mode)

실제 로봇 IP 주소를 기반으로 연동하여 가동하는 순서입니다.

### ① 실제 두산 로봇 및 그리퍼 구동
그리퍼는 로봇의 **Tool Flange Serial** 포트에 연결되어 있으며 `Baudrate: 57600`, `Slave ID: 1` 상태의 **Modbus RTU**로 제어됩니다.
```bash
# 실제 로봇과 연동하여 로봇 드라이버 및 그리퍼 Modbus 제어 노드 구동
ros2 launch e0509_gripper_description bringup.launch.py mode:=real host:=<ROBOT_IP>
```

### ② 디지털 트윈 동기화 실행 (Real Robot ➔ Isaac Sim)
실제 로봇의 움직임을 Isaac Sim 및 RViz 환경에 실시간 미러링(동기화)하여 디지털 트윈 모니터링 시스템을 시작합니다.
```bash
# 터미널 2: ROS2 -> JSON 파일 기반 브릿지 실행 (30Hz 동기화)
python3 ~/doosan_ws/src/smart-shelf-robot/src/external/e0509_gripper_description/scripts/digital_twin_bridge.py

# 터미널 3: Isaac Sim 가상환경에서 디지털 트윈 클로즈드 루프 실행
# (Isaac Sim python.sh 경로를 반드시 활용하십시오)
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh ~/doosan_ws/src/smart-shelf-robot/src/external/e0509_gripper_description/scripts/digital_twin.py
```

---

## 8. 비전 및 모션 통합 테스트 가이드

모든 시스템 구축이 정상 완료되면, 비전 감지부터 적재 완수까지의 시나리오 루프를 트리거하고 모니터링합니다.

### ① 전체 시스템 런칭 (Launch)
```bash
# RealSense 카메라 드라이버, YOLO 감지 노드, 3D Pose 노드, cuRobo 제어, FSM 통합 기동
ros2 launch smart-shelf-robot bringup.launch.py
```

### ② 모니터링 및 디버깅 팁
1. **토픽 통신 확인**:
   ```bash
   # 감지된 물체 정보 확인
   ros2 topic echo /object_class
   
   # 캘리브레이션이 완료된 로봇 베이스 기준의 물체 3D 공간 좌표
   ros2 topic echo /object_pose
   ```
2. **비상 제어 및 모션 계획 에러**:
   * cuRobo 연산 오차 혹은 모션 도중 가판대 충돌 위기 상황이 감지되면 `emergency_safety_guard` 노드가 실시간으로 판단하여 로봇의 속도를 0으로 긴급 강하(Hold)시킵니다.
   * 이때는 즉시 펜던트 조작기나 Rviz에서 홈 자세로 롤백 명령 서비스를 트리거하십시오.
   ```bash
   ros2 service call /dsr01/motion/move_joint dsr_msgs2/srv/MoveJoint "{pos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vel: 30.0, acc: 30.0}"
   ```
