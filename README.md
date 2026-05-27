# smart-shelf-robot

상품을 감지하고 물성에 맞는 파지힘으로 매대에 자동 정리하는 로봇 시스템

- **로봇**: Doosan E0509 + RH-P12-RN-A 그리퍼
- **센서**: Intel RealSense D435 x2 (Eye-in-hand / Eye-to-hand)
- **환경**: Ubuntu 22.04 + ROS2 Humble

---

## 환경 스펙

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 22.04 |
| GPU| NVIDIA RTX 5080 Laptop (16GB VRAM) |
| NVIDIA 드라이버 | 580.x 이상 (535 이상 필수) |
| CUDA | 12.8 (호스트 cuRobo용) / 13.0 (Isaac Sim 내장) |
| Isaac Sim | 5.1.0-rc.19 |
| IsaacLab | 2.3.2 |
| Python | 3.11.13 (Isaac Sim 내장) / 3.10 (시스템) |
| PyTorch | 2.7.0+cu128 |

---

## 팀 구성 및 담당 브랜치

| 이름 | 역할 | 담당 브랜치 |
|------|------|------------|
| 심예영 | 제어 통합 | `feat/integration-controller` |
| 김민성 | 모션플래닝, 캘리브레이션 | `feat/motion` |
| 남상훈 | 비전 (감지) | `feat/vision1` |
| 이현호 | 비전 (포즈) | `feat/vision2` |
| 김인영 | sim2real | `feat/simtoreal` |
| 남정혁 | isaacsim/isaaclab 강화학습 | `feat/rl` |

---

## 디렉토리 구조

```
smart-shelf-robot/
├── src/
│   ├── vision/      # -> 비전팀
│   │   ├── detection_node.py       # 물체 감지 (YOLO)
│   │   └── pose_estimation_node.py # 3D 포즈 추정
│   ├── motion/      # -> 모션플래닝
│   │   ├── arm_controller_node.py  # 로봇 팔 경로 계획
│   │   └── gripper_node.py         # 그리퍼 파지힘 제어
│   ├── rl/          # -> 강화학습팀
│   │   └── policy_node.py          # 강화학습 policy 추론
│   └── integration/
│       └── main_controller_node.py # 전체 상태머신
├── launch/
│   └── bringup.launch.py           # 전체 시스템 실행
├── config/
│   ├── grasp_force_params.yaml     # 물성별 파지힘 파라미터
│   └── place_targets.yaml          # 매대 적재 위치 설정
└── docs/
    └── interface_definition.md     # 토픽/서비스 인터페이스 정의
```

---

## 시작하기

### 1. 레포 클론

```bash
cd ~/doosan_ws/src
git clone https://github.com/username/smart-shelf-robot.git
cd smart-shelf-robot
```

### 2. 의존성 설치

```bash
# 호스트 환경
pip install -r requirements/requirements_host.txt

# Isaac Sim 환경
~/isaacsim/python.sh -m pip install -r requirements/requirements_isaac.txt

# ROS2 의존성
rosdep install --from-paths src --ignore-src -r -y
```

### 3. 담당 브랜치로 체크아웃

```bash
# 본인 담당 브랜치로 이동 (예: 강화학습 담당)
git checkout feat/rl
```

### 4. ROS2 환경 설정

```bash
# 모든 팀원 동일하게 설정
export ROS_DOMAIN_ID=x
source /opt/ros/humble/setup.bash
source ~/doosan_ws/install/setup.bash
```

---

## 네트워크 설정

같은 와이파이 + 같은 ROS_DOMAIN_ID면 자동으로 통신

```bash
# ~/.bashrc에 추가
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 연결 확인
ros2 node list
```

---

## 브랜치 규칙

```
main     → 최종 발표용. PR만 허용, 직접 push 금지
develop  → 통합 테스트용. feat 브랜치에서 PR
feat/*   → 각자 작업 브랜치
```

**작업 흐름:**
```bash
# 1. develop 최신 상태로 동기화
git checkout develop
git pull origin develop

# 2. 담당 브랜치로 이동
git checkout feat/vision-detection

# 3. 작업 후 커밋
git add .
git commit -m "feat: Add YOLO object detection"

# 4. push 후 GitHub에서 develop으로 PR
git push origin feat/vision-detection
```

**커밋 컨벤션:**
```
feat:     새 기능
fix:      버그 수정
docs:     문서 수정
refactor: 리팩토링
chore:    설정, 의존성 등
```

---

## 인터페이스

자세한 토픽/서비스 정의는 `docs/interface_definition.md` 참고.

**핵심 토픽:**
```
/object_class   → 감지된 물체 클래스 (bread/snack/bottle/can)
/object_pose    → 물체 3D 위치 및 자세
/grasp_force    → 파지힘 (N)
/place_target   → 매대 적재 목표 위치
/policy/action  → 강화학습 policy 출력
```

---

## TODO 구현 방법

각 노드 파일에 `# TODO:` 주석이 있는 곳만 구현하면 됩니다.

```python
def _detect(self, image):
    # TODO: YOLO 추론 후 detection 결과 반환
    # results = self.model(image)
    # return results
    return []
```

주석 해제하고 본인 코드로 채우면 됩니다.

---

## 통합 테스트 (3주차 1일차)

```bash
# 전체 시스템 실행
ros2 launch smart-shelf-robot bringup.launch.py

# 토픽 모니터링
rqt_graph

# 각 토픽 확인
ros2 topic echo /object_class
ros2 topic echo /object_pose
ros2 topic hz /object_pose
```
