# Interface Definition

## Topics

| 토픽 | 타입 | 발행자 | 구독자 | 설명 |
|------|------|--------|--------|------|
| `/object_class` | `std_msgs/String` | vision | integration, gripper | 감지된 물체 클래스 (bread/snack/bottle/can) |
| `/object_pose` | `geometry_msgs/PoseStamped` | vision | integration, motion | 물체 3D 위치 및 자세 |
| `/grasp_force` | `std_msgs/Float32` | integration | gripper | 물성 기반 파지힘 (N) |
| `/place_target` | `geometry_msgs/PoseStamped` | integration | motion | 매대 적재 목표 위치 |
| `/policy/action` | `std_msgs/Float32MultiArray` | rl | integration | 강화학습 policy 출력 |
| `/joint_command` | `sensor_msgs/JointState` | motion | robot | 로봇 관절 명령 |
| `/gripper_command` | `sensor_msgs/JointState` | gripper | robot | 그리퍼 관절 명령 |

## Services

| 서비스 | 타입 | 서버 | 클라이언트 | 설명 |
|--------|------|------|-----------|------|
| `/move_to_pick` | `std_srvs/Trigger` | motion | integration | 파지 위치로 이동 |
| `/move_to_place` | `std_srvs/Trigger` | motion | integration | 적재 위치로 이동 |
| `/move_to_home` | `std_srvs/Trigger` | motion | integration | 홈 포지션으로 이동 |
| `/gripper/open` | `std_srvs/Trigger` | gripper | integration | 그리퍼 열기 |
| `/gripper/close` | `std_srvs/Trigger` | gripper | integration | 그리퍼 닫기 |
| `/gripper/grasp` | `std_srvs/Trigger` | gripper | integration | 물성 기반 파지 |

## Object Classes

| 클래스 | 파지힘 (N) | 최대 파지힘 (N) | 비고 |
|--------|-----------|---------------|------|
| `bread` | 10.0 | 15.0 | 소프트바디, 상한 제어 |
| `snack` | 20.0 | 30.0 | 끝단 파지 |
| `bottle` | 40.0 | 60.0 | 원통형, 적정 파지힘 |
| `can` | 80.0 | 100.0 | 금속, 최대 파지힘 |

## Task State Machine

```
IDLE → DETECTING → MOVING_PICK → GRASPING → MOVING_PLACE → PLACING → DONE → IDLE
                                                                          ↓
                                                                        ERROR → IDLE
```
