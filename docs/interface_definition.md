# Interface Definition

## Topics

| 토픽 | 타입 | 발행자 | 구독자 | 설명 |
|------|------|--------|--------|------|
| `/shelf/empty_slot` | `std_msgs/String` | vision_shelf | vision_product, integration | 빈 매대 슬롯 정보 (슬롯 번호 + 클래스, 예: "1,can") |
| `/shelf/slot_status` | `std_msgs/String` | vision_shelf | integration | 전체 매대 상태 (예: "1:empty,2:full,3:empty,4:full") |
| `/object_class` | `std_msgs/String` | vision_product | integration, gripper | 검출된 상품 클래스 (bread/snack/bottle/can) |
| `/object_pose` | `geometry_msgs/PoseStamped` | vision_product | integration, motion | 상품 3D 위치 및 자세 |
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

## Camera Roles

| 카메라 | 위치 | 역할 | 발행 토픽 |
|--------|------|------|----------|
| `vision_shelf` | Eye-to-hand (매대 전방 고정) | 매대 슬롯별 상품 존재 여부 감지, 빈 슬롯 + 필요 클래스 반환 | `/shelf/empty_slot`, `/shelf/slot_status` |
| `vision_product` | Eye-in-hand (로봇 손목) | 빈 슬롯 신호 수신 후 상품 영역에서 해당 클래스 검출, 위치 반환 | `/object_class`, `/object_pose` |

## Task State Machine

```
IDLE → SHELF_DETECTING → PRODUCT_DETECTING → MOVING_PICK → GRASPING → MOVING_PLACE → PLACING → DONE → IDLE
            ↓                    ↓                                                                   ↓
       (빈 슬롯 없음)      (상품 미검출)                                                           ERROR → IDLE
            ↓                    ↓
           IDLE              SHELF_DETECTING (재탐색)
```

### 단계별 설명

| 단계 | 담당 | 설명 |
|------|------|------|
| `SHELF_DETECTING` | vision_shelf | 매대 전체 스캔, 빈 슬롯 + 필요 클래스 `/shelf/empty_slot`으로 발행 |
| `PRODUCT_DETECTING` | vision_product | 빈 슬롯 신호 수신 후 상품 영역에서 해당 클래스 검출, `/object_pose` 발행 |
| `MOVING_PICK` | motion | 검출된 상품 위치로 이동 |
| `GRASPING` | gripper | 물성 기반 파지힘으로 파지 |
| `MOVING_PLACE` | motion | 빈 매대 슬롯 위치로 이동 |
| `PLACING` | gripper | 그리퍼 열어 상품 적재 |
