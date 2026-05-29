# smart-shelf-robot 기능명세서

---

## 1. 시스템 개요

| 항목 | 내용 |
|------|------|
| 프로젝트명 | smart-shelf-robot |
| 목적 | 상품을 감지하고 물성에 맞는 파지힘으로 매대에 자동 정리 |
| 로봇 | Doosan E0509 + RH-P12-RN-A 그리퍼 |
| 카메라 | RealSense D455f x1 (Eye-in-hand) |
| 환경 | Ubuntu 22.04 + ROS2 Humble |

---

## 2. 노드 관계도


![graph_image](image-2.png)


---

## 3. 노드별 기능명세

### 3-1. vision 노드

| 항목 | 내용 |
|------|------|
| 담당 | 비전팀 (남상훈, 이현호) |
| 카메라 | Eye-in-hand (로봇 손목) |
| 입력 | `/camera/color/image_raw`, `/camera/depth/image_rect_raw` |
| 출력 | `/shelf/empty_slot`, `/shelf/slot_status`, `/object_class`, `/object_pose` |

| 기능 ID | 기능명 | 설명 | 동작 시점 |
|---------|--------|------|----------|
| VS-01 | 매대 슬롯 감지 | 매대 4개 슬롯 상품 존재 여부 감지 | SHELF_DETECTING |
| VS-02 | 빈 슬롯 발행 | 빈 슬롯 번호 + 필요 클래스 발행 (예: "1,can") | SHELF_DETECTING |
| VS-03 | 전체 상태 발행 | 전체 슬롯 상태 주기적 발행 | SHELF_DETECTING |
| VS-04 | 상품 검출 | 상품 구역에서 해당 클래스 YOLO 검출 | PRODUCT_DETECTING |
| VS-05 | 3D 포즈 추정 | depth + 카메라 내부 파라미터로 상품 3D 좌표 계산 | PRODUCT_DETECTING |
| VS-06 | 좌표계 변환 | 카메라 좌표 → 로봇 base_link 좌표계 변환 (TF) | PRODUCT_DETECTING |
| VS-07 | 미검출 처리 | 해당 클래스 미검출 시 integration 노드에 알림 | PRODUCT_DETECTING |

**슬롯-클래스 매핑 (place_targets.yaml 기준)**

| 슬롯 번호 | 상품 클래스 |
|----------|-----------|
| 1 | can |
| 2 | bottle |
| 3 | snack |
| 4 | bread |

---

### 3-2. integration 노드

| 항목 | 내용 |
|------|------|
| 담당 | 심예영 |
| 입력 | `/shelf/empty_slot`, `/object_class`, `/object_pose`, `/policy/action` |
| 출력 | `/grasp_force`, `/place_target` |
| 서비스 클라이언트 | `/move_to_shelf_view`, `/move_to_product_view`, `/move_to_pick`, `/move_to_place`, `/move_to_home`, `/gripper/open`, `/gripper/grasp` |

| 기능 ID | 기능명 | 설명 |
|---------|--------|------|
| IN-01 | 상태머신 관리 | IDLE → MOVE_TO_SHELF_VIEW → SHELF_DETECTING → MOVE_TO_PRODUCT_VIEW → PRODUCT_DETECTING → MOVING_PICK → GRASPING → MOVING_PLACE → PLACING → DONE 순서 제어 |
| IN-02 | 파지힘 계산 | 상품 클래스에 따라 파지힘 결정 후 /grasp_force 발행 |
| IN-03 | 적재 위치 결정 | 빈 슬롯 번호에 따라 place_targets.yaml에서 좌표 조회 후 /place_target 발행 |
| IN-04 | 에러 처리 | 상품 미검출, 파지 실패, 이동 실패 시 안전 복귀 |
| IN-05 | 작업 루프 | DONE 후 IDLE로 복귀하여 다음 빈 슬롯 처리 반복 |

**상태머신 전이 조건**

| 현재 상태 | 전이 조건 | 다음 상태 |
|----------|----------|----------|
| IDLE | 시스템 시작 | MOVE_TO_SHELF_VIEW |
| MOVE_TO_SHELF_VIEW | 이동 완료 | SHELF_DETECTING |
| SHELF_DETECTING | 빈 슬롯 감지 | MOVE_TO_PRODUCT_VIEW |
| SHELF_DETECTING | 빈 슬롯 없음 | IDLE |
| MOVE_TO_PRODUCT_VIEW | 이동 완료 | PRODUCT_DETECTING |
| PRODUCT_DETECTING | 상품 검출 성공 | MOVING_PICK |
| PRODUCT_DETECTING | 상품 미검출 | MOVE_TO_SHELF_VIEW |
| MOVING_PICK | 이동 완료 | GRASPING |
| MOVING_PICK | 이동 실패 | ERROR |
| GRASPING | 파지 성공 | MOVING_PLACE |
| GRASPING | 파지 실패 | ERROR |
| MOVING_PLACE | 이동 완료 | PLACING |
| MOVING_PLACE | 이동 실패 | ERROR |
| PLACING | 적재 완료 | DONE |
| DONE | - | IDLE |
| ERROR | 안전 복귀 완료 | IDLE |

---

### 3-3. motion 노드

| 항목 | 내용 |
|------|------|
| 담당 | 김민성 |
| 입력 | `/object_pose`, `/place_target` |
| 출력 | `/joint_command` |
| 서비스 서버 | `/move_to_shelf_view`, `/move_to_product_view`, `/move_to_pick`, `/move_to_place`, `/move_to_home` |

| 기능 ID | 기능명 | 설명 |
|---------|--------|------|
| MO-01 | 매대 관찰 위치 이동 | /move_to_shelf_view 수신 시 홈(매대 정면) 포지션으로 이동 |
| MO-02 | 상품 구역 이동 | /move_to_product_view 수신 시 상품 구역 관찰 위치로 이동 |
| MO-03 | 파지 위치 이동 | /move_to_pick 수신 시 object_pose 기반 경로 계획 및 이동 |
| MO-04 | 적재 위치 이동 | /move_to_place 수신 시 place_target 기반 경로 계획 및 이동 |
| MO-05 | 홈 복귀 | /move_to_home 수신 시 홈 포지션으로 이동 |
| MO-06 | 충돌 회피 | MoveIt/cuRobo 기반 충돌 없는 경로 계획 |
| MO-07 | 캘리브레이션 | Eye-in-hand 카메라 ↔ 로봇 좌표계 캘리브레이션 |

---

### 3-4. gripper 노드

| 항목 | 내용 |
|------|------|
| 담당 | 김민성 |
| 입력 | `/object_class`, `/grasp_force` |
| 출력 | `/gripper_command` |
| 서비스 서버 | `/gripper/open`, `/gripper/close`, `/gripper/grasp` |

| 기능 ID | 기능명 | 설명 |
|---------|--------|------|
| GR-01 | 그리퍼 열기 | /gripper/open 수신 시 그리퍼 완전 개방 |
| GR-02 | 물성 기반 파지 | /gripper/grasp 수신 시 grasp_force_params.yaml 기반 파지힘 적용 |
| GR-03 | 파지힘 상한 제어 | 클래스별 최대 파지힘 초과 방지 (bread: 15N, snack: 30N, bottle: 60N, can: 100N) |
| GR-04 | 파지 성공 판단 | joint_states 피드백으로 파지 성공 여부 판단 후 서비스 응답 |

---

### 3-5. rl (policy) 노드

| 항목 | 내용 |
|------|------|
| 담당 | 김인영, 남정혁 |
| 입력 | `/object_pose`, `/joint_states` |
| 출력 | `/policy/action` |

| 기능 ID | 기능명 | 설명 |
|---------|--------|------|
| RL-01 | policy 로드 | 학습된 ONNX/PT 모델 로드 |
| RL-02 | 관측값 구성 | joint_positions + object_pose → 관측 벡터 구성 |
| RL-03 | policy 추론 | 관측값 → action 추론 (10Hz) |
| RL-04 | action 발행 | 추론된 action /policy/action 으로 발행 |
| RL-05 | 폴백 처리 | 강화학습 실패 시 cuRobo 기반 모션플래닝으로 전환 |

---

## 4. 모션 포인트 정의

| 포인트 | 설명 | 관련 서비스 |
|--------|------|-----------|
| home / shelf_view | 매대 정면, SHELF_DETECTING 수행 위치 | `/move_to_shelf_view` |
| product_view | 상품 구역 위, PRODUCT_DETECTING 수행 위치 | `/move_to_product_view` |
| pick | 상품 파지 위치 (vision에서 실시간 계산) | `/move_to_pick` |
| place | 빈 슬롯 적재 위치 (place_targets.yaml) | `/move_to_place` |

---

## 5. 물성별 파지 전략

| 클래스 | 파지힘 | 최대 파지힘 | 전략 | 비고 |
|--------|--------|-----------|------|------|
| bread | 10.0N | 15.0N | soft | 소프트바디, 변형 방지 |
| snack | 20.0N | 30.0N | edge | 봉지 끝단 파지 |
| bottle | 40.0N | 60.0N | cylinder | 원통형 형상 기반 |
| can | 80.0N | 100.0N | max | 금속, 미끄럼 방지 |

---

## 6. 비기능 요구사항

| 항목 | 요구사항 |
|------|---------|
| 통신 | 전 노드 ROS_DOMAIN_ID=100, 동일 와이파이(ASUS_20) |
| 주기 | vision 10Hz, policy 추론 10Hz, 상태머신 5Hz |
| 안전 | 에러 발생 시 그리퍼 즉시 개방 후 홈 복귀 |
| 배포 | 로봇 제어 코드만 레포 포함, 비전/RL은 모델 파일만 업로드 |
| 인터페이스 | 모든 토픽/서비스 타입은 docs/interface_definition.md 준수 |