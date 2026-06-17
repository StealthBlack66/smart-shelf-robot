# Smart Shelf Robot — Web Dashboard

상용 수준 운영 대시보드. 모든 노드 상태·로봇/그리퍼/비전 실시간 텔레메트리·카메라 라이브 피드·
성능지표·로그를 한 화면에서 보고, 그리퍼/로봇/파지/E-STOP까지 전부 제어한다.

## 아키텍처

```
ROS2 그래프 ──▶ dashboard_node (rclpy + FastAPI)
                  · 모든 토픽 구독 + posx 서비스 폴링 → StateStore
                  · SafeGrasp 결과 관측 → 성능지표 집계(SQLite)
                  · WebSocket(JSON, 15Hz): 텔레메트리/그래프/지표
                  · MJPEG(HTTP): 카메라 영상 전용 분리 채널
                  · REST: 제어(/api/control/*)
                ──▶ 브라우저 (React + Vite + uPlot)
```

영상은 WebSocket이 아니라 MJPEG HTTP로 분리 → 그래프 갱신이 영상 부하에 영향받지 않음.

| 파일 | 역할 |
|---|---|
| `server.py` | FastAPI(WS/MJPEG/REST) + dashboard_node 엔트리 |
| `ros_bridge.py` | 토픽 구독 · posx 폴링 · 스레드 안전 스냅샷 |
| `control.py` | 그리퍼/로봇 서비스·액션 호출 래퍼 |
| `metrics.py` | 파지/택타임/에러 집계 + SQLite |
| `camera.py` | CompressedImage → MJPEG 중계 |

## 설치 & 실행

```bash
pip install -r src/dashboard/requirements.txt
cd web && npm install && npm run build && cd ..

source /opt/ros/humble/setup.bash && source ~/doosan_ws/install/setup.bash
export ROS_DOMAIN_ID=100
python3 src/dashboard/server.py        # → http://localhost:8080
```

빌드된 프론트(`web/dist`)를 백엔드가 같은 출처로 서빙하므로 vite dev 서버는 불필요.

## 제어 API (요약)

| 엔드포인트 | 동작 |
|---|---|
| `POST /api/control/move_named {name}` | home/shelf_view/product_view/place |
| `POST /api/control/move_joint {pos_deg,vel,acc,relative}` | 조인트 조그/이동 |
| `POST /api/control/estop {mode}` | 비상정지(MoveStop) |
| `POST /api/control/gripper/set_position {position}` | 그리퍼 위치(0~1150) |
| `POST /api/control/gripper/safe_grasp {target_position,max_current,...}` | 힘 제어 파지 |
| `POST /api/control/gripper/torque {enable}` | 토크 ON/OFF |
| `POST /api/control/gripper/motion_profile {velocity,acceleration}` | 모션 프로파일 |
