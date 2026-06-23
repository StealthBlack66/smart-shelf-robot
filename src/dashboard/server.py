"""dashboard_node 엔트리 — ROS2 집계기 + FastAPI(WS/MJPEG/REST) 통합.

실행 구조
  · ROS2 단일 노드(RosBridge) + MultiThreadedExecutor → 데몬 스레드에서 spin
  · 같은 노드에 ControlClients / CameraBridge / MetricsStore attach (executor 공유)
  · uvicorn(FastAPI) 은 메인 스레드 (시그널 처리)
  · WebSocket: 15Hz 로 snapshot+전류시계열+지표 push (영상 제외)
  · MJPEG: /api/camera/stream (multipart) — 영상 전용 분리 채널
  · REST: /api/control/* — 그리퍼/로봇/파이프라인 제어

실행:
  source /opt/ros/humble/setup.bash && source ~/doosan_ws/install/setup.bash
  export ROS_DOMAIN_ID=100
  python3 src/dashboard/server.py            # → http://0.0.0.0:8080
"""
from __future__ import annotations

import asyncio
import os
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 패키지/스크립트 양쪽 실행 지원
try:
    from .ros_bridge import RosBridge, StateStore
    from .control import ControlClients, POSITION_MAX, CURRENT_MAX
    from .camera import CameraBridge
    from .metrics import MetricsStore
except ImportError:                     # 스크립트 직접 실행
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ros_bridge import RosBridge, StateStore     # type: ignore
    from control import ControlClients, POSITION_MAX, CURRENT_MAX  # type: ignore
    from camera import CameraBridge                   # type: ignore
    from metrics import MetricsStore                  # type: ignore

WS_HZ = 15.0
WEB_HOST = os.environ.get('DASH_HOST', '0.0.0.0')
WEB_PORT = int(os.environ.get('DASH_PORT', '8080'))
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), '..', 'web', 'dist')


class Backend:
    """ROS 측 객체 묶음. FastAPI 라우트가 이걸 통해 ROS 에 접근한다."""

    def __init__(self) -> None:
        self.store = StateStore()
        self.bridge = RosBridge(self.store)
        self.metrics = MetricsStore()
        # safe_grasp 결과 → 지표 자동 집계 (현재 감지 클래스를 라벨로)
        self.control = ControlClients(
            self.bridge, on_grasp_result=self._on_grasp_result)
        self.camera = CameraBridge(self.bridge)
        self._last_grasp_target: int | None = None
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.bridge)
        self._spin_thread: threading.Thread | None = None

    def _on_grasp_result(self, result: dict) -> None:
        self.metrics.record_grasp(
            result,
            object_class=self.store.grasp_class,
            target_pos=self._last_grasp_target,
        )

    def start_ros(self) -> None:
        self._spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True, name='ros-spin')
        self._spin_thread.start()

    def shutdown(self) -> None:
        self.executor.shutdown()
        self.bridge.destroy_node()


backend: Backend | None = None
app = FastAPI(title='smart-shelf-robot dashboard')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'],
                   allow_headers=['*'])


# ── 요청 모델 ────────────────────────────────────────────────────────
class NamedMove(BaseModel):
    name: str                       # home | shelf_view | product_view | place


class JointMove(BaseModel):
    pos_deg: list[float]
    vel: float = 30.0
    acc: float = 30.0
    relative: bool = False


class GripperPos(BaseModel):
    position: int
    timeout: float = 5.0


class SafeGraspReq(BaseModel):
    target_position: int
    max_current: int
    current_delta_threshold: int = 20
    timeout: float = 5.0


class TorqueReq(BaseModel):
    enable: bool


class MotionProfileReq(BaseModel):
    velocity: int
    acceleration: int


class EStopReq(BaseModel):
    mode: int = 0


class PlaceMark(BaseModel):
    success: bool
    detail: str = ''


class OperatorCmd(BaseModel):
    cmd: str                        # start | restock:<class> | confirm | abort | reset


# ── 텔레메트리 WebSocket ─────────────────────────────────────────────
@app.websocket('/ws')
async def ws_telemetry(ws: WebSocket) -> None:
    await ws.accept()
    period = 1.0 / WS_HZ
    try:
        while True:
            payload = {
                'snapshot': backend.store.snapshot(),
                'current_series': backend.store.current_series(),
                'metrics': backend.metrics.summary(),
                'camera_available': backend.camera.available and backend.camera.has_frame(),
            }
            await ws.send_json(payload)
            await asyncio.sleep(period)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ── 카메라 MJPEG (영상 전용 분리 채널) ───────────────────────────────
@app.get('/api/camera/stream')
def camera_stream():
    gen = backend.camera.mjpeg_generator()
    return StreamingResponse(
        gen, media_type='multipart/x-mixed-replace; boundary=frame')


# ── RViz 화면 MJPEG (충돌구체 포함 3D 뷰) ────────────────────────────
_rviz_cap = None


def _get_rviz_cap():
    global _rviz_cap
    if _rviz_cap is None:
        from rviz_capture import RvizCapture
        _rviz_cap = RvizCapture(win_name="rviz", fps=3)
    return _rviz_cap


@app.get('/api/rviz/stream')
def rviz_stream():
    try:
        cap = _get_rviz_cap()
        return StreamingResponse(
            cap.mjpeg_generator(),
            media_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=503)


# ── 조회 REST (WS 못 쓰는 클라이언트/디버그용) ──────────────────────
@app.get('/api/state')
def get_state():
    return backend.store.snapshot()


@app.get('/api/metrics')
def get_metrics():
    return {
        'summary': backend.metrics.summary(),
        'error_causes': backend.metrics.error_causes(),
    }


@app.get('/api/logs')
def get_logs(n: int = 50):
    return {
        'rosout_errors': backend.store.errors(n),
        'events': backend.metrics.recent_events(n),
    }


# ── 제어 REST ────────────────────────────────────────────────────────
@app.post('/api/control/move_named')
def move_named(req: NamedMove):
    return JSONResponse(backend.control.move_named(req.name))


@app.post('/api/control/move_joint')
def move_joint(req: JointMove):
    return JSONResponse(backend.control.move_joint(
        req.pos_deg, req.vel, req.acc, req.relative))


@app.post('/api/control/estop')
def estop(req: EStopReq):
    return JSONResponse(backend.control.estop(req.mode))


@app.post('/api/control/gripper/set_position')
def gripper_set_position(req: GripperPos):
    return JSONResponse(backend.control.set_position(req.position, req.timeout))


@app.post('/api/control/gripper/safe_grasp')
def gripper_safe_grasp(req: SafeGraspReq):
    backend._last_grasp_target = req.target_position
    return JSONResponse(backend.control.safe_grasp(
        req.target_position, req.max_current, req.current_delta_threshold, req.timeout))


@app.post('/api/control/gripper/torque')
def gripper_torque(req: TorqueReq):
    return JSONResponse(backend.control.set_torque(req.enable))


@app.post('/api/control/gripper/motion_profile')
def gripper_motion_profile(req: MotionProfileReq):
    return JSONResponse(backend.control.set_motion_profile(
        req.velocity, req.acceleration))


@app.post('/api/operator/cmd')
def operator_cmd(req: OperatorCmd):
    return JSONResponse(backend.control.operator_cmd(req.cmd))


@app.post('/api/metrics/place')
def mark_place(req: PlaceMark):
    backend.metrics.record_place(req.success, req.detail)
    return {'ok': True}


@app.post('/api/metrics/cycle/start')
def cycle_start():
    backend.metrics.cycle_start()
    return {'ok': True}


@app.post('/api/metrics/cycle/end')
def cycle_end(success: bool = True):
    backend.metrics.cycle_end(success)
    return {'ok': True}


@app.post('/api/metrics/reset')
def metrics_reset():
    backend.metrics.reset()
    return {'ok': True}


@app.get('/api/limits')
def limits():
    return {'position_max': POSITION_MAX, 'current_max': CURRENT_MAX}


# ── 정적 프론트엔드(빌드 결과) 서빙 ──────────────────────────────────
if os.path.isdir(FRONTEND_DIST):
    app.mount('/', StaticFiles(directory=FRONTEND_DIST, html=True), name='web')


def main() -> None:
    global backend
    rclpy.init()
    backend = Backend()
    backend.start_ros()
    backend.bridge.get_logger().info(
        f'대시보드 웹서버 시작 → http://{WEB_HOST}:{WEB_PORT}')
    try:
        uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level='warning')
    finally:
        backend.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
