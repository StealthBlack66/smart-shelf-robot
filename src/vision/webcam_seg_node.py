#!/usr/bin/env python3
"""
=================================================================================
  현재 사용 중인 비전 모델 (파일 업데이트 시마다 갱신 — 최신: 2026-06-04)
=================================================================================

  ┌─ GPU/CPU 점유 (RTX 4060 Laptop 8GB) ──────────────────────────────────────
  │  YOLO11s-seg (pose_robust_seg)  ≈ 0.1 GB   ← 메인 검출+분할+분류
  │  GroundingDINO SwinT            ≈ 1.5 GB   ← open-vocab 보조 검출
  │  SAM2 Hiera Large               ≈ 2.4 GB   ← 정밀 마스크 (백그라운드 워커)
  │  PyTorch overhead               ≈ 1.2 GB
  │  rembg ISNet (ONNX, CPU)        ≈ 0.7 GB RAM  fallback only
  │  Qwen2.5-VL / HQ-SAM            비활성 / 미설치
  └────────────────────────────────────────────────────────────────────────────

  ▣ 핵심 설계 (2026-06-04)
    - 검출/분할/분류 PRIMARY = YOLO11s-seg (학습 가능 → 우리 객체 특화)
    - SAM2 = 학습 안 된 객체(GD-only) 정밀 마스크 + refinement
    - rx/ry/rz = 두산 ZYZ top-down [0, 180, yaw] (로보티즈/두산 파지 규격)
    - 학습 파이프라인 내장 (분리·겹침 객체 → fine-tune)

─────────────────────────────────────────────────────────────────────────────────
[1] PRIMARY 검출+분할+분류 — YOLO11s-seg (pose_robust_seg.pt)
─────────────────────────────────────────────────────────────────────────────────
  모델       : YOLO11s-seg fine-tuned, 4-class {0:bottle,1:can,2:snack_bag,3:bread}
  파일       : ../models/pose_robust_seg.pt (~20 MB,  원본 .bak 백업 보존)
  device     : GPU, 매 프레임 추론
  출력       : instance mask(res.masks) + bbox + class + conf
  학습 이력  : 2026-06-04 재학습 — (a) 캔/병 구분 (b) 겹친 객체 분리
               데이터: 분리배치(높이규칙 라벨) + 겹친배치(SAM2 point-prompt 라벨)
  후처리     : 클래스 시간투표(track별 최빈값) — can↔bottle 프레임 flip 완화
  ★ 선택 이유:
     - instance seg 한 모델로 검출+마스크+클래스 동시 → 파이프라인 단순·빠름
     - fine-tune 으로 우리 객체(캔/병/과자) 특화 → open-vocab 의 캔↔병 혼동 해결
     - 겹친 객체 분리를 "학습"으로 해결 (GD/SAM box-prompt 의 merge 한계 극복)
     - 가벼움(20MB) → 매 프레임 추론해도 ~18 FPS

─────────────────────────────────────────────────────────────────────────────────
[2] 보조 검출 — GroundingDINO SwinT (open-vocabulary)
─────────────────────────────────────────────────────────────────────────────────
  파일       : ~/models/groundingdino_swint_ogc.pth (172 MB)
  실행 주기  : gd_interval(기본 3s) 백그라운드 thread
  prompt     : "can . bottle . snack bag"  (cup 제거 — 2026-06-04 사용자 요청)
  후처리     : 강한 NMS + conf 하한 + detection persistence
  ★ 선택 이유:
     - open-vocabulary → YOLO seg 가 학습 안 한 종류(예: 투명포장 과자)도 검출
     - YOLO seg 가 놓친 객체를 SAM2 로 마스킹해 외곽선 보강([5] GD-only)
     - 어수선한 배경 과검출은 conf 하한·NMS·persistence 로 정리

─────────────────────────────────────────────────────────────────────────────────
[3] 정밀 마스크 — SAM2 Hiera Large (백그라운드 워커)
─────────────────────────────────────────────────────────────────────────────────
  파일       : ~/models/sam2_hiera_large.pt (config sam2_hiera_l.yaml)
  활성       : ENABLE_SAM2=1 (launch_seg.sh)
  역할       : (a) GD-only 객체(YOLO seg 미검출) box-prompt 마스크
               (b) 외곽 refinement 캐시 / 학습 라벨(point-prompt) 생성
  구조       : 별도 스레드 _sam_refine_worker — 메인 비블록, _sam_lock 직렬화,
               ~1.5s throttle (GPU 경합 회피, py-spy 로 병목 실측 후 적용)
  ★ 선택 이유:
     - HQ-SAM(segment_anything_hq) 패키지 미설치 → SAM2 로 대체
     - foundation 모델 → 라벨/반사/그림자에 안 속고 객체 정확 분할
       (ISNet saliency 는 광택 캔 라벨을 타서 외곽 망가짐 → primary 부적합)
     - box/point prompt 로 임의 객체 즉시 분할 (겹침 라벨링에 핵심)
     - GPU 무거워 매 프레임은 FPS 폭락 → 백그라운드 워커+throttle 로 해결

─────────────────────────────────────────────────────────────────────────────────
[4] 마스크 FALLBACK — rembg ISNet (CPU)
─────────────────────────────────────────────────────────────────────────────────
  파일       : ~/.u2net/isnet-general-use.onnx (170 MB), CPUExecutionProvider
  ★ 선택 이유: SAM2/HQ-SAM 둘 다 없을 때만. saliency 기반이라 광택·인접객체에
               약함 → primary 아닌 최후 fallback.

─────────────────────────────────────────────────────────────────────────────────
[5] 외곽선 파이프라인 (현재)
─────────────────────────────────────────────────────────────────────────────────
  base       : raw YOLO seg mask (학습됨·신뢰) → 구멍 메우고 최대 연결성분
  depth gate : bimodal "빈 간격(valley)" 있을 때만 컷 → 그림자/테이블 제거,
               캔 밑변(테이블까지 연속 depth) 보존
  안정화     : 마스크 시간 EMA + 근접 track key → 깜빡임 제거
  GD-only    : SAM2 워커 마스크 + detection persistence(3s) → 경계 conf 깜빡임 제거
  ★ 선택 이유:
     - ISNet/convex hull/단순 depth-threshold 는 캔 라벨 타거나 밑변 자르거나
       그림자 포함 → OUTLINE_DEBUG 비교 후 raw YOLO + gap-depth 로 정착
     - grid key EMA 는 bbox 경계 jitter 로 리셋→깜빡 → 근접 track key 로 해결

─────────────────────────────────────────────────────────────────────────────────
[6] 파지 자세 rx/ry/rz — 두산 ZYZ euler [0, 180, yaw]
─────────────────────────────────────────────────────────────────────────────────
  규격       : _RealProject_1 자료 — top-down 파지 (그리퍼 -Z 아래)
               rx=0, ry=180(top-down), rz=물체 yaw
  yaw        : base 평면 투영 점군의 cv2.minAreaRect 긴 변 각도 (원근왜곡 제거)
  ★ 선택 이유:
     - 두산 e0509 + 로보티즈 RH-P12-RN-A 파지 = ZYZ [0,180,yaw] (자료 규격)
     - PCA tilt 추정은 원통/단일뷰에서 불안정 → 규격대로 [0,180,yaw] 고정
     - 외부 코드 미참조, 현재 프로젝트 자원(mask_to_base_cloud+minAreaRect)으로 구현

─────────────────────────────────────────────────────────────────────────────────
[7] 좌표 계산 기준 — 센터 / 꼭지점 (어떤 픽셀·깊이를 base mm 로 변환하나)
─────────────────────────────────────────────────────────────────────────────────
  ◆ 공통 deproject 원리 (픽셀 → 로봇 base mm)
      1) RealSense intrinsics(fx,fy,ppx,ppy) 로 픽셀(u,v)+깊이 z →
         rs2_deproject_pixel_to_point → 카메라 좌표 xyz(m)
      2) 캘리브 T_cam2base(4×4) @ [xyz,1] → robot base 좌표, ×1000 = mm
      함수: pixel_to_base_xyz() / mask_to_base_xyz() / mask_to_base_cloud()

  ◆ 센터포인트mm (빨강 점 + 노랑 좌표) — mask_to_base_xyz()
      · 픽셀 위치 = mask 픽셀의 (x평균, y평균) = mask 중심(centroid)
      · 깊이 z   = mask 내 유효 depth 의 25퍼센타일 (물체 표면=가까운 쪽 대표)
      · 위 deproject → base xyz mm
      ※ 평균(중심)+퍼센타일: 단일 픽셀 depth 노이즈 회피, 대표점 안정

  ◆ 꼭지점mm (주황 4점) — refined 외곽선 minAreaRect
      · refined mask 의 최대 contour → cv2.minAreaRect → 회전사각형 4 corner
      · 각 corner 를 "가장 가까운 contour 점"으로 snap (코너는 외곽 밖 →
        외곽선 위로 당겨 표기)
      · 깊이 = mask depth 의 median (평면 가정, 코너 depth 구멍 회피)
      · 각 snap 픽셀 deproject → base mm, 시계방향 [우상,우하,좌하,좌상]
        (box 중심 기준 사분면으로 순서 결정)
  ★ 선택 이유: minAreaRect = 외곽 픽셀 기준 회전사각형 (GD bbox 핑크 축정렬과 별개).
               코너 snap 으로 빈 공간이 아닌 실제 외곽선 위에 표기.

─────────────────────────────────────────────────────────────────────────────────
[8] GraspGen export — 'g' 키 (graspgen_export.py)
─────────────────────────────────────────────────────────────────────────────────
  출력       : 물체별 base-frame point cloud .npz (키 'point_cloud', float32 (N,3),
               meter, base_link) + ZMQ 전송(send_zmq)
  ★ 선택 이유: 비전/그래스프팀 GraspGen 규격(키·단위·라벨 enum) 그대로 → 6DOF
               grasp 연동. bottle→pet_bottle 매핑, bread 제외.

─────────────────────────────────────────────────────────────────────────────────
[9] OBB (yolo26obb_can_pen) — 화면 그리기 비활성
─────────────────────────────────────────────────────────────────────────────────
  ★ 선택 이유: 재학습 seg 외곽선+꼭지점이 정확해 OBB 노란 회전박스는 중복 clutter
               → 그리기 off (obb_list 계산은 내부 angle 매칭용으로만 유지).

─────────────────────────────────────────────────────────────────────────────────
[10] Qwen2.5-VL / OCR — 비활성
─────────────────────────────────────────────────────────────────────────────────
  ★ 선택 이유: VRAM ~5GB 차지 → SAM2/GD 위해 양보. 라벨은 YOLO 클래스로 충분.

─────────────────────────────────────────────────────────────────────────────────
[11] Depth + 좌표 변환 — RealSense D435
─────────────────────────────────────────────────────────────────────────────────
  스트림     : color 1280x720 + depth, align(color), intrinsics 동적 추출
  변환       : pixel→cam xyz→base xyz (T_cam2base), depth 단위 mm
  ★ 선택 이유: RGBD 한 센서로 검출+3D 좌표 + depth 로 그림자/테이블 분리까지 처리.

─────────────────────────────────────────────────────────────────────────────────
[12] 캘리브레이션 — calibration_result.npz (key 'T_cam2base', 4×4)
─────────────────────────────────────────────────────────────────────────────────
  출처       : 08_카메라_핸드아이_캘리브레이션.py 생성

─────────────────────────────────────────────────────────────────────────────────
[13] 표시 색상 (visualization)
─────────────────────────────────────────────────────────────────────────────────
  노랑 외곽선 : seg/SAM2 refined mask contour
  주황 꼭지점 : 외곽 minAreaRect 4코너 (base mm)
  초록 점     : rx/ry/rz 계산 점군 / 빨강 점 : 마스크 중심
  azure 텍스트: rx/ry/rz [0,180,yaw] / 핑크 사각형 : GD bbox

─────────────────────────────────────────────────────────────────────────────────
[14] 학습 파이프라인 (2026-06-04 내장)
─────────────────────────────────────────────────────────────────────────────────
  수집       : /tmp/collect_can.py (RealSense N프레임, hash 중복 skip)
  라벨       : 분리배치 → GD박스+높이규칙 / 겹친배치 → SAM2 point-prompt (클래스 강제)
  학습       : pose_robust_seg fine-tune (백본 freeze, 강한 aug, 4-class 헤드 유지)
  교체       : best.pt → pose_robust_seg.pt (.bak 백업 후)
  ★ 선택 이유: GD/SAM 은 캔↔병 혼동·겹침 merge → "정답 라벨 학습"이 정공법.
               겹침은 SAM2 point-prompt 로 객체별 분리 라벨(GD 박스 라벨은 부정확).
               백본 freeze + 4-class 유지로 snack/bread 망각 최소화.

  데이터 흐름 (1 frame, ~15-22 FPS):
    RealSense → frame_bgr + depth_arr
      ├─[BG thread] GroundingDINO (3s) → gd_results [(bbox,phrase,score)]
      ├─[BG thread] SAM2 워커 → 객체별 정밀 mask 캐시 (throttle 1.5s)
      └─[Main loop, 매 frame] YOLO11s-seg 추론
            ├─ seg mask → 외곽선(노랑) + 구멍채움 + depth gap gate + 시간 EMA
            ├─ base xyz(센터) + rx/ry/rz[0,180,yaw] + 외곽 꼭지점(주황)
            ├─ GD-only(YOLO 미검출) → SAM2 마스크로 동일 표기 + persistence(3s)
            └─ vis → cv2.imshow + ROS image publish,  'g' = GraspGen export(.npz)

  실행:
    supervisor : bash /tmp/run_seg_durable.sh   (launch_seg.sh = ENABLE_SAM2=1)
    직접       : ENABLE_SAM2=1 python3 .../webcam_seg_node.py
=================================================================================

Webcam Segmentation Node — _RealProject_1/webcam_seg.py 의 ROS2 노드 포팅판.

UI/마커/키조작은 webcam_seg.py 그대로:
  - YOLO seg (pen_detecting 모델) 실시간 추론 + 화면 overlay
  - 마우스: `+q+좌클릭 = probe 마커, 우클릭 = undo
  - 키:    1-9=lock, g=graspgen, p=파지, v=매대확인, h=home, r=취소, c=clear, f=full, m=refine, ESC=종료
  - pynput OS 레벨 키 hold (한영 자판 매핑 포함)

로봇 제어/그리퍼/캘리브/RealSense 는 _RealProject_0 방식:
  - calibration_result.npz (key T_cam2base) — script-relative 로드
  - PICK('p') → /dsr01/curobo/pick_pose 발행 (curobo_planner_node 가 full 시퀀스 실행)
  - 그리퍼: /dsr01/gripper/{open,close} 서비스

사용:
  # 노드 단독 (curobo_planner_node + gripper_service_node 가 이미 떠 있어야 함)
  python3 ~/doosan_ws/src/e0509_gripper_description/scripts/webcam_seg_node.py

  # 통합 launch (curobo + gripper + webcam_seg 한 번에)
  ros2 launch e0509_gripper_description webcam_seg.launch.py

전제: 두산 본체 (dsr_bringup2) 는 별도 터미널에서 떠 있어야 함.
  두산 본체 (dsr_bringup2) 는 별도 터미널에서 수동으로 띄워야 함.



"""
import os
import sys
import subprocess
import signal

# 디버거 console 이 tty 인 경우 노드가 background process group 이 되어
# stdout/stderr 출력 시 SIGTTOU/SIGTTIN 받아 T (stopped) 됨 (창 옮길 때 멈춤 증상).
# SIG_IGN 으로 정지 방지.
try:
    signal.signal(signal.SIGTTOU, signal.SIG_IGN)
    signal.signal(signal.SIGTTIN, signal.SIG_IGN)
except Exception:
    pass

'''
객체의 각도는 몇도회전되어있는지 각도 표시하고 그리고 사각형 꼭지점 위치도 표기 ㄱㄱㄱ 
각도랑 사각형 글씨색은 센터포인트 글자색과 동일하게해주고 꼭지점 글자는 해당 꼭지점에 글자를 표기해주고 각도글자는 센터포인터 글자 바로 오른쪽에 , 각도 로 표기 해주셈.

꼭지점 글자는 리얼센스 + 켈리브레이션(센터포인트 연산 때 사용했던) 으로 연산된 실제좌표 를 기준으로 꼭지점이 어디인지 를 계산해주면 댐. 그리고 사물의 가로길이와 세로길이도 계산해서 표시해주셈

가로길이와 세로길이도 센터포인터를 기준으로 하면 연산가능함. 가로길이글자와 세로길이글자색상은 핑크색으로 해주셈


항상 위치는 리얼센스 + 켈리브레이션 등.. 으로 계산된 실제 좌표를 말하는거임


값들이 모두 계산이 됐다면 해당 dict 에 값들을 저장해주고 저장된 값들을 웹캠에 표시해주는거임 ㅇㅇ 이해함?
'''

# ─────────────────────────────────────────────────────────────────────────
# GraspGen 인터페이스 라벨 규격 (그래스프팀 합의 enum = 3종)
#   seg 모델 실측 names = {0:bottle, 1:can, 2:snack_bag, 3:bread}
#   → GraspGen enum 으로 매핑: bottle→pet_bottle, can/snack_bag 동일, bread 제외.
# 이 dict 가 라벨 변환의 단일 소스. export(.npz/ZMQ) 시 항상 to_graspgen_label() 경유.
GRASPGEN_LABEL_MAP = {
    'bottle':    'pet_bottle',
    'can':       'can',
    'snack_bag': 'snack_bag',
    # 'bread' 는 GraspGen enum(3종)에 없음 → None 반환되어 export 에서 skip
}
GRASPGEN_LABELS = ('snack_bag', 'can', 'pet_bottle')   # 확정 enum (단위 meter, base frame)


def to_graspgen_label(seg_name):
    """seg 모델 클래스명 → GraspGen enum 라벨. enum 밖(bread 등)이면 None."""
    return GRASPGEN_LABEL_MAP.get(str(seg_name).strip().lower())


def _rotm_to_quat(R):
    """3x3 회전행렬 → 쿼터니언 [x,y,z,w]."""
    import numpy as _np
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = _np.sqrt(t + 1.0) * 2
        w = 0.25 * s; x = (R[2, 1]-R[1, 2])/s; y = (R[0, 2]-R[2, 0])/s; z = (R[1, 0]-R[0, 1])/s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = _np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1]-R[1, 2])/s; x = 0.25*s; y = (R[0, 1]+R[1, 0])/s; z = (R[0, 2]+R[2, 0])/s
    elif R[1, 1] > R[2, 2]:
        s = _np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2]-R[2, 0])/s; x = (R[0, 1]+R[1, 0])/s; y = 0.25*s; z = (R[1, 2]+R[2, 1])/s
    else:
        s = _np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0]-R[0, 1])/s; x = (R[0, 2]+R[2, 0])/s; y = (R[1, 2]+R[2, 1])/s; z = 0.25*s
    q = _np.array([x, y, z, w], dtype=float)
    return q / (_np.linalg.norm(q) + 1e-9)


obj_dicts = {}

obj_dicts['이름'] = { # 이름은 예를들어 콜라 로 감지가되면 콜라 로 적어주셈
    '각도': '',
    '각도r': {
        'rx': 0.0,
        'ry': 0.0,
        'rz': 0.0
    },
    '가로길이mm':'',
    '세로길이mm':'',
    '꼭지점mm':[['','',''],['','',''],['','',''],['','','']], # [['우상값x','우상값y','우상값z'], ['우하값x','우하값y','우하값z'], ['좌하값x','좌하값y','좌하값z'], ['좌상값x','좌상값y','좌상값z'] ] 형식으로 ㄱ 즉 시계방향
    '외곽선':{
        '꼭지점mm':[['','',''],['','',''],['','',''],['','','']], # [['우상값x','우상값y','우상값z'], ['우하값x','우하값y','우하값z'], ['좌하값x','좌하값y','좌하값z'], ['좌상값x','좌상값y','좌상값z'] ] 형식으로 ㄱ 즉 시계방향
    },

    '센터포인트mm':''
}




# launch_webcam.sh 동일 환경을 단독 실행 시에도 자동 설정.
# setup.bash 의 모든 환경변수를 그대로 import (수십개 — PATH, LD_LIBRARY_PATH,
# AMENT_PREFIX_PATH, CMAKE_PREFIX_PATH, PYTHONPATH, ROS_DISTRO 등).
def _detect_active_display():
    """활성 GUI 세션의 DISPLAY 자동 탐지 (gnome-shell/Xorg 프로세스 environ).
    하드코딩 :1 이 세션 바뀌면 틀려서(실제 :2 등) 자동 감지로 대체."""
    import glob
    for pat in ('gnome-shell', 'gnome-session', 'Xorg'):
        for pid_dir in glob.glob('/proc/[0-9]*'):
            try:
                with open(f'{pid_dir}/comm') as f:
                    if pat not in f.read():
                        continue
                with open(f'{pid_dir}/environ', 'rb') as f:
                    for kv in f.read().decode('utf-8', 'ignore').split('\x00'):
                        if kv.startswith('DISPLAY=') and kv[8:]:
                            return kv[8:]
            except Exception:
                continue
    return None


def _autosource_ros_env():
    if not os.environ.get('DISPLAY'):
        os.environ['DISPLAY'] = _detect_active_display() or ':0'
    os.environ.setdefault('XAUTHORITY', '/run/user/1000/gdm/Xauthority')
    # 이미 ROS_DISTRO 설정돼있으면 source 안 함 (이미 source 된 환경에서 실행됨)
    if os.environ.get('ROS_DISTRO'):
        return
    ros_setup = '/opt/ros/humble/setup.bash'
    if not os.path.exists(ros_setup):
        return
    ws_setup = os.path.expanduser('~/doosan_ws/install/setup.bash')
    sources = [f'source {ros_setup}']
    if os.path.exists(ws_setup):
        sources.append(f'source {ws_setup}')
    cmd = ' && '.join(sources) + ' && env'
    try:
        # PATH 안전하게 설정 (bash 가 source 시 dirname/sed 필요)
        env = {'PATH': '/usr/local/bin:/usr/bin:/bin', **os.environ}
        out = subprocess.run(['bash', '-c', cmd], env=env,
                             capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            # setdefault 가 아니라 항상 override (setup.bash 결과 우선)
            os.environ[k] = v
    except Exception as _e:
        print(f'[autosource] ROS env 자동 source 실패: {_e}', file=sys.stderr)
    # PYTHONPATH → sys.path 동기화
    for p in os.environ.get('PYTHONPATH', '').split(':'):
        if p and p not in sys.path and os.path.isdir(p):
            sys.path.insert(0, p)


_autosource_ros_env()


def _maybe_respawn_clean():
    """debugger / IDE 환경 감지 시 nohup 으로 자신 재실행 + exit.
    어떤 방식으로 띄워도 결국 동일한 nohup detached 환경에서 cv2 동작.
    """
    if os.environ.get('WSN_RESPAWNED'):
        return
    in_debugger = (
        any('debugpy' in a for a in sys.argv) or
        'debugpy' in sys.modules or
        bool(os.environ.get('VSCODE_PID')) or
        bool(os.environ.get('TERM_PROGRAM') == 'vscode')
    )
    if not in_debugger:
        return
    print('[respawn] debugger/IDE 감지 — nohup detached 로 재실행. '
          '로그: /tmp/wsn_launch.log',
          file=sys.stderr)
    env = os.environ.copy()
    env['WSN_RESPAWNED'] = '1'
    log_path = '/tmp/wsn_launch.log'
    with open(log_path, 'w') as logf:
        subprocess.Popen(
            ['/usr/bin/nohup', '/usr/bin/python3',
             os.path.abspath(__file__)],
            stdout=logf, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env,
            start_new_session=True)
    sys.exit(0)


# _maybe_respawn_clean()  # 비활성화: 사용자 디버거 attach 가능하게

import importlib.util
import threading
import time

import cv2
import numpy as np
from pynput import keyboard as pynput_keyboard
from ultralytics import YOLO
from PIL import Image as PILImage_for_draw, ImageDraw, ImageFont

# 한글 폰트
_KFONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
_KFONT_CACHE = {}


def _kfont(size):
    if size not in _KFONT_CACHE:
        try:
            _KFONT_CACHE[size] = ImageFont.truetype(_KFONT_PATH, size)
        except Exception:
            _KFONT_CACHE[size] = ImageFont.load_default()
    return _KFONT_CACHE[size]


def draw_korean(img_bgr, text, pos_xy, size=18, color_bgr=(0, 255, 255), bg=True):
    """OpenCV img 에 한글 텍스트 그리기 (PIL 거쳐서). pos_xy = (x, y) 좌상단."""
    if not text:
        return img_bgr
    pil = PILImage_for_draw.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _kfont(size)
    # 배경 박스 (가독성)
    if bg:
        try:
            bbox = draw.textbbox(pos_xy, text, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0))
        except Exception:
            pass
    # BGR → RGB
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos_xy, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# GroundingDINO — open-vocab detector (background thread)
try:
    from groundingdino.util.inference import load_model as gd_load_model, predict as gd_predict
    import torch as _torch
    _GD_AVAIL = True
except Exception:
    _GD_AVAIL = False

# SAM 2 Tiny — Meta 2024. 정확한 instance segmentation.
try:
    from sam2.build_sam import build_sam2 as _build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor as _Sam2Predictor
    _SAM2_AVAIL = True
except Exception:
    _SAM2_AVAIL = False

# rembg (ISNet/BiRefNet) — DIS 기반 SOTA edge segmentation. 매우 정확한 외곽선.
try:
    from rembg import new_session as _rembg_new_session
    from rembg import remove as _rembg_remove
    _REMBG_AVAIL = True
except Exception:
    _REMBG_AVAIL = False

# HQ-SAM — SAM boundary 개선 모델 (high-quality token). 작은 객체 외곽 sharp.
try:
    from segment_anything_hq import sam_model_registry as _hqsam_reg
    from segment_anything_hq import SamPredictor as _HqSamPredictor
    _HQSAM_AVAIL = True
except Exception:
    _HQSAM_AVAIL = False

# PaddleOCR — 한국어+영어 OCR (background thread). Qwen2.5-VL 통합 후엔 기본 OFF.
try:
    from paddleocr import PaddleOCR
    _OCR_AVAIL = True
except Exception:
    _OCR_AVAIL = False

# Qwen2.5-VL — 한국어 정밀 비전 (background, 5초 주기). Gemini 와 비슷한 VLM.
try:
    from transformers import (Qwen2_5_VLForConditionalGeneration,
                              AutoProcessor)
    from PIL import Image as PILImage
    _QWEN_AVAIL = True
except Exception:
    _QWEN_AVAIL = False

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Point, PoseArray, Pose
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker
from sensor_msgs.msg import CompressedImage   # 대시보드 카메라 피드 발행용
try:
    from dsr_gripper_tcp_interfaces.srv import SetPosition  # 그리퍼 열기(브리지)
    _SETPOS_AVAIL = True
except Exception:
    _SETPOS_AVAIL = False
try:
    from dsr_msgs2.srv import MoveJoint  # 'h' 키 — product_view 자세 복귀
    _MOVEJOINT_AVAIL = True
except Exception:
    _MOVEJOINT_AVAIL = False
try:
    from dsr_msgs2.srv import MoveStop   # 스페이스바 비상정지 (Quick stop)
    _MOVESTOP_AVAIL = True
except Exception:
    _MOVESTOP_AVAIL = False

# GraspGen ZMQ 클라이언트 (object_tracking 워크플로우 이식). ~/GraspGen path 추가.
_GG_ROOT = os.path.expanduser("~/GraspGen")
if os.path.isdir(_GG_ROOT) and _GG_ROOT not in sys.path:
    sys.path.insert(0, _GG_ROOT)
try:
    from grasp_gen.serving.zmq_client import GraspGenClient
    _GRASPGEN_AVAIL = True
except Exception as _gge:
    _GRASPGEN_AVAIL = False
    _GRASPGEN_ERR = str(_gge)

import pyrealsense2 as rs



# _RealProject_0 루트 (00_두산로봇_리눅스_실물_연결.py, doosan_config.py 위치)
# scripts/ → e0509_gripper_description/ → src/ → doosan_ws/ → _RealProject_0/
_PROJECT_ROOT = os.path.abspath( os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))


def _here(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def _resolve(rel_local, home_fallback):
    local = _here(rel_local)
    return local if os.path.exists(local) else os.path.expanduser(home_fallback)


MARKER_COLORS = {
    'probe': (0, 255, 0),
    'pick':  (255, 255, 0),
    'place': (255, 0, 255),
}
MARKER_LABELS = {'probe': 'P', 'pick': 'PICK', 'place': 'PLACE'}


ACTION_COOLDOWN_SEC = 3.0

HANGUL_TO_EN = {
    'ㅂ': 'q', 'ㅃ': 'q',
    'ㅈ': 'w', 'ㅉ': 'w',
    'ㄷ': 'e', 'ㄸ': 'e',
    'ㄱ': 'r', 'ㄲ': 'r',
    'ㅅ': 't', 'ㅆ': 't',
    'ㅛ': 'y', 'ㅕ': 'u', 'ㅑ': 'i', 'ㅐ': 'o', 'ㅒ': 'o',
    'ㅔ': 'p', 'ㅖ': 'p',
    'ㅁ': 'a', 'ㄴ': 's', 'ㅇ': 'd', 'ㄹ': 'f', 'ㅎ': 'g',
    'ㅗ': 'h', 'ㅓ': 'j', 'ㅏ': 'k', 'ㅣ': 'l',
    'ㅋ': 'z', 'ㅌ': 'x', 'ㅊ': 'c', 'ㅍ': 'v',
    'ㅠ': 'b', 'ㅜ': 'n', 'ㅡ': 'm',
}


# --------- 두산 본체 (dsr_bringup2) 자동 연결 ---------


def _load_p00():
    """_RealProject_0/00_두산로봇_리눅스_실물_연결.py 동적 로드."""
    p00_path = os.path.join(_PROJECT_ROOT, '00_두산로봇_리눅스_실물_연결.py')
    if not os.path.exists(p00_path):
        return None, f'not found: {p00_path}'
    # doosan_config 도 같은 폴더에 있어야 import 됨
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    try:
        spec = importlib.util.spec_from_file_location('p00', p00_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, None
    except Exception as e:
        return None, f'load failed: {e}'



class WebcamSegNode(Node):
    def __init__(self):
        super().__init__('webcam_seg_node')
        self._declare_parameters()

        weights    = self.get_parameter('weights').value
        calib_path = self.get_parameter('calibration_path').value
        self.conf  = float(os.environ.get('WSN_CONF', self.get_parameter('conf').value))
        self.iou   = float(self.get_parameter('iou').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.width  = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.approach_height = float(self.get_parameter('approach_height').value)
        self.safe_z = float(self.get_parameter('safe_z').value)

        if not os.path.exists(calib_path):
            raise SystemExit(f'calibration 없음: {calib_path}')

        self._load_vision_models(weights)
        self._load_calibration(calib_path)
        self._setup_realsense()
        self._setup_ros_interfaces()
        self._init_state()
        self._setup_keyboard()
        self._setup_window()


    def _declare_parameters(self):
        self.declare_parameter('weights', _resolve('models/pose_robust_seg.pt', ''))
        self.declare_parameter('weights_obb', _resolve('models/yolo26obb_can_pen.pt', ''))
        self.declare_parameter('weights_yoloe', '')
        self.declare_parameter('yoloe_prompts',
                               'green can,red can,coca cola,bottle,cup,pen,marker')
        self.declare_parameter('gd_config',
                               os.path.expanduser('~/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'))
        self.declare_parameter('gd_weights',
                               os.path.expanduser('~/models/groundingdino_swint_ogc.pth'))
        self.declare_parameter('gd_prompts', 'can . bottle . snack bag')
        self.declare_parameter('gd_interval', 3.0)
        self.declare_parameter('gd_box_thr', 0.30)
        self.declare_parameter('gd_text_thr', 0.25)
        self.declare_parameter('ocr_lang', '')
        self.declare_parameter('ocr_interval', 4.0)
        self.declare_parameter('ocr_min_score', 0.5)
        self.declare_parameter('qwen_enable', False)
        self.declare_parameter('qwen_4bit', True)
        self.declare_parameter('qwen_model', 'Qwen/Qwen2.5-VL-7B-Instruct')
        self.declare_parameter('qwen_interval', 6.0)
        self.declare_parameter('qwen_prompt',
            '이 사진에 보이는 모든 객체를 한 줄에 하나씩, 짧고 정확하게 나열해줘. '
            '브랜드/제품명이 있으면 포함. 예: "초록색 칠성사이다 캔", "농심 포테토칲 봉지". '
            '한국어로만 답변하고 추가 설명은 하지마.')
        self.declare_parameter('calibration_path', _here('calibration_result.npz'))
        self.declare_parameter('conf', 0.40)
        self.declare_parameter('iou', 0.5)
        self.declare_parameter('imgsz', 960)
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('approach_height', 0.08)
        self.declare_parameter('safe_z', 0.15)

    def _load_vision_models(self, weights):
        # YOLO seg
        self.yolo = None
        if weights and os.path.exists(weights):
            self.get_logger().info(f'YOLO seg 로드: {weights}')
            self.yolo = YOLO(weights)
            self.get_logger().info(f'클래스: {self.yolo.names}')
        else:
            self.get_logger().info('YOLO seg 비활성 (weights 없음/미지정) — GD/obb 만 사용')

        # YOLO obb
        self.yolo_obb = None
        obb_w = self.get_parameter('weights_obb').value
        if obb_w and os.path.exists(obb_w):
            try:
                self.yolo_obb = YOLO(obb_w)
                self.get_logger().info(f'YOLO obb 로드: {obb_w}')
            except Exception as e:
                self.get_logger().warn(f'YOLO obb 로드 실패: {e}')

        # YOLOE
        self.yoloe = None
        self.yoloe_prompts_str = ''
        yoloe_w = self.get_parameter('weights_yoloe').value
        if yoloe_w and os.path.exists(yoloe_w):
            try:
                self.yoloe = YOLO(yoloe_w)
                prompts = [p.strip() for p in
                           self.get_parameter('yoloe_prompts').value.split(',')
                           if p.strip()]
                if hasattr(self.yoloe, 'set_classes'):
                    self.yoloe.set_classes(prompts, self.yoloe.get_text_pe(prompts))
                self.yoloe_prompts_str = ', '.join(prompts)
                self.get_logger().info(f'YOLOE 로드: {yoloe_w} prompts={prompts}')
            except Exception as e:
                self.get_logger().warn(f'YOLOE 로드 실패: {e}')

        # GroundingDINO
        self.gd_model = None
        self.gd_prompts  = self.get_parameter('gd_prompts').value
        self.gd_interval = float(self.get_parameter('gd_interval').value)
        self.gd_box_thr  = float(self.get_parameter('gd_box_thr').value)
        self.gd_text_thr = float(self.get_parameter('gd_text_thr').value)
        self.gd_results  = []
        self.gd_lock     = threading.Lock()
        # 매대확인용: YOLO seg 원본 2D 검출 [(name, conf), ...]. depth 투영 전이라
        #   매대(먼 거리)에서도 잡힘 — self.detections는 depth 필요해 매대거리서 비어버림.
        self.yolo_seg_results = []
        self.gd_busy     = False
        self.gd_last_t   = 0.0
        gd_cfg = self.get_parameter('gd_config').value
        gd_w   = self.get_parameter('gd_weights').value
        if _GD_AVAIL and os.path.exists(gd_cfg) and os.path.exists(gd_w):
            try:
                self.get_logger().info(f'GroundingDINO 로드: {gd_w}')
                self.gd_model = gd_load_model(gd_cfg, gd_w)
                self.get_logger().info(
                    f'  prompts: {self.gd_prompts}  interval={self.gd_interval}s')
            except Exception as e:
                self.get_logger().warn(f'GroundingDINO 로드 실패: {e}')
        else:
            self.get_logger().warn(
                f'GroundingDINO 비활성 (avail={_GD_AVAIL}, cfg={os.path.exists(gd_cfg)}, '
                f'weights={os.path.exists(gd_w)})')

        # PaddleOCR
        self.ocr_model    = None
        self.ocr_interval = float(self.get_parameter('ocr_interval').value)
        self.ocr_min_score = float(self.get_parameter('ocr_min_score').value)
        self.ocr_results  = []
        self.ocr_lock     = threading.Lock()
        self.ocr_busy     = False
        self.ocr_last_t   = 0.0
        lang = self.get_parameter('ocr_lang').value
        if _OCR_AVAIL and lang:
            try:
                self.get_logger().info(f'PaddleOCR 로드 중 (lang={lang})...')
                self.ocr_model = PaddleOCR(
                    use_angle_cls=True, lang=lang, use_gpu=True, show_log=False)
                self.get_logger().info('PaddleOCR OK')
            except Exception as e:
                self.get_logger().warn(f'PaddleOCR 로드 실패: {e}')
        else:
            self.get_logger().info('PaddleOCR 비활성 (Qwen2.5-VL 대체)')

        # Qwen2.5-VL — state
        self.qwen_model       = None
        self.qwen_processor   = None
        self.qwen_interval    = float(self.get_parameter('qwen_interval').value)
        self.qwen_prompt      = self.get_parameter('qwen_prompt').value
        self.qwen_result_text = ''
        self.qwen_phrase_label = {}
        self.qwen_lock        = threading.Lock()
        self.qwen_busy        = False
        self.last_detections  = {}
        self.rect_smooth_cache = {}
        self.qwen_last_t      = 0.0

        # HQ-SAM (vit_l > vit_b > vit_tiny 우선순위)
        self.hqsam_predictor = None
        self.sam2_predictor  = None
        for arch, ckpt, label in [
            ('vit_l',    os.path.expanduser('~/models/sam_hq_vit_l.pth'),    'Large'),
            ('vit_b',    os.path.expanduser('~/models/sam_hq_vit_b.pth'),    'Base'),
            ('vit_tiny', os.path.expanduser('~/models/sam_hq_vit_tiny.pth'), 'Tiny'),
        ]:
            if not (_HQSAM_AVAIL and os.path.exists(ckpt)):
                continue
            try:
                self.get_logger().info(f'HQ-SAM ({arch}, {label}) 로드 중...')
                hq = _hqsam_reg[arch](checkpoint=ckpt)
                hq.to(device='cuda')
                hq.eval()
                self.hqsam_predictor = _HqSamPredictor(hq)
                self.get_logger().info(f'HQ-SAM {label} OK')
                break
            except Exception as e:
                self.get_logger().warn(f'HQ-SAM {label} 로드 실패: {e}')

        # SAM 2 (HQ-SAM 없을 때 fallback)
        if self.hqsam_predictor is None and os.environ.get('ENABLE_SAM2', '1') != '0':
            for ckpt, cfg, label in [
                (os.path.expanduser('~/models/sam2_hiera_large.pt'), 'configs/sam2/sam2_hiera_l.yaml', 'Large'),
                (os.path.expanduser('~/models/sam2_hiera_small.pt'), 'configs/sam2/sam2_hiera_s.yaml', 'Small'),
                (os.path.expanduser('~/models/sam2_hiera_tiny.pt'),  'configs/sam2/sam2_hiera_t.yaml', 'Tiny'),
            ]:
                if not (_SAM2_AVAIL and os.path.exists(ckpt)):
                    continue
                try:
                    self.get_logger().info(f'SAM 2 {label} 로드 중...')
                    sam2 = _build_sam2(cfg, ckpt, device='cuda', mode='eval')
                    self.sam2_predictor = _Sam2Predictor(sam2)
                    self.get_logger().info(f'SAM 2 {label} OK')
                    break
                except Exception as e:
                    self.get_logger().warn(f'SAM 2 {label} 로드 실패: {e}')

        # rembg ISNet
        self.rembg_session      = None
        self.rembg_session_name = None
        if _REMBG_AVAIL:
            try:
                self.rembg_session = _rembg_new_session(
                    'isnet-general-use', providers=['CPUExecutionProvider'])
                self.rembg_session_name = 'isnet-general-use'
                self.get_logger().info('rembg ISNet OK (CPU)')
            except Exception as e:
                self.get_logger().warn(f'rembg ISNet 로드 실패: {e}')

        self._isnet_cache       = {}
        self._isnet_req         = None
        self._isnet_worker_stop = False
        self._sam_lock          = threading.Lock()
        if (self.hqsam_predictor is not None or self.sam2_predictor is not None
                or self.rembg_session is not None):
            self._isnet_worker_thread = threading.Thread(
                target=self._sam_refine_worker, daemon=True)
            self._isnet_worker_thread.start()
            _wname = ('HQ-SAM' if self.hqsam_predictor is not None
                      else 'SAM2' if self.sam2_predictor is not None else 'ISNet')
            self.get_logger().info(f'정밀외곽 워커 시작 ({_wname}, 백그라운드)')

        # Qwen2.5-VL — model load
        qwen_enable = bool(self.get_parameter('qwen_enable').value)
        qwen_4bit   = bool(self.get_parameter('qwen_4bit').value)
        if not qwen_enable:
            self.get_logger().info('Qwen2.5-VL 비활성 (qwen_enable=False)')
        elif _QWEN_AVAIL:
            try:
                qm = self.get_parameter('qwen_model').value
                quant_label = '4-bit NF4' if qwen_4bit else 'bfloat16'
                self.get_logger().info(f'Qwen2.5-VL 로드 중 ({qm}, {quant_label})...')
                if qwen_4bit:
                    from transformers import BitsAndBytesConfig
                    bnb = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=_torch.bfloat16,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_use_double_quant=True,
                    )
                    self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        qm, quantization_config=bnb, device_map='cuda')
                else:
                    self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        qm, torch_dtype=_torch.bfloat16, device_map='cuda')
                self.qwen_processor = AutoProcessor.from_pretrained(qm)
                self.get_logger().info(
                    f'Qwen2.5-VL OK ({quant_label}) — interval={self.qwen_interval}s')
            except Exception as e:
                self.get_logger().warn(f'Qwen2.5-VL 로드 실패: {e}')
        else:
            self.get_logger().warn('Qwen2.5-VL 비활성 (transformers/Qwen 모듈 없음)')

    def _load_calibration(self, calib_path):
        self.get_logger().info(f'캘리브 로드: {calib_path}')
        T = np.load(calib_path)['T_cam2base']
        self.T_cam2base = T
        self.get_logger().info(
            f'  translation (mm) = {np.round(T[:3, 3] * 1000, 1).tolist()}')
        self._eih_active = False
        self.create_subscription(
            Float64MultiArray, '/eih/T_cam2base', self._eih_tcam_cb, 10)
        self.get_logger().info(
            '  /eih/T_cam2base 구독 — 로봇+eih_fk 떠있으면 실시간 변환(eye-in-hand) 사용')

    def _setup_realsense(self):
        self.get_logger().info('RealSense 시작...')
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.profile = self.pipe.start(cfg)
        self.align   = rs.align(rs.stream.color)
        self.intr    = (self.profile.get_stream(rs.stream.color)
                                    .as_video_stream_profile().get_intrinsics())
        for _ in range(10):
            self.pipe.wait_for_frames()
        self.get_logger().info(f'  RealSense OK ({self.width}x{self.height})')

    def _setup_ros_interfaces(self):
        cb_group = ReentrantCallbackGroup()
        # latched QoS — g 한 번 발행하면 늦게 붙는 RViz/구독자도 마지막 메시지를 받음
        # (one-shot volatile 이면 타이밍 어긋날 때 화살표/후보 안 뜸).
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.pub_pick        = self.create_publisher(PoseStamped, '/dsr01/curobo/pick_pose',   10)
        self.pub_grasp_class = self.create_publisher(String,      '/dsr01/curobo/grasp_class', 10)
        # [실험] GraspGen 후보 N개(EE pose) → curobo plan_grasp(goalset) 선택용 (latched)
        self.pub_grasp_candidates = self.create_publisher(
            PoseArray, '/dsr01/curobo/grasp_candidates', latched)
        self.pub_obstacles   = self.create_publisher(String,      '/dsr01/curobo/obstacles',   10)
        # ★대시보드 "매대 정리 시작" 버튼 → /dashboard/operator_cmd 구독 → 'a'(auto_restock) 트리거
        self.create_subscription(String, '/dashboard/operator_cmd', self._operator_cmd_cb, 10)
        # ★대시보드 카메라 라이브 피드 — 화면(vis_show)을 JPEG CompressedImage 로 발행
        self.pub_dash_cam = self.create_publisher(
            CompressedImage, '/dashboard/camera/compressed', 1)
        self._dash_cam_n = 0
        # ★대시보드 매대 품목 재고 (캔/바틀/스낵 각 0/1) — 매대확인+진열로 갱신해 발행
        self.pub_shelf_inv = self.create_publisher(
            String, '/dashboard/shelf_inventory', 1)
        self.shelf_inv = {'can': 0, 'bottle': 0, 'snack': 0}
        self.cli_open = (self.create_client(
            SetPosition, '/gripper_service/set_position',
            callback_group=cb_group) if _SETPOS_AVAIL else None)
        self.cli_product_view = None
        if _MOVEJOINT_AVAIL:
            self.cli_product_view = self.create_client(
                MoveJoint, '/dsr01/motion/move_joint', callback_group=cb_group)
        # 🛑 비상정지 (스페이스바) — 두산 MoveStop(Quick stop). 물리 E-stop 의 보조.
        self.cli_stop = None
        if _MOVESTOP_AVAIL:
            self.cli_stop = self.create_client(
                MoveStop, '/dsr01/motion/move_stop', callback_group=cb_group)
        self._estop_last = 0.0
        # 자동 진열('a' 키)용 — curobo place/home 서비스(Trigger). place 는 grasp_class 기준 슬롯.
        self.cli_place = self.create_client(Trigger, '/move_to_place', callback_group=cb_group)
        self.cli_home  = self.create_client(Trigger, '/move_to_home',  callback_group=cb_group)
        self._auto_running = False
        _hp = os.environ.get('WSN_PRODUCT_VIEW', '0.0,-36.0,56.0,5.0,110.0,0.0')
        self.product_view_pose  = [float(v) for v in _hp.split(',')]
        self._product_view_last = 0.0

    def _init_state(self):
        self.marker_pub     = self.create_publisher(
            Marker, '/graspgen/preview_marker',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       history=HistoryPolicy.KEEP_LAST))   # latched — 늦게 붙는 RViz도 받음
        self.detections        = []
        self.shelf_missing     = None   # 'v' 매대재고: 없는(집을) 제품 리스트
        # 'v' 매대뷰(home) 자세 — 여기로 이동 후 매대재고 확인 (place_targets.yaml home)
        self.shelf_view_pose   = [float(v) for v in os.environ.get(
            'WSN_SHELF_VIEW', '-6.73,8.12,104.62,80.22,93.13,-23.49').split(',')]
        self.selected_idx      = 0
        self.locked            = False
        self.locked_idx        = None
        self._det_track        = {}
        self.locked_tk         = None
        self.pending_grasp_pose = None
        self.gripper_offset    = 0.11
        self.pregrasp_standoff = 0.06   # 's' 프리그래스프 standoff (6cm, p/curobo와 통일)
        _gfz = os.environ.get('WSN_GRASP_FIXED_Z', '57.5')
        self.grasp_fixed_z = (None if _gfz in ('auto', 'off', '') else float(_gfz) / 1000.0)
        self.gg = None
        if _GRASPGEN_AVAIL:
            try:
                self.gg = GraspGenClient('localhost', 5556,
                                         timeout_ms=60000, wait_for_server=False)
                self.get_logger().info('GraspGen 클라이언트 준비 (:5556)')
            except Exception as e:
                self.get_logger().warn(f'GraspGen 클라이언트 init 실패: {e}')
        else:
            self.get_logger().warn(f'GraspGen 모듈 없음: {_GRASPGEN_ERR}')
        self._mask_refine     = False
        self.pending_clicks   = []
        self.probe_markers    = []
        self.action_trigger   = {'fire': None, 'last_t': 0.0}
        self.keys_held        = set()
        self.motion_lock      = threading.Lock()

    # ---------- 키/마우스 ----------
    def _normalize(self, key):
        try:
            c = key.char
        except AttributeError:
            return None
        if not c:
            return None
        c = c.lower()
        return HANGUL_TO_EN.get(c, c)

    def _go_product_view(self):
        """'h' 키 — product_view 자세로 이동. 디바운스 1s."""
        now_ = time.time()
        if now_ - self._product_view_last < 1.0:
            return
        self._product_view_last = now_
        if self.cli_product_view is None:
            self.get_logger().error('product_view 서비스 없음 (dsr_msgs2/MoveJoint)')
            return
        if not self.cli_product_view.service_is_ready():
            if not self.cli_product_view.wait_for_service(timeout_sec=0.5):
                self.get_logger().error('move_joint 서비스 미연결 — 브링업 확인')
                return
        try:
            req = MoveJoint.Request()
            req.pos = [float(v) for v in self.product_view_pose]
            req.vel = 30.0
            req.acc = 30.0
            req.time = 0.0
            req.radius = 0.0
            req.mode = 0       # ABSOLUTE
            req.blend_type = 0
            req.sync_type = 1
            self.cli_product_view.call_async(req)
            self.get_logger().info(
                f"[h] product_view 이동 → {[round(v,1) for v in self.product_view_pose]}°")
        except Exception as e:
            self.get_logger().error(f'product_view 이동 실패: {e}')

    def _emergency_stop(self):
        """🛑 스페이스바 비상정지 — 두산 MoveStop(Quick stop). 디바운스 0.3s.
        ※ 물리 E-stop(TP 빨간버튼)이 1순위. 이건 소프트 보조."""
        now_ = time.time()
        if now_ - self._estop_last < 0.3:
            return
        self._estop_last = now_
        if self.cli_stop is None:
            self.get_logger().error('🛑 비상정지 서비스 없음 (dsr_msgs2/MoveStop) — 물리 E-stop 사용!')
            return
        try:
            req = MoveStop.Request()
            req.stop_mode = 1   # DR_QSTOP : Quick stop
            self.cli_stop.call_async(req)
            self.get_logger().warn('🛑🛑🛑 [SPACE] 비상정지 — MoveStop(QSTOP) 호출됨')
        except Exception as e:
            self.get_logger().error(f'🛑 비상정지 호출 실패: {e} — 물리 E-stop 사용!')

    def _on_key_press(self, key):
        # 스페이스바 = 비상정지 (OS레벨 — 창 포커스 무관하게 즉시 작동)
        try:
            if key == pynput_keyboard.Key.space:
                self._emergency_stop()
                return
        except Exception:
            pass
        c = self._normalize(key)
        # ('h' product_view는 cv2.waitKey 핸들러에서만 처리 — 여기서 또 하면 move_joint 이중발사
        #  → 충돌로 로봇이 엉뚱한 자세(하늘)로 감. OS레벨 중복 금지.)
        if not c:
            return
        self.keys_held.add(c)
        if '`' in self.keys_held and c == 'w':
            now_ = time.time()
            if now_ - self.action_trigger['last_t'] > ACTION_COOLDOWN_SEC:
                self.action_trigger['fire'] = 'pick'
                self.action_trigger['last_t'] = now_
                self.get_logger().info(
                    f'[key] `+{c} → {self.action_trigger["fire"]} armed')
        # (ctrl+c 화면캡처+클립보드 복사 제거됨 — 사용자 요청)

    def _on_key_release(self, key):
        c = self._normalize(key)
        if c:
            self.keys_held.discard(c)

    def _setup_keyboard(self):
        self.key_listener = pynput_keyboard.Listener(
            on_press=self._on_key_press, on_release=self._on_key_release)
        self.key_listener.start()

    def _window_to_src(self, x, y):
        """창 좌표(letterbox 적용된) → 원본 프레임 좌표.

        창이 fullscreen 으로 늘어나도 원본 640x480 좌표계로 정확히 매핑.
        검은 letterbox 영역 클릭은 None 반환.
        """
        di = self._display_info
        if di is None or di['scale'] <= 0:
            return int(x), int(y)
        sx = (x - di['ox']) / di['scale']
        sy = (y - di['oy']) / di['scale']
        sw_src, sh_src = di['src']
        if not (0 <= sx < sw_src and 0 <= sy < sh_src):
            return None
        return int(sx), int(sy)

    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mapped = self._window_to_src(x, y)
            if mapped is None:
                return
            sx, sy = mapped
            backtick = '`' in self.keys_held
            ctrl = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
            armed = (backtick and 'q' in self.keys_held) or ctrl
            if armed:
                self.pending_clicks.append((sx, sy, 'probe'))
            else:
                self.get_logger().info(
                    f'[click ignored] ({sx},{sy}) keys={sorted(self.keys_held)}')
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.probe_markers:
                removed = self.probe_markers.pop()
                self.get_logger().info(f'[undo] {removed[4]} 마커 제거')
            else:
                self.get_logger().info('[undo] 제거할 마커 없음')

    def _force_maximize_periodically(self):
        """background thread: 시작 후 첫 10초간 매초 wmctrl maximize 강제."""
        env = os.environ.copy()
        env.setdefault('DISPLAY', ':1')
        env.setdefault('XAUTHORITY', '/run/user/1000/gdm/Xauthority')
        time.sleep(2)
        for _ in range(10):
            try:
                subprocess.run(
                    ['wmctrl', '-r', 'webcam_seg_node', '-b',
                     'add,maximized_vert,maximized_horz'],
                    env=env, timeout=2, capture_output=True)
            except Exception:
                pass
            time.sleep(1)

    def _setup_window(self):
        self.win = 'webcam_seg_node (1-9=lock g=graspgen p=pick v=shelf a=AUTO o=open h=home r=cancel c=clear f=full m=refine ESC=quit)'
        # WINDOW_NORMAL + resizeWindow 로 widget 1600x1200 시작 사이즈.
        # 영상은 640x480 그대로 imshow → cv2 가 widget 에 KEEPRATIO 로 스케일 (가벼움).
        # widget 비율 1600:1200 = 영상 640:480 = 4:3 → 패딩 X.
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)
        self._display_info = None
        self._fullscreen = False
        self._fullscreen_reapplied = True
        self._show_w, self._show_h = 1600, 1200
        dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.imshow(self.win, dummy)
        cv2.waitKey(1)
        cv2.resizeWindow(self.win, self._show_w, self._show_h)
        cv2.waitKey(1)
        self.get_logger().info(
            f'[window] 시작 사이즈 = {self._show_w}x{self._show_h} (NORMAL)')

    def _update_display_info(self, src_w, src_h):
        """창 크기에 맞춰 letterbox 파라미터 갱신. 창 크기 변화 감지 시만 갱신."""
        try:
            rect = cv2.getWindowImageRect(self.win)
        except cv2.error:
            return
        sw, sh = int(rect[2]), int(rect[3])
        if sw <= 0 or sh <= 0:
            return
        if (self._display_info is not None
                and self._display_info['screen'] == (sw, sh)
                and self._display_info['src'] == (src_w, src_h)):
            return  # 변화 없음
        scale = min(sw / src_w, sh / src_h)
        nw = int(src_w * scale)
        nh = int(src_h * scale)
        ox = (sw - nw) // 2
        oy = (sh - nh) // 2
        self._display_info = dict(
            screen=(sw, sh), src=(src_w, src_h),
            scale=scale, ox=ox, oy=oy, nw=nw, nh=nh)

    def _letterbox(self, img):
        """화면 가득 채우기 (stretch). 비율 무시, 검정/흰색 띠 없음."""
        di = self._display_info
        if di is None:
            return img
        sw, sh = di['screen']
        return cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)

    def _sam2_prepare(self, frame_bgr):
        """프레임 1회당 image encoder 호출 — 모든 박스가 재사용.
        HQ-SAM/SAM2 는 백그라운드 워커(_sam_refine_worker)가 소유하므로 메인
        스레드는 set_image 하지 않음(predictor state 경쟁 방지)."""
        if self.hqsam_predictor is not None or self.sam2_predictor is not None:
            return                      # 워커가 SAM 담당 → 메인은 건드리지 않음
        return

    def _estimate_rect_from_bbox(self, frame_bgr, x1, y1, x2, y2, depth_arr=None,
                                  phrase=None):
        """GD bbox crop 안에서 회전 사각형 (corners + angle + pixel w/h) 추정.
        SAM 2 mask + depth-based refinement + Canny edge alignment 으로 정확한 외곽.
        반환 dict {'corners': [(x,y)*4 영상 절대 픽셀], 'angle_deg': 0~180°,
                   'pixel_w': long-side, 'pixel_h': short-side} 또는 None.
        """
        h, w = frame_bgr.shape[:2]
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        mask = None
        # isnet_used=True 시 GrabCut/Depth/Morphology/Bilateral 후처리 skip →
        # ISNet sharp boundary 보존 (후처리하면 박스로 무뎌짐).
        isnet_used = False
        bw0 = x2 - x1; bh0 = y2 - y1
        cx_p = (x1 + x2) // 2; cy_p = (y1 + y2) // 2
        # 0) rembg ISNet (DIS) — SOTA sharp boundary. HQ-SAM/SAM2 보다 우선.
        # 메모리 reference_rembg_isnet_outline: 병/봉지/캔 정확.
        if self.rembg_session is not None:
            try:
                # v7 BEST: padding 8, threshold 64, fill 0.06~0.92, boxiness+solidity.
                # TTA 평균은 mask 흐려져서 박스화 → 단일 호출 유지.
                pad = 8
                px1 = max(0, x1 - pad); py1 = max(0, y1 - pad)
                px2 = min(w, x2 + pad); py2 = min(h, y2 + pad)
                crop_pad = frame_bgr[py1:py2, px1:px2]
                rgb_pad = cv2.cvtColor(crop_pad, cv2.COLOR_BGR2RGB)
                pil_pad = PILImage.fromarray(rgb_pad)
                isnet_mask = _rembg_remove(
                    pil_pad, session=self.rembg_session,
                    only_mask=True, post_process_mask=True)
                isnet_arr = np.array(isnet_mask)
                if isnet_arr.ndim == 3:
                    isnet_arr = isnet_arr[..., 0]
                # threshold 32 (이전 64) — alpha 약한 영역도 mask 에 포함 → 객체 boundary
                # 더 넓게 잡음. 이후 GrabCut 가 그 안에서 edge color gradient 따라 정밀 분리.
                _, isnet_bin = cv2.threshold(
                    isnet_arr, 32, 255, cv2.THRESH_BINARY)
                offy = y1 - py1; offx = x1 - px1
                bw_local = x2 - x1; bh_local = y2 - y1
                mask_isnet = isnet_bin[offy:offy + bh_local,
                                       offx:offx + bw_local]
                bbox_area_local = float(bw_local * bh_local)
                fill = (float((mask_isnet > 0).sum())
                        / max(bbox_area_local, 1.0))
                # ISNet mask 가 너무 박스에 가까운지 검증 (박스성 측정):
                # contour perimeter / sqrt(area) 비율 — 박스는 ~4, 복잡한 외곽은 >5.5
                # 박스성 높으면 ISNet 거부 → SAM2 fallback (둘 다 박스면 어차피 같음).
                # GrabCut 가 박스 mask 도 봉지 모양으로 refine → 박스성 check 완화.
                # fill upper 0.95 (이전 0.92) — 큰 객체도 수용.
                if 0.05 <= fill <= 0.95 and mask_isnet.shape == (bh_local, bw_local):
                    mask = mask_isnet.copy()
                    isnet_used = True
            except Exception:
                mask = None
        # 0a) HQ-SAM — boundary sharp. ISNet fail 시 fallback.
        #     HQ-SAM 은 워커와 공유 → _sam_lock 으로 직렬화하고, 현재 frame 으로
        #     set_image 후 predict (워커 frame 으로 오염 방지).
        if mask is None and self.hqsam_predictor is not None:
            try:
                with self._sam_lock:
                    self.hqsam_predictor.set_image(
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
                    masks_hq, scores_hq, _ = self.hqsam_predictor.predict(
                        box=box_np[None, :],
                        point_coords=None, point_labels=None,
                        multimask_output=False,
                        hq_token_only=True)
                self._sam_frame_fid = None    # 워커 set_image 로 무효화됨 표시
                m_hq = (masks_hq[0] > 0).astype(np.uint8) * 255
                cand = m_hq[y1:y2, x1:x2]
                fill = float(cand.sum()) / max(bw0 * bh0 * 255.0, 1.0)
                if 0.05 <= fill <= 0.95:
                    mask = cand.copy()
            except Exception as e:
                mask = None
        # 1) SAM 2 — bbox prompt + multi-point + negative corners.
        # GD bbox shrink 5% (객체 외곽보다 안쪽으로 들여서 그림자 배제)
        # ISNet 이 mask 못 잡았을 때 fallback.
        if mask is None and self.sam2_predictor is not None:
            try:
                bw0 = x2 - x1; bh0 = y2 - y1
                sx1 = x1 + int(bw0 * 0.03); sx2 = x2 - int(bw0 * 0.03)
                sy1 = y1 + int(bh0 * 0.03); sy2 = y2 - int(bh0 * 0.03)
                box_np = np.array([sx1, sy1, sx2, sy2], dtype=np.float32)
                cx_p = (x1 + x2) // 2; cy_p = (y1 + y2) // 2
                # 다중 positive — 객체 ID 강화 (negative 는 봉지 잘릴 위험 있어 제거)
                pos = [
                    [cx_p, cy_p],
                    [cx_p, y1 + int(bh0 * 0.30)],
                    [cx_p, y1 + int(bh0 * 0.70)],
                    [x1 + int(bw0 * 0.4), cy_p],
                    [x1 + int(bw0 * 0.6), cy_p],
                ]
                pt = np.array([pos], dtype=np.float32)
                lbl = np.array([[1]*len(pos)], dtype=np.int32)
                with self._sam_lock:    # 워커와 직렬화 + 현재 frame 으로 set_image
                    self.sam2_predictor.set_image(
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    masks_, scores_, _ = self.sam2_predictor.predict(
                        box=box_np, point_coords=pt, point_labels=lbl,
                        multimask_output=True)
                _ = (cx_p, cy_p)  # used below for diagnostic
                if masks_.ndim == 4:
                    masks_ = masks_[0]
                sc = scores_[0] if scores_.ndim == 2 else scores_
                bw = x2 - x1; bh = y2 - y1
                bbox_area = float(bw * bh)
                K = masks_.shape[0]
                best_idx = -1
                best_metric = -1e9
                for k in range(K):
                    m_full = (masks_[k] > 0).astype(np.uint8)
                    m_crop = m_full[y1:y2, x1:x2]
                    area = float(m_crop.sum())
                    fill = area / max(bbox_area, 1.0)
                    if fill < 0.10 or fill > 0.95:
                        continue
                    # bbox 4 변 경계 mask 닿은 비율 (이상적 30% bbox 보다 작게 차길)
                    edge = (
                        m_crop[0, :].sum() + m_crop[-1, :].sum() +
                        m_crop[:, 0].sum() + m_crop[:, -1].sum())
                    edge_ratio = edge / (2.0 * (bw + bh))
                    # metric: score 높을수록 +, fill 이 30~70% 사이일수록 +, edge_ratio 낮을수록 +
                    fill_bonus = -abs(fill - 0.5) * 0.8
                    edge_pen = -min(edge_ratio, 0.5) * 0.6
                    metric = float(sc[k]) + fill_bonus + edge_pen
                    if metric > best_metric:
                        best_metric = metric
                        best_idx = k
                if best_idx < 0:
                    # 모두 거부됐으면 score 최고 (best-effort)
                    best_idx = int(np.argmax(sc))
                full_m = (masks_[best_idx] > 0).astype(np.uint8) * 255
                mask = full_m[y1:y2, x1:x2].copy()
                # 가장 큰 connected component
                n_l, lbl_img, stats, _ = cv2.connectedComponentsWithStats(
                    mask, connectivity=8)
                if n_l > 1:
                    largest_l = 1 + int(
                        np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    mask = ((lbl_img == largest_l)
                            .astype(np.uint8)) * 255
                if mask.sum() < bbox_area * 0.03 * 255:
                    mask = None
            except Exception:
                mask = None
        # 1b) GrabCut refinement — mask 를 init 으로 실제 edge snap.
        # ISNet 사용 시에도 적용 — sharp boundary 가 봉지 같은 객체 edge 에 더 정확히 fit.
        # 봉지의 zip-seal 같은 미세 외각이 edge color gradient 따라 잡힘.
        if mask is not None:
            try:
                gc_mask = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
                k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                # v21: erode 7 / dilate 12 — fg sure 더 안쪽, bg sure 더 멀리.
                # probable region 더 넓어져 GrabCut 가 edge gradient 따라 더 정밀 자름.
                fg_sure = cv2.erode(mask, k3, iterations=7)
                bg_sure_inv = cv2.dilate(mask, k3, iterations=12)
                gc_mask[mask > 0] = cv2.GC_PR_FGD
                gc_mask[fg_sure > 0] = cv2.GC_FGD
                gc_mask[bg_sure_inv == 0] = cv2.GC_BGD
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                # 2 iterations — 속도 회복용 (이전 5 → 2). 5 와 외곽선 큰 차이 X.
                cv2.grabCut(crop, gc_mask, None, bgd_model, fgd_model,
                            2, cv2.GC_INIT_WITH_MASK)
                refined = np.where(
                    (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                    255, 0).astype(np.uint8)
                if refined.sum() > (x2-x1)*(y2-y1)*0.03 * 255:
                    mask = refined
            except Exception:
                pass
        # 1c) Depth-based 정밀 refinement — RealSense depth.
        # 투명 객체 (bottle/cup) 는 IR depth 가 unreliable → skip.
        is_transparent = phrase is not None and any(
            t in str(phrase).lower() for t in ('bottle', 'cup', 'glass'))
        if (mask is not None and depth_arr is not None
                and not is_transparent and not isnet_used):
            try:
                depth_crop = depth_arr[y1:y2, x1:x2].astype(np.float32)
                # mask 내 평균 depth (대표 깊이)
                mask_pixels = depth_crop[mask > 0]
                valid = mask_pixels[(mask_pixels > 200) & (mask_pixels < 1500)]
                if valid.size > 50:
                    med = float(np.median(valid))
                    # ±25mm 깊이 범위 (실제 객체 두께 정도)
                    depth_ok = ((depth_crop > med - 25) &
                                (depth_crop < med + 25)).astype(np.uint8) * 255
                    # SAM mask ∩ depth mask
                    refined = cv2.bitwise_and(mask, depth_ok)
                    # 너무 작아지면 fallback (SAM mask 유지)
                    if (refined > 0).sum() > mask.sum() * 0.4 // 255 * 255:
                        mask = refined
            except Exception:
                pass
        if mask is not None:
            # 가장 큰 connected component
            n_l, lbl_img, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
            if n_l > 1:
                largest_l = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                mask = ((lbl_img == largest_l).astype(np.uint8)) * 255
            # ISNet sharp boundary 사용 시 morphology/bilateral skip → 외곽 보존.
            if not isnet_used:
                # 가장자리 부드럽게 (지글거림 제거)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                # Edge alignment: jointBilateralFilter 로 mask boundary 를 색 경계에 맞춤
                # (mask 영역 자르지 않고 boundary 만 부드럽게 정렬)
                try:
                    mask_f = mask.astype(np.float32)
                    # color-guided bilateral: 약한 sigma 로 mask 가장자리만 정렬
                    mask_bf = cv2.bilateralFilter(mask_f, 5, 35, 7)
                    _, mask = cv2.threshold(
                        mask_bf.astype(np.uint8), 80, 255, cv2.THRESH_BINARY)
                    # closing 으로 작은 구멍 채움
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                except Exception:
                    pass
        # 2) Otsu fallback
        if mask is None:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (mask == 255).sum() > (mask == 0).sum():
            mask = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 50:
            return None
        rect = cv2.minAreaRect(largest)  # ((cx,cy),(w,h),angle)
        (rw, rh) = rect[1]
        angle = rect[2]
        if rw < rh:
            angle += 90
            rw, rh = rh, rw
        angle = angle % 180
        # mask 의 axis-aligned bbox
        ys_m, xs_m = np.where(mask > 0)
        if xs_m.size > 0:
            mask_bbox = (int(xs_m.min()) + x1, int(ys_m.min()) + y1,
                         int(xs_m.max()) + x1, int(ys_m.max()) + y1)
        else:
            mask_bbox = None
        # mask 외곽 contour (영상 절대 좌표) — 핑크선용 + 노란 꼭지점 추출용 (동일!)
        # ISNet 사용 시 GaussianBlur skip 해서 sharp edge 보존.
        # SAM2/HQ-SAM 사용 시 약한 blur 로 지글거림만 정리.
        mask_contour = None
        c_largest = largest  # default
        mask_for_ray = mask  # default, ray cast 에 사용할 mask (drawn contour 와 동일)
        if contours:
            if isnet_used:
                cnt_use, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                c_largest = (max(cnt_use, key=cv2.contourArea)
                             if cnt_use else largest)
                # mask_for_ray 는 mask 그대로
            else:
                mask_smooth = cv2.GaussianBlur(mask, (5, 5), 0)
                _, mask_smooth = cv2.threshold(mask_smooth, 127, 255,
                                               cv2.THRESH_BINARY)
                cnt_smooth, _ = cv2.findContours(
                    mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                c_largest = (max(cnt_smooth, key=cv2.contourArea)
                             if cnt_smooth else largest)
                mask_for_ray = mask_smooth  # drawn contour 와 동일 mask 로 ray
            # 가장 큰 contour, crop-local → 영상 절대
            c = c_largest.astype(np.int32).copy()
            c[:, :, 0] += x1
            c[:, :, 1] += y1
            mask_contour = c
        # 꼭지점 — 센터(drawn contour 중심) 에서 4방향(↑→↓←) ray cast.
        # mask_for_ray 는 drawn contour 와 동일한 mask → 점이 정확히 외곽선 위.
        cnt_pts = c_largest.reshape(-1, 2).astype(np.float32)
        if cnt_pts.shape[0] >= 4:
            # drawn contour 의 centroid (mask 가 아닌 contour 모먼트)
            Mc = cv2.moments(c_largest)
            if Mc['m00'] > 0:
                ccx = Mc['m10'] / Mc['m00']
                ccy = Mc['m01'] / Mc['m00']
            else:
                ccx = (x2 - x1) / 2.0
                ccy = (y2 - y1) / 2.0
            mh, mw = mask_for_ray.shape[:2]
            cx_i = int(round(ccx)); cy_i = int(round(ccy))
            # 4방향 ray cast — mask_for_ray 안에서 boundary 까지
            def _ray(dx, dy):
                x, y = cx_i, cy_i
                last_in = (cx_i, cy_i)
                for _ in range(max(mw, mh)):
                    x += dx; y += dy
                    if x < 0 or x >= mw or y < 0 or y >= mh:
                        return last_in
                    if mask_for_ray[y, x] > 0:
                        last_in = (x, y)
                    else:
                        return last_in
                return last_in
            up    = _ray(0, -1)
            right = _ray(1,  0)
            down  = _ray(0,  1)
            left  = _ray(-1, 0)
            corners = [
                (float(up[0])    + x1, float(up[1])    + y1),
                (float(right[0]) + x1, float(right[1]) + y1),
                (float(down[0])  + x1, float(down[1])  + y1),
                (float(left[0])  + x1, float(left[1])  + y1),
            ]
        else:
            box = cv2.boxPoints(rect)
            corners = [(float(p[0]) + x1, float(p[1]) + y1) for p in box]
        return {
            'corners': corners,
            'angle_deg': float(angle),
            'pixel_w': float(rw),
            'pixel_h': float(rh),
            'mask_bbox': mask_bbox,
            'mask_contour': mask_contour,
        }

    # ---------- GroundingDINO background worker ----------
    def _qwen_worker(self, frame_bgr):
        """background: GD 박스 중 라벨 안 된 phrase 1개 처리 (cycling).
        crop 후 Qwen 에 "이 객체 한국어 이름 + 브랜드" 단답 요청.
        결과는 self.qwen_phrase_label[phrase] 에 캐시.
        """
        try:
            with self.gd_lock:
                gd_snap = list(self.gd_results)
            # 라벨 안 된 첫 박스 선택
            target = None
            for det in gd_snap:
                phrase = det[4]
                if phrase not in self.qwen_phrase_label:
                    target = det
                    break
            if target is None:
                # 모든 phrase 라벨 됨 → 첫 박스 재추론으로 갱신 (refresh)
                if gd_snap:
                    target = gd_snap[0]
                else:
                    return
            x1, y1, x2, y2, phrase, score = target
            H, W = frame_bgr.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            prompt = (f"이 사진의 객체를 한국어 한 줄로 정확히 답해. "
                      f"형식: '<색깔> <브랜드/제품명> <종류>'. 예: '초록색 칠성사이다 캔', "
                      f"'농심 포테토칲 봉지'. 추가 설명 없이 한 줄만.")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = self.qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.qwen_processor(
                text=[text], images=[pil_img],
                padding=True, return_tensors="pt").to('cuda')
            with _torch.no_grad():
                output_ids = self.qwen_model.generate(
                    **inputs, max_new_tokens=40, do_sample=False)
            gen_ids = [out[len(inp):] for inp, out
                       in zip(inputs.input_ids, output_ids)]
            response = self.qwen_processor.batch_decode(
                gen_ids, skip_special_tokens=True)[0].strip()
            # 한 줄만, 너무 길면 자름
            response = response.split('\n')[0].strip()[:40]
            # 후처리: 양끝 따옴표 제거 + 첫 단어가 "X색" 패턴이면 제거
            response = response.strip().strip("'").strip('"').strip("'").strip('"').strip()
            parts = response.split(maxsplit=1)
            if parts and parts[0].endswith('색'):
                response = parts[1] if len(parts) > 1 else ''
            response = response.strip().strip("'").strip('"').strip()
            with self.qwen_lock:
                self.qwen_phrase_label[phrase] = response
                self.qwen_result_text = response
            self.get_logger().info(f'[Qwen2.5-VL] {phrase} → {response}')
        except Exception as e:
            self.get_logger().warn(f'[Qwen2.5-VL] 추론 실패: {e}')
        finally:
            self.qwen_busy = False

    def _ocr_worker(self, frame_bgr):
        """background thread: PaddleOCR 추론 → self.ocr_results 갱신."""
        try:
            res = self.ocr_model.ocr(frame_bgr, cls=True)
            results = []
            if res and res[0]:
                for line in res[0]:
                    box, (text, score) = line
                    if score < self.ocr_min_score or not text.strip():
                        continue
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
                    results.append((x1, y1, x2, y2, text.strip(), float(score)))
            with self.ocr_lock:
                self.ocr_results = results
            if results:
                self.get_logger().info(
                    f'[OCR] {len(results)} texts: '
                    f'{[(r[4], round(r[5],2)) for r in results[:8]]}')
        except Exception as e:
            self.get_logger().warn(f'[OCR] 추론 실패: {e}')
        finally:
            self.ocr_busy = False

    def _gd_worker(self, frame_bgr):
        """background thread: 1 frame 추론 → self.gd_results 갱신."""
        try:
            import torchvision.transforms.functional as TF
            H, W = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t = TF.to_tensor(rgb)
            # GD 가 기대하는 normalize
            t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            boxes, logits, phrases = gd_predict(
                model=self.gd_model, image=t,
                caption=self.gd_prompts,
                box_threshold=self.gd_box_thr,
                text_threshold=self.gd_text_thr)
            # boxes: cxcywh normalized → x1y1x2y2 pixel
            raw = []
            for box, logit, phrase in zip(boxes, logits, phrases):
                cx, cy, w, h = box.tolist()
                x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
                x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
                raw.append((x1, y1, x2, y2, phrase, float(logit)))
            # NMS: 같은 객체 2 detection 제거 (IoU > 0.1 또는 containment > 0.5)
            def _iou(a, b):
                ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
                ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
                iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                inter = iw * ih
                area_a = max(0, (a[2]-a[0])*(a[3]-a[1]))
                area_b = max(0, (b[2]-b[0])*(b[3]-b[1]))
                union = area_a + area_b - inter
                return inter, area_a, area_b, (inter / union if union > 0 else 0)
            order = sorted(range(len(raw)), key=lambda i: -raw[i][5])
            keep = []
            for i in order:
                dup = False
                for j in keep:
                    inter, area_i, area_j, iou = _iou(raw[i][:4], raw[j][:4])
                    # 매우 강한 NMS — 같은 위치 객체 dedup
                    if (iou > 0.02
                            or (area_i > 0 and inter / area_i > 0.2)
                            or (area_j > 0 and inter / area_j > 0.2)):
                        dup = True
                        break
                if not dup:
                    keep.append(i)
            results = [raw[i] for i in keep][:6]  # 화면 정리: 상위 6개만
            with self.gd_lock:
                self.gd_results = results
            self.get_logger().info(f'[GD] {len(results)} detections: '
                                    f'{[(r[4], round(r[5],2)) for r in results]}')
        except Exception as e:
            self.get_logger().warn(f'[GD] 추론 실패: {e}')
        finally:
            self.gd_busy = False

    # ---------- 좌표 변환 ----------
    def _eih_tcam_cb(self, msg):
        """eih_fk_publisher 의 /eih/T_cam2base (Float64MultiArray 16) → 실시간 T_cam2base.
        eye-in-hand: 카메라가 그리퍼에 달려 움직이므로 FK 로 매 순간 갱신된 변환 사용.
        참조 교체는 GIL 하 원자적이라 별도 lock 불필요 (spin_camera 가 읽음)."""
        if len(msg.data) >= 16:
            self.T_cam2base = np.array(msg.data[:16], dtype=float).reshape(4, 4)
            if not self._eih_active:
                self._eih_active = True
                self.get_logger().info(
                    '[eih] /eih/T_cam2base 수신 시작 — 실시간 eye-in-hand 변환으로 전환')

    # ---------- object_tracking 식 GraspGen 워크플로우 (1-9 lock + g/s/p) ----------
    def _selected(self):
        """현재 lock된 검출 dict (트랙키 기준, 없으면 None)."""
        if self.locked and self.locked_tk is not None:
            for d in self.detections:
                if d.get('_tk') == self.locked_tk:
                    return d
            return None
        return self.detections[0] if self.detections else None

    @staticmethod
    def _apply_grasp_yaw(R_grasp, approach):
        """can/bottle 파지를 월드 Z축 기준 XY평면 안에서 yaw 회전 (수직 안 들고 수평 유지,
        approach 방향만 꺾음). WSN_GRASP_YAW(deg, 기본 90). approach 도 회전 반영해 반환."""
        _yaw = np.radians(float(os.environ.get('WSN_GRASP_YAW', '90')))
        if abs(_yaw) < 1e-6:
            return R_grasp, approach
        _cz, _sz = np.cos(_yaw), np.sin(_yaw)
        _Rz = np.array([[_cz, -_sz, 0.0], [_sz, _cz, 0.0], [0.0, 0.0, 1.0]])
        R2 = _Rz @ np.asarray(R_grasp, dtype=float)
        return R2, R2[:, 2].copy()

    def _level_grasp_orient(self, approach, R_grasp, ggl, Cxy):
        """can/bottle 옆면 수평 레벨링 → (quat[x,y,z,w], approach). 단일·goalset 후보 공용."""
        approach = np.asarray(approach, dtype=float)
        approach = approach / (np.linalg.norm(approach) + 1e-9)
        if ggl in ('can', 'pet_bottle') and os.environ.get('WSN_LEVEL_GRASP', '1') != '0':
            if (os.environ.get('WSN_RADIAL_APPROACH', '0') != '0'
                    and np.linalg.norm(Cxy) > 0.05):
                a_h = np.array([Cxy[0], Cxy[1], 0.0])
            else:
                a_h = np.array([approach[0], approach[1], 0.0])
            n = np.linalg.norm(a_h)
            if n > 1e-3:
                approach = a_h / n
            up = np.array([0.0, 0.0, 1.0])
            x = np.cross(up, approach); x /= (np.linalg.norm(x) + 1e-9)
            y = np.cross(approach, x); y /= (np.linalg.norm(y) + 1e-9)
            if os.environ.get('WSN_LEVEL_SWAP', '1') != '0':
                x, y = y, -x
            R_grasp = np.column_stack([x, y, approach])
            R_grasp, approach = self._apply_grasp_yaw(R_grasp, approach)  # XY평면 90° 꺾기
        return _rotm_to_quat(R_grasp), approach

    def _publish_grasp_candidates(self, grasps, confs, idx, center, C, det):
        """[goalset 실험] 필터 통과 후보들의 EE pose(C앵커 - gripper_offset·approach) →
        PoseArray 발행 → curobo plan_grasp 이 무충돌·도달 best 선택."""
        if idx is None or len(idx) == 0:
            idx = np.arange(len(grasps))
        idx = np.asarray(idx)
        idx = idx[np.argsort(-confs[idx])][:int(os.environ.get('WSN_GOALSET_N', '20'))]
        ggl = det.get('gg_label')
        Cxy = np.asarray(det.get('center_base'), dtype=float)[:2]
        pa = PoseArray()
        pa.header.frame_id = 'base_link'
        pa.header.stamp = self.get_clock().now().to_msg()
        for _i in idx:
            _Ti = grasps[int(_i)].copy(); _Ti[:3, 3] += center
            _qi, _ai = self._level_grasp_orient(_Ti[:3, 2], _Ti[:3, :3], ggl, Cxy)
            _ee = np.asarray(C, dtype=float) - self.gripper_offset * np.asarray(_ai)
            p = Pose()
            p.position.x, p.position.y, p.position.z = float(_ee[0]), float(_ee[1]), float(_ee[2])
            p.orientation.x, p.orientation.y = float(_qi[0]), float(_qi[1])
            p.orientation.z, p.orientation.w = float(_qi[2]), float(_qi[3])
            pa.poses.append(p)
        self.pub_grasp_candidates.publish(pa)
        self.get_logger().info(
            f"[g] goalset 후보 {len(pa.poses)}개 발행 (/dsr01/curobo/grasp_candidates)")

    # ───────────────────────── 자동 진열 ('a' 키) ─────────────────────────
    def _operator_cmd_cb(self, msg):
        """대시보드 /dashboard/operator_cmd 수신 → 'a'(auto_restock) 와 동일 동작.
        start/auto/restock/매대정리 → 시작, abort/stop/cancel → 중단."""
        cmd = (msg.data or '').strip().lower()
        self.get_logger().info(f"[operator_cmd] 수신: '{cmd}'")
        if cmd in ('start', 'auto', 'restock', 'a', '매대정리', '매대 정리 시작'):
            self.get_logger().info("[operator_cmd] → 자동 진열 시작 (대시보드 버튼)")
            self.auto_restock()
        elif cmd in ('abort', 'stop', 'cancel', '중단', '정지'):
            self.get_logger().info("[operator_cmd] → 자동 진열 중단")
            self._auto_running = False

    def auto_restock(self):
        """'a' 키: 전자동 진열 루프.
        v(매대확인)→없는 제품 파악→h(테이블뷰)→없는 제품 매칭 물체 중 y작은순(동률 x작은순)
        선택→lock→g(첫 파지)→p(픽)→픽완료 대기→place(매대 슬롯)→다 채울 때까지 반복.
        백그라운드 스레드 (디스플레이 비블록). 중단: 다시 'a' 또는 SPACE(비상정지)."""
        if getattr(self, '_auto_running', False):
            self.get_logger().warn('[auto] 이미 실행 중 — 무시 (멈추려면 SPACE 비상정지)')
            return
        self._auto_running = True
        threading.Thread(target=self._auto_restock_worker, daemon=True).start()

    def _auto_wait_settle(self, timeout=30.0, settle_n=8, min_move=0.03):
        """팔(eye-in-hand 카메라 T_cam2base)이 '움직였다가 멈출' 때까지 대기 = 모션 완료 감지.
        ★중요: 픽은 그리퍼 열기(Step1)로 시작해서 팔이 잠깐 안 움직임 → 그 정지를 모션완료로
        오판하면 place 가 너무 일찍 호출돼 빈손으로 매대로 감(사용자 지적 2026-06-19).
        그래서 시작위치 대비 min_move(기본 3cm) 이상 '움직인 뒤'에만 정지를 완료로 인정."""
        time.sleep(1.2)                     # 이동 시작 대기(출발 전 '정지' 오판 방지)
        t0 = time.time(); last = None; stable = 0
        start_p = None; moved = False
        while time.time() - t0 < timeout:
            T = self.T_cam2base
            if T is not None:
                p = np.asarray(T)[:3, 3].astype(float)
                if start_p is None:
                    start_p = p
                if np.linalg.norm(p - start_p) > min_move:
                    moved = True            # 팔이 실제로 움직이기 시작함
                if last is not None and np.linalg.norm(p - last) < 0.002:
                    stable += 1
                    if moved and stable >= settle_n:   # ★움직인 뒤 멈췄을 때만 완료
                        return True
                else:
                    stable = 0
                last = p
            time.sleep(0.1)
        return False

    @staticmethod
    def _missing_match(det_name, missing_key):
        """매대재고 키(bottle/can/snack) ↔ 검출 이름(bottle/pet_bottle/can/snack_bag/coffee) 매칭."""
        n = (det_name or '').lower()
        if missing_key == 'bottle':
            return 'bottle' in n or 'pet' in n
        if missing_key == 'can':
            return 'can' in n
        if missing_key == 'snack':
            return 'snack' in n
        return False

    def _auto_restock_worker(self):
        log = self.get_logger()
        try:
            log.info('[auto] ===== 자동 진열 시작 =====')
            # 1) 매대 확인 → 없는 제품 리스트
            self.shelf_missing = None
            self._go_shelf_and_check()                       # 내부 스레드(이동+재고확인)
            t0 = time.time()
            while self.shelf_missing is None and time.time() - t0 < 40.0:
                if not self._auto_running:
                    return
                time.sleep(0.3)
            remaining = list(self.shelf_missing or [])
            log.info(f'[auto] 매대에 없는(채울) 제품: {remaining}')
            if not remaining:
                log.info('[auto] 매대 다 채워짐 — 종료'); return

            max_loops = len(remaining) + 3
            loop = 0
            while remaining and loop < max_loops and self._auto_running:
                loop += 1
                # 2) 테이블 뷰로 이동(h) + 검출 안정 대기
                self._go_product_view()
                self._auto_wait_settle(timeout=25.0)
                time.sleep(1.5)

                # 3) ★검출이 나타날 때까지 최대 10초 폴링 후 선택 (테이블뷰 도착 직후엔
                #   검출이 아직 갱신 안 돼 비어있음 → 바로 break 되던 문제. 사용자 2026-06-19)
                #   남은 제품과 매칭되는 검출 중 y작은→x작은 순으로 선택.
                #   ★최소 수집시간(_det_min) 동안 모든 물체가 다 뜰 때까지 모은 뒤 선택 →
                #   일찍 골라서 엉뚱한 거(아직 안 뜬 y작은 물체 대신 먼저 뜬 것) 잡던 문제 해결.
                _det_wait = float(os.environ.get('WSN_AUTO_DETECT_WAIT', '15'))
                _det_min = float(os.environ.get('WSN_AUTO_DETECT_MIN', '10'))   # 10초 본 뒤 선택 (사용자)
                cands = []
                _t_det = time.time()
                while time.time() - _t_det < _det_wait and self._auto_running:
                    cands = []
                    for d in list(self.detections):
                        cb = d.get('center_base')
                        if cb is None:
                            continue
                        for mk in remaining:
                            if self._missing_match(d.get('name'), mk):
                                cands.append((float(cb[0]), float(cb[1]), mk, d))
                                break
                    # 후보 있고 + 최소 수집시간 지났으면 선택 (그 전엔 계속 모음 = 모든 물체 등장 대기)
                    if cands and (time.time() - _t_det) >= _det_min:
                        log.info(f'[auto] 검출 확보: {len(cands)}개 (대기 {time.time()-_t_det:.1f}s)')
                        break
                    time.sleep(0.3)
                if not cands:
                    log.warn(f'[auto] 테이블에서 {remaining} 매칭 물체 못 찾음 ({_det_wait:.0f}초 대기 후) — 종료')
                    break
                cands.sort(key=lambda c: (c[1], c[0]))       # y작은→동률시 x작은 (사용자 2026-06-18)
                x, y, mk, det = cands[0]
                log.info(f'[auto] 선택: {det.get("name")} (→{mk}) '
                         f'x={x*1000:.0f} y={y*1000:.0f}mm')

                # 4) lock → g(첫 파지 생성) → p(픽 발행)
                self.locked = True
                self.locked_tk = det.get('_tk')
                self.locked_idx = None
                self.locked_class = str(det.get('name', '?'))   # lock 순간 클래스 고정
                self.send_graspgen()
                time.sleep(2.0)
                if self.pending_grasp_pose is None:
                    log.warn('[auto] g 파지 생성 실패 — 이 물체 skip')
                    self.locked = False; self.locked_tk = None
                    time.sleep(1.0); continue
                if not self.advance_and_grip():
                    log.warn('[auto] p(픽) 발행 실패 — skip')
                    self.locked = False; self.locked_tk = None
                    time.sleep(1.0); continue

                # 5) 픽 전체(open→pregrasp→전진→잡기→lift ≈47s) 완료까지 충분히 고정 대기.
                #   ★중간 스텝 정지(pregrasp 후 2s, 잡기 중 팔 정지 등)를 모션완료로 오판해
                #   place 를 일찍 불러서 — 안 잡고 매대로 가고, place의 그리퍼열기가 잡기와
                #   충돌하던 문제 → 고정대기로 확실히 분리. (사용자 2026-06-19)
                _pw = float(os.environ.get('WSN_AUTO_PICK_WAIT', '55'))
                log.info(f'[auto] 픽 완료 대기 {_pw:.0f}s (open→pregrasp→전진→잡기→lift)...')
                _t_pw = time.time()
                while time.time() - _t_pw < _pw and self._auto_running:
                    time.sleep(0.5)

                # 6) place — curobo /move_to_place (grasp_class=물체이름 기준 슬롯)
                if self.cli_place is not None and self.cli_place.service_is_ready():
                    log.info('[auto] place 호출 (매대 슬롯에 넣기)...')
                    fut = self.cli_place.call_async(Trigger.Request())
                    t1 = time.time()
                    while not fut.done() and time.time() - t1 < 70.0:
                        time.sleep(0.2)
                    ok = bool(fut.done() and fut.result() and fut.result().success)
                    log.info(f'[auto] place 결과: {"성공" if ok else "실패/타임아웃"}')
                else:
                    log.warn('[auto] /move_to_place 서비스 미가용 — place 건너뜀')

                self.locked = False; self.locked_tk = None
                self.clear_grasp_preview()
                remaining.remove(mk)
                # 대시보드 매대 재고: 진열(옮김) 완료한 품목 → 1
                if mk in self.shelf_inv:
                    self.shelf_inv[mk] = 1
                    self._publish_shelf_inv()
                log.info(f'[auto] {mk} 진열 완료. 남은: {remaining}')

            log.info(f'[auto] ===== 자동 진열 종료 (남은: {remaining if remaining else "없음"}) =====')
        except Exception as e:
            log.error(f'[auto] 예외 — 중단: {e}')
        finally:
            self._auto_running = False
            # ★락/미리보기 보장 해제 — 예외·중단으로 빠져나가도 stuck-lock(전부 빨강 OBS,
            #   SEL 없음, 장애물 계속 발행) 방지. auto 끝나면 무조건 풀린 상태로. (2026-06-22)
            self.locked = False
            self.locked_tk = None
            self.locked_idx = None
            try:
                self.clear_grasp_preview()
            except Exception:
                pass

    def _go_shelf_and_check(self):
        """'v' 키: home(매대뷰) 자세로 이동 → 도착 후 매대재고 확인.
        디스플레이 안 멈추게 백그라운드 스레드에서 이동·대기."""
        if self.cli_product_view is None:
            self.get_logger().error('[v] move_joint 서비스 없음 — 브링업 확인'); return
        if not self.cli_product_view.service_is_ready():
            if not self.cli_product_view.wait_for_service(timeout_sec=0.5):
                self.get_logger().error('[v] move_joint 서비스 미연결'); return

        def _worker():
            req = MoveJoint.Request()
            req.pos = [float(v) for v in self.shelf_view_pose]
            req.vel = 25.0; req.acc = 25.0; req.time = 0.0
            req.radius = 0.0; req.mode = 0; req.blend_type = 0; req.sync_type = 1
            self.get_logger().info(
                f"[v] 매대뷰(home) 이동 → {[round(x,1) for x in self.shelf_view_pose]}°")
            self.cli_product_view.call_async(req)
            # ★도착 감지: 카메라 pose(eih T_cam2base)가 멈출 때까지 대기 (이동 끝난 후 확인).
            #   고정 sleep 이 아니라 실제 정지 감지 — 이동 중 확인 방지.
            time.sleep(1.0)   # 이동 시작 대기(출발 전 '정지'로 오판 방지)
            t0 = time.time(); last = None; stable = 0
            while time.time() - t0 < 20.0:
                T = self.T_cam2base
                if T is not None:
                    p = np.asarray(T)[:3, 3].astype(float)
                    if last is not None and np.linalg.norm(p - last) < 0.002:  # 2mm
                        stable += 1
                        if stable >= 8:        # ~0.8s 연속 정지 = 도착
                            break
                    else:
                        stable = 0
                    last = p
                time.sleep(0.1)
            self.get_logger().info("[v] 매대뷰 도착 — 매대재고 확인 (3초 안정화 판정)")
            time.sleep(0.3)   # 짧은 settle (실제 판정은 아래 3초 멀티프레임 최댓값)
            self.shelf_inventory_check()

        threading.Thread(target=_worker, daemon=True).start()

    def shelf_inventory_check(self):
        """'v' 키: 매대(홈뷰)에서 3종 제품(bottle/can/snack) present/absent 판정.
        ★fine-tuned YOLO '원본 2D 검출'(self.yolo_seg_results) 사용. self.detections 는 depth
        투영이 필요해서 매대(먼 거리)에선 비어버림(tracks=0) → present/absent 못 봄. 원본 2D 는
        depth 불필요라 먼 매대도 잡히고, 학습모델이라 배경(소화기 등) 오검출도 없음.
        N초간 보고 프레임 다수에서 보이면 present (깜빡임 무시). (WSN_SHELF_VIEW_SEC 기본 5초)"""
        view_sec = float(os.environ.get('WSN_SHELF_VIEW_SEC', '5.0'))
        thr = float(os.environ.get('WSN_SHELF_CONF', '0.45'))
        KW = {'bottle': ('bottle', 'pet'), 'can': ('can',), 'snack': ('snack',)}
        seen = {k: 0 for k in KW}
        _frames = 0
        t0 = time.time()
        while time.time() - t0 < view_sec:
            ys = list(getattr(self, 'yolo_seg_results', []))   # 학습 YOLO 원본 2D [(name,conf)]
            _frames += 1
            _hit = {k: False for k in KW}
            for (nm, cf) in ys:
                if cf < thr:
                    continue
                nml = str(nm).lower()
                for k, kws in KW.items():
                    if any(w in nml for w in kws):
                        _hit[k] = True
            for k in KW:
                if _hit[k]:
                    seen[k] += 1
            time.sleep(0.15)
        # 전체 프레임의 20% 이상(최소 2프레임) 보이면 present — 한두 프레임 깜빡임은 무시
        need = max(2, int(_frames * 0.2))
        present = {k: (seen[k] >= need) for k in KW}
        missing = [k for k in KW if not present[k]]
        self.shelf_missing = missing
        # 대시보드 매대 재고: 처음 매대 갔을 때 있으면 1, 없으면 0
        for k in KW:
            self.shelf_inv[k] = 1 if present[k] else 0
        self._publish_shelf_inv()
        self.get_logger().info(
            f"[매대재고] (YOLO2D {view_sec:.0f}s/{_frames}프레임 conf>={thr}, present>={need}) "
            + "  ".join(f"{k}={'O' if present[k] else 'X'}({seen[k]}/{_frames})" for k in KW)
            + f"  → 바닥에서 집을것={missing if missing else '없음(다 채워짐)'}")
        return missing

    def _publish_shelf_inv(self):
        """매대 재고(캔/바틀/스낵 0/1) JSON 을 대시보드로 발행."""
        try:
            import json
            m = String(); m.data = json.dumps(self.shelf_inv)
            self.pub_shelf_inv.publish(m)
        except Exception:
            pass

    def _snack_topdown_grasp(self, det, cloud):
        """과자봉지: 검출 중심에서 수직(top-down) 파지. approach=-Z, 핑거축=봉지 장축(rz).
        GraspGen·방위필터 우회 (누운 봉지는 위에서 수직 중심을 잡아야 잡힘).
        z=검출 높이(봉지 표면) — 캔용 grasp_fixed_z 안 씀."""
        C = np.asarray(det.get('center_base'), dtype=float)
        approach = np.array([0.0, 0.0, -1.0])                  # 위 → 아래
        rz_deg = 0.0
        bbox = det.get('bbox')
        if bbox is not None:
            rxyz = self.cloud_to_rxryrz(
                cloud, key=self._isnet_grid_key(*bbox), cname='snack_bag')
            if rxyz is not None:
                rz_deg = float(rxyz[2])
        rz = np.deg2rad(rz_deg)
        x = np.array([np.cos(rz), np.sin(rz), 0.0])            # 핑거축 (XY평면, 봉지 장축)
        y = np.cross(approach, x); y /= (np.linalg.norm(y) + 1e-9)
        R = np.column_stack([x, y, approach])
        quat = _rotm_to_quat(R)
        # 스낵 X 오프셋 + 축방향(아래) 추가 하강 — C 에 적용 (standoff 가 C 기준).
        C = np.asarray(C, dtype=float)
        C[0] += float(os.environ.get('WSN_SNACK_X_OFFSET', '0.02'))
        C = C + float(os.environ.get('WSN_SNACK_DOWN_EXTRA', '0.02')) * np.asarray(approach)
        self.pending_grasp_pose = (C, np.asarray(quat), np.asarray(approach))
        self.publish_grasp_marker(C, quat, action=Marker.ADD)
        # goalset 후보 비움 → 'p'가 standoff 단일경로(수직 하강)로 가게
        try:
            _pa = PoseArray(); _pa.header.frame_id = 'base_link'
            _pa.header.stamp = self.get_clock().now().to_msg()
            self.pub_grasp_candidates.publish(_pa)
        except Exception:
            pass
        self.get_logger().info(
            f"[g] snack_bag 수직(top-down) 파지: "
            f"중심=({C[0]*1000:.0f},{C[1]*1000:.0f},{C[2]*1000:.0f})mm "
            f"rz={rz_deg:.0f}° (위에서 수직 하강)")

    def send_graspgen(self):
        """'g' 키: 선택 물체 cloud(base,m) → GraspGen → best 6DOF 파지(오프셋 없는 '파지점').
        pending=(파지점pos, quat, 접근축) 저장 + RViz 미리보기 (로봇 안 움직임).
        's'=파지점으로 이동, 'p'=12cm 전진+집기, 'r'=취소."""
        det = self._selected()
        if det is None:
            self.get_logger().warn('선택 물체 없음 (1-9 로 lock)'); return
        if self.gg is None:
            self.get_logger().warn('GraspGen 클라이언트 없음 (서버 :5556 확인)'); return
        cloud = det.get('cloud_m')
        if cloud is None or len(cloud) < 50:
            self.get_logger().warn('포인트클라우드 부족'); return
        cloud = np.asarray(cloud, dtype=np.float32)
        # ── 과자봉지: 무조건 수직(top-down) 파지. GraspGen/방위필터 우회.
        #   gg_label 이 'snack_bag' 이거나 이름에 'snack' 포함이면 항상 수직 (수평 절대 X).
        _nm0 = str(det.get('name', '')).lower()
        if det.get('gg_label') == 'snack_bag' or 'snack' in _nm0:
            self._snack_topdown_grasp(det, cloud)
            return
        # ── 캔·바틀이 '누워있으면' 수직(top-down) 파지 (모션 확인용 활성화 2026-06-19) ──
        #   서있는 캔/바틀은 옆면 수평 파지(아래 기본 로직)지만, 누우면 위에서 수직으로 잡아야 함.
        #   center_base z 가 WSN_CYL_LIE_Z(기본 5.5cm) 미만이면 바닥에 누운 것으로 간주.
        #   _snack_topdown_grasp 가 top-down(approach=-Z, 핑거축=장축) 템플릿이라 그대로 재활용.
        # ★2단계 누움 판정 (사용자 2026-06-22): ① center z 먼저 — 높으면 확실히 서있음.
        #   ② z 가 낮으면(누움 의심) SAM 포인트클라우드 주축(PCA)으로 확인 — 주축이 수평이면
        #   진짜 누움(top-down), 수직이면 서있는데 z 만 낮게 읽힌 것(반투명 바틀이 테이블 읽음)
        #   → 옆면 파지. z 단독 오판(반투명 바틀이 뒤 테이블 읽어 낮게 나옴) 을 축으로 교정.
        _is_cyl0 = (det.get('gg_label') in ('can', 'pet_bottle')
                    or 'can' in _nm0 or 'bottle' in _nm0)
        _cz = float(np.asarray(det.get('center_base', [0, 0, 1.0]), dtype=float)[2])
        _lie_th = float(os.environ.get('WSN_CYL_LIE_Z', '0.055'))   # 5.5cm
        if _is_cyl0 and _cz < _lie_th:
            # z 낮음(누움 의심) → SAM 포인트클라우드 주축으로 진짜 누움인지 확인
            _axv = 1.0; _why = f"z={_cz*100:.1f}cm"
            try:
                _cc = np.asarray(cloud, dtype=float)
                if _cc.ndim == 2 and len(_cc) >= 20:
                    _, _, _vt = np.linalg.svd(_cc - _cc.mean(axis=0), full_matrices=False)
                    _axv = abs(float(_vt[0][2]))    # 주축 수직성분 (1=수직=서있음, 0=수평=누움)
                    _why = f"z={_cz*100:.1f}cm axisZ={_axv:.2f}"
            except Exception as _e:
                _axv = 0.0; _why = f"z={_cz*100:.1f}cm PCA실패({_e})"
            _vth = float(os.environ.get('WSN_LIE_AXIS_VERT', '0.5'))
            if _axv < _vth:      # 주축 수평 = 진짜 누움 → top-down
                self.get_logger().info(f"[g] {_nm0} 누움 확정 ({_why}) → 수직 top-down 파지")
                self._snack_topdown_grasp(det, cloud)   # 위에서 수직 중심 파지 (누운 원통)
                return
            else:                # 주축 수직 = 서있음(z만 낮게 읽힘, 반투명) → 옆면 파지
                self.get_logger().info(f"[g] {_nm0} z낮지만 축 수직=서있음 ({_why}) → 옆면 파지")
        center = cloud.mean(axis=0)
        pc = (cloud - center).astype(np.float32)   # 중심정규화 후 추론
        try:
            grasps, confs = self.gg.infer(pc)
        except Exception as e:
            self.get_logger().error(f'GraspGen infer 실패: {e}'); return
        if grasps is None or len(grasps) == 0:
            self.get_logger().warn('GraspGen: 파지 0개'); return
        grasps = np.asarray(grasps, dtype=float); confs = np.asarray(confs, dtype=float)
        # ── 파지 선택. 캔/병(원통)은 옆면 수평 파지 — approach 가 XY평면에 평행
        # (base z 성분 az≈0)인 것 우선. 그 외(스낵 등)는 conf 최고.
        _ggl = det.get('gg_label')
        _nm = str(det.get('name', '')).lower()
        # 캔/병/커피(원통형) = 무조건 옆면 수평 파지. gg_label 누락 대비 이름으로도 매칭.
        _is_cyl = (_ggl in ('can', 'pet_bottle')
                   or 'can' in _nm or 'bottle' in _nm)
        az = grasps[:, 2, 2]   # 각 grasp approach(Z열)의 base z 성분 (0=수평, ±1=수직)
        # ── 접근 방위각 제한: 기준방향 ±range 안의 grasp만 (반대쪽/뒤 접근 제외).
        # 기준 기본 = radial(베이스→물체). WSN_GRASP_AZ_REF(deg)로 고정, RANGE(기본90).
        _cxy = np.asarray(det.get('center_base'), dtype=float)[:2]
        _refdeg = os.environ.get('WSN_GRASP_AZ_REF', '')
        if _refdeg not in ('', 'auto'):
            _rr = np.radians(float(_refdeg)); _ref = np.array([np.cos(_rr), np.sin(_rr)])
        elif np.linalg.norm(_cxy) > 0.05:
            _ref = _cxy / np.linalg.norm(_cxy)
        else:
            _ref = None
        _azok = np.ones(len(grasps), dtype=bool)
        if _ref is not None:
            _axy = grasps[:, :2, 2]
            _nn = np.linalg.norm(_axy, axis=1) + 1e-9
            _dotr = (_axy[:, 0]*_ref[0] + _axy[:, 1]*_ref[1]) / _nn
            _azok = _dotr >= np.cos(np.radians(
                float(os.environ.get('WSN_GRASP_AZ_RANGE', '90'))))
        if _is_cyl:
            # 방위 OK + 수평-ish(|az|<0.45) 중 선택. 없으면 단계적 완화.
            _ok = np.where((np.abs(az) < 0.45) & _azok)[0]
            if _ok.size == 0:
                _ok = np.where(_azok)[0]
            if _ok.size == 0:
                _ok = np.arange(len(grasps))
            # 원기둥은 모든 옆면이 동등(conf 비슷) → max-conf 뽑으면 매번 방향 튐.
            # 선호방위(WSN_GRASP_AZ_REF, _ref)에 '가장 가까운' 후보 선택 → 안정적 + 선호방향 우선.
            if _ref is not None:
                best = int(_ok[np.argmax(_dotr[_ok])])
                _sel = "선호방위近"
            else:
                best = int(_ok[np.argmax(confs[_ok])]); _sel = "conf最高"
            self.get_logger().info(
                f"[g] {_ggl} 옆면 파지[{_sel}]: az={az[best]:+.2f} conf={confs[best]:.2f} "
                f"approach방위={np.degrees(np.arctan2(grasps[best,1,2],grasps[best,0,2])):.0f}° "
                f"(기준 {os.environ.get('WSN_GRASP_AZ_REF','radial')}° / 범위±{os.environ.get('WSN_GRASP_AZ_RANGE','90')}° OK {int(_azok.sum())}/{len(grasps)})")
        else:
            _ok = np.where(_azok)[0]
            best = int(_ok[np.argmax(confs[_ok])]) if _ok.size else int(np.argmax(confs))
        T = grasps[best].copy()
        T[:3, 3] += center                      # 중심정규화 복원 (base frame, m)
        pos = T[:3, 3].copy()                    # GraspGen 파지점 (오프셋 X)
        approach = T[:3, 2]; approach = approach / (np.linalg.norm(approach) + 1e-9)
        R_grasp = T[:3, :3].copy()
        # 캔/병 옆면 파지: 그리퍼를 XY평면에 평행(수평)하게 강제.
        #   approach 의 수직성분 제거 → 완전 수평 접근축. 핑거축도 수평(XY평면).
        #   binormal = 수직(world +Z). → 그리퍼 전체가 XY평면에 평평하게 누움.
        # (WSN_LEVEL_GRASP=0 으로 off, 핑거축 90° 어긋나면 WSN_LEVEL_SWAP=1)
        if (_is_cyl
                and os.environ.get('WSN_LEVEL_GRASP', '1') != '0'):
            # 방위각: 캔/병은 원통이라 어느 방향서 잡아도 됨 → 베이스→물체 직선(radial)
            # 방향으로 고정. GraspGen 의 45° 대각 방위 제거 + 손목 회전(spin) 최소화.
            # (WSN_RADIAL_APPROACH=0 이면 GraspGen 방위 투영 사용)
            _Cxy = np.asarray(det.get('center_base'), dtype=float)[:2]
            if (os.environ.get('WSN_RADIAL_APPROACH', '0') != '0'   # 기본 OFF (g 마다 방향 다양)
                    and np.linalg.norm(_Cxy) > 0.05):
                a_h = np.array([_Cxy[0], _Cxy[1], 0.0])       # radial (베이스→물체)
            else:
                a_h = np.array([approach[0], approach[1], 0.0])  # GraspGen 방위 투영
            n = np.linalg.norm(a_h)
            if n > 1e-3:
                approach = a_h / n                            # 완전 수평 접근축
                up = np.array([0.0, 0.0, 1.0])
                x = np.cross(up, approach); x /= (np.linalg.norm(x) + 1e-9)  # 수평 핑거축
                y = np.cross(approach, x); y /= (np.linalg.norm(y) + 1e-9)   # 수직 binormal
                if os.environ.get('WSN_LEVEL_SWAP', '1') != '0':   # 기본 swap ON
                    x, y = y, -x                                   # 핑거축 90° (RH-P12 보정)
                R_grasp = np.column_stack([x, y, approach])
                R_grasp, approach = self._apply_grasp_yaw(R_grasp, approach)  # XY평면 90° 꺾기
        quat = _rotm_to_quat(R_grasp)
        # 앵커 = 안정적인 캔 검출 중심 C (GraspGen pos 는 노이즈 심해 안 씀).
        C = np.asarray(det.get('center_base'), dtype=float)
        if self.grasp_fixed_z is not None:
            # ★bottle 은 키 커서 더 높이 잡음(z=70mm) → 낮은 자세 도달실패 회피. 캔 등은 기본(57.5).
            #   (사용자 2026-06-22). WSN_GRASP_FIXED_Z_BOTTLE 로 조정.
            _nm = str(det.get('name', '')).lower()
            if 'bottle' in _nm:
                C[2] = float(os.environ.get('WSN_GRASP_FIXED_Z_BOTTLE', '70')) / 1000.0
            else:
                C[2] = self.grasp_fixed_z   # 잡는 높이 고정 (검출 z 무시)
        # ★X축 오프셋은 더 이상 C 에 섞지 않음 — curobo 가 '축방향 전진 전에' 별도 단계로
        #   월드 X 0.5cm 옆이동(CUROBO_PICK_X_SHIFT). C 에 섞으면 cuRobo 전진경로가
        #   X 로 갔다 돌아오며 흔들려서(사용자 지적 2026-06-18) 분리함. 여기선 C=물체중심.
        _xo = float(os.environ.get('WSN_S_X_OFFSET', '0.0'))
        if _xo and (_ggl == 'pet_bottle' or 'bottle' in str(det.get('name', '')).lower()):
            _xo += float(os.environ.get('WSN_BOTTLE_X_EXTRA', '0.0'))
        C[0] += _xo
        # pending = (캔중심 C, quat, approach) — s/p 모두 C 기준으로 계산.
        self.pending_grasp_pose = (C, np.asarray(quat), np.asarray(approach))
        self.publish_grasp_marker(C, quat, action=Marker.ADD)
        _pt = C - self.gripper_offset * np.asarray(approach)   # p 그리퍼밑동 타겟
        _cb = np.asarray(det.get('center_base', [0, 0, 0])) * 1000.0  # 캔 검출 중심(mm)
        self.get_logger().info(
            f"[g] {det['name']} conf={float(confs[best]):.3f} "
            f"캔중심=({_cb[0]:.0f},{_cb[1]:.0f},{_cb[2]:.0f}) "
            f"pos=({pos[0]*1000:.0f},{pos[1]*1000:.0f},{pos[2]*1000:.0f})mm "
            f"approach=({approach[0]:+.2f},{approach[1]:+.2f},{approach[2]:+.2f}) "
            f"[az<0=아래로] p타겟(밑동)=({_pt[0]*1000:.0f},{_pt[1]*1000:.0f},{_pt[2]*1000:.0f})mm "
            f"오프셋={self.gripper_offset*100:.0f}cm")
        # [goalset 실험] 필터 통과 후보(_ok) 전부 발행 → curobo plan_grasp 선택
        self._publish_grasp_candidates(grasps, confs, _ok, center, C, det)

    def _depth_obstacles(self, exclude_xy=None, exclude_r=0.07):
        """분류기(YOLO/GD)가 못 잡아도 depth 클라우드로 '테이블 위 모든 물체'를
        클러스터링해 장애물 박스로 반환. top-down 홈뷰에서 seg 가 0개여도 동작.
        exclude_xy(m): 타깃 중심 — 그 반경(exclude_r m) 안 점은 제외(잡을 물체)."""
        depth_arr = getattr(self, '_last_depth', None)
        if depth_arr is None or self.T_cam2base is None \
                or getattr(self, 'intr', None) is None:
            return []
        H, W = depth_arr.shape
        step = max(1, W // 140)                       # ~140px 가로 해상도(촘촘)
        sub = depth_arr[::step, ::step].astype(np.float32) * 0.001   # m
        vv, uu = np.mgrid[0:H:step, 0:W:step]
        m = (sub > 0.1) & (sub < 1.2)
        if int(m.sum()) < 30:
            return []
        Z = sub[m]; U = uu[m].astype(np.float32); V = vv[m].astype(np.float32)
        fx, fy = self.intr.fx, self.intr.fy
        cx, cy = self.intr.ppx, self.intr.ppy
        X = (U - cx) / fx * Z; Y = (V - cy) / fy * Z
        cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=0)
        base = (self.T_cam2base @ cam)[:3].T * 1000.0   # mm, Nx3
        bx, by, bz = base[:, 0], base[:, 1], base[:, 2]
        # 장애물 후보 영역 — 작업영역보다 좁게(가장자리 펑보드/뒷판 배경 배제).
        # env 로 조정 가능(WSN_OBS_X/Y).
        XMIN = float(os.environ.get('WSN_OBS_XMIN', '300'))
        XMAX = float(os.environ.get('WSN_OBS_XMAX', '750'))
        YMIN = float(os.environ.get('WSN_OBS_YMIN', '-450'))
        YMAX = float(os.environ.get('WSN_OBS_YMAX', '550'))
        TABLE = -30.                                       # 테이블 base z(mm)
        keep = ((bx > XMIN) & (bx < XMAX) & (by > YMIN) & (by < YMAX)
                & (bz > TABLE + 40.) & (bz < 350.))        # 테이블 위 4cm~35cm
        bx, by, bz = bx[keep], by[keep], bz[keep]
        pu, pv = U[keep], V[keep]                          # 원본 픽셀(시각화용)
        if bx.size < 20:
            return []
        if exclude_xy is not None:
            far = ((bx - exclude_xy[0] * 1000.) ** 2
                   + (by - exclude_xy[1] * 1000.) ** 2) > (exclude_r * 1000.) ** 2
            bx, by, bz = bx[far], by[far], bz[far]
            pu, pv = pu[far], pv[far]
        if bx.size < 12:
            return []
        # 2cm XY 점유격자 → 연결요소 클러스터
        res = 20.0
        GW = int((XMAX - XMIN) / res) + 1
        GH = int((YMAX - YMIN) / res) + 1
        gx = np.clip(((bx - XMIN) / res).astype(int), 0, GW - 1)
        gy = np.clip(((by - YMIN) / res).astype(int), 0, GH - 1)
        occ = np.zeros((GH, GW), np.uint8)
        occ[gy, gx] = 255
        occ = cv2.dilate(occ, np.ones((3, 3), np.uint8), iterations=1)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(occ, 8)
        labels_pt = lab[gy, gx]
        obs = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 2:           # 노이즈 셀
                continue
            sel = labels_pt == i
            if int(sel.sum()) < 5:
                continue
            ox, oy, oz = bx[sel], by[sel], bz[sel]
            su, sv = pu[sel], pv[sel]
            # 5~95 백분위로 stray 점 제거(박스 풍선현상 방지)
            x5, x95 = np.percentile(ox, [5, 95])
            y5, y95 = np.percentile(oy, [5, 95])
            top = float(np.percentile(oz, 95))
            if top < TABLE + 30.:          # 테이블보다 3cm 미만 = 바닥 노이즈, 물체 아님
                continue
            cxm = float(np.median(ox)) / 1000.; cym = float(np.median(oy)) / 1000.
            _w0 = float(x95 - x5); _d0 = float(y95 - y5)
            # 벽/판 배제: 한 변이 25cm 넘거나 종횡비 3:1 넘는 길쭉한 면 = 물체 아님
            _lo = max(min(_w0, _d0), 1.0); _hi = max(_w0, _d0)
            if _hi > 250. or _hi / _lo > 3.0:
                continue
            wmm = min(max(_w0 + 30., 40.), 250.)
            dmm = min(max(_d0 + 30., 40.), 250.)
            hmm = min(max(top - TABLE, 40.), 400.)        # 테이블~top 까지 채움
            obs.append({'name': 'obj',
                        'pos': [cxm, cym, ((top + TABLE) / 2.) / 1000.],
                        'dims': [wmm / 1000., dmm / 1000., hmm / 1000.],
                        'px_bbox': (int(np.percentile(su, 5)),
                                    int(np.percentile(sv, 5)),
                                    int(np.percentile(su, 95)),
                                    int(np.percentile(sv, 95)))})
        return obs

    def _dino_obstacles(self, exclude_xy=None, exclude_r=0.07):
        """[WSN_ALL_DINO_OBS] DINO(gd_results)가 인식한 것 전부를 obstacle 로.
        bbox 중심 depth→base 3D. 타깃(exclude_xy 반경) 만 제외."""
        depth_arr = getattr(self, '_last_depth', None)
        if depth_arr is None:
            return []
        with self.gd_lock:
            dets = list(self.gd_results)
        # 번호 게이트(gd_only_min_conf)와 동일 임계 — 헛검출(0.3~0.4 bottle)이
        # 장애물로 발행되지 않게. "전부 obs"는 '확신한 DINO 전부' 의미.
        _obs_minc = float(self._otun('gd_only_min_conf', 0.42))
        out = []
        for d in dets:
            try:
                x1, y1, x2, y2, phrase, score = d[0], d[1], d[2], d[3], d[4], d[5]
            except Exception:
                continue
            if float(score) < _obs_minc:
                continue
            cu = int((float(x1) + float(x2)) / 2); cv = int((float(y1) + float(y2)) / 2)
            xyz, _st = self.pixel_to_base_xyz(cu, cv, depth_arr)
            if xyz is None:
                continue
            c = np.asarray(xyz, dtype=float).ravel() / 1000.0   # mm→m
            if exclude_xy is not None and \
               (c[0]-exclude_xy[0])**2 + (c[1]-exclude_xy[1])**2 < exclude_r**2:
                continue                                          # 잡을 타깃은 장애물 아님
            out.append({'name': 'dino:' + str(phrase).split()[0][:8],
                        'pos': [float(c[0]), float(c[1]), float(c[2])],
                        'dims': [0.08, 0.08, 0.15]})
        return out

    def _publish_obstacles(self, exclude_det, log=True):
        """타깃(exclude_det) 제외한 물체를 curobo 장애물로 발행.
        seg 검출(self.detections) + DINO 검출 전부(WSN_ALL_DINO_OBS) + depth 클러스터 합집합."""
        import json as _json
        obs = []
        ex_tk = exclude_det.get('_tk') if exclude_det else None
        ex_c = exclude_det.get('center_base') if exclude_det else None
        for d in self.detections:
            if ex_tk is not None and d.get('_tk') == ex_tk:
                continue                       # 타깃 제외 (잡을 물체는 장애물 아님)
            c = d.get('center_base')
            if c is None:
                continue
            cloud = d.get('cloud_m')
            if cloud is not None and len(cloud) >= 10:
                cl = np.asarray(cloud, dtype=float)
                dims = (cl.max(axis=0) - cl.min(axis=0)) + 0.03   # 3cm 여유
                dims = np.clip(dims, 0.04, 0.30)
            else:
                dims = np.array([0.08, 0.08, 0.15])
            obs.append({'name': str(d.get('name', 'obj')),
                        'pos': [float(c[0]), float(c[1]), float(c[2])],
                        'dims': [float(dims[0]), float(dims[1]), float(dims[2])]})
        # depth 클러스터 장애물 — seg 가 못 잡는 물체 보강. 단 펑보드/top-down 등
        # 노이즈 환경에선 거짓 장애물이 curobo plan 을 방해할 수 있어 기본 OFF.
        # 켜려면 WSN_DEPTH_OBS=1. (화면 빨간 OBS 시각화는 항상 ON, curobo 발행만 게이트)
        if os.environ.get('WSN_DEPTH_OBS', '0') != '0':
            _ex_xy = (float(ex_c[0]), float(ex_c[1])) if ex_c is not None else None
            for dob in self._depth_obstacles(exclude_xy=_ex_xy):
                _p = dob['pos']
                _dup = any((_p[0] - o['pos'][0]) ** 2 + (_p[1] - o['pos'][1]) ** 2
                           < 0.08 ** 2 for o in obs)
                if not _dup:
                    obs.append(dob)
        # ★DINO 가 인식한 것 전부를 obstacle 로 (사용자 요청). 기본 ON, 타깃 반경 제외.
        if os.environ.get('WSN_ALL_DINO_OBS', '1') != '0':
            _ex_xy = (float(ex_c[0]), float(ex_c[1])) if ex_c is not None else None
            _nd = 0
            for dob in self._dino_obstacles(exclude_xy=_ex_xy):
                _p = dob['pos']
                if not any((_p[0]-o['pos'][0])**2 + (_p[1]-o['pos'][1])**2 < 0.08**2
                           for o in obs):
                    obs.append(dob); _nd += 1
        m = String(); m.data = _json.dumps(obs)
        self.pub_obstacles.publish(m)
        if log:
            self.get_logger().info(
                f"[obstacles] 장애물 {len(obs)}개 발행 (seg+DINO전부+depth, 타깃 제외)")

    def advance_and_grip(self):
        """'p' 키: 파지점에서 접근축 방향으로 12cm 전진(카메라↔그리퍼끝 보정) + 집기 + 15cm 수직 lift.
        (파지점+12cm) 를 /dsr01/curobo/pick_pose 발행 → curobo: descend→close→lift."""
        if self.pending_grasp_pose is None:
            self.get_logger().warn("[p] 대기 파지 없음 — 'g' 먼저"); return False
        C, quat, approach = self.pending_grasp_pose   # C=캔중심
        # 타깃 외 물체 장애물 발행 → curobo world 갱신 대기 후 pick 발행
        self._publish_obstacles(self._selected())
        # 파지 대상 클래스 발행 → curobo 가 물성별 전류(grasp_force_params.yaml) + 매대좌표 적용.
        #   ★lock 순간 고정 저장된 locked_class 우선 (화면 표시와 동일 = live 재분류로 안 어긋남).
        _sel = self._selected()
        _cls_pub = getattr(self, 'locked_class', None) or (_sel.get('name') if _sel else None)
        if _cls_pub:
            self.pub_grasp_class.publish(String(data=str(_cls_pub)))
            self.get_logger().info(f"[p] 파지 클래스 발행: {_cls_pub} (lock 고정)")
        time.sleep(0.3)
        # 집기: 그리퍼밑동 = 캔중심 - 그리퍼길이*approach (손가락이 캔중심에 닿음).
        adv = np.asarray(C, dtype=float) - self.gripper_offset * np.asarray(approach)
        # X 오프셋·하강 오프셋은 이미 파지중심 C 에 반영됨(send_graspgen/_snack_topdown).
        # → 여기선 그대로 C - 그리퍼길이·approach 만. (goalset candidates 와 일관)
        msg = self._pose_msg(adv, quat)
        self.pub_pick.publish(msg)
        self.get_logger().info(
            f"[p] 집기(캔중심-{self.gripper_offset*100:.0f}cm·approach, X오프셋은 C에 반영됨) "
            f"→ /dsr01/curobo/pick_pose "
            f"({adv[0]*1000:.0f},{adv[1]*1000:.0f},{adv[2]*1000:.0f})mm (이후 15cm 수직 lift)")
        self.clear_grasp_preview()
        return True

    def _pose_msg(self, pos, quat):
        msg = PoseStamped()
        msg.header.frame_id = 'base_link'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pos[0]); msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat[0]); msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2]); msg.pose.orientation.w = float(quat[3])
        return msg

    def clear_grasp_preview(self):
        if self.pending_grasp_pose is None:
            return False
        self.pending_grasp_pose = None
        self.publish_grasp_marker(None, None, action=Marker.DELETE)
        return True

    def publish_grasp_marker(self, C, quat_xyzw, action=Marker.ADD):
        """RViz ARROW: 그리퍼 접근 경로 → 캔중심(C). 꼬리=프리그래스프, 머리=캔중심."""
        m = Marker()
        m.header.frame_id = 'base_link'; m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'graspgen_preview'; m.id = 0; m.type = Marker.ARROW; m.action = action
        if action == Marker.ADD:
            rot = self._quat_to_rotm(quat_xyzw)
            approach = rot[:, 2]
            # 캔중심(C)을 향한 화살표: 꼬리=프리그래스프(밑동-standoff 뒤), 머리=캔중심.
            C = np.asarray(C)
            start = C - (self.gripper_offset + self.pregrasp_standoff) * approach  # s
            end = C                                                                # 캔중심
            m.points = [
                Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
                Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))]
            m.scale.x = 0.012; m.scale.y = 0.025; m.scale.z = 0.0
            m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.2; m.color.a = 0.9
            m.lifetime.sec = 0
        self.marker_pub.publish(m)

    @staticmethod
    def _quat_to_rotm(q):
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

    def pixel_to_base_xyz(self, u, v, depth_arr, window=15, forced_depth=None):
        H, W = depth_arr.shape
        u, v = int(u), int(v)
        if not (0 <= u < W and 0 <= v < H):
            return None, 'out-of-frame'
        if forced_depth is not None and forced_depth > 0:
            z_m = float(forced_depth) * 0.001
        else:
            h = window // 2
            patch = depth_arr[max(0, v - h):min(H, v + h + 1),
                              max(0, u - h):min(W, u + h + 1)]
            valid = patch[patch > 0]
            if valid.size < 5:
                return None, f'depth hole ({valid.size}/{patch.size})'
            # 금속/반사 물체는 윗면 depth 구멍 → 유효값이 옆면·테이블·먼배경에
            # 쏠려 median 이 '먼 쪽'으로 편향(카메라 아래보면 z 과음수). 가까운
            # 표면(물체 자신)으로 편향되게 25분위 사용 (mask_to_base_xyz 와 통일).
            z_m = float(np.percentile(valid, 25)) * 0.001
            if z_m <= 0.05:
                return None, f'too close (z={z_m*1000:.0f}mm)'
        cam = rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], z_m)
        p = np.array([cam[0], cam[1], cam[2], 1.0])
        base_m = self.T_cam2base @ p
        return base_m[:3] * 1000.0, None      # mm

    def mask_to_base_xyz(self, mask_uint8, depth_arr):
        ys, xs = np.where(mask_uint8 > 0)
        if xs.size < 10:
            return None
        cu, cv = int(xs.mean()), int(ys.mean())
        vals = depth_arr[mask_uint8 > 0]
        vals = vals[vals > 0]
        if vals.size < 10:
            return None
        z_m = float(np.percentile(vals, 25)) * 0.001
        if z_m <= 0.05:
            return None
        cam = rs.rs2_deproject_pixel_to_point(self.intr, [float(cu), float(cv)], z_m)
        p = np.array([cam[0], cam[1], cam[2], 1.0])
        base_m = self.T_cam2base @ p
        return float(cu), float(cv), base_m[:3] * 1000.0    # mm

    def _in_work_zone(self, base_xyz):
        """검출의 base 좌표(mm)가 작업 가능 영역 안인지 판정.
        밖이면 배경/먼 물체(사람·모니터·로봇베이스 등 false positive)로 보고
        검출을 통째로 버린다. 범위는 WSN_ZONE_* env 로 조정 가능.
        실측 작업물체 범위(x 347~637, y -230~460, z -87~215mm) + 마진."""
        if base_xyz is None:
            return False
        try:
            x, y, z = float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])
        except Exception:
            return False

        def _e(name, default):
            try:
                return float(os.environ.get(name, default))
            except Exception:
                return default
        xmin = _e('WSN_ZONE_XMIN', 150.0);  xmax = _e('WSN_ZONE_XMAX', 850.0)
        ymin = _e('WSN_ZONE_YMIN', -550.0); ymax = _e('WSN_ZONE_YMAX', 650.0)
        zmin = _e('WSN_ZONE_ZMIN', -450.0); zmax = _e('WSN_ZONE_ZMAX', 450.0)
        return (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax)

    def _is_gray_nonsnack(self, frame_bgr, mask_uint8):
        """마스크 영역이 회색/저채도(로봇베이스 등)면 True → snack 오검출로 보고 억제.
        과자봉지=컬러풀(채도≥~49), 로봇베이스=회색(채도~26). 임계 WSN_SNACK_SAT(기본42).
        실측: snack sat min 48.9 / 회색물체 sat mean 26.5."""
        try:
            ys, xs = np.where(mask_uint8 > 0)
            if xs.size < 30:
                return False
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            msat = float(hsv[ys, xs, 1].mean())
            thr = float(os.environ.get('WSN_SNACK_SAT', '42'))
            gray = msat < thr
            if gray:
                self.get_logger().info(f"[snack채도] msat={msat:.1f}<{thr:.0f} → skip(회색)")
            return gray
        except Exception:
            return False

    def mask_to_base_cloud(self, mask_uint8, depth_arr, max_pts=500,
                           return_pix=False):
        """마스크 내부 픽셀들을 base 프레임 (N,3) point cloud(meter)로 deproject.
        GraspGen 입력이자 rx/ry/rz 평면 추정용. 속도 위해 max_pts 로 subsample.
        return_pix=True 면 (cloud, pix(N,2 u,v)) 반환 — 화면 점 표시용."""
        fail = (None, None) if return_pix else None
        ys, xs = np.where(mask_uint8 > 0)
        if xs.size < 30:
            return fail
        d = depth_arr[ys, xs].astype(np.float32)
        ok = d > 0
        ys, xs, d = ys[ok], xs[ok], d[ok]
        if d.size < 30:
            return fail
        # depth 이상치 제거 (중앙값 ±60mm) — 배경/엣지 노이즈 컷
        med = float(np.median(d))
        keep = np.abs(d - med) < 60.0
        ys, xs, d = ys[keep], xs[keep], d[keep]
        if d.size < 30:
            return fail
        if d.size > max_pts:                       # 균일 subsample
            idx = np.linspace(0, d.size - 1, max_pts).astype(int)
            ys, xs, d = ys[idx], xs[idx], d[idx]
        z_m = d * 0.001
        fx, fy = self.intr.fx, self.intr.fy
        ppx, ppy = self.intr.ppx, self.intr.ppy
        x_cam = (xs - ppx) / fx * z_m
        y_cam = (ys - ppy) / fy * z_m
        cam_pts = np.stack([x_cam, y_cam, z_m, np.ones_like(z_m)], axis=0)  # (4,N)
        base_pts = (self.T_cam2base @ cam_pts)[:3].T.astype(np.float32)     # (N,3) m
        if return_pix:
            return base_pts, np.stack([xs, ys], axis=1)
        return base_pts

    def _isnet_grid_key(self, x1, y1, x2, y2):
        return (((x1 + x2) // 2) // 40, ((y1 + y2) // 2) // 40)

    def _track_key(self, x1, y1, x2, y2, tol=80.0):
        """bbox 중심을 근접한 기존 track 에 연결 (grid 경계 jitter 로 key 가 바뀌어
        EMA smoothing 이 리셋→깜빡이던 문제 방지). tol 안이면 같은 track 재사용."""
        cx = (x1 + x2) * 0.5; cy = (y1 + y2) * 0.5
        if not hasattr(self, '_tracks'):
            self._tracks = {}
        now = time.time()
        best = None; bestd = 1e18
        for k, (tx, ty, tt) in list(self._tracks.items()):
            if now - tt > 3.0:
                self._tracks.pop(k, None); continue
            d = (cx - tx) ** 2 + (cy - ty) ** 2
            if d < bestd:
                bestd = d; best = k
        if best is not None and bestd <= tol * tol:
            self._tracks[best] = (cx, cy, now)
            return best
        nk = (int(cx), int(cy))
        self._tracks[nk] = (cx, cy, now)
        return nk

    def _label_box(self, vis, text, x, y, scale, color):
        """라벨을 어두운 반투명 배경 위에 그려 노란 외곽선과 구분·가독성 확보.
        이미 그린 라벨과 겹치면 아래로 밀어 겹침 방지(밀집 구역 정리).
        (x,y)=텍스트 baseline 좌상 기준."""
        f = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), bl = cv2.getTextSize(text, f, scale, 1)
        H, W = vis.shape[:2]
        x = max(2, min(x, W - tw - 4)); y = max(th + 4, min(y, H - 4))
        bh = th + bl + 6
        if not hasattr(self, '_label_rects'):
            self._label_rects = []
        # 충돌 회피: 기존 라벨과 겹치면 아래로 한 칸씩 이동(최대 12회)
        for _ in range(12):
            x1, y1 = x - 3, y - th - 3; x2, y2 = x + tw + 3, y + bl
            hit = False
            for (rx1, ry1, rx2, ry2) in self._label_rects:
                if x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1:
                    hit = True; break
            if not hit:
                break
            y += bh
            if y > H - 4:
                y = max(th + 4, y - bh * 12); x = min(x + tw // 2, W - tw - 4)
        y = max(th + 4, min(y, H - 4))
        x1, y1 = x - 3, y - th - 3; x2, y2 = x + tw + 3, y + bl
        self._label_rects.append((x1, y1, x2, y2))
        roi = vis[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if roi.size:
            roi[:] = (roi.astype(np.float32) * 0.35).astype(np.uint8)  # 어둡게
        cv2.putText(vis, text, (x, y), f, scale, color, 1, cv2.LINE_AA)

    def _smooth_bbox(self, key, bbox, a=0.4):
        """track 별 bbox 를 EMA 로 안정화. 검출 박스가 프레임마다 흔들리면 SAM2
        프롬프트(box+points)가 매번 달라져 마스크가 지글거리며 jitter → 박스를
        부드럽게 고정해 일관된 외곽선 확보. 박스가 크게 점프하면(다른 물체/이동)
        즉시 추종."""
        if not hasattr(self, '_bbox_ema'):
            self._bbox_ema = {}
        cur = np.array(bbox, dtype=np.float32)
        prev = self._bbox_ema.get(key)
        if prev is not None:
            # 중심 이동이 박스 크기 대비 크면 이동 → 즉시 추종
            pcx = (prev[0] + prev[2]) * 0.5; pcy = (prev[1] + prev[3]) * 0.5
            ccx = (cur[0] + cur[2]) * 0.5; ccy = (cur[1] + cur[3]) * 0.5
            diag = max(1.0, ((cur[2]-cur[0])**2 + (cur[3]-cur[1])**2) ** 0.5)
            moved = ((pcx-ccx)**2 + (pcy-ccy)**2) ** 0.5 > 0.5 * diag
            aa = 1.0 if moved else a
            cur = (1.0 - aa) * prev + aa * cur
        self._bbox_ema[key] = cur
        if len(self._bbox_ema) > 48:
            self._bbox_ema.pop(next(iter(self._bbox_ema)))
        return (int(round(cur[0])), int(round(cur[1])),
                int(round(cur[2])), int(round(cur[3])))

    def _otun(self, key, default):
        """외곽선 튜닝 파라미터를 /tmp/outline_params.json 에서 live 로 읽음
        (재시작 없이 반복 디버깅). mtime 캐시."""
        try:
            p = '/tmp/outline_params.json'
            st = os.stat(p)
            c = getattr(self, '_otun_cache', None)
            if c is None or c[0] != st.st_mtime:
                import json as _json
                with open(p) as _f:
                    self._otun_cache = (st.st_mtime, _json.load(_f))
            return self._otun_cache[1].get(key, default)
        except Exception:
            return default

    def _sample_mask_points(self, mask_full, n=5):
        """YOLO mask 내부에서 foreground 점 n개 샘플 (SAM2 point-prompt 용).
        erode 로 경계 피하고, 중심 + 분산 점 → specular 물체도 전체 분할 유도.
        반환: [(x,y), ...] (full-frame px) 또는 None."""
        try:
            m = (mask_full > 0).astype(np.uint8)
            if int(m.sum()) < 50:
                return None
            er = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=2)
            if int(er.sum()) < 20:
                er = m
            ys, xs = np.where(er > 0)
            if xs.size == 0:
                return None
            idx = np.linspace(0, xs.size - 1, min(n, xs.size)).astype(int)
            return [(int(xs[i]), int(ys[i])) for i in idx]
        except Exception:
            return None

    def _sam_refine_worker(self):
        """백그라운드 정밀-마스크 워커. 메인 루프가 올린 최신 (frame, bbox 목록) 을
        받아 객체별 정밀 mask 를 self._isnet_cache 에 채운다 (메인 비블록).
        우선순위: HQ-SAM(box-prompt, foundation — 라벨/반사 무관, sharp) > ISNet.
        predictor 는 _sam_lock 으로 메인 스레드(_estimate_rect)와 직렬화."""
        while not self._isnet_worker_stop:
            req = self._isnet_req
            if req is None:
                time.sleep(0.03)
                continue
            self._isnet_req = None
            frame_bgr, boxes = req
            h, w = frame_bgr.shape[:2]
            now = time.time()
            try:
                if self.hqsam_predictor is not None:
                    with self._sam_lock:
                        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        self.hqsam_predictor.set_image(rgb)
                        for item in boxes:
                            key, x1, y1, x2, y2 = item[:5]
                            pts = item[5] if len(item) > 5 else None
                            ex = int((x2 - x1) * 0.15); ey = int((y2 - y1) * 0.15)
                            box_np = np.array(
                                [max(0, x1 - ex), max(0, y1 - ey),
                                 min(w, x2 + ex), min(h, y2 + ey)],
                                dtype=np.float32)
                            if pts:
                                pc = np.array(pts, dtype=np.float32)
                                pl = np.ones(len(pts), dtype=np.int32)
                            else:
                                pc = None; pl = None
                            masks_hq, _, _ = self.hqsam_predictor.predict(
                                box=box_np[None, :], point_coords=pc,
                                point_labels=pl, multimask_output=False,
                                hq_token_only=True)
                            m = (masks_hq[0] > 0).astype(np.uint8) * 255
                            if m.shape == (h, w):
                                self._isnet_cache[key] = (now, m)
                elif self.sam2_predictor is not None:
                    with self._sam_lock:
                        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        self.sam2_predictor.set_image(rgb)
                        for item in boxes:
                            key, x1, y1, x2, y2 = item[:5]
                            pts = item[5] if len(item) > 5 else None
                            # box 만 주면 금속캔 specular 에 속아 일부만 잡음 → box 를
                            # 15% 확장(여유) + YOLO mask 내부 점(foreground anchor)
                            # 함께 prompt → 반사/누운 캔도 전체 분할.
                            ex = int((x2 - x1) * 0.15); ey = int((y2 - y1) * 0.15)
                            ebox = np.array(
                                [max(0, x1 - ex), max(0, y1 - ey),
                                 min(w, x2 + ex), min(h, y2 + ey)],
                                dtype=np.float32)
                            if pts:
                                pc = np.array(pts, dtype=np.float32)
                                pl = np.ones(len(pts), dtype=np.int32)
                            else:
                                pc = None; pl = None
                            masks_, _, _ = self.sam2_predictor.predict(
                                box=ebox[None, :], point_coords=pc,
                                point_labels=pl, multimask_output=False)
                            mm = masks_[0] if masks_.ndim == 3 else masks_
                            m = (mm > 0).astype(np.uint8) * 255
                            if m.shape == (h, w):
                                self._isnet_cache[key] = (now, m)
                elif self.rembg_session is not None:
                    for item in boxes:
                        key, x1, y1, x2, y2 = item[:5]
                        pad = 8
                        px1 = max(0, x1 - pad); py1 = max(0, y1 - pad)
                        px2 = min(w, x2 + pad); py2 = min(h, y2 + pad)
                        rgb_pad = cv2.cvtColor(frame_bgr[py1:py2, px1:px2],
                                               cv2.COLOR_BGR2RGB)
                        isnet_mask = _rembg_remove(
                            PILImage.fromarray(rgb_pad),
                            session=self.rembg_session,
                            only_mask=True, post_process_mask=True)
                        isnet_arr = np.array(isnet_mask)
                        if isnet_arr.ndim == 3:
                            isnet_arr = isnet_arr[..., 0]
                        _, isnet_bin = cv2.threshold(isnet_arr, 64, 255,
                                                     cv2.THRESH_BINARY)
                        full = np.zeros((h, w), dtype=np.uint8)
                        full[py1:py2, px1:px2] = isnet_bin
                        self._isnet_cache[key] = (now, full)
            except Exception:
                pass
            if len(self._isnet_cache) > 64:              # 오래된 캐시 정리
                for k in [k for k, v in self._isnet_cache.items()
                          if time.time() - v[0] > 8.0]:
                    self._isnet_cache.pop(k, None)

    def _refine_object_mask(self, frame_bgr, raw_mask, bbox, depth_arr=None,
                            use_isnet=True, block=False, cname=None):
        """YOLO seg raw mask → 정밀 외곽 mask(uint8 0/255).
        ISNet(sharp boundary) ∩ YOLO(위치) + depth gating + morphology + 최대성분.
        block=False: ISNet 은 워커가 채운 캐시만 사용(메인 비블록). 캐시 없으면
        기하 정제(depth+morphology)만 적용 → 워커가 채우면 다음 프레임부터 sharp."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return raw_mask
        # mkey0 는 원본 bbox 기준(워커 캐시 키와 일치). ROI 는 15% 확장 — SAM2 가
        # 박스 밖(누운 캔/포장 밖으로 삐져나온 부분)에서 찾은 픽셀도 외곽선이 박스에
        # 안 잘리게. ← "노란 외곽선이 핑크박스 밖으로 못 나간다" 버그 대응.
        mkey0 = self._isnet_grid_key(x1, y1, x2, y2)
        _mx = int((x2 - x1) * 0.15); _my = int((y2 - y1) * 0.15)
        ex1 = max(0, x1 - _mx); ey1 = max(0, y1 - _my)
        ex2 = min(w, x2 + _mx); ey2 = min(h, y2 + _my)
        # ── 모든 연산을 확장 ROI 안에서만 (full-frame 연산은 FPS 폭락).
        rmask = raw_mask[ey1:ey2, ex1:ex2].copy()
        droi = depth_arr[ey1:ey2, ex1:ex2] if depth_arr is not None else None

        def _fill_largest(m):
            """최대 외곽 contour 만 채워 solid 실루엣 (내부 구멍 제거)."""
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                return m
            big = max(cs, key=cv2.contourArea)
            o = np.zeros_like(m)
            cv2.drawContours(o, [big], -1, 255, -1)
            return o

        # 1) base mask 선택: HQ-SAM 워커가 채운 정밀 mask 우선(foundation, sharp,
        #    라벨/반사 무관). 단 SAM 이 YOLO 대비 면적 비정상(너무 크거나 작음)이면
        #    배경 새거나 일부만 → raw YOLO 로 fallback. 둘 다 확장 ROI 로 자름.
        c0 = self._isnet_cache.get(mkey0)
        base = None
        used_sam = False
        yolo_a = float((rmask > 0).sum()) + 1.0
        if c0 is not None and (time.time() - c0[0]) < 8.0:
            sam_roi = c0[1][ey1:ey2, ex1:ex2]
            sam_a = float((sam_roi > 0).sum())
            # YOLO mask 와 겹침(IoU)으로 '같은 물체' 검증 — grid 키 충돌/엉뚱한 SAM
            #  거부. SAM2 는 YOLO 보다 tight 해 면적 게이트는 넓게(0.25~5x) 두고,
            #  IoU 로 진짜 매칭만 채택. → 날카로운 SAM2 경계를 외곽선으로 사용.
            inter = float(np.logical_and(sam_roi > 0, rmask > 0).sum())
            uni = float(np.logical_or(sam_roi > 0, rmask > 0).sum()) + 1.0
            if 0.25 * yolo_a <= sam_a <= 5.0 * yolo_a and inter / uni > 0.35:
                base = sam_roi.copy()
                used_sam = True
        if base is None:
            base = rmask
        # SAM2 는 이미 sharp → 약한 close(3x3)로 경계 보존. YOLO fallback 은 거칠어
        # 5x5 close 로 메움.
        _k = 3 if used_sam else 5
        roi = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((_k, _k), np.uint8))
        roi = _fill_largest(roi)
        # 2) depth bimodal gating — 물체는 테이블보다 솟아(카메라에 가까움=작은 depth),
        #    그림자는 테이블 평면(멀음=큰 depth)에 있음. 둘 사이 간격이 충분하면
        #    중간에서 잘라 '먼(테이블/그림자) 클러스터'만 제거. 캔 밑변은 테이블과
        #    만나는 지점까지 자연히 남고, specular depth 구멍(d=0)은 건드리지 않음.
        #    평평한 봉지(near≈far)는 분리 불가라 그대로 둠.
        if droi is not None and not used_sam:
            ys, xs = np.where(roi > 0)
            if xs.size > 30:
                d = droi[ys, xs].astype(np.float32)
                dv = np.sort(d[d > 0])
                # depth 유효 픽셀이 충분(35%+)할 때만 gate. 금속/반사 캔처럼 depth
                # 대부분 무효(d=0)면 gap 판정이 노이즈라 마스크를 조각냄 → skip.
                if dv.size > 30 and dv.size > 0.35 * xs.size:
                    # 물체→테이블/그림자 분리: 정렬 depth 상위 구간에서 '빈 간격(valley)'
                    # 을 찾아, 명확한 간격이 있을 때만 먼 클러스터(그림자/테이블)를 제거.
                    # 캔은 밑변(테이블 접촉)까지 depth 가 연속이라 간격이 없어 컷 안 됨
                    # → 밑변 보존. bottle 그림자는 따로 떨어진 클러스터라 간격 있음 → 컷.
                    k0 = int(0.35 * dv.size)
                    upper = dv[k0:]
                    if upper.size > 3:
                        gaps = np.diff(upper)
                        gi = int(np.argmax(gaps))
                        gap = float(gaps[gi]); gap_at = float(upper[gi])
                        if gap > float(self._otun('depth_gap_mm', 20.0)):
                            cut = gap_at + 0.5 * gap
                            bad = (d > 0) & (d > cut)      # 떨어진 먼 클러스터만 컷
                            roi[ys[bad], xs[bad]] = 0
                            roi = _fill_largest(roi)
        # 3) 안정화. SAM2 마스크는 캐시(1.5s)라 프레임간 이미 일정 → 시간 EMA 를
        #    적용하면 갱신 경계에서 서로 다른 마스크를 섞어 경계가 지글지글해짐
        #    → SAM2 는 EMA 생략, 가벼운 blur smoothing 만(깨끗한 단일 외곽 보존).
        #    YOLO fallback 만 움직임 인지 EMA(깜빡임/이동 대응).
        if used_sam:
            sm = cv2.medianBlur(roi, 7)         # 지글거림 제거(반사 캔 경계)
            sm = cv2.morphologyEx(
                sm, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            roi = _fill_largest((sm > 127).astype(np.uint8) * 255)
        else:
            mkey = self._track_key(x1, y1, x2, y2)
            if not hasattr(self, '_mask_ema'):
                self._mask_ema = {}
            cur = roi.astype(np.float32)
            prev = self._mask_ema.get(mkey)
            if prev is not None and prev.shape != cur.shape:
                prev = cv2.resize(prev, (cur.shape[1], cur.shape[0]))
            if prev is None:
                ema = cur
            else:
                pb = prev > 128; cb = cur > 128
                inter = float(np.logical_and(pb, cb).sum())
                uni = float(np.logical_or(pb, cb).sum()) + 1.0
                iou = inter / uni
                base_a = float(self._otun('mask_ema_alpha', 0.35))
                # 정적(겹침 큼)=base_a 부드럽게, 이동/신규(겹침 작음)=0.85 추종
                a = base_a if iou > 0.55 else 0.85
                ema = (1.0 - a) * prev + a * cur
            self._mask_ema[mkey] = ema
            if len(self._mask_ema) > 48:
                self._mask_ema.pop(next(iter(self._mask_ema)))
            roi = _fill_largest((ema > 128).astype(np.uint8) * 255)
        self._dbg_used_sam = used_sam
        out = np.zeros((h, w), dtype=np.uint8)
        out[ey1:ey2, ex1:ex2] = roi
        return out

    def _render_sam_object(self, vis, frame, base_mask_full, bbox, cname,
                           depth_arr):
        """GD 가 잡았지만 YOLO seg 가 놓친 객체를 SAM2 mask 로 렌더.
        seg 루프와 동일 표기: 노랑 외곽선 + 센터좌표 + 초록점 + rx/ry/rz + 주황꼭지점
        + obj_dicts. base_mask_full = full-frame uint8 mask(SAM2 워커 결과)."""
        vH, vW = vis.shape[:2]
        refined = self._refine_object_mask(
            frame, base_mask_full, bbox, depth_arr, use_isnet=True,
            block=False, cname=cname)
        # 작업영역 게이트 — base 좌표가 작업 범위 밖(사람·모니터·배경)이면
        # GD-only 객체도 외곽선·라벨·dict 전부 skip.
        _gate = self.mask_to_base_xyz((refined > 0).astype(np.uint8), depth_arr)
        if _gate is None or not self._in_work_zone(_gate[2]):
            return
        # snack 채도 필터 — snack_bag 인데 회색/저채도(로봇베이스)면 오검출 → skip
        if (str(cname) == 'snack_bag'
                and self._is_gray_nonsnack(frame, (refined > 0).astype(np.uint8))):
            return
        cnts, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        if os.environ.get('DEBUG_CAN'):
            _ars = sorted([int(cv2.contourArea(c)) for c in cnts],
                          reverse=True)[:4]
            print(f'[DBG GD-only] {cname} bbox={bbox} '
                  f'used_sam={getattr(self,"_dbg_used_sam",None)} '
                  f'ncnt={len(cnts)} areas={_ars}', flush=True)
        drew = False
        for c in cnts:
            if cv2.contourArea(c) < 50:
                continue
            self._pending_outlines.append(c)   # 맨 마지막에 그림
            drew = True
        if not drew:
            return
        mask01 = (refined > 0).astype(np.uint8)
        out = self.mask_to_base_xyz(mask01, depth_arr)
        if out is None:
            return
        cu, cv_y, base_xyz = out
        gg_label = to_graspgen_label(cname)
        cloud, pix = self.mask_to_base_cloud(mask01, depth_arr, return_pix=True)
        if pix is not None and len(pix) > 0:
            _st = max(1, len(pix) // 24)
            _pd = pix[::_st]
            for _gx, _gy in zip(np.clip(_pd[:, 0], 0, vW - 1),
                                np.clip(_pd[:, 1], 0, vH - 1)):
                cv2.circle(vis, (int(_gx), int(_gy)), 1, (0, 255, 0), -1)
        label = (f'{cname} ({base_xyz[0]:+.0f}, {base_xyz[1]:+.0f}, '
                 f'{base_xyz[2]:+.0f}) mm')
        cv2.circle(vis, (int(cu), int(cv_y)), 4, (0, 0, 255), -1)
        self._label_box(vis, label, int(cu) + 6, int(cv_y) - 8, 0.52,
                        (255, 255, 255))
        # ★GD-only 객체도 self.detections 에 등록 → 번호/lock 가능.
        #   (YOLO seg 가 놓치고 GD만 잡은 snack 등이 번호 안 매겨지던 문제 해결)
        self.detections.append({
            'name': cname, 'gg_label': gg_label,
            'center_base': np.asarray(base_xyz, dtype=float) / 1000.0,
            'center_px': (int(cu), int(cv_y)),
            'mask': mask01, 'cloud_m': cloud,
            'bbox': tuple(int(v) for v in bbox),
        })
        bb = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        rxyz = self.cloud_to_rxryrz(
            cloud, key=self._isnet_grid_key(*bb), cname=cname)
        if rxyz is not None:
            rx, ry, rz = rxyz
            self._label_box(vis, f'rx{rx:+.0f} ry{ry:+.0f} rz{rz:+.0f}',
                            int(cu) + 6, int(cv_y) + 16, 0.5, (120, 220, 255))
            obj_dicts[cname] = {
                '각도': f'{rz:.1f}',
                '각도r': {'rx': rx, 'ry': ry, 'rz': rz},
                '센터포인트mm': [float(base_xyz[0]), float(base_xyz[1]),
                              float(base_xyz[2])],
                'graspgen_label': gg_label,
            }
        # 주황 꼭지점 (외곽선 minAreaRect → contour snap)
        try:
            cc = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cc) >= 50:
                box = cv2.boxPoints(cv2.minAreaRect(cc))
                bcx = float(box[:, 0].mean()); bcy = float(box[:, 1].mean())
                ordered = [None, None, None, None]
                for _p in box:
                    if _p[0] >= bcx and _p[1] < bcy:
                        ordered[0] = _p
                    elif _p[0] >= bcx and _p[1] >= bcy:
                        ordered[1] = _p
                    elif _p[0] < bcx and _p[1] >= bcy:
                        ordered[2] = _p
                    else:
                        ordered[3] = _p
                dvm = depth_arr[mask01 > 0]; dvm = dvm[dvm > 0]
                od = float(np.median(dvm)) if dvm.size > 10 else None
                cc_pts = cc.reshape(-1, 2); cmm = []
                for _p in ordered:
                    if _p is None:
                        cmm.append(['', '', '']); continue
                    _di = ((cc_pts[:, 0] - _p[0]) ** 2
                           + (cc_pts[:, 1] - _p[1]) ** 2)
                    _sp = cc_pts[int(_di.argmin())]
                    px, py = int(_sp[0]), int(_sp[1])
                    cwv, _e = self.pixel_to_base_xyz(
                        px, py, depth_arr, forced_depth=od)
                    if cwv is not None:
                        cmm.append([round(float(cwv[0]), 1),
                                    round(float(cwv[1]), 1),
                                    round(float(cwv[2]), 1)])
                        _o = (0, 140, 255)
                        cv2.circle(vis, (px, py), 5, _o, -1)
                        cv2.circle(vis, (px, py), 6, (0, 0, 0), 1)
                        _t = f'{cwv[0]:+.0f},{cwv[1]:+.0f},{cwv[2]:+.0f}'
                        ox = 8 if px >= bcx else -95
                        oy = -8 if py < bcy else 16
                        cv2.putText(vis, _t, (px + ox, py + oy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(vis, _t, (px + ox, py + oy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _o, 1,
                                    cv2.LINE_AA)
                    else:
                        cmm.append(['', '', ''])
                if cname in obj_dicts:
                    obj_dicts[cname]['꼭지점mm'] = cmm
        except Exception:
            pass

    def cloud_to_rxryrz(self, cloud_m, key=None, cname=None):
        """base 프레임 point cloud → 두산 RPY(ZYZ euler, deg) [rx, ry, rz].

        두산 e0509 + 로보티즈 RH-P12-RN-A top-down 파지 자세 규격(_RealProject_1
        자료 기준): 그리퍼가 -Z(아래) 향함 → ZYZ euler [0, 180, yaw].
          - rx = 0, ry = 180  (top-down, 그리퍼 Z 아래)
          - rz = 물체 yaw = base 평면에 투영한 외곽의 긴 변 각도
        yaw 는 현재 프로젝트 내부 자원만 사용: mask_to_base_cloud() 가 만든 base
        프레임 점들의 XY 에 cv2.minAreaRect(노드가 이미 외곽에 쓰는 방식) → 긴 변.
        원근 왜곡은 이미 base 평면 투영으로 제거됨. key 면 rz EMA 스무딩."""
        if cloud_m is None or len(cloud_m) < 30:
            return None
        xy = np.ascontiguousarray(cloud_m[:, :2], dtype=np.float32)
        try:
            box = cv2.boxPoints(cv2.minAreaRect(xy))      # base 평면 회전사각형 4점
        except Exception:
            return None
        p0, p1, p2 = box[0], box[1], box[2]
        e01 = float(np.hypot(*(p1 - p0)))
        e12 = float(np.hypot(*(p2 - p1)))
        pa, pb = (p0, p1) if e01 >= e12 else (p1, p2)     # 긴 변
        rz = float(np.degrees(np.arctan2(pb[1] - pa[1], pb[0] - pa[0])))
        while rz > 90.0:   rz -= 180.0                    # 180° 대칭(평행조) 정규화
        while rz <= -90.0: rz += 180.0
        rx, ry = 0.0, 180.0                               # 두산 ZYZ top-down
        # rz 프레임간 EMA (180° 주기 unwrap) — 정적 객체 yaw 안정화
        if key is not None:
            if not hasattr(self, '_rz_ema'):
                self._rz_ema = {}
            prev = self._rz_ema.get(key)
            if prev is not None:
                d = rz - prev
                while d > 90.0:  rz -= 180.0; d = rz - prev
                while d < -90.0: rz += 180.0; d = rz - prev
                rz = 0.35 * rz + 0.65 * prev
                while rz > 90.0:   rz -= 180.0
                while rz <= -90.0: rz += 180.0
            self._rz_ema[key] = rz
        return float(rx), float(ry), float(rz)

    # ---------- 로봇 제어 (단일 비행) ----------
    def _make_top_down_pose(self, base_xyz_mm):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position.x = float(base_xyz_mm[0]) * 0.001
        msg.pose.position.y = float(base_xyz_mm[1]) * 0.001
        msg.pose.position.z = float(base_xyz_mm[2]) * 0.001
        # top-down: π rotation around X-axis → quat (xyzw) = (1, 0, 0, 0)
        msg.pose.orientation.x = 1.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 0.0
        return msg

    def _gripper_open_call(self, label='place/open', position=0):
        """그리퍼 열기 — 브리지 set_position(0). fire-and-forget(UI 비블록)."""
        client = self.cli_open
        if client is None:
            self.get_logger().warn(f'[{label}] set_position 미가용 (브리지 소스 안됨)')
            return False
        if not client.service_is_ready():
            self.get_logger().warn(f'[{label}] set_position service not ready')
            return False
        req = SetPosition.Request()
        req.position = int(position)
        req.timeout_sec = 0.5
        future = client.call_async(req)
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f'[{label}] set_position resp: '
                f'{f.result().success if f.result() else "?"}'))
        return True

    def _trigger_call(self, client, label):
        if not client.service_is_ready():
            self.get_logger().warn(f'[{label}] service not ready')
            return False
        req = Trigger.Request()
        future = client.call_async(req)
        # 빠르게 fire-and-forget — UI block X
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f'[{label}] resp: {f.result().success if f.result() else "?"}'))
        return True

    def execute_pick(self, base_xyz_mm):
        if not self.motion_lock.acquire(blocking=False):
            self.get_logger().warn('[pick] 모션 진행중 — 무시')
            return
        try:
            msg = self._make_top_down_pose(base_xyz_mm)
            self.get_logger().info(
                f'[pick] publish /pick_pose '
                f'pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, '
                f'{msg.pose.position.z:.3f}) m')
            self.pub_pick.publish(msg)
            # curobo_planner_node 가 approach → descend → close → lift 전체 처리
            time.sleep(0.5)  # publish 가 처리되는 동안 lock 유지
        finally:
            self.motion_lock.release()

    # ---------- 메인 루프 ----------
    def spin_camera(self):
        t_prev = time.time()
        fps_ema = 0.0
        try:
            while rclpy.ok():
                if hasattr(self, '_contour_frame_id'):
                    self._contour_frame_id += 1
                else:
                    self._contour_frame_id = 0
                frames = self.align.process(self.pipe.wait_for_frames())
                cf = frames.get_color_frame()
                df = frames.get_depth_frame()
                if not cf or not df:
                    continue
                frame = np.asanyarray(cf.get_data())
                depth_arr = np.asanyarray(df.get_data()).copy()
                # 🛑 [좌표 정확도] depth garbage 컷 — RealSense 가 무효 픽셀을 65535mm
                # (=65m, uint16 overflow) 나 먼 배경(벽/메시 너머)으로 내보냄. 이게
                # cloud 에 섞이면 점이 수십 m 밖으로 날아가 centroid·파지점이 박살남
                # (파지점 454→787 점프 원인). 작업범위(기본 1.2m) 초과는 전부 0(무효).
                _dmax = float(os.environ.get('WSN_DEPTH_MAX_MM', '850'))
                depth_arr[depth_arr > _dmax] = 0
                self._last_depth = depth_arr   # 장애물 depth 클러스터링용

                # ── [좌표 정확도 디버그] 화면 중앙 픽셀(거의 카메라 바로 아래) 을
                # deproject → 카메라위치와 비교. 정확하면 z≈테이블(-30), xy≈카메라xy.
                self._dbg_n = getattr(self, '_dbg_n', 0) + 1
                if os.environ.get('DBG_COORD') and self._dbg_n % 30 == 0:
                    H0, W0 = depth_arr.shape
                    cpos = self.T_cam2base[:3, 3] * 1000.0
                    cd = int(depth_arr[H0 // 2, W0 // 2])
                    bc, _e = self.pixel_to_base_xyz(W0 // 2, H0 // 2, depth_arr, window=9)
                    bc_s = ('(%.0f,%.0f,%.0f)' % tuple(bc)) if bc is not None else f'fail({_e})'
                    _valid = depth_arr[depth_arr > 0]
                    _frac = 100.0 * _valid.size / depth_arr.size
                    if _valid.size > 0:
                        _dstat = (f'유효{_frac:.0f}% 범위[{int(_valid.min())}~'
                                  f'{int(_valid.max())}]중앙값{int(np.median(_valid))}mm')
                    else:
                        _dstat = '유효0% (depth 전체 0!)'
                    self.get_logger().info(
                        f"[DBG_COORD] 카메라pos=({cpos[0]:.0f},{cpos[1]:.0f},{cpos[2]:.0f})mm "
                        f"중앙깊이={cd}mm 중앙base={bc_s} | depth {_dstat}")

                # 메인 YOLO seg (있을 때만). plot() 는 class별 컬러 (red 포함)
                # 칠해서 외곽선 색상 통일 안되므로 raw frame 사용. 우리는 자체
                # 핑크 polylines 만 그림.
                if self.yolo is not None:
                    res = self.yolo.predict(
                        frame, conf=self.conf, iou=self.iou,
                        imgsz=self.imgsz, verbose=False)[0]
                    vis = frame.copy()
                    # 매대확인용 원본 2D(클래스+conf) 저장 — depth 투영 전이라 먼 매대도 잡힘
                    try:
                        _ys = []
                        if res is not None and res.boxes is not None:
                            for _b in res.boxes:
                                _ys.append((str(self.yolo.names[int(_b.cls)]),
                                            float(_b.conf)))
                        self.yolo_seg_results = _ys
                    except Exception:
                        self.yolo_seg_results = []
                else:
                    res = None
                    vis = frame.copy()
                # 외곽선은 모았다가 맨 마지막에 그림 — 라벨 박스(어두운 배경)가
                # 그 아래 노란 외곽선을 덮어 끊는 것 방지(외곽선이 항상 최상위).
                self._pending_outlines = []
                self._label_rects = []      # 라벨 충돌 회피용(frame 마다 리셋)
                self._seg_seen = set()      # 이번 frame 에 그린 seg track (지속성용)

                # YOLO26-obb: 회전 박스 (노란색 4 corner). corners + angle + size 보존.
                obb_list = []  # [{'cx','cy','corners','angle_deg','pixel_w','pixel_h'}]
                if self.yolo_obb is not None:
                    try:
                        res_obb = self.yolo_obb.predict(
                            frame, conf=self.conf, imgsz=self.imgsz,
                            verbose=False)[0]
                        if hasattr(res_obb, 'obb') and res_obb.obb is not None:
                            polys = res_obb.obb.xyxyxyxy.cpu().numpy()
                            xywhr = res_obb.obb.xywhr.cpu().numpy()
                            for poly, params in zip(polys, xywhr):
                                pts_pix = poly.reshape(-1, 2)
                                # OBB 노란 회전박스 그리기 비활성 — 재학습 seg 모델의
                                # tight 외곽선+꼭지점으로 대체(중복 clutter 제거).
                                # cv2.polylines(vis, [pts_pix.astype(np.int32)],
                                #               True, (0, 255, 255), 2)
                                obb_list.append({
                                    'cx': float(params[0]),
                                    'cy': float(params[1]),
                                    'corners': [
                                        (float(p[0]), float(p[1]))
                                        for p in pts_pix],
                                    'angle_deg': float(
                                        np.degrees(params[4])) % 180,
                                    'pixel_w': float(params[2]),
                                    'pixel_h': float(params[3]),
                                })
                    except Exception as e:
                        pass

                # GroundingDINO: background thread (gd_interval 초마다 1회 trigger)
                if self.gd_model is not None:
                    now_ = time.time()
                    if (not self.gd_busy
                            and now_ - self.gd_last_t >= self.gd_interval):
                        self.gd_busy = True
                        self.gd_last_t = now_
                        threading.Thread(
                            target=self._gd_worker, args=(frame.copy(),),
                            daemon=True).start()
                    # 캐시된 결과 그리기 (빨간색) + 센터포인트 base 좌표
                    with self.gd_lock:
                        gd_snap = list(self.gd_results)

                # Qwen2.5-VL background trigger (5초마다)
                if self.qwen_model is not None:
                    now_q = time.time()
                    if (not self.qwen_busy
                            and now_q - self.qwen_last_t >= self.qwen_interval):
                        self.qwen_busy = True
                        self.qwen_last_t = now_q
                        threading.Thread(
                            target=self._qwen_worker, args=(frame.copy(),),
                            daemon=True).start()
                    with self.qwen_lock:
                        qwen_phrase_label = dict(self.qwen_phrase_label)
                    # 각 GD 박스 밑 한국어 라벨은 GD 그리기 시 phrase 로 lookup
                else:
                    qwen_phrase_label = {}

                # PaddleOCR background trigger
                if self.ocr_model is not None:
                    now_o = time.time()
                    if (not self.ocr_busy
                            and now_o - self.ocr_last_t >= self.ocr_interval):
                        self.ocr_busy = True
                        self.ocr_last_t = now_o
                        threading.Thread(
                            target=self._ocr_worker, args=(frame.copy(),),
                            daemon=True).start()
                    # OCR 캐시
                    with self.ocr_lock:
                        ocr_snap = list(self.ocr_results)
                else:
                    ocr_snap = []
                # OCR 결과 그리기 (마젠타색)
                for (ox1, oy1, ox2, oy2, otext, oscore) in ocr_snap:
                    cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), (255, 0, 255), 1)
                    olabel = f'OCR: {otext}'
                    (otw, oth), _ = cv2.getTextSize(
                        olabel, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(vis, (ox1, oy2 + 2),
                                  (ox1 + otw + 4, oy2 + oth + 8),
                                  (0, 0, 0), -1)
                    cv2.putText(vis, olabel, (ox1 + 2, oy2 + oth + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 0, 255), 1, cv2.LINE_AA)
                # GD bbox 와 OCR 매칭 (GD 박스 안에 들어가는 OCR 텍스트 모으기)
                gd_ocr_match = {}
                for gi, (x1, y1, x2, y2, *_rest) in enumerate(gd_snap):
                    matched = []
                    for (ox1, oy1, ox2, oy2, otext, oscore) in ocr_snap:
                        ocu = (ox1 + ox2) / 2
                        ocv = (oy1 + oy2) / 2
                        if x1 <= ocu <= x2 and y1 <= ocv <= y2:
                            matched.append(otext)
                    if matched:
                        gd_ocr_match[gi] = ' | '.join(matched[:3])
                    for gi, (x1, y1, x2, y2, phrase, score) in enumerate(gd_snap):
                        # 빨간 박스는 rect 계산 후 (mask_bbox 있으면 그것 사용)
                        cu, cv_y = (x1 + x2) // 2, (y1 + y2) // 2
                        # 작업영역 게이트 — GD 검출 중심 base 좌표가 작업 범위 밖
                        # (사람·모니터·배경)이거나 depth 불가면 핑크박스·라벨 전부 skip.
                        _gdw = max(15, min((x2 - x1) // 2, (y2 - y1) // 2))
                        _gb, _ = self.pixel_to_base_xyz(
                            cu, cv_y, depth_arr, window=_gdw)
                        if not self._in_work_zone(_gb):
                            continue
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        # 이 GD 박스 안을 YOLO seg 가 이미 검출했으면(seg_covers)
                        # GD 핑크박스+phrase 라벨을 숨김 — seg 가 정확한 클래스(can)로
                        # 표시하므로 GD 의 오분류(bottle) 중복 라벨 제거.
                        seg_covers = False
                        if res is not None and res.boxes is not None:
                            try:
                                for _yb in res.boxes.xyxy.cpu().numpy():
                                    _yx = (_yb[0] + _yb[2]) / 2
                                    _yy = (_yb[1] + _yb[3]) / 2
                                    if x1 <= _yx <= x2 and y1 <= _yy <= y2:
                                        seg_covers = True; break
                            except Exception:
                                seg_covers = False

                        # 1) 박스 위에 이름 + score + OCR 텍스트 매칭 (빨간색, 얇게)
                        # phrase 에서 색상 단어 (green/red/blue 등) 제거
                        _COLORS = {'green', 'red', 'blue', 'yellow', 'purple',
                                   'white', 'black', 'pink', 'orange', 'brown',
                                   'gray', 'grey', 'cyan', 'magenta', 'violet'}
                        # 색상 제거 + 첫 객체 명사 1개만 (can can → can)
                        _non_color = [w for w in phrase.split()
                                      if w.lower() not in _COLORS]
                        phrase_clean = _non_color[0] if _non_color else phrase
                        # ── snack 오탐 억제 (GD-only) — 'snack' 인데 박스 영역이
                        # 회색/저채도(로봇베이스·그림자 등)면 skip. 실측 snack sat≥49,
                        # 회색물체 sat~26. seg 가 담당(seg_covers)하는 건 seg 필터가 처리.
                        if (not seg_covers and 'snack' in phrase_clean.lower()):
                            _sx1, _sy1 = max(0, x1), max(0, y1)
                            _sx2 = min(frame.shape[1], x2)
                            _sy2 = min(frame.shape[0], y2)
                            if _sx2 - _sx1 >= 4 and _sy2 - _sy1 >= 4:
                                _scrop = frame[_sy1:_sy2, _sx1:_sx2]
                                _ssat = float(cv2.cvtColor(
                                    _scrop, cv2.COLOR_BGR2HSV)[..., 1].mean())
                                if _ssat < float(os.environ.get(
                                        'WSN_SNACK_SAT', '42')):
                                    continue
                        ocr_extra = gd_ocr_match.get(gi, '')
                        if ocr_extra:
                            name_label = f'{phrase_clean} {score:.2f} [{ocr_extra}]'
                        else:
                            name_label = f'{phrase_clean} {score:.2f}'
                        if not seg_covers:   # seg 가 담당하면 GD 라벨 숨김
                            (nw, nh), _ = cv2.getTextSize(name_label, font, 0.5, 1)
                            cv2.rectangle(vis, (x1, y1 - nh - 8),
                                          (x1 + nw + 4, y1 - 2), (0, 0, 0), -1)
                            cv2.putText(vis, name_label, (x1 + 2, y1 - 6),
                                        font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                        # 2) 센터 → base xyz (mm). window 를 bbox 크기에 맞춰 동적.
                        # default 15 픽셀로는 bbox 큰 객체 center 주변 그림자에 걸려 depth fail.
                        # bbox 의 short-side 절반 까지 확장 → bbox 안 valid depth median.
                        dyn_window = max(15, min((x2 - x1) // 2, (y2 - y1) // 2))
                        base_xyz, err = self.pixel_to_base_xyz(
                            cu, cv_y, depth_arr, window=dyn_window)
                        cv2.circle(vis, (cu, cv_y), 5, (0, 255, 0), -1)   # 초록 점
                        if base_xyz is not None:
                            bx, by, bz = float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])
                            xyz_label = f'({bx:+.0f},{by:+.0f},{bz:+.0f})mm'
                        else:
                            xyz_label = '(depth fail)'
                        # 회전 사각형: obb 매칭 우선, 없으면 image processing fallback
                        rect = None
                        for o in obb_list:
                            if x1 <= o['cx'] <= x2 and y1 <= o['cy'] <= y2:
                                rect = {
                                    'corners': o['corners'],
                                    'angle_deg': o['angle_deg'],
                                    'pixel_w': o['pixel_w'],
                                    'pixel_h': o['pixel_h'],
                                }
                                break
                        if rect is None:
                            # contour 캐시 (phrase 별 4 frame TTL) — lag 잡기
                            ck = f'{phrase_clean}_contour'
                            cached = getattr(self, '_contour_cache', {}).get(ck)
                            if not hasattr(self, '_contour_cache'):
                                self._contour_cache = {}
                            if not hasattr(self, '_contour_frame_id'):
                                self._contour_frame_id = 0
                            # bbox 크기 변화 < 15% & 위치 < 30 px 이면 재사용
                            reuse = False
                            if cached is not None:
                                cx1, cy1, cx2, cy2 = cached['bbox']
                                if (abs(cx1 - x1) < 30
                                        and abs(cy1 - y1) < 30
                                        and abs(cx2 - x2) < 30
                                        and abs(cy2 - y2) < 30
                                        and cached['fid'] >= self._contour_frame_id - 60):
                                    reuse = True
                            if reuse:
                                rect = cached['rect']
                            elif not getattr(self, '_mask_refine', True):
                                # 'm' 키 토글 OFF — ISNet+GrabCut skip. GD bbox 4-corner 만.
                                # 꼭지점(노란 원) + 좌표 표시 유지 위해 rect dict 구성.
                                rect = {
                                    'corners': [(float(x1), float(y1)),
                                                (float(x2), float(y1)),
                                                (float(x2), float(y2)),
                                                (float(x1), float(y2))],
                                    'angle_deg': 0.0,
                                    'pixel_w': float(x2 - x1),
                                    'pixel_h': float(y2 - y1),
                                    'mask_bbox': (x1, y1, x2, y2),
                                    'mask_contour': None,
                                }
                            else:
                                # 메인 thread block 방지 위해 한 frame 에 객체 1개만
                                # ISNet+GrabCut 호출 (round-robin). 나머지는 stale cache.
                                # → 객체 N개일 때 한 객체당 N frame 마다 갱신.
                                if not hasattr(self, '_refine_token'):
                                    self._refine_token = 0
                                self._refine_token += 1
                                # phrase_clean hash 로 token 매칭
                                slot = hash(phrase_clean) & 0xFFFF
                                if (slot % 4) == (self._contour_frame_id % 4):
                                    self._sam2_prepare(frame)
                                    rect = self._estimate_rect_from_bbox(
                                        frame, x1, y1, x2, y2,
                                        depth_arr=depth_arr, phrase=phrase_clean)
                                    if rect is not None:
                                        self._contour_cache[ck] = {
                                            'rect': rect, 'bbox': (x1, y1, x2, y2),
                                            'fid': self._contour_frame_id,
                                        }
                                elif cached is not None:
                                    # stale cache 도 fall back (bbox 가 멀어도 외곽선 표시)
                                    rect = cached['rect']
                                else:
                                    rect = None

                        # 외곽선 — mask contour. seg_covers(이 GD bbox 안 YOLO seg
                        # 검출)면 seg 경로가 외곽선을 그리므로 GD 외곽선·핑크박스 skip
                        # (이중 외곽선 + 오분류 라벨 중복 제거).
                        if (not seg_covers and rect is not None
                                and rect.get('mask_contour') is not None):
                            self._pending_outlines.append(rect['mask_contour'])
                        # GD bbox 핑크 사각형 — seg 미검출(GD-only) 객체만 표시
                        if not seg_covers:
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)

                        # smoothing: pixel_w, pixel_h, angle 도 frame 간 안정화
                        if rect is not None:
                            sm_size_key = phrase_clean + '_size'
                            cached_sz = self.rect_smooth_cache.get(sm_size_key)
                            if cached_sz is not None:
                                a_sz = 0.1
                                rect['pixel_w'] = (
                                    a_sz * rect.get('pixel_w', cached_sz['pixel_w'])
                                    + (1 - a_sz) * cached_sz['pixel_w'])
                                rect['pixel_h'] = (
                                    a_sz * rect.get('pixel_h', cached_sz['pixel_h'])
                                    + (1 - a_sz) * cached_sz['pixel_h'])
                            self.rect_smooth_cache[sm_size_key] = {
                                'pixel_w': rect.get('pixel_w', 50),
                                'pixel_h': rect.get('pixel_h', 50),
                            }

                        # IIR smoothing — 꼭지점 frame 간 변동 완화 (key=phrase_clean)
                        # 꼭지점 = top/right/bottom/left ray cast (이미 정해진 의미)
                        # → 추가 시계방향 sort 금지 (idx 0=top, 1=right, 2=bottom, 3=left 고정)
                        if rect is not None and len(rect.get('corners', [])) == 4:
                            import math as _math
                            sm_cx = (x1 + x2) / 2
                            sm_cy = (y1 + y2) / 2
                            sm_key = phrase_clean
                            cached = self.rect_smooth_cache.get(sm_key)
                            if cached and len(cached['corners']) == 4:
                                # 변동 검증: 평균 corner 이동 거리 > bbox 의 30% 면 outlier
                                bbox_diag = (
                                    (x2-x1)**2 + (y2-y1)**2) ** 0.5
                                tot_d = sum(
                                    ((nc[0]-oc[0])**2 + (nc[1]-oc[1])**2) ** 0.5
                                    for nc, oc in zip(rect['corners'], cached['corners']))
                                avg_d = tot_d / 4
                                if avg_d > bbox_diag * 0.3:
                                    # 큰 jump → 새 corners 무시 (이전 유지)
                                    rect['corners'] = list(cached['corners'])
                                    rect['angle_deg'] = cached['angle_deg']
                                else:
                                    alpha = 0.7  # 새 70% + 옛 30% — contour 즉시 추적
                                    smoothed = []
                                    for new_c, old_c in zip(rect['corners'], cached['corners']):
                                        smoothed.append((
                                            alpha * new_c[0] + (1-alpha) * old_c[0],
                                            alpha * new_c[1] + (1-alpha) * old_c[1]))
                                    rect['corners'] = smoothed
                                    old_a = cached['angle_deg']
                                    new_a = rect['angle_deg']
                                    if abs(new_a - old_a) > 90:
                                        new_a += 180 if new_a < old_a else -180
                                    rect['angle_deg'] = (
                                        alpha * new_a + (1-alpha) * old_a) % 180
                            self.rect_smooth_cache[sm_key] = {
                                'corners': list(rect['corners']),
                                'angle_deg': rect['angle_deg'],
                            }

                        # dict 에 모든 값 저장
                        det_data = {
                            'phrase': phrase, 'score': float(score),
                            'center_pixel': (cu, cv_y),
                            'center_world_mm': (
                                None if base_xyz is None
                                else (float(base_xyz[0]),
                                      float(base_xyz[1]),
                                      float(base_xyz[2]))),
                        }
                        if rect is not None:
                            det_data['angle_deg'] = rect['angle_deg']
                            # 1) 픽셀 corners — cv2.boxPoints (center+W/H/angle 수학 계산)
                            pixel_w = rect.get('pixel_w', max(1, x2-x1))
                            pixel_h = rect.get('pixel_h', max(1, y2-y1))
                            box_input = ((float(cu), float(cv_y)),
                                         (float(pixel_w), float(pixel_h)),
                                         float(rect['angle_deg']))
                            calc_pix = cv2.boxPoints(box_input)
                            det_data['corners_pixel'] = [
                                (float(p[0]), float(p[1])) for p in calc_pix]
                            # 2) 가로/세로 (mm) — 픽셀 W/H 의 base 거리로 추정.
                            #    가까운 픽셀 변의 base xy 거리 / 픽셀 거리 * mm 환산.
                            w_mm = h_mm = None
                            if base_xyz is not None:
                                # 두 인접 corner 쌍의 base xy 거리 평균
                                def _bxy(px, py):
                                    cw_, _ = self.pixel_to_base_xyz(
                                        int(round(px)), int(round(py)), depth_arr)
                                    return cw_
                                p0 = det_data['corners_pixel'][0]
                                p1 = det_data['corners_pixel'][1]
                                p2 = det_data['corners_pixel'][2]
                                b0 = _bxy(*p0); b1 = _bxy(*p1); b2 = _bxy(*p2)
                                if b0 is not None and b1 is not None:
                                    e0 = ((b0[0]-b1[0])**2 + (b0[1]-b1[1])**2
                                          + (b0[2]-b1[2])**2) ** 0.5
                                else:
                                    e0 = None
                                if b1 is not None and b2 is not None:
                                    e1 = ((b1[0]-b2[0])**2 + (b1[1]-b2[1])**2
                                          + (b1[2]-b2[2])**2) ** 0.5
                                else:
                                    e1 = None
                                if e0 is not None and e1 is not None:
                                    w_mm = max(e0, e1)
                                    h_mm = min(e0, e1)
                            det_data['width_mm'] = w_mm
                            det_data['height_mm'] = h_mm
                            # 3) 꼭지점 월드 = RealSense depth + 캘리브 (정확).
                            #    단 depth 는 모든 꼭지점에 객체 중심 depth 통일 (안정).
                            corners_world = [None] * 4
                            if base_xyz is not None and depth_arr is not None:
                                # 객체 중심 픽셀의 depth (raw mm) 사용
                                cu_i, cv_i = int(cu), int(cv_y)
                                if (0 <= cu_i < depth_arr.shape[1]
                                        and 0 <= cv_i < depth_arr.shape[0]):
                                    center_depth = int(depth_arr[cv_i, cu_i])
                                else:
                                    center_depth = 0
                                # depth 0 면 주변 영역 평균
                                if center_depth == 0:
                                    pad = 10
                                    cy_s = max(0, cv_i-pad)
                                    cy_e = min(depth_arr.shape[0], cv_i+pad)
                                    cx_s = max(0, cu_i-pad)
                                    cx_e = min(depth_arr.shape[1], cu_i+pad)
                                    patch = depth_arr[cy_s:cy_e, cx_s:cx_e]
                                    valid_d = patch[patch > 0]
                                    if valid_d.size > 0:
                                        center_depth = int(valid_d.mean())
                                if center_depth > 0:
                                    for i, (px, py) in enumerate(
                                            det_data['corners_pixel']):
                                        cw, _ = self.pixel_to_base_xyz(
                                            int(round(px)), int(round(py)),
                                            depth_arr, forced_depth=center_depth)
                                        if cw is not None:
                                            corners_world[i] = (
                                                float(cw[0]), float(cw[1]),
                                                float(cw[2]))
                            det_data['corners_world_mm'] = corners_world
                        self.last_detections[phrase] = det_data

                        # ---- 화면 표시 ----
                        # (a) 각도 → xyz_label 옆에 ,각도 (초록)
                        if 'angle_deg' in det_data:
                            xyz_label += f',{det_data["angle_deg"]:.1f}deg'
                        # (b) 4 꼭지점 글자 — 비활성. seg 외곽선 minAreaRect 기준 주황
                        #     코너(아래 seg 루프)로 대체해 겹침/이중표기 제거.
                        if False and 'corners_pixel' in det_data and 'corners_world_mm' in det_data:
                            H_, W_ = vis.shape[:2]
                            ccx, ccy = (x1 + x2) // 2, (y1 + y2) // 2
                            for (px, py), cw in zip(
                                    det_data['corners_pixel'],
                                    det_data['corners_world_mm']):
                                if cw is not None:
                                    cx_label = f'{cw[0]:+.0f},{cw[1]:+.0f},{cw[2]:+.0f}'
                                else:
                                    cx_label = 'depth?'
                                (cw_, ch_), _ = cv2.getTextSize(
                                    cx_label, font, 0.3, 1)
                                ix, iy = int(px), int(py)
                                # bbox 중심에서 꼭지점 방향으로 글자 offset (바깥쪽)
                                dx = ix - ccx; dy = iy - ccy
                                # 좌/우: 우면 글자 오른쪽, 좌면 왼쪽
                                ox = 6 if dx >= 0 else -(cw_ + 6)
                                # 상/하: 아래면 글자 더 아래, 위면 글자 위
                                oy = ch_ + 6 if dy >= 0 else -4
                                tx = ix + ox; ty = iy + oy
                                # 화면 안 클램프
                                tx = max(2, min(W_ - cw_ - 4, tx))
                                ty = max(ch_ + 4, min(H_ - 2, ty))
                                # 꼭지점 마커 + 글자 모두 노란
                                # 꼭지점 — 노랑 (외곽선 색과 동일) + 검정 outline 으로 시인성
                                cv2.circle(vis, (ix, iy), 7, (0, 255, 255), -1)
                                cv2.circle(vis, (ix, iy), 8, (0, 0, 0), 2)
                                cv2.rectangle(
                                    vis, (tx - 2, ty - ch_ - 3),
                                    (tx + cw_ + 3, ty + 2),
                                    (0, 0, 0), -1)
                                cv2.putText(
                                    vis, cx_label, (tx + 1, ty - 2),
                                    font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
                        # (c) 가로/세로 길이 (핑크)
                        if (det_data.get('width_mm') is not None
                                and det_data.get('height_mm') is not None):
                            wh_label = (f'W{det_data["width_mm"]:.0f}mm '
                                        f'H{det_data["height_mm"]:.0f}mm')
                            (wlw, wlh), _ = cv2.getTextSize(
                                wh_label, font, 0.4, 1)
                            wlx = cu - wlw // 2
                            wly = cv_y - 14  # 센터 위쪽
                            cv2.rectangle(
                                vis, (wlx - 2, wly - wlh - 3),
                                (wlx + wlw + 3, wly + 2),
                                (0, 0, 0), -1)
                            cv2.putText(
                                vis, wh_label, (wlx, wly),
                                font, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
                        # 초록 글자, 초록점 아래, 가운데 정렬
                        (tw, th), _ = cv2.getTextSize(xyz_label, font, 0.45, 1)
                        tx = cu - tw // 2
                        ty = cv_y + th + 10
                        cv2.rectangle(vis, (tx - 3, ty - th - 3),
                                      (tx + tw + 3, ty + 3),
                                      (0, 0, 0), -1)
                        cv2.putText(vis, xyz_label, (tx, ty), font, 0.45,
                                    (0, 255, 0), 1, cv2.LINE_AA)
                        # 초록 점 — 모든 라벨 그린 후 다시 그림 (라벨 박스 가림 방지).
                        # 검정 outline 으로 어떤 배경에서도 보이도록.
                        cv2.circle(vis, (cu, cv_y), 8, (0, 0, 0), -1)
                        cv2.circle(vis, (cu, cv_y), 6, (0, 255, 0), -1)

                        # 3) Qwen 한국어 라벨 (빨간색 박스 바로 위)
                        kor = qwen_phrase_label.get(phrase, '')
                        if kor:
                            ksize = 18
                            # 빨간색 글자 위쪽 (y1 - nh - 8) 위에 한글 그림 (top 좌표).
                            kor_y = y1 - nh - 8 - ksize - 4
                            kor_y = max(2, kor_y)  # 화면 위로 잘리지 않게 클램프
                            vis = draw_korean(
                                vis, kor, (x1, kor_y),
                                size=ksize, color_bgr=(255, 255, 0), bg=True)

                # ★GroundingDINO 검출 시각화 — 매대재고 판정에 쓰는 그 검출(gd_results) 주황 박스.
                #   conf ≥ 매대임계(WSN_SHELF_CONF) 인 것 = "매대에 있다"로 카운트(밝은주황+[재고O]).
                #   매대뷰에서 화면엔 없는데 '있다'고 판정되는 GD 오검출을 눈으로 확인 가능. WSN_GD_VIZ=0 끔.
                if os.environ.get('WSN_GD_VIZ', '1') != '0':
                    _sthr = float(os.environ.get('WSN_SHELF_CONF', '0.55'))
                    _smax = float(os.environ.get('WSN_SHELF_MAX_MM', '1000'))
                    _dep = getattr(self, '_last_depth', None)
                    with self.gd_lock:
                        _gdv = list(self.gd_results)
                    for _r in _gdv:
                        gx1, gy1, gx2, gy2, gph, gcf = (int(_r[0]), int(_r[1]), int(_r[2]),
                                                        int(_r[3]), str(_r[4]), float(_r[5]))
                        # ★far(배경) GD 는 화면에도 안 그림 (매대점검 depth 기준과 동일). 사용자 2026-06-22
                        if _dep is not None:
                            _cu = (gx1 + gx2) // 2; _cv = (gy1 + gy2) // 2
                            _Hd, _Wd = _dep.shape[:2]
                            _dmm = 0.0
                            if 0 <= _cv < _Hd and 0 <= _cu < _Wd:
                                _pt = _dep[max(0, _cv-4):_cv+5, max(0, _cu-4):_cu+5]
                                _vd = _pt[_pt > 0]
                                _dmm = float(np.median(_vd)) if _vd.size else 0.0
                            if _dmm <= 0 or _dmm > _smax:
                                continue   # 배경(멀거나 depth무효) → 화면 표시 안 함
                        _on = (gcf >= _sthr)
                        _gc = (0, 140, 255) if _on else (110, 130, 160)   # 밝은주황 vs 흐림
                        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), _gc, 2)
                        cv2.putText(vis, f"GD:{gph} {gcf:.2f}",
                                    (gx1, gy2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _gc, 2)

                # YOLOE: open-vocab 검출 (시안색 박스 + 라벨)
                if self.yoloe is not None:
                    try:
                        res_e = self.yoloe.predict(
                            frame, conf=0.10, imgsz=self.imgsz, verbose=False)[0]
                        if res_e.boxes is not None:
                            for box, cls_id, conf in zip(
                                    res_e.boxes.xyxy.cpu().numpy(),
                                    res_e.boxes.cls.cpu().numpy().astype(int),
                                    res_e.boxes.conf.cpu().numpy()):
                                x1, y1, x2, y2 = box.astype(int)
                                cls_name = res_e.names.get(int(cls_id), str(cls_id))
                                cv2.rectangle(vis, (x1, y1), (x2, y2),
                                              (255, 255, 0), 2)
                                cv2.putText(vis,
                                            f'E:{cls_name} {conf:.2f}',
                                            (x1, y1 - 4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (255, 255, 0), 1)
                    except Exception as e:
                        pass

                # 클릭 처리
                while self.pending_clicks:
                    u, v, kind = self.pending_clicks.pop(0)
                    base_xyz, err = self.pixel_to_base_xyz(u, v, depth_arr)
                    if base_xyz is None:
                        self.get_logger().info(
                            f'[{kind}] ({u},{v}) FAIL: {err}')
                        self.probe_markers.append(
                            (int(u), int(v), None, err, kind))
                    else:
                        bx, by, bz = float(base_xyz[0]), float(base_xyz[1]), \
                                     float(base_xyz[2])
                        self.probe_markers.append(
                            (int(u), int(v), (bx, by, bz), None, kind))
                        self.get_logger().info(
                            f'[{kind}] ({u},{v}) → base=({bx:+.1f}, {by:+.1f}, '
                            f'{bz:+.1f}) mm')

                # 키 trigger
                if self.action_trigger['fire']:
                    action = self.action_trigger['fire']
                    self.action_trigger['fire'] = None
                    if action == 'pick':
                        latest = None
                        for m in reversed(self.probe_markers):
                            if m[4] == 'probe' and m[2] is not None:
                                latest = m
                                break
                        if latest is None:
                            self.get_logger().warn('[pick] probe 없음 — `+q+좌클릭 먼저')
                        else:
                            _, _, base_xyz, _, _ = latest
                            threading.Thread(
                                target=self.execute_pick, args=(base_xyz,), daemon=True).start()

                # 마커 그리기
                for marker in self.probe_markers:
                    u, v, base_xyz, err, kind = marker
                    kind_label = MARKER_LABELS.get(kind, '?')
                    if base_xyz is None:
                        cv2.drawMarker(vis, (u, v), (0, 0, 255),
                                       cv2.MARKER_TILTED_CROSS, 22, 2)
                        label = f'{kind_label}-FAIL {err}'
                        color = (0, 0, 255)
                    else:
                        color = MARKER_COLORS.get(kind, (0, 255, 0))
                        bx, by, bz = base_xyz
                        cv2.circle(vis, (u, v), 4, color, -1)
                        label = f'{kind_label} ({bx:+.0f}, {by:+.0f}, {bz:+.0f}) mm'
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(vis, (u + 12, v - th - 8),
                                  (u + 12 + tw + 4, v - 4), (0, 0, 0), -1)
                    cv2.putText(vis, label, (u + 14, v - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # YOLO seg mask → 정밀 외곽(노랑) + base 좌표 + rx/ry/rz + 점 표시.
                # 하나의 루프에서 정밀 refine mask 를 외곽선·좌표·평면추정에 공통 사용.
                vH, vW = vis.shape[:2]
                isnet_boxes = []     # SAM2 워커에 올릴 (key, x1,y1,x2,y2)
                seg_centers = []     # YOLO seg 가 처리한 객체 중심 (GD-only 중복 방지)
                seg_boxes = []       # YOLO seg bbox — GD 박스가 이 안에 들면 같은 물체로 병합
                # 1-9 선택용 인덱스 검출 리스트 (매 프레임 재수집; eye-in-hand 라
                # base 좌표/cloud 는 현재 T_cam2base 로 매 프레임 새로 계산됨 — 캐싱 X)
                self.detections = []
                if res is not None and res.masks is not None:
                    try:
                        masks_data = res.masks.data.cpu().numpy()
                        yolo_boxes = res.boxes.xyxy.cpu().numpy()
                        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                        confs = res.boxes.conf.cpu().numpy()
                        # 마스크 IoU 기반 중복 제거 — 한 물체에 겹쳐 찍힌 중복
                        # 인스턴스(예: 봉지/포장 위 허위 can 다중검출) 제거. 큰 물체
                        # 위 작은 실물체(캔)는 마스크 IoU 낮아 보존됨(박스 NMS 와 차이).
                        _mfull = []
                        for _mi in range(masks_data.shape[0]):
                            _mm = masks_data[_mi] > 0.5
                            if _mm.shape != (vH, vW):
                                _mm = cv2.resize(
                                    _mm.astype(np.uint8), (vW, vH),
                                    interpolation=cv2.INTER_NEAREST) > 0
                            _mfull.append(_mm)
                        # snack_bag(cls 2)는 캔/병을 저신뢰로 오분류하는 일 많음 →
                        # 높은 conf 만 인정(WSN_SNACK_MIN_CONF, 기본 0.45). 실제 과자는 고신뢰라 유지,
                        # 캔이 snack 으로 오분류돼 vertical 로 잡히는 것 차단. (bottle/can 은 낮은 conf 유지)
                        _snack_min = float(os.environ.get('WSN_SNACK_MIN_CONF', '0.45'))
                        _keep = []
                        for _i in list(np.argsort(-confs)):
                            if int(cls_ids[_i]) == 2 and float(confs[_i]) < _snack_min:
                                continue
                            _ai = _mfull[_i]; _aa = int(_ai.sum()) + 1
                            _dup = False
                            for _j in _keep:
                                _inter = int(np.logical_and(_ai, _mfull[_j]).sum())
                                _uni = int(np.logical_or(_ai, _mfull[_j]).sum()) + 1
                                if _inter / _uni > 0.5 or _inter / _aa > 0.7:
                                    _dup = True; break
                            if not _dup:
                                _keep.append(int(_i))
                        keep_mi = set(_keep)
                        # GD bbox 와 YOLO mask 매칭 (center 포함 검사)
                        gd_bboxes = [(int(g[0]), int(g[1]), int(g[2]), int(g[3]))
                                     for g in gd_snap]
                        dbg_items = []     # [OUTLINE_DEBUG] (bbox, refined_mask, cname)
                        for mi in range(masks_data.shape[0]):
                            if mi not in keep_mi:    # 중복 인스턴스 skip
                                continue
                            # 번호 대상 제한 — 과자(snack_bag)/캔(can)/바틀(bottle)만.
                            # bread 등 그 외 클래스는 외곽선·번호·검출 전부 skip (사용자 요청).
                            if self.yolo.names.get(int(cls_ids[mi]),
                                                   '') not in ('bottle', 'can', 'snack_bag'):
                                continue
                            m = (masks_data[mi] > 0.5).astype(np.uint8) * 255
                            if m.shape != (vH, vW):
                                m = cv2.resize(m, (vW, vH),
                                               interpolation=cv2.INTER_NEAREST)
                            # bbox 선택: 이 객체 자신의 YOLO box 가 기본. center 포함
                            # GD box 는 '크기가 비슷할 때만'(같은 물체) 채택 — 안 그러면
                            # 작은 캔이 큰 snack bag 의 GD box 를 받아 외곽선이 그 박스
                            # (핑크선) 밖으로 못 나가고 잘림(사용자 지적 버그).
                            _yb = yolo_boxes[mi]
                            ycx = (_yb[0] + _yb[2]) / 2
                            ycy = (_yb[1] + _yb[3]) / 2
                            _ya = max(1.0, (_yb[2] - _yb[0]) * (_yb[3] - _yb[1]))
                            bbox = (int(_yb[0]), int(_yb[1]), int(_yb[2]), int(_yb[3]))
                            for gb in gd_bboxes:
                                if gb[0] <= ycx <= gb[2] and gb[1] <= ycy <= gb[3]:
                                    _ga = (gb[2] - gb[0]) * (gb[3] - gb[1])
                                    if _ga <= 1.8 * _ya:   # 같은 물체 크기일 때만
                                        bbox = gb
                                    break
                            # bbox 안정화 — 검출 박스 프레임간 흔들림 제거(SAM2
                            # 프롬프트 일관 → 외곽선 지글거림/jitter 억제).
                            bbox = self._smooth_bbox(
                                self._track_key(int(bbox[0]), int(bbox[1]),
                                                int(bbox[2]), int(bbox[3])), bbox)
                            # ISNet 워커에 이 객체 bbox 등록 (다음 프레임 sharp)
                            _bx1 = max(0, min(vW, int(bbox[0])))
                            _by1 = max(0, min(vH, int(bbox[1])))
                            _bx2 = max(0, min(vW, int(bbox[2])))
                            _by2 = max(0, min(vH, int(bbox[3])))
                            if _bx2 - _bx1 >= 8 and _by2 - _by1 >= 8:
                                # YOLO mask 내부에서 foreground 점 샘플 → SAM2 가 box
                                # 만으로 specular(금속캔)에 속아 일부만 잡는 것 방지.
                                # box + interior points 함께 prompt → 전체 객체 분할.
                                _pts = self._sample_mask_points(m, n=5)
                                isnet_boxes.append(
                                    (self._isnet_grid_key(_bx1, _by1, _bx2, _by2),
                                     _bx1, _by1, _bx2, _by2, _pts))
                            # ── 정밀 외곽 refine. ISNet 은 백그라운드 워커가 채운
                            # 캐시만 사용(메인 비블록), 캐시 없으면 기하 정제(빠름).
                            refined = self._refine_object_mask(
                                frame, m, bbox, depth_arr, use_isnet=True,
                                block=False,
                                cname=self.yolo.names.get(
                                    int(cls_ids[mi]), str(cls_ids[mi])))
                            # ── 작업영역 게이트 — base 좌표가 작업 범위 밖이면
                            # (사람·모니터·로봇베이스 등 배경/먼 물체) 외곽선·라벨·dict
                            # 전부 skip. 외곽선 그리기 전에 컷해서 노란선도 안 남게.
                            _gate = self.mask_to_base_xyz(
                                (refined > 0).astype(np.uint8), depth_arr)
                            if _gate is None or not self._in_work_zone(_gate[2]):
                                continue
                            # ── snack 채도 필터 — snack_bag(2) 인데 회색/저채도면
                            # (로봇베이스 등) 오검출 → skip (외곽선 전에 컷).
                            if (int(cls_ids[mi]) == 2
                                    and self._is_gray_nonsnack(
                                        frame, (refined > 0).astype(np.uint8))):
                                continue
                            # 외곽선 (tight) — 노랑
                            cnts, _ = cv2.findContours(
                                refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            if os.environ.get('DEBUG_CAN'):
                                _cn = self.yolo.names.get(int(cls_ids[mi]),
                                                          str(cls_ids[mi]))
                                _ars = sorted([int(cv2.contourArea(c))
                                               for c in cnts], reverse=True)[:4]
                                print(f'[DBG seg] {_cn} bbox={bbox} '
                                      f'used_sam={getattr(self,"_dbg_used_sam",None)} '
                                      f'ncnt={len(cnts)} areas={_ars}', flush=True)
                            _kept = [c for c in cnts if cv2.contourArea(c) >= 50]
                            self._pending_outlines.extend(_kept)
                            # 검출 지속성: 이 track 외곽선 캐시 — 붐비는 장면에서 한
                            # 프레임 놓쳐도 ~0.6s 유지해 외곽선 깜빡임(검출됐다 안됐다)
                            # 방지. (정적 객체 기준; track key 로 위치 매칭)
                            _tkp = self._track_key(int(bbox[0]), int(bbox[1]),
                                                   int(bbox[2]), int(bbox[3]))
                            if not hasattr(self, '_outline_persist'):
                                self._outline_persist = {}
                            if _kept:
                                self._outline_persist[_tkp] = (time.time(), _kept)
                            self._seg_seen.add(_tkp)
                            dbg_items.append((bbox, refined.copy(), m.copy(),
                                              self.yolo.names.get(int(cls_ids[mi]),
                                                                  str(cls_ids[mi]))))
                            # ── base 좌표 + rx/ry/rz + 점, 모두 동일 refined mask 기준
                            mask01 = (refined > 0).astype(np.uint8)
                            out = self.mask_to_base_xyz(mask01, depth_arr)
                            if out is None:
                                continue
                            cu, cv_y, base_xyz = out
                            # ★테이블 표면/펑보드 구멍 오검출 제거: z(높이)가 바닥이면 skip.
                            #   실물(snack≈+38, can≈+71, bottle≈+151)은 z≥~35mm, 테이블 표면/
                            #   구멍은 ~-30~+15mm → WSN_DET_ZMIN(기본 18) 미만은 버림. (사용자 2026-06-19)
                            if float(base_xyz[2]) < float(os.environ.get('WSN_DET_ZMIN', '18')):
                                continue
                            seg_centers.append((cu, cv_y))   # GD-only 중복 방지용
                            try:
                                seg_boxes.append((int(bbox[0]), int(bbox[1]),
                                                  int(bbox[2]), int(bbox[3])))
                            except Exception:
                                pass
                            # ── [DBG_COORD] 이 물체가 실제로 읽는 depth vs base z.
                            # 캔이면 depth≈350mm(윗면)이어야 z≈+100. depth≈500(테이블)
                            # 이면 z 음수 = 캔 뒤를 읽는 것(반사 구멍).
                            if os.environ.get('DBG_COORD'):
                                _mvals = depth_arr[mask01 > 0]
                                _mvals = _mvals[_mvals > 0]
                                if _mvals.size > 0:
                                    _cn = self.yolo.names.get(int(cls_ids[mi]), '?')
                                    self.get_logger().info(
                                        f"[DBG_OBJ] {_cn} px=({int(cu)},{int(cv_y)}) "
                                        f"depth[min{int(_mvals.min())} p25 "
                                        f"{int(np.percentile(_mvals,25))} med"
                                        f"{int(np.median(_mvals))} max{int(_mvals.max())}]mm "
                                        f"→ base z={base_xyz[2]:.0f}mm "
                                        f"(서있는캔이면 z>0 정상)")
                            # 클래스 시간투표 (track 별 최빈값) — can↔bottle 프레임간
                            # flip 깜빡임 완화. (일관 오분류는 재학습 필요)
                            _cid = int(cls_ids[mi])
                            _tk = self._track_key(int(bbox[0]), int(bbox[1]),
                                                  int(bbox[2]), int(bbox[3]))
                            if not hasattr(self, '_cls_vote'):
                                self._cls_vote = {}
                            _vv = self._cls_vote.setdefault(_tk, [])
                            _vv.append(_cid)
                            if len(_vv) > 15:
                                _vv.pop(0)
                            _cid = max(set(_vv), key=_vv.count)   # 최빈 클래스
                            cname = self.yolo.names.get(_cid, str(_cid))
                            gg_label = to_graspgen_label(cname)  # bread→None
                            cloud, pix = self.mask_to_base_cloud(
                                mask01, depth_arr, return_pix=True)
                            # [GraspGen export] 이 프레임 물체별 base-frame PC 적재
                            # (mi==0 에서 리셋). 'g' 키로 .npz 저장/ZMQ 전송.
                            if mi == 0:
                                self._export_buf = []
                            if (gg_label is not None and cloud is not None
                                    and len(cloud) >= 30):
                                self._export_buf.append({
                                    'label': gg_label,
                                    'cloud_m': cloud,
                                    'center_mm': [float(base_xyz[0]),
                                                  float(base_xyz[1]),
                                                  float(base_xyz[2])],
                                })
                            # 1-9 선택용 검출 등록 (mask/center/cloud/bbox — GraspGen 입력)
                            self.detections.append({
                                'name': cname,
                                'gg_label': gg_label,
                                'center_base': np.asarray(base_xyz, dtype=float) / 1000.0,  # mm→m
                                'center_px': (int(cu), int(cv_y)),
                                'mask': mask01,
                                'cloud_m': cloud,          # base frame (m), GraspGen 입력
                                'bbox': tuple(int(v) for v in bbox),
                            })
                            # rx/ry/rz 계산에 쓰인 픽셀을 초록 점으로 표시. 너무 많으면
                            # 객체·외곽선을 가림 → ~24개만 sparse 하게(작게) 표시.
                            if pix is not None and len(pix) > 0:
                                _st = max(1, len(pix) // 24)
                                _pd = pix[::_st]
                                uu = np.clip(_pd[:, 0], 0, vW - 1)
                                vv = np.clip(_pd[:, 1], 0, vH - 1)
                                for _gx, _gy in zip(uu, vv):
                                    cv2.circle(vis, (int(_gx), int(_gy)), 1,
                                               (0, 255, 0), -1)
                            label = (f'{cname} ({base_xyz[0]:+.0f}, '
                                     f'{base_xyz[1]:+.0f}, {base_xyz[2]:+.0f}) mm')
                            cv2.circle(vis, (int(cu), int(cv_y)), 4, (0, 0, 255), -1)
                            # 라벨 = 어두운 배경 + 흰 글씨 (노란 외곽선과 구분·가독성)
                            self._label_box(vis, label, int(cu) + 6, int(cv_y) - 8,
                                            0.52, (255, 255, 255))
                            rxyz = self.cloud_to_rxryrz(
                                cloud, key=self._isnet_grid_key(
                                    int(bbox[0]), int(bbox[1]),
                                    int(bbox[2]), int(bbox[3])),
                                cname=cname)
                            if rxyz is not None:
                                rx, ry, rz = rxyz
                                self._label_box(
                                    vis, f'rx{rx:+.0f} ry{ry:+.0f} rz{rz:+.0f}',
                                    int(cu) + 6, int(cv_y) + 16, 0.5,
                                    (120, 220, 255))
                                obj_dicts[cname] = {
                                    '각도': f'{rz:.1f}',
                                    '각도r': {'rx': rx, 'ry': ry, 'rz': rz},
                                    '센터포인트mm': [float(base_xyz[0]),
                                                  float(base_xyz[1]),
                                                  float(base_xyz[2])],
                                    'graspgen_label': gg_label,
                                }
                            # 외곽선 꼭지점(minAreaRect 4모서리) → base mm + 웹캠 표기.
                            # 시계방향 [우상, 우하, 좌하, 좌상]. 코너 depth 구멍 회피 위해
                            # 물체 표면 대표깊이(mask median)로 평면 투영.
                            try:
                                cc = (max(cnts, key=cv2.contourArea)
                                      if cnts else None)
                                if cc is not None and cv2.contourArea(cc) >= 50:
                                    box = cv2.boxPoints(cv2.minAreaRect(cc))
                                    bcx = float(box[:, 0].mean())
                                    bcy = float(box[:, 1].mean())
                                    ordered = [None, None, None, None]
                                    for _p in box:
                                        if _p[0] >= bcx and _p[1] < bcy:
                                            ordered[0] = _p     # 우상
                                        elif _p[0] >= bcx and _p[1] >= bcy:
                                            ordered[1] = _p     # 우하
                                        elif _p[0] < bcx and _p[1] >= bcy:
                                            ordered[2] = _p     # 좌하
                                        else:
                                            ordered[3] = _p     # 좌상
                                    dvm = depth_arr[mask01 > 0]
                                    dvm = dvm[dvm > 0]
                                    od = (float(np.median(dvm))
                                          if dvm.size > 10 else None)
                                    cc_pts = cc.reshape(-1, 2)
                                    corner_mm = []
                                    for _p in ordered:
                                        if _p is None:
                                            corner_mm.append(['', '', ''])
                                            continue
                                        # minAreaRect 코너는 외곽 밖 → 가장 가까운
                                        # contour 점으로 snap (외곽선 위에 찍히게).
                                        _di = ((cc_pts[:, 0] - _p[0]) ** 2
                                               + (cc_pts[:, 1] - _p[1]) ** 2)
                                        _sp = cc_pts[int(_di.argmin())]
                                        px, py = int(_sp[0]), int(_sp[1])
                                        cwv, _e = self.pixel_to_base_xyz(
                                            px, py, depth_arr, forced_depth=od)
                                        if cwv is not None:
                                            corner_mm.append([
                                                round(float(cwv[0]), 1),
                                                round(float(cwv[1]), 1),
                                                round(float(cwv[2]), 1)])
                                            # 주황(0,140,255) — 외곽선(노랑)/GD박스
                                            # (핑크)와 구분. 라벨은 박스 중심 반대쪽
                                            # 으로 offset 해 서로/핑크와 안 겹치게.
                                            _orng = (0, 140, 255)
                                            cv2.circle(vis, (px, py), 5, _orng, -1)
                                            cv2.circle(vis, (px, py), 6,
                                                       (0, 0, 0), 1)
                                            _txt = (f'{cwv[0]:+.0f},'
                                                    f'{cwv[1]:+.0f},{cwv[2]:+.0f}')
                                            ox = 8 if px >= bcx else -95
                                            oy = -8 if py < bcy else 16
                                            cv2.putText(
                                                vis, _txt, (px + ox, py + oy),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                                (0, 0, 0), 3, cv2.LINE_AA)
                                            cv2.putText(
                                                vis, _txt, (px + ox, py + oy),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                                _orng, 1, cv2.LINE_AA)
                                        else:
                                            corner_mm.append(['', '', ''])
                                    if cname in obj_dicts:
                                        obj_dicts[cname]['꼭지점mm'] = corner_mm
                            except Exception:
                                pass
                        # [OUTLINE_DEBUG] 오버레이 없는 raw vs 외곽선 clean 덤프
                        # (정밀 일치 판단용 — OUTLINE_DEBUG=1, 2초 throttle)
                        if os.environ.get('OUTLINE_DEBUG') == '1' and dbg_items:
                            _now_d = time.time()
                            if _now_d - getattr(self, '_odbg_last', 0.0) > 2.0:
                                self._odbg_last = _now_d
                                try:
                                    os.makedirs('/tmp/outline_dbg', exist_ok=True)
                                    for _di, (bb, rmask, rawm, cnm) in \
                                            enumerate(dbg_items):
                                        PAD = 24
                                        x1 = max(0, int(bb[0]) - PAD)
                                        y1 = max(0, int(bb[1]) - PAD)
                                        x2 = min(vW, int(bb[2]) + PAD)
                                        y2 = min(vH, int(bb[3]) + PAD)
                                        if x2 - x1 < 8 or y2 - y1 < 8:
                                            continue
                                        raw_c = frame[y1:y2, x1:x2].copy()
                                        # panel2: raw YOLO/SAM mask 외곽(빨강)
                                        ov_raw = raw_c.copy()
                                        cr, _ = cv2.findContours(
                                            rawm[y1:y2, x1:x2],
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                                        cv2.drawContours(ov_raw, cr, -1,
                                                         (0, 0, 255), 1)
                                        # panel3: refined(ISNet+depth) 외곽(노랑)
                                        ov_ref = raw_c.copy()
                                        cf, _ = cv2.findContours(
                                            rmask[y1:y2, x1:x2],
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                                        cv2.drawContours(ov_ref, cf, -1,
                                                         (0, 255, 255), 1)
                                        comp = np.hstack([raw_c, ov_raw, ov_ref])
                                        comp = cv2.resize(
                                            comp, None, fx=2.5, fy=2.5,
                                            interpolation=cv2.INTER_NEAREST)
                                        cv2.imwrite(
                                            f'/tmp/outline_dbg/clean_{_di}_{cnm}.png',
                                            comp)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # GD 가 잡았지만 YOLO seg 가 놓친 객체(예: 학습 안 된 스낵) → SAM2
                # 마스크로 외곽선/좌표/rx-ry-rz 생성. SAM2 워커에 GD bbox 등록 후,
                # 캐시 채워지면 _render_sam_object 로 동일 표기. seg 처리분은 skip.
                try:
                    if not hasattr(self, '_gd_persist'):
                        self._gd_persist = {}   # key -> (bbox, cn, last_t)
                    _nowp = time.time()
                    gd_done = []     # 이미 렌더한 GD-only 중심 (중복 제거)
                    rendered_keys = set()
                    _minc = float(self._otun('gd_only_min_conf', 0.42))   # 노이즈(≤0.40) 컷, 실제 뒤 객체(0.44~0.50) coffee 로 잡음
                    _ttl = float(self._otun('gd_persist_s', 3.0))
                    # conf 높은 순 — 같은 물체 중복검출 시 최고 conf 1개만 렌더
                    for g in sorted(gd_snap,
                                    key=lambda d: -(float(d[5]) if len(d) > 5
                                                    else 1.0)):
                        gx1, gy1, gx2, gy2 = (int(g[0]), int(g[1]),
                                              int(g[2]), int(g[3]))
                        if gx2 - gx1 < 8 or gy2 - gy1 < 8:
                            continue
                        conf = float(g[5]) if len(g) > 5 else 1.0
                        if conf < _minc:
                            continue                      # 너무 낮은 conf 노이즈 컷
                        gcx, gcy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
                        if any(abs(gcx - sx) < 70 and abs(gcy - sy) < 70
                               for (sx, sy) in seg_centers):
                            continue                      # seg 가 이미 처리
                        # GD 박스 중심이 YOLO bbox 안(20px 여유)이면 같은 물체 → 병합(skip).
                        # 키 큰 캔에서 중심 70px 넘어도 박스 안이면 중복으로 처리.
                        if any(bx[0]-20 <= gcx <= bx[2]+20 and bx[1]-20 <= gcy <= bx[3]+20
                               for bx in seg_boxes):
                            continue
                        if any(abs(gcx - dx) < 70 and abs(gcy - dy) < 70
                               for (dx, dy) in gd_done):
                            continue                      # GD 중복(같은 물체) skip
                        ph = str(g[4]).lower()
                        # 여긴 YOLO 가 못 잡은 GD-only 검출만 옴(위 seg_centers 필터).
                        # GD-only 도 실제 라벨(bottle/can/snack)로 등록 (coffee 제거).
                        cn = ('snack_bag' if 'snack' in ph
                              else 'bottle' if 'bottle' in ph
                              else 'can' if 'can' in ph
                              else None)
                        if cn is None:
                            continue
                        gd_done.append((gcx, gcy))
                        bx1 = max(0, min(vW, gx1)); by1 = max(0, min(vH, gy1))
                        bx2 = max(0, min(vW, gx2)); by2 = max(0, min(vH, gy2))
                        bb = (bx1, by1, bx2, by2)
                        key = self._isnet_grid_key(*bb)
                        isnet_boxes.append((key, bx1, by1, bx2, by2, None))
                        self._gd_persist[key] = (bb, cn, _nowp)   # 검출 기억
                        rendered_keys.add(key)
                        c = self._isnet_cache.get(key)
                        if c is not None and (_nowp - c[0]) < 8.0:
                            self._render_sam_object(
                                vis, frame, c[1], bb, cn, depth_arr)
                    # 지속성: 최근 _ttl 초 내 검출됐으나 이번 GD 에서 빠진 객체도
                    # SAM2 캐시로 계속 렌더 → 경계 conf 깜빡임 제거.
                    for k, (bb, cn, lt) in list(self._gd_persist.items()):
                        if _nowp - lt > _ttl:
                            self._gd_persist.pop(k, None); continue
                        if k in rendered_keys:
                            continue
                        pcx = (bb[0] + bb[2]) // 2; pcy = (bb[1] + bb[3]) // 2
                        if any(abs(pcx - sx) < 70 and abs(pcy - sy) < 70
                               for (sx, sy) in seg_centers):
                            continue
                        if any(bx[0]-20 <= pcx <= bx[2]+20 and bx[1]-20 <= pcy <= bx[3]+20
                               for bx in seg_boxes):
                            continue
                        if any(abs(pcx - dx) < 70 and abs(pcy - dy) < 70
                               for (dx, dy) in gd_done):
                            continue
                        isnet_boxes.append((k, bb[0], bb[1], bb[2], bb[3], None))
                        c = self._isnet_cache.get(k)
                        if c is not None and (_nowp - c[0]) < 8.0:
                            gd_done.append((pcx, pcy))
                            self._render_sam_object(vis, frame, c[1], bb,
                                                    cn, depth_arr)
                except Exception:
                    pass

                # SAM2 워커 제출 (seg + GD-only boxes). GPU 무거워 ~1.5s throttle.
                _now_req = time.time()
                if (isnet_boxes and self._isnet_req is None
                        and _now_req - getattr(self, '_isnet_req_last', 0.0)
                        > float(self._otun('sam_period_s', 1.5))):
                    self._isnet_req = (frame.copy(), isnet_boxes)
                    self._isnet_req_last = _now_req

                # 상태바
                now = time.time()
                dt = now - t_prev
                t_prev = now
                inst_fps = 1.0 / dt if dt > 0 else 0.0
                fps_ema = (0.9 * fps_ema + 0.1 * inst_fps
                           if fps_ema else inst_fps)
                n_det = 0 if (res is None or res.boxes is None) else len(res.boxes)
                armed = ('`' in self.keys_held) and ('q' in self.keys_held)
                # 검출 지속성: 이번 frame 에 놓친 seg 객체도 최근 봤으면 외곽선 유지 →
                # 깜빡임 제거. conf 낮아 YOLO 가 자주 놓치는 물체는 TTL 늘려 안정화.
                # (WSN_OUTLINE_TTL, 기본 2.0s. 정적 장면이라 길게 둬도 OK)
                if hasattr(self, '_outline_persist'):
                    _np = time.time()
                    _ttl_o = float(os.environ.get('WSN_OUTLINE_TTL', '2.0'))
                    for _tk, (_t, _cl) in list(self._outline_persist.items()):
                        if _np - _t > _ttl_o:
                            self._outline_persist.pop(_tk, None); continue
                        if _tk not in self._seg_seen:
                            self._pending_outlines.extend(_cl)
                # 모아둔 외곽선을 맨 마지막에 그림 → 라벨 박스에 안 가리고 최상위.
                for _oc in getattr(self, '_pending_outlines', []):
                    cv2.polylines(vis, [_oc], True, (0, 255, 255), 2)

                # ── 검출 지속성(깜빡임 방지) + 위치기반 안정 lock ──
                # 매 프레임 self.detections 를 통째로 새로 만들면 YOLO 가 한 프레임만
                # 놓쳐도 번호가 사라짐(깜빡임) + 인덱스가 흔들려 lock 이 빗나감.
                # → base XY 6cm 그리드를 트랙키로, 0.7s 동안 유지 + 키 순 안정정렬.
                _now = time.time()
                # 각 검출을 가장 가까운 기존 트랙(10cm 이내)에 매칭 → 그리드 경계
                # 깜빡임 제거. 없으면 신규 트랙. (base XY 거리 기준)
                for _d in self.detections:
                    _cb = _d['center_base']
                    # 매칭 임계 5cm: jitter(~2cm)보단 크고 물체간격(~10cm)보단 작게 →
                    # 같은 물체는 추종, 다른 물체는 분리 (10cm면 인접 물체 합쳐짐).
                    _best_tk = None; _best_d = 0.05
                    for _tk2, _vv in self._det_track.items():
                        _oc = _vv['det']['center_base']
                        _dist = ((_cb[0]-_oc[0])**2 + (_cb[1]-_oc[1])**2) ** 0.5
                        if _dist < _best_d:
                            _best_d = _dist; _best_tk = _tk2
                    if _best_tk is None:
                        _best_tk = (round(float(_cb[0]), 3), round(float(_cb[1]), 3))
                    # 고정 번호: 기존 트랙이면 그 번호 유지, 신규면 1-9 중 빈 번호.
                    _ex = self._det_track.get(_best_tk)
                    if _ex and _ex.get('num'):
                        _num = _ex['num']
                    else:
                        _usednum = {v.get('num') for v in self._det_track.values()}
                        _num = next((n for n in range(1, 10) if n not in _usednum), 0)
                    # ── 라벨 안정화: track별 최근 9프레임 다수결 (can↔bottle 깜빡임 제거).
                    #   자동화에서 라벨이 흔들리면 안 되므로 다수결 라벨로 고정.
                    _votes = list((_ex.get('votes') if _ex else None) or [])
                    _votes.append(str(_d.get('name', '')))
                    _votes = _votes[-9:]
                    _maj = max(set(_votes), key=_votes.count)
                    if _maj and _maj != _d.get('name'):
                        _d['name'] = _maj
                        _d['gg_label'] = to_graspgen_label(_maj)
                    _d['_tk'] = _best_tk
                    _d['num'] = _num
                    self._det_track[_best_tk] = {'det': _d, 't': _now,
                                                 'num': _num, 'votes': _votes}
                # 만료: 일반 2.5s. 단 잠긴 물체(locked_tk)는 만료 안 함(락 유지) —
                # seg 가 한참 놓쳐도 LOCKED 표시·초록박스 안 사라지게.
                for _k in [kk for kk, vv in self._det_track.items()
                           if _now - vv['t'] > 8.0   # 8초 유지 (seg 간헐 검출 마스킹)
                           and not (self.locked and kk == self.locked_tk)]:
                    del self._det_track[_k]
                _keys = sorted(self._det_track.keys())
                self.detections = [self._det_track[k]['det'] for k in _keys]
                # DBG: 검출 개수/이름/좌표 1초마다 — 장애물 후보 추적용
                if os.environ.get('DBG_DET') and now - getattr(self, '_dbg_det_t', 0.0) > 1.0:
                    self._dbg_det_t = now
                    _info = [f"{d.get('name')}({d['center_base'][0]*1000:.0f},"
                             f"{d['center_base'][1]*1000:.0f},"
                             f"{d['center_base'][2]*1000:.0f})"
                             for d in self.detections]
                    _dobs = self._depth_obstacles()
                    self.get_logger().info(
                        f"[DBG_DET] tracks={len(self.detections)} {_info} "
                        f"| depth_clusters={len(_dobs)} "
                        f"{[[round(o['pos'][0]*1000),round(o['pos'][1]*1000)] for o in _dobs]} "
                        f"| YOLO2D={[(n, round(c,2)) for (n,c) in getattr(self,'yolo_seg_results',[])]}")
                # 잠긴 물체를 인덱스가 아닌 트랙키(위치)로 추종 → 리스트가 재정렬돼도 유지
                if self.locked and self.locked_tk is not None:
                    if self.locked_tk in self._det_track:
                        self.locked_idx = _keys.index(self.locked_tk)
                        self.selected_idx = self.locked_idx
                    elif self.detections:
                        self.locked_idx = min(self.locked_idx or 0,
                                              len(self.detections) - 1)
                    else:
                        self.locked_idx = None

                # ★락 무결성 — locked인데 현재 검출에 유효 SEL 타깃이 없으면(키 None/유령/
                #   트랙 churn) 같은 클래스 실물로 재획득. 그래도 없고 auto 아니면 자동 언락.
                #   → "전부 빨강 OBS·아무것도 SEL 안 됨" stuck 원천 차단. (2026-06-22)
                _valid_lock = (self.locked and self.locked_tk is not None
                               and any(_dd.get('_tk') == self.locked_tk
                                       for _dd in self.detections))
                if self.locked and not _valid_lock \
                        and not getattr(self, '_auto_running', False):
                    _reacq = next((_dd for _dd in self.detections
                                   if self.locked_class
                                   and str(_dd.get('name', '')) == self.locked_class),
                                  None)
                    if _reacq is not None:
                        self.locked_tk = _reacq.get('_tk')      # 재획득 → SEL 복구
                        _valid_lock = True
                    else:
                        self.locked = False; self.locked_tk = None
                        self.locked_idx = None
                        self.get_logger().info('[lock] 유효 타깃 없음 → 자동 언락')

                # ── 고정 번호(num) 표시 + lock 하이라이트 (선택은 트랙키 기준) ──
                for _d in self.detections:
                    _x1, _y1, _x2, _y2 = _d['bbox']
                    _is_sel = (self.locked and _d.get('_tk') == self.locked_tk)
                    # 락 걸린 상태에서 타깃 외 검출 = 장애물(빨강 OBS). 미락이면 회색.
                    # 번호 박스에 검출 클래스명 표기 → 어떤 클래스로 규명됐는지 확인용 (사용자 2026-06-19)
                    _nm = str(_d.get('name', '?'))
                    _num = _d.get('num', '?')
                    if _is_sel:
                        _col = (0, 255, 0); _lab = f"[{_num}] {_nm} <SEL>"
                    elif _valid_lock:                # 진짜 타깃 있을 때만 나머지=장애물
                        _col = (0, 0, 255)            # 빨강 = curobo 장애물
                        _lab = f"[{_num}] {_nm} OBS"
                    else:
                        _col = (200, 200, 200); _lab = f"[{_num}] {_nm}"
                    cv2.rectangle(vis, (_x1, _y1), (_x2, _y2), _col,
                                  3 if _is_sel else 2)
                    cv2.putText(vis, _lab, (_x1, _y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _col, 2)
                # 락 중이면 장애물 상시 발행(~3Hz) — s/p 순간 외에도 curobo world 최신 유지
                # (유효 SEL 타깃이 있을 때만 — 끊긴 락으로 엉뚱한 장애물 발행 방지)
                if _valid_lock and self._selected() is not None:
                    if now - getattr(self, '_last_obs_pub', 0.0) > 0.33:
                        self._publish_obstacles(self._selected(), log=False)
                        self._last_obs_pub = now
                # depth 장애물 시각화 — 분류기가 못 잡는 물체도 빨간 OBS 박스로.
                # 단 잡동사니(키보드/손/펑보드) 거짓 클러스터로 화면이 어지러워지므로
                # curobo 발행과 동일하게 WSN_DEPTH_OBS 게이트(기본 OFF). 켜야 보임.
                if os.environ.get('WSN_DEPTH_OBS', '0') != '0' \
                        and now - getattr(self, '_viz_obs_t', 0.0) > 0.3:
                    self._viz_obs_t = now
                    _ex = None
                    if self.locked and self._selected() is not None:
                        _c = self._selected().get('center_base')
                        _ex = (float(_c[0]), float(_c[1])) if _c is not None else None
                    self._viz_obs = self._depth_obstacles(exclude_xy=_ex)
                elif os.environ.get('WSN_DEPTH_OBS', '0') == '0':
                    self._viz_obs = []
                for _ob in getattr(self, '_viz_obs', []):
                    _pb = _ob.get('px_bbox')
                    if _pb is None:
                        continue
                    cv2.rectangle(vis, (_pb[0], _pb[1]), (_pb[2], _pb[3]),
                                  (0, 0, 255), 2)
                    cv2.putText(vis, "OBS", (_pb[0], _pb[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if self.locked and self.locked_tk is not None:
                    _ltrk = self._det_track.get(self.locked_tk, {})
                    _lnum = _ltrk.get('num', '?')
                    # ★lock 순간 고정 저장된 클래스 표시 (live 검출 아님 — 전류/매대좌표에 쓰는 그 클래스)
                    _lcls = str(getattr(self, 'locked_class', None) or '?')
                    cv2.putText(vis, f"LOCKED [{_lnum}] = {_lcls}  (r=unlock)",
                                (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if self.pending_grasp_pose is not None:
                    _pb = self.pending_grasp_pose[0]
                    # 접근축 z성분(=curobo approach_z) 으로 수평/탑다운 미리 표시.
                    # |az|<0.5 = 수평(손목 spin 없음), 그 외 = 탑다운/비스듬(spin 가능).
                    _appr = np.asarray(self.pending_grasp_pose[2], dtype=float)
                    _az = float(_appr[2]) if _appr.size >= 3 else 0.0
                    if abs(_az) < 0.5:
                        _ot = f"HORIZONTAL OK (az={_az:+.2f})"; _oc = (0, 255, 0)
                    else:
                        _ot = f"TOP-DOWN/TILT spin-risk (az={_az:+.2f})"; _oc = (0, 165, 255)
                    cv2.putText(vis,
                                f"GRASP PREVIEW ({_pb[0]*1000:.0f},{_pb[1]*1000:.0f},{_pb[2]*1000:.0f})mm  p=advance+grasp r=cancel",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    cv2.putText(vis, f"  approach: {_ot}",
                                (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _oc, 2)
                status = (f'FPS {fps_ema:5.1f}  det={n_det}  '
                          f'probes={len(self.probe_markers)}')
                if armed:
                    status += '  [PROBE ARMED]'
                cv2.putText(vis, status, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255) if armed else (0, 255, 0), 2)
                cv2.putText(vis,
                            'SPACE=E-STOP  a=AUTO  o=open  h=HOME  v=shelf-check  1-9=lock  g=graspgen  p=advance+grasp  r=cancel  q=quit',
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 255), 1)
                if self.shelf_missing is not None:
                    _inv = ("SHELF: ALL STOCKED"
                            if not self.shelf_missing
                            else f"SHELF EMPTY -> PICK {len(self.shelf_missing)}: "
                                 + ", ".join(self.shelf_missing))
                    _ic = (0, 255, 0) if not self.shelf_missing else (0, 140, 255)
                    # 배경 박스 + 큰 글씨로 눈에 띄게
                    (_tw, _th), _ = cv2.getTextSize(
                        _inv, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(vis, (8, 84), (16 + _tw, 84 + _th + 14),
                                  (0, 0, 0), -1)
                    cv2.putText(vis, _inv, (12, 84 + _th + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, _ic, 2)

                # 고정 1600x1200 으로 resize. 1280x720 입력 → 1.25배 upscale 이라
                # INTER_LINEAR 로도 mosaic 없음 (FPS 우선).
                vis_show = cv2.resize(
                    vis, (self._show_w, self._show_h),
                    interpolation=cv2.INTER_LINEAR)
                cv2.imshow(self.win, vis_show)
                # ★대시보드 카메라 피드: 2프레임마다 JPEG 발행(대역폭 절감)
                self._dash_cam_n += 1
                if self._dash_cam_n % 2 == 0:
                    try:
                        _ok, _jpg = cv2.imencode(
                            '.jpg', vis_show, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if _ok:
                            _cm = CompressedImage()
                            _cm.format = 'jpeg'
                            _cm.data = _jpg.tobytes()
                            self.pub_dash_cam.publish(_cm)
                    except Exception:
                        pass
                if os.environ.get('WSN_DUMP_FRAME', '0') != '0':   # 디버그: 화면을 파일로(원격확인)
                    self._frame_dump_n = getattr(self, '_frame_dump_n', 0) + 1
                    if self._frame_dump_n % 8 == 0:
                        try:
                            cv2.imwrite('/tmp/webcam_latest.jpg', vis_show,
                                        [cv2.IMWRITE_JPEG_QUALITY, 70])
                        except Exception:
                            pass
                k = cv2.waitKey(1) & 0xFF
                if k == 32:   # 🛑 스페이스바 = 비상정지 (창 포커스 시 백업; pynput 이 OS레벨 주)
                    self._emergency_stop()
                if k == 27 or (k == ord('q') and '`' not in self.keys_held):
                    break
                # ── object_tracking 식 키: 1-9 lock / p pick / r cancel ──
                if ord('1') <= k <= ord('9'):
                    _wantnum = k - ord('0')   # '1'→1 ... '9'→9 (고정 번호)
                    _match = next((d for d in self.detections
                                   if d.get('num') == _wantnum), None)
                    if _match is not None:
                        self.locked = True
                        self.locked_tk = _match.get('_tk')
                        self.locked_idx = None
                        # ★lock 순간 클래스 고정 저장 → 화면/전류/매대좌표 모두 이걸로 (사용자 2026-06-19)
                        self.locked_class = str(_match.get('name', '?'))
                        self.get_logger().info(
                            f"LOCKED [{_wantnum}] {self.locked_class}")
                if k == ord('h') and '`' not in self.keys_held:
                    # 'h': 높은 scout 자세(카메라가 depth 범위 위)로 복귀
                    self._go_product_view()
                if k == ord('p') and '`' not in self.keys_held:
                    # 'p': 파지점에서 축방향 전진 + 집기 + 15cm 수직 lift
                    self.advance_and_grip()
                if k == ord('r') and '`' not in self.keys_held:
                    self.locked = False
                    self.locked_idx = None
                    self.locked_tk = None
                    if self.clear_grasp_preview():
                        self.get_logger().info('미리보기 취소됨')
                    self.get_logger().info('UNLOCKED')
                if k == ord('c') and '`' not in self.keys_held:
                    if self.probe_markers:
                        self.get_logger().info(
                            f'[clear] {len(self.probe_markers)}개 마커 삭제')
                    self.probe_markers.clear()
                if k == ord('f') and '`' not in self.keys_held:
                    self._fullscreen = not self._fullscreen
                    cv2.setWindowProperty(
                        self.win, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if self._fullscreen
                        else cv2.WINDOW_NORMAL)
                    if not self._fullscreen:
                        # windowed 로 복귀 시 원본 영상 크기로
                        vh, vw = vis.shape[:2]
                        cv2.resizeWindow(self.win, vw, vh)
                    self._display_info = None  # 다음 프레임에 재계산
                    self.get_logger().info(
                        f'[f] fullscreen = {self._fullscreen}')
                if k == ord('m') and '`' not in self.keys_held:
                    # 'm' 토글: ISNet+GrabCut 외곽선 정밀화 ON/OFF.
                    # ON  = 정밀 (FPS ~1.5, 봉지 모양 fit)
                    # OFF = 빠름 (FPS ~5+, GD bbox 만)
                    self._mask_refine = not getattr(self, '_mask_refine', True)
                    self._contour_cache = {}  # 캐시 비움 (다음 프레임부터 새 모드)
                    self.get_logger().info(
                        f'[m] mask_refine = {self._mask_refine} '
                        f'({"정밀" if self._mask_refine else "빠름"})')
                if k == ord('g') and '`' not in self.keys_held:
                    # 'g': 선택 물체 GraspGen 추론 → RViz 미리보기 (로봇 안 움직임)
                    self.send_graspgen()
                if k == ord('v') and '`' not in self.keys_held:
                    # 'v': home(매대뷰)로 이동 → 매대재고 확인 (없는 제품 = 바닥에서 집을 것)
                    self._go_shelf_and_check()
                if k == ord('a') and '`' not in self.keys_held:
                    # 'a': 전자동 진열 (v→없는제품→h→x작은순 파지→place→반복)
                    self.auto_restock()
                if k == ord('o') and '`' not in self.keys_held:
                    # 'o': 그리퍼 열기
                    if self._gripper_open_call(label='key-o', position=0):
                        self.get_logger().info('[o] 그리퍼 열기')
        finally:
            try:
                self.key_listener.stop()
            except Exception:
                pass
            try:
                self.pipe.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = WebcamSegNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.spin_camera()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
