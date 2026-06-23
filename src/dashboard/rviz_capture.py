"""RViz 창 캡처 → MJPEG. GNOME 컴포지터라 가려져도(occluded) 윈도우 백킹에서 캡처됨.
python-xlib XGetImage 로 창 픽맵을 읽어 JPEG 로 인코딩. 대시보드 /api/rviz/stream 용.
"""
from __future__ import annotations

import os
import time
import threading

import numpy as np
import cv2
from Xlib import display, X

_DISPLAY = os.environ.get('DISPLAY', ':2')
_PLACEHOLDER = None


def _make_placeholder(text='RViz 대기'):
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (140, 190), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (80, 80, 80), 2)
    ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpg.tobytes() if ok else b''


class RvizCapture:
    """RViz 창을 주기적으로 캡처해 최신 JPEG 프레임 보유."""

    def __init__(self, win_name='RViz', fps=3):
        # ★fps 낮춤(8→3) + 캡처 다운스케일 → 로봇 RT 제어 CPU 경합 완화 (2026-06-23)
        self.win_name = win_name
        self.period = 1.0 / float(fps)
        self._down = float(os.environ.get('RVIZ_CAP_DOWNSCALE', '0.6'))  # 캡처 축소비
        self._disp = None
        self._win = None
        self._lock = threading.Lock()
        self._frame = _make_placeholder()
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _connect(self):
        if self._disp is None:
            self._disp = display.Display(_DISPLAY)

    def _name_of(self, w):
        try:
            n = w.get_wm_name()
            if n:
                return str(n)
        except Exception:
            pass
        try:
            net = self._disp.intern_atom('_NET_WM_NAME')
            utf8 = self._disp.intern_atom('UTF8_STRING')
            p = w.get_full_property(net, utf8)
            if p and p.value:
                return p.value.decode('utf-8', 'ignore')
        except Exception:
            pass
        return None

    def _find_window(self):
        """이름에 win_name(또는 .rviz) 포함된 창들 중 '가장 큰' 창 반환
        (3x3 'Qt Selection Owner' 같은 유틸창 회피 → 실제 메인 RViz 창)."""
        self._connect()
        root = self._disp.screen().root
        wins = [root]
        best = None; best_area = 0
        i = 0
        while i < len(wins) and i < 5000:
            w = wins[i]; i += 1
            try:
                nm = self._name_of(w)
                if nm and ('rviz' in nm.lower()):
                    g = w.get_geometry()
                    area = g.width * g.height
                    if area > best_area:
                        best_area = area; best = w
                wins.extend(w.query_tree().children)
            except Exception:
                continue
        return best

    def _grab(self, w):
        geo = w.get_geometry()
        ww, hh = geo.width, geo.height
        if ww <= 0 or hh <= 0:
            return None
        # 큰 창은 max-request 초과 → 가로 스트립으로 나눠 캡처
        strip = max(16, 2_000_000 // (ww * 4))
        rows = []
        y = 0
        while y < hh:
            h = min(strip, hh - y)
            raw = w.get_image(0, y, ww, h, X.ZPixmap, 0xffffffff)
            arr = np.frombuffer(raw.data, dtype=np.uint8)
            if arr.size < ww * h * 4:
                return None
            arr = arr[:ww * h * 4].reshape(h, ww, 4)   # BGRX
            rows.append(arr[:, :, :3])                 # BGR
            y += h
        return np.vstack(rows)

    def _loop(self):
        while not self._stop:
            t0 = time.time()
            try:
                if self._win is None:
                    self._win = self._find_window()
                if self._win is not None:
                    bgr = self._grab(self._win)
                    if bgr is not None:
                        if self._down < 0.99:
                            bgr = cv2.resize(bgr, None, fx=self._down, fy=self._down,
                                             interpolation=cv2.INTER_AREA)
                        ok, jpg = cv2.imencode(
                            '.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        if ok:
                            with self._lock:
                                self._frame = jpg.tobytes()
                    else:
                        self._win = None     # 캡처 실패 → 재검색
                else:
                    with self._lock:
                        self._frame = _make_placeholder('RViz 창 없음')
            except Exception:
                self._win = None
                self._disp = None
            dt = self.period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    def latest(self):
        with self._lock:
            return self._frame

    def mjpeg_generator(self):
        boundary = b'--frame\r\n'
        while True:
            f = self.latest()
            yield (boundary + b'Content-Type: image/jpeg\r\n'
                   + b'Content-Length: ' + str(len(f)).encode() + b'\r\n\r\n'
                   + f + b'\r\n')
            time.sleep(self.period)


if __name__ == '__main__':
    # 단독 테스트: 1프레임 저장
    cap = RvizCapture()
    time.sleep(2.0)
    with open('/tmp/rviz_cap_test.jpg', 'wb') as fp:
        fp.write(cap.latest())
    print('saved /tmp/rviz_cap_test.jpg', len(cap.latest()), 'bytes')
