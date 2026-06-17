"""작업 성능 지표 집계 + SQLite 영속화.

컨트롤러(main_controller_node.py)가 미구현이므로 지표는 어디서도 집계되지 않는다.
→ 대시보드 백엔드가 직접 이벤트를 관측해서 집계한다.

집계 정의
  · 총 시도(attempts)         : safe_grasp 호출/결과 1건 = 1 시도
  · 파지 성공률(전류 기반)     : grasp_detected == True 비율  (전류 delta 임계 도달)
  · 파지 성공률(자세 기반)     : final_position 이 target 근처로 도달 == True 비율
  · place 성공률              : place 이벤트 success 비율 (컨트롤러 연동 전까진 수동 마킹)
  · 평균 택 타임              : cycle_start ~ cycle_end 간 경과 평균
  · ERROR 횟수 / 원인          : rosout ERROR + 파지 실패 사유 집계

스키마는 단순하게 이벤트 로그 테이블 1개 + 집계는 쿼리로.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Any, Optional

# 자세 기반 성공 판정: 목표 대비 허용 오차(틱). object_lost 면 실패.
POSE_SUCCESS_TOL = 60


class MetricsStore:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'dashboard_metrics.db')
        self._lock = threading.Lock()
        # 멀티스레드 접근 → check_same_thread=False + 자체 락
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_schema()
        self._cycle_start: Optional[float] = None   # 현재 진행 중 사이클 시작 ts

    def _init_schema(self) -> None:
        with self._lock, self._db:
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          REAL NOT NULL,
                    kind        TEXT NOT NULL,      -- grasp | place | error | cycle
                    success     INTEGER,            -- 0/1/NULL
                    detail      TEXT,               -- json/문자열
                    grasp_detected INTEGER,         -- 전류 기반
                    pose_ok     INTEGER,            -- 자세 기반
                    target_pos  INTEGER,
                    final_pos   INTEGER,
                    final_current INTEGER,
                    object_class TEXT,
                    duration    REAL                -- cycle 용 택타임(초)
                )
            """)

    def _insert(self, **kw) -> None:
        cols = ('ts', 'kind', 'success', 'detail', 'grasp_detected', 'pose_ok',
                'target_pos', 'final_pos', 'final_current', 'object_class', 'duration')
        row = {c: kw.get(c) for c in cols}
        row['ts'] = row['ts'] or time.time()
        with self._lock, self._db:
            self._db.execute(
                f"INSERT INTO events ({','.join(cols)}) "
                f"VALUES ({','.join(':' + c for c in cols)})", row)

    # ── 이벤트 기록 ──────────────────────────────────────────────
    def record_grasp(self, result: dict, object_class: Optional[str] = None,
                     target_pos: Optional[int] = None) -> None:
        """ControlClients.safe_grasp 결과 콜백에서 호출."""
        grasp_detected = bool(result.get('grasp_detected'))
        final_pos = result.get('final_position')
        object_lost = bool(result.get('object_lost'))
        pose_ok = False
        if not object_lost and final_pos is not None and target_pos is not None:
            pose_ok = abs(int(final_pos) - int(target_pos)) <= POSE_SUCCESS_TOL
        self._insert(
            kind='grasp',
            success=1 if result.get('success') else 0,
            grasp_detected=1 if grasp_detected else 0,
            pose_ok=1 if pose_ok else 0,
            target_pos=target_pos,
            final_pos=final_pos,
            final_current=result.get('final_current'),
            object_class=object_class,
            detail=result.get('message'),
        )
        if not (grasp_detected or pose_ok):
            self._insert(kind='error', success=0,
                         detail=f"grasp_fail: {result.get('message') or 'unknown'} "
                                f"(lost={object_lost})",
                         object_class=object_class)

    def record_place(self, success: bool, detail: str = '') -> None:
        self._insert(kind='place', success=1 if success else 0, detail=detail)

    def record_error(self, detail: str) -> None:
        self._insert(kind='error', success=0, detail=detail)

    # 택 타임 사이클 마킹
    def cycle_start(self) -> None:
        self._cycle_start = time.time()

    def cycle_end(self, success: bool = True) -> None:
        if self._cycle_start is None:
            return
        dur = time.time() - self._cycle_start
        self._insert(kind='cycle', success=1 if success else 0, duration=dur)
        self._cycle_start = None

    # ── 집계 조회 ────────────────────────────────────────────────
    def _scalar(self, q: str, *args) -> Any:
        with self._lock:
            cur = self._db.execute(q, args)
            r = cur.fetchone()
            return r[0] if r else None

    def summary(self) -> dict:
        attempts = self._scalar("SELECT COUNT(*) FROM events WHERE kind='grasp'") or 0
        grasp_ok = self._scalar(
            "SELECT COUNT(*) FROM events WHERE kind='grasp' AND grasp_detected=1") or 0
        pose_ok = self._scalar(
            "SELECT COUNT(*) FROM events WHERE kind='grasp' AND pose_ok=1") or 0
        place_n = self._scalar("SELECT COUNT(*) FROM events WHERE kind='place'") or 0
        place_ok = self._scalar(
            "SELECT COUNT(*) FROM events WHERE kind='place' AND success=1") or 0
        avg_tact = self._scalar(
            "SELECT AVG(duration) FROM events WHERE kind='cycle' AND duration IS NOT NULL")
        error_n = self._scalar("SELECT COUNT(*) FROM events WHERE kind='error'") or 0

        def rate(num, den):
            return round(100.0 * num / den, 1) if den else None

        return {
            'attempts': attempts,
            'grasp_success_rate_current': rate(grasp_ok, attempts),
            'grasp_success_rate_pose': rate(pose_ok, attempts),
            'place_success_rate': rate(place_ok, place_n),
            'avg_tact_time': round(avg_tact, 2) if avg_tact else None,
            'error_count': error_n,
        }

    def recent_events(self, n: int = 50, kind: Optional[str] = None) -> list[dict]:
        with self._lock:
            if kind:
                cur = self._db.execute(
                    "SELECT * FROM events WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, n))
            else:
                cur = self._db.execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ?", (n,))
            return [dict(r) for r in cur.fetchall()]

    def error_causes(self) -> list[dict]:
        """파지 실패/에러 원인 집계 (원인 문자열 앞부분 기준)."""
        with self._lock:
            cur = self._db.execute(
                "SELECT detail, COUNT(*) AS n FROM events "
                "WHERE kind='error' GROUP BY detail ORDER BY n DESC LIMIT 20")
            return [{'cause': r['detail'], 'count': r['n']} for r in cur.fetchall()]

    def reset(self) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM events")
