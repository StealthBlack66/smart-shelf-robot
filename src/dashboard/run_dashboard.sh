#!/bin/bash
# 대시보드 백엔드 실행용 런처 (ROS 환경 source 포함).
# 일반 실행:   bash src/dashboard/run_dashboard.sh   → http://localhost:8080
#
# ※ 자동 상주(systemd) 는 의도적으로 설정하지 않음 — 필요할 때 수동 실행.
source /opt/ros/humble/setup.bash
[ -f /home/user/doosan_ws/install/setup.bash ] && source /home/user/doosan_ws/install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"
export DASH_HOST="${DASH_HOST:-0.0.0.0}"
export DASH_PORT="${DASH_PORT:-8080}"
exec python3 -u /home/user/doosan_ws/src/smart-shelf-robot/src/dashboard/server.py
