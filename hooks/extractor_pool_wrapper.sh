#!/usr/bin/env bash
# ── iter260: extractor pool manager wrapper ──
# OS 类比：systemd service unit — 确保 extractor_pool 常驻进程在后台持续运行，
#   在首次调用时启动，在 crash 后自动重启（类似 Restart=on-failure）。
#
# 调用方式：
#   此脚本不被 Claude hook 直接调用。由 iterate.sh 在 session 启动时唤起。
#   也可由 loader.py SessionStart 在检测到 pool 未运行时触发。
#
# 健康检查机制：
#   1. 检查 extractor_pool.pid — 进程是否存在
#   2. 检查 extractor_pool.heartbeat — 心跳是否新鲜（< 30s）
#   3. 如果 pool 不健康：SIGKILL 旧进程 → 清理文件 → 重新启动
#
# OS 类比细化：
#   check_health  → watchdog timer 检查（WDOG_RESET 前 kick 超时判断）
#   restart logic → systemd Restart=on-failure + RestartSec=2
#   nohup         → systemd ExecStart= 进程分离
#   flock         → 防止并发启动（类似 systemd single-instance 语义）

_HOOKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POOL_PY="${_HOOKS_DIR}/extractor_pool.py"
_LOG_DIR="${MEMORY_OS_DIR:-$HOME/.claude/memory-os}"
PID_FILE="${_LOG_DIR}/extractor_pool.pid"
HEARTBEAT_FILE="${_LOG_DIR}/extractor_pool.heartbeat"
LOCK_FILE="${_LOG_DIR}/extractor_pool.lock"
LOG_FILE="${_LOG_DIR}/extractor_pool.log"

mkdir -p "$_LOG_DIR"

# ── 检查 pool 是否健康 ────────────────────────────────────────────────────────
_pool_healthy() {
    # 1. pid 文件存在
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null)" || return 1
    [ -n "$pid" ] || return 1
    # 2. 进程存活
    kill -0 "$pid" 2>/dev/null || return 1
    # 3. 心跳文件存在且新鲜（< 30 seconds）
    [ -f "$HEARTBEAT_FILE" ] || return 1
    local hb_mtime now age
    hb_mtime="$(stat -c '%Y' "$HEARTBEAT_FILE" 2>/dev/null)" || return 1
    now="$(date +%s)"
    age=$(( now - hb_mtime ))
    [ "$age" -le 30 ] || return 1
    return 0
}

# ── 强制停止 pool ─────────────────────────────────────────────────────────────
_pool_stop() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid="$(cat "$PID_FILE" 2>/dev/null)"
        if [ -n "$pid" ]; then
            kill -TERM "$pid" 2>/dev/null || true
            # 等最多 5 秒
            for _i in $(seq 1 50); do
                sleep 0.1
                kill -0 "$pid" 2>/dev/null || break
            done
            # 仍存活则 SIGKILL
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
    rm -f "$PID_FILE" "$HEARTBEAT_FILE" 2>/dev/null || true
}

# ── 启动 pool ─────────────────────────────────────────────────────────────────
_pool_start() {
    (
        flock -n 9 2>/dev/null && {
            if ! _pool_healthy; then
                nohup python3 "$POOL_PY" start >>"$LOG_FILE" 2>&1 &
                echo $! > "$PID_FILE"
                # 等心跳出现（最多 5 秒）
                for _i in $(seq 1 50); do
                    sleep 0.1
                    [ -f "$HEARTBEAT_FILE" ] && break
                done
            fi
        }
    ) 9>"$LOCK_FILE" 2>/dev/null || true
}

# ── 主命令逻辑 ────────────────────────────────────────────────────────────────
CMD="${1:-ensure}"

case "$CMD" in
    start)
        _pool_start
        if _pool_healthy; then
            echo "extractor_pool started (pid=$(cat "$PID_FILE" 2>/dev/null))"
        else
            echo "extractor_pool failed to start" >&2
            exit 1
        fi
        ;;

    stop)
        _pool_stop
        echo "extractor_pool stopped"
        ;;

    restart)
        _pool_stop
        sleep 0.5
        _pool_start
        if _pool_healthy; then
            echo "extractor_pool restarted (pid=$(cat "$PID_FILE" 2>/dev/null))"
        else
            echo "extractor_pool failed to restart" >&2
            exit 1
        fi
        ;;

    status)
        if _pool_healthy; then
            echo "extractor_pool: running (pid=$(cat "$PID_FILE" 2>/dev/null))"
            if [ -f "$HEARTBEAT_FILE" ]; then
                hb_mtime="$(stat -c '%Y' "$HEARTBEAT_FILE" 2>/dev/null)"
                now="$(date +%s)"
                age=$(( now - hb_mtime ))
                echo "  last heartbeat: ${age}s ago"
            fi
            exit 0
        else
            echo "extractor_pool: stopped"
            exit 1
        fi
        ;;

    ensure)
        # 确保运行中，未运行则启动（供 iterate.sh 和 SessionStart 调用）
        if ! _pool_healthy; then
            _pool_stop  # 清理可能的僵尸 pid 文件
            _pool_start
        fi
        # 静默退出（不打印任何内容，避免干扰 hook 输出）
        ;;

    health)
        # 健康检查（供 extractor.py 调用，返回 JSON）
        python3 "$POOL_PY" health
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|ensure|health}" >&2
        exit 1
        ;;
esac
