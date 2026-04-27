#!/usr/bin/env bash
# ── iter162: retriever daemon wrapper ──
# OS 类比：Linux glibc vDSO wrapper — 用户态薄层，优先走快速路径（socket），
#   失败时 fallback 到系统调用（direct Python）。
#
# 性能目标：
#   daemon running:  ~2ms (bash + nc + socket roundtrip)
#   first call:      ~60ms (启动 daemon 同时用 direct Python 处理本次请求)
#   daemon fallback: ~55ms (direct Python，退化到 iter161 水平)
#
# daemon 响应语义：
#   {} 或空    → SKIP/TLB hit，无注入（正常情况，直接退出 0）
#   {"hookSpecificOutput": ...} → 有注入内容，打印后退出 0
#   连接失败    → fallback 到 direct Python
#
# iter259: Heartbeat + Auto-restart
#   DEAD_COUNT_FILE 记录连续心跳失败次数。
#   失败 ≥ 2 次：强制杀死旧 daemon（可能挂死），清理残留文件，重新启动。
#   OS 类比：systemd watchdog — WatchdogSec 超时触发 SIGKILL + Restart=on-failure。

SOCKET="${MEMORY_OS_DAEMON_SOCK:-/tmp/memory-os-retriever.sock}"
_HOOKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON_PY="${_HOOKS_DIR}/retriever_daemon.py"
RETRIEVER_PY="${_HOOKS_DIR}/retriever.py"
PID_FILE="${SOCKET}.pid"
LOCK_FILE="${SOCKET}.lock"
DEAD_COUNT_FILE="${SOCKET}.dead"

# ── 读 stdin（只能读一次）──
INPUT="$(cat)"

# ── 辅助函数：检查 daemon 是否运行 ──
_daemon_running() {
    if [ ! -f "$PID_FILE" ]; then
        return 1
    fi
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null)" || return 1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# ── 辅助函数：通过 socket 发送请求 ──
# 返回值：0 = socket 通信成功（响应已输出），1 = 失败
_try_socket() {
    local response
    response="$(printf '%s\n' "$INPUT" | nc -U -N -w 3 "$SOCKET" 2>/dev/null)"
    local nc_exit=$?

    if [ $nc_exit -ne 0 ]; then
        return 1  # nc 失败（连接被拒、超时等）
    fi

    # 响应为空 → SKIP/TLB hit（daemon 处理了请求但不注入）
    # 响应为 {} → 同上（daemon 显式返回空对象）
    # 其他 → 有注入内容，打印
    if [ -n "$response" ] && [ "$response" != "{}" ]; then
        printf '%s\n' "$response"
    fi
    return 0
}

# ── iter259: Heartbeat 探针 ──
# 快速 ping（100ms timeout），确认 daemon 真正可响应，而非仅 socket 文件残留。
# OS 类比：TCP keepalive probe — 验证连接活跃，而非仅检查 fd 是否存在。
_heartbeat_ok() {
    echo '{"ping":1}' | nc -U -N -w 1 "$SOCKET" 2>/dev/null | grep -q '"pong"' 2>/dev/null
}

# ── iter259: 强制重启 daemon（挂死时使用）──
_force_restart_daemon() {
    local _log_dir="${MEMORY_OS_DIR:-$HOME/.claude/memory-os}"
    # 杀死旧进程
    if [ -f "$PID_FILE" ]; then
        local _old_pid
        _old_pid="$(cat "$PID_FILE" 2>/dev/null)"
        if [ -n "$_old_pid" ]; then
            kill -9 "$_old_pid" 2>/dev/null || true
        fi
    fi
    # 清理残留文件
    rm -f "$SOCKET" "$PID_FILE" "$DEAD_COUNT_FILE" 2>/dev/null || true
    # 重新启动
    mkdir -p "$_log_dir"
    nohup python3 "$DAEMON_PY" >>"${_log_dir}/daemon.log" 2>&1 &
    # 等待 socket 就绪（最多 3 秒）
    for _i in $(seq 1 30); do
        sleep 0.1
        [ -S "$SOCKET" ] && break
    done
    echo 0 > "$DEAD_COUNT_FILE" 2>/dev/null || true
}

# ── Fast Path：daemon socket 存在时尝试通过 socket 发送 ──
if [ -S "$SOCKET" ]; then
    if _try_socket; then
        # 成功：清零 dead count
        echo 0 > "$DEAD_COUNT_FILE" 2>/dev/null || true
        exit 0
    fi
    # socket 存在但请求失败 → 心跳计数加一，判断是否需要强制重启
    # OS 类比：watchdog kick 未收到 → 递增失败计数器
    _dead_count="$(cat "$DEAD_COUNT_FILE" 2>/dev/null || echo 0)"
    _dead_count=$((_dead_count + 1))
    echo "$_dead_count" > "$DEAD_COUNT_FILE" 2>/dev/null || true
    if [ "$_dead_count" -ge 2 ]; then
        # 连续 2 次失败 → 强制重启 daemon
        _force_restart_daemon
    fi
fi

# ── Daemon 启动逻辑（仅当 daemon 未运行时）──
if ! _daemon_running; then
    # 使用 flock 防止并发启动多个 daemon
    (
        flock -n 9 2>/dev/null && {
            # double-check after acquiring lock
            if ! _daemon_running; then
                _LOG_DIR="${MEMORY_OS_DIR:-$HOME/.claude/memory-os}"
                mkdir -p "$_LOG_DIR"
                nohup python3 "$DAEMON_PY" \
                    >>"${_LOG_DIR}/daemon.log" 2>&1 &
                # 等待 daemon socket 就绪（最多等 3 秒）
                for _i in $(seq 1 30); do
                    sleep 0.1
                    [ -S "$SOCKET" ] && break
                done
                echo 0 > "$DEAD_COUNT_FILE" 2>/dev/null || true
            fi
        }
    ) 9>"$LOCK_FILE" 2>/dev/null || true
fi

# ── Second Chance：daemon 现在可能已启动，再试一次 ──
if [ -S "$SOCKET" ]; then
    if _try_socket; then
        exit 0
    fi
fi

# ── Fallback：直接运行 retriever.py（退化到 iter161 水平）──
# OS 类比：vDSO miss → fallback to syscall
printf '%s' "$INPUT" | exec python3 "$RETRIEVER_PY"
