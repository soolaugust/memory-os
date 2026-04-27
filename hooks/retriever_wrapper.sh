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

SOCKET="${MEMORY_OS_DAEMON_SOCK:-/tmp/memory-os-retriever.sock}"
_HOOKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON_PY="${_HOOKS_DIR}/retriever_daemon.py"
RETRIEVER_PY="${_HOOKS_DIR}/retriever.py"
PID_FILE="${SOCKET}.pid"
LOCK_FILE="${SOCKET}.lock"

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

# ── Fast Path：daemon socket 存在时尝试通过 socket 发送 ──
if [ -S "$SOCKET" ]; then
    if _try_socket; then
        exit 0
    fi
    # socket 存在但请求失败（daemon 刚退出），继续走 fallback
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
                    >"${_LOG_DIR}/daemon.log" 2>&1 &
                # 等待 daemon socket 就绪（最多等 3 秒）
                for _i in $(seq 1 30); do
                    sleep 0.1
                    [ -S "$SOCKET" ] && break
                done
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
