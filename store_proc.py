"""
store_proc.py — /proc Virtual Filesystem + dmesg Ring Buffer

从 store_core.py 拆分（迭代26-29 功能集）。
包含：proc_stats()（运行时可观测性）、dmesg 日志系统。

OS 类比：Linux /proc (Plan 9 -> Linux 1992) + /dev/kmsg ring buffer
"""
import json
from datetime import datetime, timezone

from store_vfs import open_db, ensure_schema, STORE_DB, MEMORY_OS_DIR

# ── /proc Virtual Filesystem — 运行时可观测性（迭代26）─────────

def proc_stats(conn=None) -> dict:
    """
    迭代26：/proc — 运行时可观测性。
    OS 类比：Linux /proc/meminfo + /proc/stat + /proc/vmstat
      cat /proc/meminfo 给出内存使用全貌，
      cat /proc/stat 给出 CPU 调度统计，
      cat /proc/vmstat 给出页面换入换出计数。
    proc_stats() 一次调用返回 memory-os 全貌：
      chunks — 总量/按项目/按类型分布（≈ /proc/meminfo）
      retrieval — 召回次数/命中率/平均延迟（≈ /proc/stat）
      staleness — 7/30/90天未访问 chunk 数（≈ /proc/vmstat pgsteal）

    返回 dict，可直接 json.dumps 输出。
    """
    own_conn = conn is None
    if own_conn:
        if not STORE_DB.exists():
            return {"error": "store.db not found"}
        conn = open_db()
        ensure_schema(conn)

    try:
        stats = {}

        # ── /proc/meminfo：chunk 分布 ──
        total = conn.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]

        by_project = conn.execute(
            "SELECT project, COUNT(*) FROM memory_chunks GROUP BY project ORDER BY COUNT(*) DESC"
        ).fetchall()

        by_type = conn.execute(
            "SELECT chunk_type, COUNT(*) FROM memory_chunks GROUP BY chunk_type ORDER BY COUNT(*) DESC"
        ).fetchall()

        stats["chunks"] = {
            "total": total,
            "by_project": {p: c for p, c in by_project},
            "by_type": {t: c for t, c in by_type},
        }

        # ── /proc/stat：召回统计 ──
        trace_total = conn.execute("SELECT COUNT(*) FROM recall_traces").fetchone()[0]
        trace_injected = conn.execute(
            "SELECT COUNT(*) FROM recall_traces WHERE injected=1"
        ).fetchone()[0]
        trace_skipped = conn.execute(
            "SELECT COUNT(*) FROM recall_traces WHERE injected=0"
        ).fetchone()[0]

        avg_latency = conn.execute(
            "SELECT AVG(duration_ms) FROM recall_traces WHERE duration_ms > 0"
        ).fetchone()[0]
        p95_latency = conn.execute(
            """SELECT duration_ms FROM recall_traces
               WHERE duration_ms > 0
               ORDER BY duration_ms DESC
               LIMIT 1 OFFSET (
                   SELECT CAST(COUNT(*) * 0.05 AS INTEGER)
                   FROM recall_traces WHERE duration_ms > 0
               )"""
        ).fetchone()

        hit_rate = (trace_injected / trace_total * 100) if trace_total > 0 else 0.0

        stats["retrieval"] = {
            "total_queries": trace_total,
            "injected": trace_injected,
            "skipped": trace_skipped,
            "hit_rate_pct": round(hit_rate, 1),
            "avg_latency_ms": round(avg_latency, 2) if avg_latency else 0.0,
            "p95_latency_ms": round(p95_latency[0], 2) if p95_latency else 0.0,
        }

        # ── /proc/vmstat：过期/活跃度统计 ──
        # 迭代147：datetime(last_accessed) 修复 ISO8601+timezone 字符串比较 bug
        stale_7d = conn.execute(
            """SELECT COUNT(*) FROM memory_chunks
               WHERE datetime(last_accessed) < datetime('now', '-7 days')"""
        ).fetchone()[0]
        stale_30d = conn.execute(
            """SELECT COUNT(*) FROM memory_chunks
               WHERE datetime(last_accessed) < datetime('now', '-30 days')"""
        ).fetchone()[0]
        stale_90d = conn.execute(
            """SELECT COUNT(*) FROM memory_chunks
               WHERE datetime(last_accessed) < datetime('now', '-90 days')"""
        ).fetchone()[0]

        avg_importance = conn.execute(
            "SELECT AVG(importance) FROM memory_chunks"
        ).fetchone()[0]
        avg_access_count = conn.execute(
            "SELECT AVG(COALESCE(access_count, 0)) FROM memory_chunks"
        ).fetchone()[0]

        stats["staleness"] = {
            "not_accessed_7d": stale_7d,
            "not_accessed_30d": stale_30d,
            "not_accessed_90d": stale_90d,
            "active_pct": round((total - stale_7d) / total * 100, 1) if total > 0 else 0.0,
        }

        stats["health"] = {
            "avg_importance": round(avg_importance, 3) if avg_importance else 0.0,
            "avg_access_count": round(avg_access_count, 1) if avg_access_count else 0.0,
        }

        # ── 迭代38：/proc/[pid]/oom_score_adj 统计 ──
        try:
            protected_count = conn.execute(
                "SELECT COUNT(*) FROM memory_chunks WHERE COALESCE(oom_adj, 0) < 0"
            ).fetchone()[0]
            disposable_count = conn.execute(
                "SELECT COUNT(*) FROM memory_chunks WHERE COALESCE(oom_adj, 0) > 0"
            ).fetchone()[0]
            locked_count = conn.execute(
                "SELECT COUNT(*) FROM memory_chunks WHERE COALESCE(oom_adj, 0) <= -1000"
            ).fetchone()[0]
            stats["oom_score"] = {
                "protected": protected_count,   # oom_adj < 0
                "locked": locked_count,         # oom_adj <= -1000 (mlock)
                "disposable": disposable_count, # oom_adj > 0
                "default": total - protected_count - disposable_count,
            }
        except Exception:
            stats["oom_score"] = {"protected": 0, "locked": 0, "disposable": 0, "default": total}

        # ── 迭代33：/proc/swaps — swap 分区统计 ──
        try:
            swap_total = conn.execute("SELECT COUNT(*) FROM swap_chunks").fetchone()[0]
            swap_by_project = conn.execute(
                "SELECT project, COUNT(*) FROM swap_chunks GROUP BY project ORDER BY COUNT(*) DESC"
            ).fetchall()
            stats["swap"] = {
                "total": swap_total,
                "by_project": {p: c for p, c in swap_by_project},
            }
        except Exception:
            stats["swap"] = {"total": 0, "by_project": {}}

        # ── 迭代36：/proc/pressure — PSI 压力统计 ──
        try:
            from store_mm import psi_stats as _psi_stats
            psi_by_project = {}
            for proj, _ in by_project:
                psi_by_project[proj] = _psi_stats(conn, proj)
            stats["pressure"] = psi_by_project
        except Exception:
            stats["pressure"] = {}

        # ── 迭代42：/proc/damon — 数据访问热度分布 ──
        try:
            zero_access = conn.execute(
                "SELECT COUNT(*) FROM memory_chunks WHERE COALESCE(access_count, 0) = 0"
            ).fetchone()[0]
            stats["access_heatmap"] = {
                "zero_access_count": zero_access,
                "zero_access_pct": round(zero_access / total * 100, 1) if total > 0 else 0.0,
                "avg_access_count": round(avg_access_count, 1) if avg_access_count else 0.0,
            }
        except Exception:
            stats["access_heatmap"] = {}

        # ── 迭代41：/proc/schedstat — Deadline I/O Scheduler 统计 ──
        try:
            deadline_traces = conn.execute(
                """SELECT COUNT(*) FROM recall_traces
                   WHERE reason LIKE '%deadline%'"""
            ).fetchone()[0]
            hard_deadline_traces = conn.execute(
                """SELECT COUNT(*) FROM recall_traces
                   WHERE reason LIKE '%hard_deadline%'"""
            ).fetchone()[0]
            stats["deadline"] = {
                "soft_deadline_skips": deadline_traces,
                "hard_deadline_hits": hard_deadline_traces,
                "total_traces": trace_total,
                "skip_rate_pct": round(deadline_traces / max(1, trace_total) * 100, 1),
            }
        except Exception:
            stats["deadline"] = {"soft_deadline_skips": 0, "hard_deadline_hits": 0}

        # ── /proc/aimd：TCP AIMD 拥塞窗口统计（迭代50）──
        try:
            from store_mm import _AIMD_STATE_FILE, _cwnd_to_policy
            aimd_data = {}
            if _AIMD_STATE_FILE.exists():
                aimd_raw = json.loads(_AIMD_STATE_FILE.read_text(encoding="utf-8"))
                if isinstance(aimd_raw, dict):
                    for proj, pdata in aimd_raw.items():
                        if isinstance(pdata, dict):
                            aimd_data[proj] = {
                                "cwnd": pdata.get("cwnd", 0),
                                "policy": _cwnd_to_policy(pdata.get("cwnd", 0.7)),
                                "hit_rate": pdata.get("hit_rate", 0),
                                "direction": pdata.get("direction", ""),
                            }
            stats["aimd"] = aimd_data if aimd_data else {"status": "no_data"}
        except Exception:
            stats["aimd"] = {"status": "error"}

        return stats

    finally:
        if own_conn:
            conn.close()

# ── dmesg Ring Buffer — 结构化事件日志（迭代29）─────────────────

# 日志级别常量（严重度从高到低）
DMESG_ERR = "ERR"       # 错误：FTS5 查询失败、配额超限触发淘汰
DMESG_WARN = "WARN"     # 警告：降级路径、接近配额
DMESG_INFO = "INFO"     # 信息：正常操作记录
DMESG_DEBUG = "DEBUG"   # 调试：详细内部状态

_LEVEL_ORDER = {DMESG_ERR: 0, DMESG_WARN: 1, DMESG_INFO: 2, DMESG_DEBUG: 3}


def dmesg_log(conn, level: str, subsystem: str,
              message: str, session_id: str = "", project: str = "",
              extra: dict = None) -> None:
    """
    迭代29：写入一条 dmesg 日志。
    OS 类比：printk(KERN_ERR "subsystem: message") — 内核子系统通过 printk
    向环形缓冲区写入带级别和子系统标签的结构化日志。

    参数：
      level — ERR/WARN/INFO/DEBUG（对应 KERN_ERR/KERN_WARNING/KERN_INFO/KERN_DEBUG）
      subsystem — 来源子系统（retriever/extractor/writer/loader/router/eviction）
      message — 日志内容（简洁，<200字）
      extra — 可选附加数据（JSON 序列化）

    环形缓冲区机制：
      写入后检查总条目数，超过 dmesg.ring_buffer_size 时删除最旧条目。
      OS 类比：__log_buf 固定大小，满时覆盖最旧记录。
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    extra_json = json.dumps(extra, ensure_ascii=False) if extra else None

    conn.execute(
        """INSERT INTO dmesg (timestamp, level, subsystem, message, session_id, project, extra)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (now_iso, level, subsystem, message[:500], session_id or "", project or "", extra_json),
    )

    # 环形缓冲区裁剪：保留最新 N 条
    try:
        from config import get as _cfg
        max_size = _cfg("dmesg.ring_buffer_size")
    except Exception:
        max_size = 500

    count = conn.execute("SELECT COUNT(*) FROM dmesg").fetchone()[0]
    if count > max_size:
        overflow = count - max_size
        conn.execute(
            "DELETE FROM dmesg WHERE id IN (SELECT id FROM dmesg ORDER BY id ASC LIMIT ?)",
            (overflow,),
        )


def dmesg_read(conn, level: str = None,
               subsystem: str = None, limit: int = 50,
               project: str = None) -> list:
    """
    迭代29：读取 dmesg 日志。
    OS 类比：dmesg | grep -i "error" — 按级别/子系统过滤内核日志。

    返回 dict 列表，按时间倒序（最新在前）。
    level 过滤：指定级别 = 该级别及更严重的级别（ERR 只返回 ERR，INFO 返回 ERR+WARN+INFO）。
    """
    sql = "SELECT id, timestamp, level, subsystem, message, session_id, project, extra FROM dmesg WHERE 1=1"
    params = []

    if level and level in _LEVEL_ORDER:
        threshold = _LEVEL_ORDER[level]
        allowed = [k for k, v in _LEVEL_ORDER.items() if v <= threshold]
        placeholders = ",".join("?" * len(allowed))
        sql += f" AND level IN ({placeholders})"
        params.extend(allowed)

    if subsystem:
        sql += " AND subsystem = ?"
        params.append(subsystem)

    if project:
        sql += " AND project = ?"
        params.append(project)

    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    result = []
    for rid, ts, lvl, sub, msg, sid, proj, ext in rows:
        entry = {
            "id": rid, "timestamp": ts, "level": lvl, "subsystem": sub,
            "message": msg, "session_id": sid, "project": proj,
        }
        if ext:
            try:
                entry["extra"] = json.loads(ext)
            except Exception:
                entry["extra"] = ext
        result.append(entry)
    return result


def dmesg_clear(conn) -> int:
    """
    清空 dmesg 缓冲区。
    OS 类比：dmesg -c（读取并清空内核日志缓冲区）。
    返回清除的条目数。
    """
    count = conn.execute("SELECT COUNT(*) FROM dmesg").fetchone()[0]
    conn.execute("DELETE FROM dmesg")
    return count
