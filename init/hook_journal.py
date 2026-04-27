"""
hook_journal.py — Hook 执行日志
OS 类比：systemd-journald / journalctl

复用 store_core.py 的 dmesg_log/dmesg_read，
提供面向 hook 的结构化日志 API：
  - log_unit_start(unit_name, event)
  - log_unit_stop(unit_name, duration_ms, exit_code)
  - log_unit_fail(unit_name, duration_ms, exit_code, reason)
  - query_journal(unit_name, since, level) → 类比 journalctl -u <unit> --since <time>
  - format_journal(entries) → 类比 journalctl 文本输出格式

数据存储：复用 dmesg 表（不新建表），subsystem 固定为 "hook-journal"，
extra 字段存 JSON：
  {"unit": "memory-os-loader", "event": "SessionStart",
   "phase": "stop", "exit_code": 0, "duration_ms": 42.3}
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from store_core import dmesg_log, dmesg_read, DMESG_INFO, DMESG_WARN, DMESG_ERR, DMESG_DEBUG

_SUBSYSTEM = "hook-journal"


class HookJournal:
    """
    Hook 执行日志
    OS 类比：journald — 收集并索引 systemd unit 生命周期事件

    使用方式：
      journal = HookJournal("/path/to/store.db")
      journal.log_unit_start("memory-os-loader", "SessionStart")
      journal.log_unit_stop("memory-os-loader", duration_ms=42.3, exit_code=0)
      entries = journal.query_journal(unit_name="memory-os-loader", limit=20)
      print(journal.format_journal(entries))
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _write(self, level: str, message: str, extra: dict,
               session_id: str = "", project: str = "") -> None:
        conn = self._conn()
        try:
            dmesg_log(conn, level, _SUBSYSTEM, message,
                      session_id=session_id, project=project, extra=extra)
            conn.commit()
        finally:
            conn.close()

    # ── 生命周期日志 API ──────────────────────────────────────────────────

    def log_unit_start(self, unit_name: str, event: str,
                       session_id: str = "", project: str = "") -> None:
        """
        记录 unit 开始执行。
        类比：journald 记录 "Started <unit>"。
        extra: {"unit": "...", "event": "...", "phase": "start"}
        """
        self._write(
            DMESG_DEBUG,
            f"[start] {unit_name} ({event})",
            {"unit": unit_name, "event": event, "phase": "start"},
            session_id, project,
        )

    def log_unit_stop(self, unit_name: str, duration_ms: float,
                      exit_code: int, event: str = "",
                      session_id: str = "", project: str = "") -> None:
        """
        记录 unit 执行成功完成。
        类比：journald 记录 "Finished <unit> in Xms"。
        extra: {"unit": "...", "event": "...", "phase": "stop",
                "exit_code": 0, "duration_ms": 42.3}
        """
        self._write(
            DMESG_INFO,
            f"[stop ] {unit_name} exit={exit_code} {duration_ms:.0f}ms",
            {
                "unit": unit_name, "event": event, "phase": "stop",
                "exit_code": exit_code, "duration_ms": round(duration_ms, 2),
            },
            session_id, project,
        )

    def log_unit_fail(self, unit_name: str, duration_ms: float,
                      exit_code: int, reason: str = "", event: str = "",
                      session_id: str = "", project: str = "") -> None:
        """
        记录 unit 执行失败。
        类比：journald 记录 "Failed to start <unit>"。
        extra: {"unit": "...", "event": "...", "phase": "fail",
                "exit_code": 1, "duration_ms": 5000.0, "reason": "timeout"}
        """
        self._write(
            DMESG_ERR,
            f"[fail ] {unit_name} exit={exit_code} {duration_ms:.0f}ms | {reason[:100]}",
            {
                "unit": unit_name, "event": event, "phase": "fail",
                "exit_code": exit_code, "duration_ms": round(duration_ms, 2),
                "reason": reason[:200] if reason else "",
            },
            session_id, project,
        )

    def log_target_complete(self, event: str, total: int, succeeded: int,
                            failed: int, skipped: int, duration_ms: float,
                            session_id: str = "", project: str = "") -> None:
        """
        记录整个 event target 的执行摘要。
        类比：journald 记录 target 的 Finished 消息。
        """
        level = DMESG_INFO if failed == 0 else DMESG_WARN
        self._write(
            level,
            (f"[target] {event}: {succeeded}/{total} ok, "
             f"{failed} failed, {skipped} skipped, {duration_ms:.0f}ms"),
            {
                "unit": f"{event}.target", "event": event,
                "phase": "target_complete", "total": total,
                "succeeded": succeeded, "failed": failed,
                "skipped": skipped, "duration_ms": round(duration_ms, 2),
            },
            session_id, project,
        )

    # ── 查询 API ──────────────────────────────────────────────────────────

    def query_journal(
        self,
        unit_name: str = None,
        event: str = None,
        phase: str = None,
        since: str = None,
        level: str = None,
        limit: int = 50,
        project: str = None,
    ) -> List[dict]:
        """
        查询 hook 执行日志。
        类比：journalctl -u <unit> --since <time> -p <level>

        参数：
          unit_name — 过滤指定 unit（精确匹配 extra.unit）
          event     — 过滤指定 event（精确匹配 extra.event）
          phase     — "start" | "stop" | "fail" | "target_complete"
          since     — ISO-8601 时间戳，只返回该时间之后的记录
          level     — ERR/WARN/INFO/DEBUG
          limit     — 最大返回条数
          project   — 过滤 project
        """
        conn = self._conn()
        try:
            raw = dmesg_read(
                conn,
                level=level,
                subsystem=_SUBSYSTEM,
                limit=limit * 5,
                project=project,
            )
        finally:
            conn.close()

        results = []
        for entry in raw:
            extra = entry.get("extra") or {}
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except Exception:
                    extra = {}

            if unit_name and extra.get("unit") != unit_name:
                continue
            if event and extra.get("event") != event:
                continue
            if phase and extra.get("phase") != phase:
                continue
            if since and entry.get("timestamp", "") < since:
                continue

            entry["extra"] = extra
            results.append(entry)
            if len(results) >= limit:
                break

        return results

    def query_unit_history(self, unit_name: str, limit: int = 20) -> List[dict]:
        """
        查询单个 unit 的完整执行历史。
        类比：journalctl -u nginx.service -n 20
        """
        return self.query_journal(unit_name=unit_name, limit=limit)

    def get_failure_summary(self, event: str = None, limit: int = 100) -> List[dict]:
        """
        获取失败记录摘要。
        类比：journalctl -p err -b
        """
        entries = self.query_journal(phase="fail", event=event,
                                     level=DMESG_ERR, limit=limit)
        return [
            {
                "unit":        e.get("extra", {}).get("unit", "?"),
                "event":       e.get("extra", {}).get("event", "?"),
                "exit_code":   e.get("extra", {}).get("exit_code", -1),
                "duration_ms": e.get("extra", {}).get("duration_ms", 0),
                "reason":      e.get("extra", {}).get("reason", ""),
                "timestamp":   e.get("timestamp", ""),
            }
            for e in entries
        ]

    # ── 格式化输出 ────────────────────────────────────────────────────────

    def format_journal(self, entries: List[dict], verbose: bool = False) -> str:
        """
        格式化为可读文本。
        类比：journalctl 默认输出：
          Apr 19 10:00:00  [INFO]  memory-os-loader          [stop ] exit=0 42ms
        """
        lines = []
        for entry in entries:
            ts_raw = entry.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts_raw)
                ts_fmt = dt.strftime("%b %d %H:%M:%S")
            except Exception:
                ts_fmt = ts_raw[:19]

            level   = entry.get("level", "INFO")
            msg     = entry.get("message", "")
            extra   = entry.get("extra", {})
            unit    = extra.get("unit", "?") if isinstance(extra, dict) else "?"
            lvl_tag = {"ERR": "ERR ", "WARN": "WARN", "INFO": "INFO",
                       "DEBUG": "DBG "}.get(level, level)

            lines.append(f"{ts_fmt}  [{lvl_tag}]  {unit:<40}  {msg}")

            if verbose and isinstance(extra, dict):
                for k, v in extra.items():
                    if k not in ("unit", "phase"):
                        lines.append(f"              {k}: {v}")

        return "\n".join(lines)

    def stats(self, event: str = None) -> dict:
        """
        返回 journal 统计摘要。
        类比：journalctl --disk-usage + systemctl list-units --failed
        """
        entries = self.query_journal(event=event, limit=1000)
        by_phase: Dict[str, int] = {}
        by_event: Dict[str, int] = {}
        stop_count = fail_count = 0

        for e in entries:
            extra = e.get("extra", {})
            if not isinstance(extra, dict):
                continue
            phase = extra.get("phase", "unknown")
            ev    = extra.get("event", "unknown")
            by_phase[phase] = by_phase.get(phase, 0) + 1
            by_event[ev]    = by_event.get(ev, 0) + 1
            if phase == "stop":
                stop_count += 1
            elif phase == "fail":
                fail_count += 1

        total_runs = stop_count + fail_count
        return {
            "total_entries": len(entries),
            "by_phase":      by_phase,
            "by_event":      by_event,
            "error_rate":    round(fail_count / total_runs, 3) if total_runs > 0 else 0.0,
        }

    def __repr__(self) -> str:
        return f"HookJournal(db={self.db_path!r})"
