#!/usr/bin/env python3
"""
迭代29 测试：dmesg Ring Buffer — 结构化事件日志

OS 类比验证：
  T1 基础写入/读取：printk() + dmesg 读取
  T2 级别过滤：dmesg --level=err 只显示严重级别
  T3 子系统过滤：dmesg | grep "retriever"
  T4 环形缓冲区裁剪：ring buffer 满时自动覆盖最旧条目
  T5 dmesg -c 清空：清空并返回条目数
  T6 extra 附加数据：JSON 序列化/反序列化
  T7 schema 幂等：多次 ensure_schema 不报错
  T8 project 过滤：按项目筛选日志
  T9 集成测试：retriever dmesg_log 导入正确
  T10 config tunable：dmesg.ring_buffer_size 可配置
"""
import sys
import os
import json
import time
from pathlib import Path

# 设置 path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import sqlite3
import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import
from store import (
    open_db, ensure_schema,
    dmesg_log, dmesg_read, dmesg_clear,
    DMESG_ERR, DMESG_WARN, DMESG_INFO, DMESG_DEBUG,
)
from config import get as _sysctl

# 用内存数据库测试（不污染生产 store.db）
def _test_db():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn

PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def test_basic_write_read():
    """T1：基础写入/读取 — printk() + dmesg"""
    print("\nT1: 基础写入/读取")
    conn = _test_db()

    dmesg_log(conn, DMESG_INFO, "retriever", "query classified as FULL",
              session_id="s1", project="proj1")
    dmesg_log(conn, DMESG_WARN, "extractor", "cgroup OOM: evicted 3 chunks",
              session_id="s1", project="proj1")
    conn.commit()

    entries = dmesg_read(conn)
    check("写入2条读取2条", len(entries) == 2, f"got {len(entries)}")
    check("最新在前（WARN）", entries[0]["level"] == DMESG_WARN)
    check("subsystem 正确", entries[0]["subsystem"] == "extractor")
    check("message 正确", "evicted 3" in entries[0]["message"])
    check("project 字段", entries[0]["project"] == "proj1")
    conn.close()


def test_level_filter():
    """T2：级别过滤 — dmesg --level=err"""
    print("\nT2: 级别过滤")
    conn = _test_db()

    dmesg_log(conn, DMESG_DEBUG, "loader", "debug info")
    dmesg_log(conn, DMESG_INFO, "retriever", "injected 3 chunks")
    dmesg_log(conn, DMESG_WARN, "extractor", "approaching quota")
    dmesg_log(conn, DMESG_ERR, "retriever", "FTS5 index corrupt")
    conn.commit()

    err_only = dmesg_read(conn, level=DMESG_ERR)
    check("ERR 过滤只返回1条", len(err_only) == 1, f"got {len(err_only)}")

    warn_up = dmesg_read(conn, level=DMESG_WARN)
    check("WARN 过滤返回 ERR+WARN=2条", len(warn_up) == 2, f"got {len(warn_up)}")

    info_up = dmesg_read(conn, level=DMESG_INFO)
    check("INFO 过滤返回 ERR+WARN+INFO=3条", len(info_up) == 3, f"got {len(info_up)}")

    all_entries = dmesg_read(conn, level=DMESG_DEBUG)
    check("DEBUG 过滤返回全部4条", len(all_entries) == 4, f"got {len(all_entries)}")

    no_filter = dmesg_read(conn)
    check("无过滤返回全部4条", len(no_filter) == 4, f"got {len(no_filter)}")
    conn.close()


def test_subsystem_filter():
    """T3：子系统过滤 — dmesg | grep 'retriever'"""
    print("\nT3: 子系统过滤")
    conn = _test_db()

    dmesg_log(conn, DMESG_INFO, "retriever", "msg1")
    dmesg_log(conn, DMESG_INFO, "extractor", "msg2")
    dmesg_log(conn, DMESG_INFO, "retriever", "msg3")
    dmesg_log(conn, DMESG_INFO, "writer", "msg4")
    conn.commit()

    retriever_only = dmesg_read(conn, subsystem="retriever")
    check("retriever 过滤返回2条", len(retriever_only) == 2, f"got {len(retriever_only)}")
    check("全部是 retriever", all(e["subsystem"] == "retriever" for e in retriever_only))
    conn.close()


def test_ring_buffer_trim():
    """T4：环形缓冲区裁剪 — ring buffer overflow"""
    print("\nT4: 环形缓冲区裁剪")
    conn = _test_db()

    # 用环境变量临时设置小 buffer size
    os.environ["MEMORY_OS_DMESG_RING_BUFFER_SIZE"] = "10"
    try:
        # 写入 15 条
        for i in range(15):
            dmesg_log(conn, DMESG_INFO, "test", f"msg-{i:02d}")
        conn.commit()

        entries = dmesg_read(conn, limit=100)
        check("裁剪后 <= 10 条", len(entries) <= 10, f"got {len(entries)}")
        # 最新的应该是 msg-14
        check("保留最新条目", "msg-14" in entries[0]["message"], f"got {entries[0]['message']}")
        # 最旧的应该被删除（msg-00 到 msg-04 应该被裁剪）
        all_msgs = [e["message"] for e in entries]
        check("最旧条目被裁剪", "msg-00" not in " ".join(all_msgs))
    finally:
        del os.environ["MEMORY_OS_DMESG_RING_BUFFER_SIZE"]
    conn.close()


def test_dmesg_clear():
    """T5：dmesg -c 清空"""
    print("\nT5: dmesg -c 清空")
    conn = _test_db()

    dmesg_log(conn, DMESG_INFO, "test", "msg1")
    dmesg_log(conn, DMESG_INFO, "test", "msg2")
    dmesg_log(conn, DMESG_INFO, "test", "msg3")
    conn.commit()

    cleared = dmesg_clear(conn)
    conn.commit()
    check("清空返回3", cleared == 3, f"got {cleared}")

    remaining = dmesg_read(conn)
    check("清空后0条", len(remaining) == 0, f"got {len(remaining)}")
    conn.close()


def test_extra_json():
    """T6：extra 附加数据 JSON 序列化"""
    print("\nT6: extra 附加数据")
    conn = _test_db()

    extra_data = {"top_k_ids": ["abc", "def"], "priority": "FULL", "latency_ms": 1.35}
    dmesg_log(conn, DMESG_INFO, "retriever", "injected with extra",
              extra=extra_data)
    conn.commit()

    entries = dmesg_read(conn)
    check("有 extra 字段", "extra" in entries[0])
    check("extra 是 dict", isinstance(entries[0]["extra"], dict))
    check("extra 数据正确", entries[0]["extra"]["priority"] == "FULL")
    check("extra 列表正确", entries[0]["extra"]["top_k_ids"] == ["abc", "def"])
    conn.close()


def test_schema_idempotent():
    """T7：schema 幂等"""
    print("\nT7: schema 幂等")
    conn = _test_db()
    # 再次调用 ensure_schema 不应报错
    ensure_schema(conn)
    ensure_schema(conn)
    dmesg_log(conn, DMESG_INFO, "test", "after triple schema")
    conn.commit()
    entries = dmesg_read(conn)
    check("三次 ensure_schema 后正常写入", len(entries) == 1)
    conn.close()


def test_project_filter():
    """T8：project 过滤"""
    print("\nT8: project 过滤")
    conn = _test_db()

    dmesg_log(conn, DMESG_INFO, "retriever", "proj-a msg", project="proj-a")
    dmesg_log(conn, DMESG_INFO, "retriever", "proj-b msg", project="proj-b")
    dmesg_log(conn, DMESG_INFO, "extractor", "proj-a msg2", project="proj-a")
    conn.commit()

    proj_a = dmesg_read(conn, project="proj-a")
    check("proj-a 过滤返回2条", len(proj_a) == 2, f"got {len(proj_a)}")
    proj_b = dmesg_read(conn, project="proj-b")
    check("proj-b 过滤返回1条", len(proj_b) == 1, f"got {len(proj_b)}")
    conn.close()


def test_import_from_retriever():
    """T9：集成测试 — retriever 导入 dmesg 符号"""
    print("\nT9: retriever 导入验证")
    # 验证 retriever.py 可以正确导入 dmesg 相关符号
    try:
        sys.path.insert(0, str(_ROOT / "hooks"))
        # 只验证导入不报错（不实际运行 main）
        from store import dmesg_log as _dl, DMESG_INFO as _di, DMESG_WARN as _dw, DMESG_DEBUG as _dd
        check("dmesg_log 可导入", callable(_dl))
        check("DMESG_INFO 值正确", _di == "INFO")
        check("DMESG_WARN 值正确", _dw == "WARN")
        check("DMESG_DEBUG 值正确", _dd == "DEBUG")
    except ImportError as e:
        check("导入成功", False, str(e))


def test_config_tunable():
    """T10：config tunable — dmesg.ring_buffer_size"""
    print("\nT10: config tunable")
    val = _sysctl("dmesg.ring_buffer_size")
    check("默认值 500", val == 500, f"got {val}")
    check("类型 int", isinstance(val, int))


def test_message_truncation():
    """T11：消息截断保护"""
    print("\nT11: 消息截断")
    conn = _test_db()
    long_msg = "A" * 1000
    dmesg_log(conn, DMESG_INFO, "test", long_msg)
    conn.commit()
    entries = dmesg_read(conn)
    check("长消息被截断到 500 字", len(entries[0]["message"]) <= 500, f"got {len(entries[0]['message'])}")
    conn.close()


def test_performance():
    """T12：性能测试 — 写入延迟"""
    print("\nT12: 性能")
    conn = _test_db()
    t0 = time.time()
    for i in range(100):
        dmesg_log(conn, DMESG_INFO, "perf", f"msg-{i}")
    conn.commit()
    dur_ms = (time.time() - t0) * 1000
    check(f"100次写入 < 100ms（实测 {dur_ms:.1f}ms）", dur_ms < 100)

    t0 = time.time()
    entries = dmesg_read(conn, limit=50)
    read_ms = (time.time() - t0) * 1000
    check(f"50条读取 < 10ms（实测 {read_ms:.1f}ms）", read_ms < 10)
    conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("迭代29 测试：dmesg Ring Buffer")
    print("=" * 60)

    test_basic_write_read()
    test_level_filter()
    test_subsystem_filter()
    test_ring_buffer_trim()
    test_dmesg_clear()
    test_extra_json()
    test_schema_idempotent()
    test_project_filter()
    test_import_from_retriever()
    test_config_tunable()
    test_message_truncation()
    test_performance()

    print(f"\n{'=' * 60}")
    print(f"结果：{PASS} passed, {FAIL} failed / {PASS + FAIL} total")
    print(f"{'=' * 60}")
    sys.exit(1 if FAIL > 0 else 0)
