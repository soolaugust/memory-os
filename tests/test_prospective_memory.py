"""
test_prospective_memory.py — iter390: Prospective Memory Trigger 单元测试

覆盖：
  PM1: _detect_prospective_intent — 中文展望信号（下次/记得/以后）→ 返回 trigger_pattern
  PM2: _detect_prospective_intent — TODO 意图信号 → 返回 trigger_pattern
  PM3: _detect_prospective_intent — 英文展望信号（remember to）→ 返回 trigger_pattern
  PM4: _detect_prospective_intent — 无展望信号 → 返回 None
  PM5: insert_trigger + query_triggers — 注册并命中触发器
  PM6: query_triggers — 关键词不匹配 → 不触发
  PM7: fire_trigger — fired_count 递增
  PM8: expires_at 过期 → 不触发
  PM9: query_triggers 批量返回上限 max_triggers

认知科学依据：
  Einstein & McDaniel (1990) Prospective Memory —
  意图性记忆：在未来某个时刻执行某个动作的意图（"下次打开 X 时记得..."）。
  触发模式：特定信号（context cue）激活相关延迟意图记忆。
OS 类比：Linux inotify/fanotify — 注册文件系统事件监听，触发条件满足时唤醒等待进程。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa

from store_vfs import ensure_schema, insert_trigger, query_triggers, fire_trigger
# Import detection function from extractor
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))
from extractor import _detect_prospective_intent


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_trigger(tid, chunk_id, project, trigger_pattern,
                  trigger_type="keyword", expires_at=None):
    return {
        "id": tid,
        "chunk_id": chunk_id,
        "project": project,
        "session_id": "s1",
        "trigger_pattern": trigger_pattern,
        "trigger_type": trigger_type,
        "created_at": _now(),
        "fired_count": 0,
        "last_fired": None,
        "expires_at": expires_at,
    }


# ── PM1: 中文展望信号 ─────────────────────────────────────────────────────

def test_pm1_chinese_prospective_signal():
    """中文展望信号（下次/记得/以后）→ 返回 trigger_pattern。"""
    texts = [
        "下次部署前记得检查数据库迁移脚本",
        "记得以后处理 Redis 连接池配置",
        "以后运行测试的时候注意内存限制",
    ]
    for text in texts:
        result = _detect_prospective_intent(text)
        assert result is not None, f"应检测到展望意图: '{text}'"
        assert len(result) >= 2, f"trigger_pattern 应有实质内容: got '{result}'"


# ── PM2: TODO 意图信号 ────────────────────────────────────────────────────

def test_pm2_todo_signal():
    """TODO 意图信号 → 返回 trigger_pattern。"""
    texts = [
        "TODO：优化 BM25 召回分数阈值",
        "待办: 清理旧版本配置文件",
    ]
    for text in texts:
        result = _detect_prospective_intent(text)
        assert result is not None, f"应检测到 TODO 意图: '{text}'"


# ── PM3: 英文展望信号 ─────────────────────────────────────────────────────

def test_pm3_english_prospective_signal():
    """英文展望信号（remember to）→ 返回 trigger_pattern。"""
    text = "remember to update the migration script before deploying"
    result = _detect_prospective_intent(text)
    assert result is not None, f"应检测到英文展望意图: '{text}'"
    assert len(result) >= 3


# ── PM4: 无展望信号 ──────────────────────────────────────────────────────

def test_pm4_no_prospective_signal():
    """普通文本（无展望意图）→ 返回 None。"""
    texts = [
        "BM25 算法的参数调整完成了",
        "FTS5 索引创建成功，耗时 120ms",
        "这是一个普通的分析结果",
    ]
    for text in texts:
        result = _detect_prospective_intent(text)
        assert result is None, f"不应检测到展望意图: '{text}' → got '{result}'"


# ── PM5: insert_trigger + query_triggers — 注册并命中 ────────────────────

def test_pm5_register_and_hit_trigger(conn):
    """注册 trigger 后，匹配 query 时返回 chunk_id。"""
    insert_trigger(conn, _make_trigger("t1", "chunk_abc", "test", "数据库迁移"))
    conn.commit()

    matches = query_triggers(conn, "test", "下次部署前检查数据库迁移脚本")
    assert len(matches) == 1, f"应命中1条 trigger，got {len(matches)}"
    cid, tid, pattern = matches[0]
    assert cid == "chunk_abc"
    assert "数据库迁移" in pattern


# ── PM6: 关键词不匹配 → 不触发 ──────────────────────────────────────────

def test_pm6_no_match_different_topic(conn):
    """query 不含 trigger_pattern 关键词 → 不触发。"""
    insert_trigger(conn, _make_trigger("t2", "chunk_xyz", "test", "Redis 连接池"))
    conn.commit()

    matches = query_triggers(conn, "test", "BM25 评分算法优化")
    assert len(matches) == 0, f"不同话题不应触发，got {len(matches)}"


# ── PM7: fire_trigger — fired_count 递增 ─────────────────────────────────

def test_pm7_fire_trigger_increments_count(conn):
    """fire_trigger 后 fired_count 递增。"""
    insert_trigger(conn, _make_trigger("t3", "chunk_def", "test", "缓存优化"))
    conn.commit()

    fire_trigger(conn, "t3")
    conn.commit()

    row = conn.execute("SELECT fired_count, last_fired FROM trigger_conditions WHERE id='t3'").fetchone()
    assert row["fired_count"] == 1, f"fired_count 应为 1，got {row['fired_count']}"
    assert row["last_fired"] is not None, "last_fired 应被设置"


# ── PM8: expires_at 过期 → 不触发 ───────────────────────────────────────

def test_pm8_expired_trigger_not_fired(conn):
    """expires_at 过期的 trigger → 不触发。"""
    expired_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    insert_trigger(conn, _make_trigger(
        "t4", "chunk_old", "test", "旧触发条件", expires_at=expired_time
    ))
    conn.commit()

    matches = query_triggers(conn, "test", "旧触发条件相关内容")
    assert len(matches) == 0, f"过期 trigger 不应触发，got {len(matches)}"


# ── PM9: 批量上限 max_triggers ───────────────────────────────────────────

def test_pm9_max_triggers_limit(conn):
    """query_triggers 返回不超过 max_triggers 条结果。"""
    for i in range(5):
        insert_trigger(conn, _make_trigger(
            f"t_{i}", f"chunk_{i}", "test", f"数据库迁移{i}"
        ))
    conn.commit()

    matches = query_triggers(conn, "test", "数据库迁移配置检查", max_triggers=3)
    assert len(matches) <= 3, f"不应超过 max_triggers=3，got {len(matches)}"
