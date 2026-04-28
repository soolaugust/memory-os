"""
test_iter429_enactment_effect.py — iter429: Enactment Effect 单元测试

覆盖：
  EE1: source_type='tool_result' → stability × 1.4
  EE2: chunk_type='tool_insight' → stability × 1.4
  EE3: content 含 shell prompt 特征 → 被检测为 enacted
  EE4: 普通文本 content（无工具特征）→ 不加成
  EE5: enactment_enabled=False → 不加成
  EE6: stability cap 遵守（永远不超过 365.0）
  EE7: boost factor 可通过 sysctl 配置
  EE8: 普通 chunk_type='decision' 且无工具特征 → 不加成
  EE9: content 含 diff 标记（+++ b/）→ 检测为 enacted
  EE10: insert_chunk 写入 source_type='tool_result' 的 chunk → stability 被提升

认知科学依据：
  Engelkamp & Zimmer (1989) "Memory for subject-performed tasks" —
    Subject-Performed Tasks (SPT) 比 Verbal Tasks (VT) 记忆留存率高约 40%。
    行动编码激活运动皮层 + 语义系统双路径，形成多模态记忆痕迹。

OS 类比：Linux writeback dirty page accounting —
  write() syscall 产生的 dirty page 比 read() 产生的 clean page
  有更高的 priority（行动 = 写入 = 更强记忆）。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tmpfs  # noqa
from store_vfs import ensure_schema, apply_enactment_effect, insert_chunk


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _insert_raw(conn, cid, project="test", chunk_type="decision",
                importance=0.7, stability=1.0, source_type=None,
                content="default content"):
    now = _now_iso()
    # source_type 列可能还不存在——如果不存在就忽略
    try:
        conn.execute(
            "INSERT OR REPLACE INTO memory_chunks "
            "(id, project, chunk_type, content, summary, importance, stability, "
            "created_at, updated_at, retrievability, source_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9, ?)",
            (cid, project, chunk_type, content, f"summary_{cid}",
             importance, stability, now, now, source_type)
        )
    except Exception:
        conn.execute(
            "INSERT OR REPLACE INTO memory_chunks "
            "(id, project, chunk_type, content, summary, importance, stability, "
            "created_at, updated_at, retrievability) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.9)",
            (cid, project, chunk_type, content, f"summary_{cid}",
             importance, stability, now, now)
        )
    conn.commit()


def _get_stability(conn, cid):
    row = conn.execute(
        "SELECT stability FROM memory_chunks WHERE id=?", (cid,)
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0


# ── EE1: source_type='tool_result' → stability × 1.4 ──────────────────────────

def test_ee1_source_type_tool_result(conn):
    """EE1: source_type='tool_result' 的 chunk 应得到 stability × 1.4 加成。"""
    _insert_raw(conn, "ee1_chunk", stability=1.0, source_type="tool_result")

    result_stab = apply_enactment_effect(conn, "ee1_chunk", base_stability=1.0)
    final_stab = _get_stability(conn, "ee1_chunk")

    assert result_stab > 1.0, f"EE1: 返回值应 > 1.0，got {result_stab}"
    assert abs(result_stab - 1.4) < 0.01, f"EE1: stability 应约为 1.4，got {result_stab}"
    assert abs(final_stab - 1.4) < 0.01, f"EE1: DB stability 应约为 1.4，got {final_stab}"


# ── EE2: chunk_type='tool_insight' → stability × 1.4 ──────────────────────────

def test_ee2_chunk_type_tool_insight(conn):
    """EE2: chunk_type='tool_insight' 应被识别为 enacted，stability × 1.4。"""
    _insert_raw(conn, "ee2_chunk", chunk_type="tool_insight", stability=2.0,
                source_type=None, content="some tool output")

    result_stab = apply_enactment_effect(conn, "ee2_chunk", base_stability=2.0)
    assert result_stab > 2.0, f"EE2: tool_insight chunk 应得到加成，got {result_stab}"
    assert abs(result_stab - 2.8) < 0.05, f"EE2: 2.0×1.4=2.8，got {result_stab}"


# ── EE3: content 含 shell prompt 特征 → 检测为 enacted ─────────────────────────

def test_ee3_shell_prompt_in_content(conn):
    """EE3: content 包含 '$ ls -la' 等 shell prompt → 被检测为 enacted。"""
    shell_content = "$ ls -la /tmp\ntotal 42\ndrwxr-xr-x 2 root root 4096 Jan 1 00:00 ."
    _insert_raw(conn, "ee3_chunk", chunk_type="decision", stability=1.0,
                source_type=None, content=shell_content)

    result_stab = apply_enactment_effect(conn, "ee3_chunk", base_stability=1.0)
    assert result_stab > 1.0, f"EE3: shell prompt 内容应被检测为 enacted，got {result_stab}"


# ── EE4: 普通文本 → 不加成 ────────────────────────────────────────────────────

def test_ee4_plain_text_no_boost(conn):
    """EE4: 普通文字内容（无工具特征）不应得到 enactment 加成。"""
    plain_content = "今天讨论了架构方案，决定使用微服务模式。"
    _insert_raw(conn, "ee4_chunk", chunk_type="decision", stability=1.5,
                source_type=None, content=plain_content)

    result_stab = apply_enactment_effect(conn, "ee4_chunk", base_stability=1.5)
    assert abs(result_stab - 1.5) < 0.01, \
        f"EE4: 普通内容不应改变 stability，got {result_stab}"


# ── EE5: enactment_enabled=False → 不加成 ─────────────────────────────────────

def test_ee5_enactment_disabled(conn):
    """EE5: store_vfs.enactment_enabled=False 时，不应用任何加成。"""
    import unittest.mock as mock
    import config as _config

    _insert_raw(conn, "ee5_chunk", stability=1.0, source_type="tool_result")

    original_get = _config.get
    def patched_get(key, project=None):
        if key == "store_vfs.enactment_enabled":
            return False
        return original_get(key, project=project)

    with mock.patch.object(_config, 'get', side_effect=patched_get):
        result_stab = apply_enactment_effect(conn, "ee5_chunk", base_stability=1.0)

    assert abs(result_stab - 1.0) < 0.01, \
        f"EE5: 禁用时不应加成，got {result_stab}"


# ── EE6: stability cap 遵守（永远不超过 365.0）─────────────────────────────────

def test_ee6_stability_cap_respected(conn):
    """EE6: 即使 stability × boost 超过 365.0，也应截断到 cap。"""
    _insert_raw(conn, "ee6_chunk", stability=300.0, source_type="tool_result")

    result_stab = apply_enactment_effect(conn, "ee6_chunk", base_stability=300.0)
    assert result_stab <= 365.0, \
        f"EE6: stability 不应超过 365.0，got {result_stab}"
    # 300 × 1.4 = 420 > 365 → 应截断为 365
    assert abs(result_stab - 365.0) < 0.01, \
        f"EE6: 应截断到 cap=365.0，got {result_stab}"


# ── EE7: boost factor 可通过 sysctl 配置 ──────────────────────────────────────

def test_ee7_custom_boost_factor(conn):
    """EE7: enactment_boost 可通过 sysctl 配置，custom boost=2.0 时 stability 应翻倍。"""
    import unittest.mock as mock
    import config as _config

    _insert_raw(conn, "ee7_chunk", stability=1.0, source_type="tool_result")

    original_get = _config.get
    def patched_get(key, project=None):
        if key == "store_vfs.enactment_boost":
            return 2.0
        return original_get(key, project=project)

    with mock.patch.object(_config, 'get', side_effect=patched_get):
        result_stab = apply_enactment_effect(conn, "ee7_chunk", base_stability=1.0)

    assert abs(result_stab - 2.0) < 0.01, \
        f"EE7: boost=2.0 时 stability 应为 2.0，got {result_stab}"


# ── EE8: 普通 decision chunk 无工具特征 → 不加成 ─────────────────────────────

def test_ee8_decision_no_tool_markers(conn):
    """EE8: chunk_type='decision' 且 content 无工具特征，不应得到 enactment 加成。"""
    _insert_raw(conn, "ee8_chunk", chunk_type="decision", stability=1.2,
                source_type=None, content="决策：选择 PostgreSQL 作为主数据库。原因：事务支持完善。")

    result_stab = apply_enactment_effect(conn, "ee8_chunk", base_stability=1.2)
    assert abs(result_stab - 1.2) < 0.01, \
        f"EE8: 普通 decision 不应加成，got {result_stab}"


# ── EE9: content 含 diff 标记 → 检测为 enacted ─────────────────────────────────

def test_ee9_diff_markers_in_content(conn):
    """EE9: content 包含 diff 标记（+++ b/ 或 --- a/）→ 识别为代码编辑工具输出。"""
    diff_content = (
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -10,3 +10,4 @@\n"
        "+    logger.info('启动服务')\n"
    )
    _insert_raw(conn, "ee9_chunk", chunk_type="reasoning_chain", stability=1.0,
                source_type=None, content=diff_content)

    result_stab = apply_enactment_effect(conn, "ee9_chunk", base_stability=1.0)
    assert result_stab > 1.0, f"EE9: diff 标记应被检测为 enacted，got {result_stab}"


# ── EE10: insert_chunk 写入 source_type='tool_result' 的 chunk ────────────────

def test_ee10_insert_chunk_integration(conn):
    """EE10: insert_chunk 写入含 source_type='tool_result' 的 chunk 后，stability 被提升。"""
    now = _now_iso()
    chunk_dict = {
        "id": "ee10_chunk",
        "project": "test",
        "chunk_type": "decision",
        "content": "执行了 git commit 操作，确认代码已提交",
        "summary": "git commit 操作结果",
        "importance": 0.7,
        "stability": 1.0,
        "source_type": "tool_result",
        "created_at": now,
        "updated_at": now,
        "source_session": "test_session",
        "retrievability": 0.9,
        "last_accessed": now,
    }
    insert_chunk(conn, chunk_dict)
    final_stab = _get_stability(conn, "ee10_chunk")

    # source_type='tool_result' 应触发 enactment effect，stability 提升
    assert final_stab > 1.0, \
        f"EE10: insert_chunk 后 stability 应被提升，got {final_stab}"
    assert final_stab <= 365.0, "EE10: stability 不应超过 cap"
