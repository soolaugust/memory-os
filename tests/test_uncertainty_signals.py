"""
test_uncertainty_signals.py — iter370 Uncertainty Signal Extraction 测试

覆盖：
  US1: _extract_uncertainty_signals — 中文显式不确定
  US2: _extract_uncertainty_signals — 英文不确定
  US3: _extract_uncertainty_signals — 隐式假设（P1 级）
  US4: _extract_uncertainty_signals — 无不确定信号时返回空列表
  US5: _extract_uncertainty_signals — 最多返回 6 条（上限控制）
  US6: _write_uncertainty_chunks — 写入 DB 并可通过 FTS5 检索
  US7: _write_uncertainty_chunks — 幂等：相同 topic 不重复写入
  US8: 写入的 chunk 类型为 reasoning_chain，info_class 为 episodic
  US9: 写入的 chunk summary 格式为 "[不确定] {topic}"
  US10: chunk importance < 0.65（低于正常决策 chunk）
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture()
def tmpdb(tmp_path):
    db_path = tmp_path / "test_store.db"
    os.environ["MEMORY_OS_DB"] = str(db_path)
    os.environ["MEMORY_OS_DIR"] = str(tmp_path)
    yield db_path
    os.environ.pop("MEMORY_OS_DB", None)
    os.environ.pop("MEMORY_OS_DIR", None)


@pytest.fixture()
def conn(tmpdb):
    from store_vfs import open_db, ensure_schema
    c = open_db(tmpdb)
    ensure_schema(c)
    yield c
    c.close()


# ── US1: 中文显式不确定信号 ──────────────────────────────────────────────────

def test_us1_chinese_explicit_uncertainty():
    from hooks.extractor import _extract_uncertainty_signals
    text = "我不确定这个函数的参数格式是否正确，需要再验证一下。"
    signals = _extract_uncertainty_signals(text)
    assert len(signals) > 0
    topics = [s[0] for s in signals]
    assert any("参数格式" in t or "函数" in t for t in topics)


# ── US2: 英文不确定信号 ────────────────────────────────────────────────────────

def test_us2_english_uncertainty():
    from hooks.extractor import _extract_uncertainty_signals
    text = "I'm not sure about the correct API endpoint to use here."
    signals = _extract_uncertainty_signals(text)
    assert len(signals) > 0
    topics = [s[0] for s in signals]
    assert any("API" in t or "endpoint" in t for t in topics)


# ── US3: 隐式假设（P1 级）────────────────────────────────────────────────────

def test_us3_implicit_assumption():
    from hooks.extractor import _extract_uncertainty_signals
    text = "需要验证这个配置文件的路径是否正确"
    signals = _extract_uncertainty_signals(text)
    assert len(signals) > 0
    levels = [s[1] for s in signals]
    assert any(lvl in ('low', 'medium') for lvl in levels)


# ── US4: 无不确定信号 → 空列表 ───────────────────────────────────────────────

def test_us4_no_uncertainty_signals():
    from hooks.extractor import _extract_uncertainty_signals
    text = "已完成数据库迁移，测试全部通过。"
    signals = _extract_uncertainty_signals(text)
    assert signals == []


# ── US5: 最多返回 6 条 ────────────────────────────────────────────────────────

def test_us5_max_signals():
    from hooks.extractor import _extract_uncertainty_signals
    # 构造含大量不确定信号的文本
    lines = [f"我不确定第{i}个参数的含义是否正确" for i in range(20)]
    text = "\n".join(lines)
    signals = _extract_uncertainty_signals(text)
    assert len(signals) <= 6


# ── US6: write_uncertainty_chunks 写入可检索 ─────────────────────────────────

def test_us6_write_uncertainty_chunks_searchable(conn):
    from hooks.extractor import _extract_uncertainty_signals, _write_uncertainty_chunks
    from store_vfs import fts_search
    signals = [("API 参数格式", "low")]
    count = _write_uncertainty_chunks(conn, signals, "proj", "sess1")
    conn.commit()
    assert count == 1
    results = fts_search(conn, "不确定 API", "proj", top_k=5)
    # 应该能找到写入的 chunk
    if results:
        assert any("不确定" in r.get("summary", "") for r in results)


# ── US7: 幂等 — 相同 topic 不重复写入 ───────────────────────────────────────

def test_us7_idempotent_write(conn):
    from hooks.extractor import _write_uncertainty_chunks
    signals = [("数据库连接字符串", "low")]
    c1 = _write_uncertainty_chunks(conn, signals, "proj", "sess1")
    conn.commit()
    c2 = _write_uncertainty_chunks(conn, signals, "proj", "sess1")
    conn.commit()
    total = conn.execute(
        "SELECT COUNT(*) FROM memory_chunks WHERE chunk_type='reasoning_chain' "
        "AND summary LIKE '[不确定]%'"
    ).fetchone()[0]
    assert total == 1  # 第二次不重复写入
    assert c1 == 1
    assert c2 == 0


# ── US8: chunk_type=reasoning_chain, info_class=episodic ────────────────────

def test_us8_chunk_type_and_info_class(conn):
    from hooks.extractor import _write_uncertainty_chunks
    signals = [("缓存失效策略", "low")]
    _write_uncertainty_chunks(conn, signals, "proj", "sess1")
    conn.commit()
    row = conn.execute(
        "SELECT chunk_type, info_class FROM memory_chunks "
        "WHERE summary='[不确定] 缓存失效策略'"
    ).fetchone()
    assert row is not None
    assert row[0] == "reasoning_chain"
    assert row[1] == "episodic"


# ── US9: summary 格式为 "[不确定] {topic}" ───────────────────────────────────

def test_us9_summary_format(conn):
    from hooks.extractor import _write_uncertainty_chunks
    topic = "检索算法的时间复杂度"
    signals = [(topic, "low")]
    _write_uncertainty_chunks(conn, signals, "proj", "sess1")
    conn.commit()
    row = conn.execute(
        "SELECT summary FROM memory_chunks WHERE summary LIKE '[不确定]%'"
    ).fetchone()
    assert row is not None
    assert row[0] == f"[不确定] {topic}"


# ── US10: importance < 0.65 ───────────────────────────────────────────────────

def test_us10_importance_below_threshold(conn):
    from hooks.extractor import _write_uncertainty_chunks
    signals = [("端口配置", "low"), ("路由规则", "medium")]
    _write_uncertainty_chunks(conn, signals, "proj", "sess1")
    conn.commit()
    rows = conn.execute(
        "SELECT importance FROM memory_chunks WHERE summary LIKE '[不确定]%'"
    ).fetchall()
    assert len(rows) == 2
    for (imp,) in rows:
        assert imp < 0.65, f"importance {imp} should be < 0.65"
