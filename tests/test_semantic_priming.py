"""
test_semantic_priming.py — iter404: Semantic Priming — Spreading Activation with Temporal Decay

覆盖：
  SP1: prime_entities — 写入 priming_state 表
  SP2: prime_entities — 空输入安全返回 0
  SP3: prime_entities — 重复 prime 保留更强的 strength（取 max）
  SP4: get_active_primes — 返回当前活跃（未衰减）的 primes
  SP5: get_active_primes — 过期的 primes 不返回（衰减到阈值以下）
  SP6: get_active_primes — 空 project 安全返回 {}
  SP7: compute_priming_boost — 有匹配 prime 时返回 boost > 0
  SP8: compute_priming_boost — 无匹配 prime 时返回 0.0
  SP9: compute_priming_boost — boost 上限 0.30
  SP10: compute_priming_boost — chunk 无 encode_context 返回 0.0
  SP11: clear_stale_primes — 清理过期条目
  SP12: clear_stale_primes — 活跃条目不被清理
  SP13: update_accessed 自动触发 priming（integration）
  SP14: prime strength 随时间衰减（指数衰减模型）
  SP15: 空/None 输入全部安全返回

认知科学依据：
  Collins & Loftus (1975) Spreading Activation Theory:
    语义网络中，激活从当前概念沿关联链扩散，扩散强度随距离衰减。
    时间维度：启动效应持续约数十分钟，随时间指数衰减。
  Meyer & Schvaneveldt (1971) Semantic Priming:
    "Bread"→"Butter" 反应更快，说明语义相关记忆被共同激活。

OS 类比：Linux page readahead —
  访问 page N 触发 [N+1, N+ra_size] 预取进 cache；
  类比：检索 chunk A → prime 相关 entities → 后续相关 chunk 有 cache 命中优势。
"""
import sys
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

import tmpfs  # noqa
from store_vfs import (
    ensure_schema,
    prime_entities,
    get_active_primes,
    compute_priming_boost,
    clear_stale_primes,
    update_accessed,
    insert_chunk,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _past(minutes=0, hours=0, days=0):
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes, hours=hours, days=days)
    return dt.isoformat()


def _insert_chunk_with_ctx(conn, chunk_id, encode_context, project="test"):
    """插入 chunk 并设置 encode_context。"""
    now = _now()
    conn.execute(
        "INSERT OR REPLACE INTO memory_chunks "
        "(id, project, chunk_type, content, summary, importance, retrievability, "
        "stability, encode_context, created_at, updated_at) "
        "VALUES (?, ?, 'decision', 'content', 'summary', 0.7, 0.5, 1.0, ?, ?, ?)",
        (chunk_id, project, encode_context, now, now)
    )
    conn.commit()


def _make_chunk(cid, content="test", summary="test", chunk_type="decision",
                project="test", stability=1.0):
    now = _now()
    return {
        "id": cid, "created_at": now, "updated_at": now, "project": project,
        "source_session": "s1", "chunk_type": chunk_type, "info_class": "world",
        "content": content, "summary": summary, "tags": [chunk_type],
        "importance": 0.7, "retrievability": 0.5, "last_accessed": now,
        "access_count": 0, "oom_adj": 0, "lru_gen": 0, "stability": stability,
        "raw_snippet": "",
    }


# ══════════════════════════════════════════════════════════════════════
# 1. prime_entities 测试
# ══════════════════════════════════════════════════════════════════════

def test_sp1_prime_entities_writes_db(conn):
    """prime_entities 写入 priming_state 表。"""
    n = prime_entities(conn, ["redis", "cache", "performance"], "test", prime_strength=0.9)
    conn.commit()
    assert n == 3, f"SP1: 应写入 3 条，got {n}"
    rows = conn.execute("SELECT entity_name FROM priming_state WHERE project='test'").fetchall()
    entity_names = {r[0] for r in rows}
    assert "redis" in entity_names, "SP1: 'redis' 应在 priming_state"
    assert "cache" in entity_names, "SP1: 'cache' 应在 priming_state"


def test_sp2_prime_entities_empty_safe(conn):
    """空输入安全返回 0。"""
    assert prime_entities(conn, [], "test") == 0
    assert prime_entities(conn, None, "test") == 0
    assert prime_entities(conn, ["redis"], "") == 0


def test_sp3_prime_entities_keeps_stronger(conn):
    """重复 prime 时保留更强的 strength（不降级已有强 prime）。"""
    # 第一次 prime：strength = 0.9
    prime_entities(conn, ["redis"], "test", prime_strength=0.9)
    conn.commit()
    # 第二次 prime：strength = 0.3（弱于现有）
    prime_entities(conn, ["redis"], "test", prime_strength=0.3)
    conn.commit()
    row = conn.execute(
        "SELECT prime_strength FROM priming_state WHERE entity_name='redis' AND project='test'"
    ).fetchone()
    # 应保留 0.9（更强的）
    assert row is not None
    assert row[0] >= 0.8, f"SP3: 应保留更强的 prime，got {row[0]}"


# ══════════════════════════════════════════════════════════════════════
# 2. get_active_primes 测试
# ══════════════════════════════════════════════════════════════════════

def test_sp4_get_active_primes_returns_recent(conn):
    """刚 prime 的 entity 应出现在 active primes 中。"""
    prime_entities(conn, ["redis", "cache"], "test", prime_strength=1.0)
    conn.commit()
    actives = get_active_primes(conn, "test")
    assert "redis" in actives, f"SP4: 'redis' 应在 active primes，got {actives}"
    assert actives["redis"] > 0.9, f"SP4: 刚 prime 的 strength 应接近 1.0"


def test_sp5_get_active_primes_excludes_expired(conn):
    """超过半衰期足够多的 prime 应衰减到阈值以下，不返回。"""
    # 插入 3 小时前的 prime（远超 30min 半衰期）
    old_time = _past(hours=3)
    conn.execute(
        "INSERT INTO priming_state (entity_name, project, primed_at, prime_strength) "
        "VALUES (?, 'test', ?, 1.0)",
        ("old_entity", old_time)
    )
    conn.commit()
    actives = get_active_primes(conn, "test")
    # 3小时后，strength = 1.0 × exp(-ln2/30 × 180min) = 1.0 × exp(-6×ln2) ≈ 0.016
    # 低于 0.05 阈值，应不出现
    assert "old_entity" not in actives, (
        f"SP5: 过期 prime 不应返回，got actives={actives}"
    )


def test_sp6_get_active_primes_empty_project(conn):
    """空 project 安全返回 {}。"""
    result = get_active_primes(conn, "")
    assert result == {}, f"SP6: 空 project 应返回空字典，got {result}"
    result2 = get_active_primes(conn, None)
    assert result2 == {}, f"SP6: None project 应返回空字典，got {result2}"


def test_sp14_prime_strength_decays_over_time(conn):
    """时间越长，prime strength 越低（指数衰减）。

    半衰期 30min：
    - 15min 后：strength ≈ 0.707（2^-0.5，半个半衰期）
    - 30min 后：strength ≈ 0.5（恰好一个半衰期）
    - 60min 后：strength ≈ 0.25（两个半衰期）
    """
    # 插入 30min 前的 prime（恰好一个半衰期）
    time_30min_ago = _past(minutes=30)
    conn.execute(
        "INSERT INTO priming_state (entity_name, project, primed_at, prime_strength) "
        "VALUES ('decay_test', 'test', ?, 1.0)",
        (time_30min_ago,)
    )
    conn.commit()
    actives = get_active_primes(conn, "test")
    if "decay_test" in actives:
        # 30min 后，strength ≈ 0.5（半衰期）
        s = actives["decay_test"]
        assert 0.35 <= s <= 0.65, f"SP14: 30min(一个半衰期)后 strength 应约 0.5，got {s}"

    # 验证：60min 前 prime 的 strength 应 < 30min 前 prime 的 strength
    time_60min_ago = _past(minutes=60)
    conn.execute(
        "INSERT INTO priming_state (entity_name, project, primed_at, prime_strength) "
        "VALUES ('decay_older', 'test', ?, 1.0)",
        (time_60min_ago,)
    )
    conn.commit()
    actives2 = get_active_primes(conn, "test")
    s_30min = actives2.get("decay_test", 0.0)
    s_60min = actives2.get("decay_older", 0.0)
    # 60min 前的 prime 应比 30min 前的更弱（或都已失效）
    assert s_30min >= s_60min, (
        f"SP14: 30min prime({s_30min:.4f}) 应 >= 60min prime({s_60min:.4f})"
    )


# ══════════════════════════════════════════════════════════════════════
# 3. compute_priming_boost 测试
# ══════════════════════════════════════════════════════════════════════

def test_sp7_priming_boost_positive_when_match(conn):
    """有匹配 prime 的 chunk 返回 boost > 0。"""
    # Prime "redis"
    prime_entities(conn, ["redis"], "test", prime_strength=1.0)
    conn.commit()
    # 插入含 "redis" encode_context 的 chunk
    _insert_chunk_with_ctx(conn, "sp7_chunk", "cache,performance,redis,setup")
    boost = compute_priming_boost(conn, "sp7_chunk", "test")
    assert boost > 0.0, f"SP7: 有匹配 prime 时 boost 应 > 0，got {boost}"
    assert boost <= 0.30, f"SP7: boost 上限 0.30，got {boost}"


def test_sp8_no_priming_when_no_match(conn):
    """无匹配 prime 时返回 0.0。"""
    # Prime "machine_learning" — 与 chunk 的 encode_context 无关
    prime_entities(conn, ["machine_learning"], "test", prime_strength=1.0)
    conn.commit()
    _insert_chunk_with_ctx(conn, "sp8_chunk", "redis,cache,cluster")
    boost = compute_priming_boost(conn, "sp8_chunk", "test")
    assert boost == 0.0, f"SP8: 无匹配 prime 时 boost 应为 0，got {boost}"


def test_sp9_priming_boost_capped_at_max(conn):
    """boost 上限 0.30。"""
    # Prime 大量匹配 entity，strength=1.0
    prime_entities(conn, ["redis", "cache", "performance", "cluster"], "test", prime_strength=1.0)
    conn.commit()
    _insert_chunk_with_ctx(conn, "sp9_chunk", "cache,cluster,performance,redis,setup,tuning")
    boost = compute_priming_boost(conn, "sp9_chunk", "test")
    assert boost <= 0.30, f"SP9: boost 上限 0.30，got {boost}"


def test_sp10_no_encode_context_returns_zero(conn):
    """chunk 无 encode_context → boost = 0.0。"""
    prime_entities(conn, ["redis"], "test", prime_strength=1.0)
    conn.commit()
    # 插入无 encode_context 的 chunk
    now = _now()
    conn.execute(
        "INSERT INTO memory_chunks (id, project, chunk_type, content, summary, "
        "importance, retrievability, stability, created_at, updated_at) "
        "VALUES ('sp10_chunk', 'test', 'decision', 'redis content', 'redis summary', "
        "0.7, 0.5, 1.0, ?, ?)",
        (now, now)
    )
    conn.commit()
    boost = compute_priming_boost(conn, "sp10_chunk", "test")
    assert boost == 0.0, f"SP10: 无 encode_context 时 boost 应为 0，got {boost}"


# ══════════════════════════════════════════════════════════════════════
# 4. clear_stale_primes 测试
# ══════════════════════════════════════════════════════════════════════

def test_sp11_clear_stale_primes_removes_old(conn):
    """clear_stale_primes 清理过期条目。"""
    # 插入 3 小时前的 prime（已过期）
    old_time = _past(hours=3)
    conn.execute(
        "INSERT INTO priming_state (entity_name, project, primed_at, prime_strength) "
        "VALUES ('old_prime', 'test', ?, 1.0)",
        (old_time,)
    )
    conn.commit()
    deleted = clear_stale_primes(conn, project="test")
    conn.commit()
    assert deleted >= 1, f"SP11: 应删除至少 1 条过期记录，got {deleted}"
    remaining = conn.execute(
        "SELECT entity_name FROM priming_state WHERE entity_name='old_prime'"
    ).fetchone()
    assert remaining is None, "SP11: 过期 prime 应被清理"


def test_sp12_clear_stale_preserves_active(conn):
    """clear_stale_primes 不清理活跃条目。"""
    prime_entities(conn, ["active_entity"], "test", prime_strength=1.0)
    conn.commit()
    clear_stale_primes(conn, project="test")
    conn.commit()
    row = conn.execute(
        "SELECT entity_name FROM priming_state WHERE entity_name='active_entity'"
    ).fetchone()
    assert row is not None, "SP12: 活跃 prime 不应被清理"


# ══════════════════════════════════════════════════════════════════════
# 5. 集成测试
# ══════════════════════════════════════════════════════════════════════

def test_sp13_update_accessed_triggers_priming(conn):
    """update_accessed 后，chunk 的 encode_context entities 被 primed。"""
    # 插入有 encode_context 的 chunk
    _insert_chunk_with_ctx(conn, "sp13_chunk", "redis,cache,performance")
    # 触发 update_accessed
    update_accessed(conn, ["sp13_chunk"])
    conn.commit()
    # 检查 priming_state 是否有 redis/cache/performance
    actives = get_active_primes(conn, "test")
    # 至少 redis 或 cache 应该被 primed
    primed_set = set(actives.keys())
    expected = {"redis", "cache", "performance"}
    overlap = primed_set & expected
    assert len(overlap) >= 1, (
        f"SP13: update_accessed 后应至少 prime 1 个相关 entity，"
        f"got actives={primed_set}, expected overlap with {expected}"
    )


def test_sp15_empty_inputs_safe(conn):
    """所有函数的空/None 输入安全处理。"""
    # compute_priming_boost
    assert compute_priming_boost(conn, "", "test") == 0.0
    assert compute_priming_boost(conn, None, "test") == 0.0
    assert compute_priming_boost(conn, "chunk_x", "") == 0.0

    # get_active_primes
    assert get_active_primes(conn, "") == {}
    assert get_active_primes(conn, None) == {}

    # clear_stale_primes
    n = clear_stale_primes(conn, project=None)
    assert isinstance(n, int), "SP15: clear_stale_primes 应返回 int"
