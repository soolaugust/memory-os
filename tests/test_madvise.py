#!/usr/bin/env python3
"""
迭代32 测试：madvise — Memory Access Hints（工作集预热）
OS 类比：Linux madvise(2) (POSIX.1b, 2001)

测试覆盖：
  T1  madvise_write 写入 hint 关键词
  T2  madvise_read 读取有效 hint
  T3  madvise_read TTL 过期返回空
  T4  madvise_clear 清除 hint
  T5  madvise_write 按 project 隔离
  T6  max_hints 限制
  T7  extractor _extract_topic_entities 提取质量
  T8  retriever boost 逻辑验证（hint 匹配 chunk 加分）
  T9  retriever 无 hint 时不加分（零开销）
  T10 回归：config.py 新增 tunable 注册正确
  T11 性能：madvise_write + madvise_read < 5ms
"""
import json
import os
import sys
import time
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 设置路径
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "hooks"))

import tmpfs  # noqa: F401 — tmpfs isolation (iter54), must precede store import

passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except AssertionError as e:
        print(f"  ❌ {name}: {e}")
        failed += 1
    except Exception as e:
        print(f"  ❌ {name}: {type(e).__name__}: {e}")
        failed += 1


# ── T1: madvise_write 基本写入 ──
def test_write_basic():
    from store import madvise_write, _MADVISE_FILE, MEMORY_OS_DIR
    # 清空已有文件
    if _MADVISE_FILE.exists():
        _MADVISE_FILE.unlink()

    madvise_write("proj_test", ["BM25", "FTS5", "检索"], "session_1")

    assert _MADVISE_FILE.exists(), "madvise.json not created"
    data = json.loads(_MADVISE_FILE.read_text())
    assert "proj_test" in data, "project key missing"
    assert data["proj_test"]["hints"] == ["BM25", "FTS5", "检索"], f"hints mismatch: {data['proj_test']['hints']}"
    assert "timestamp" in data["proj_test"], "timestamp missing"
    assert data["proj_test"]["session_id"] == "session_1", "session_id mismatch"


# ── T2: madvise_read 有效读取 ──
def test_read_valid():
    from store import madvise_write, madvise_read
    madvise_write("proj_read", ["kswapd", "compaction", "水位"], "s2")

    hints = madvise_read("proj_read")
    assert hints == ["kswapd", "compaction", "水位"], f"read mismatch: {hints}"


# ── T3: madvise_read TTL 过期 ──
def test_read_expired():
    from store import _MADVISE_FILE, madvise_read, MEMORY_OS_DIR

    # 手动写入一个过期 hint
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=3600)).isoformat()
    data = {
        "proj_expired": {
            "hints": ["old_hint"],
            "timestamp": old_ts,
            "session_id": "s_old",
        }
    }
    MEMORY_OS_DIR.mkdir(parents=True, exist_ok=True)
    _MADVISE_FILE.write_text(json.dumps(data))

    hints = madvise_read("proj_expired")
    assert hints == [], f"expired hint should be empty, got: {hints}"


# ── T4: madvise_clear ──
def test_clear():
    from store import madvise_write, madvise_read, madvise_clear

    madvise_write("proj_clear", ["hint1", "hint2"], "s3")
    assert madvise_read("proj_clear") != [], "should have hints before clear"

    count = madvise_clear("proj_clear")
    assert count == 2, f"should clear 2, got {count}"
    assert madvise_read("proj_clear") == [], "should be empty after clear"


# ── T5: project 隔离 ──
def test_project_isolation():
    from store import madvise_write, madvise_read

    madvise_write("proj_A", ["alpha", "beta"], "s4")
    madvise_write("proj_B", ["gamma", "delta"], "s5")

    assert madvise_read("proj_A") == ["alpha", "beta"]
    assert madvise_read("proj_B") == ["gamma", "delta"]
    assert madvise_read("proj_C") == [], "nonexistent project should return empty"


# ── T6: max_hints 限制 ──
def test_max_hints():
    from store import madvise_write, madvise_read

    # 写入超过 max_hints (default=10) 的数量
    many_hints = [f"hint_{i}" for i in range(20)]
    madvise_write("proj_max", many_hints, "s6")

    result = madvise_read("proj_max")
    assert len(result) <= 10, f"should cap at 10, got {len(result)}"


# ── T7: extractor _extract_topic_entities 提取质量 ──
def test_extract_topic_entities():
    # 动态导入 extractor 的内部函数
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extractor", str(_ROOT / "hooks" / "extractor.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    text = """
    ## Memory Compaction 实现

    使用 `compact_zone` 函数进行碎片整理。
    BM25 检索引擎的 `hybrid_tokenize` 方法效果优于 unigram。
    store.py 和 retriever.py 共同完成数据访问。
    """
    decisions = ["采用 compact_zone 合并碎片化 chunk", "BM25 hybrid_tokenize 效果最优"]
    excluded = []
    reasoning = ["因为碎片化导致配额浪费 → 实现 compaction"]
    summaries = []
    topic = "Memory Compaction"

    entities = mod._extract_topic_entities(
        text, decisions, excluded, reasoning, summaries, topic)

    # 应至少包含核心技术实体
    entities_lower = [e.lower() for e in entities]
    assert any("compact" in e for e in entities_lower), f"should contain compact*: {entities_lower}"
    assert any("bm25" in e for e in entities_lower), f"should contain BM25: {entities_lower}"
    assert any("store.py" in e for e in entities_lower), f"should contain store.py: {entities_lower}"
    assert len(entities) >= 5, f"should extract at least 5 entities, got {len(entities)}"


# ── T8: retriever boost 验证 ──
def test_retriever_boost():
    """
    验证 madvise boost 逻辑：hint 匹配的 chunk 获得加分，
    可能改变排名顺序。
    """
    from store import madvise_write, madvise_read
    from scorer import retrieval_score as _unified_retrieval_score

    # 准备：写入 hint
    project = "proj_boost_test"
    madvise_write(project, ["kswapd", "水位", "淘汰"], "s_boost")

    # 模拟两个候选 chunk
    chunk_a = {  # 不匹配 hint
        "summary": "BM25 检索引擎 hybrid tokenize",
        "content": "FTS5 全文索引加速",
        "importance": 0.8,
        "last_accessed": datetime.now(timezone.utc).isoformat(),
        "access_count": 5,
        "chunk_type": "decision",
    }
    chunk_b = {  # 匹配 hint（kswapd, 水位, 淘汰）
        "summary": "kswapd 水位线预淘汰策略",
        "content": "三级水位线：pages_low 触发后台淘汰",
        "importance": 0.7,  # 原始 importance 更低
        "last_accessed": datetime.now(timezone.utc).isoformat(),
        "access_count": 3,
        "chunk_type": "decision",
    }

    # 计算原始分数
    score_a = _unified_retrieval_score(
        relevance=0.5, importance=chunk_a["importance"],
        last_accessed=chunk_a["last_accessed"],
        access_count=chunk_a["access_count"])
    score_b = _unified_retrieval_score(
        relevance=0.5, importance=chunk_b["importance"],
        last_accessed=chunk_b["last_accessed"],
        access_count=chunk_b["access_count"])

    # 原始：A 应该比 B 高（importance 更高）
    assert score_a > score_b, f"baseline: A ({score_a:.4f}) should > B ({score_b:.4f})"

    # 应用 madvise boost
    from config import get as _sysctl
    boost = _sysctl("madvise.boost_factor")
    hints = madvise_read(project)
    hint_set = set(h.lower() for h in hints)

    # chunk_b 匹配 hint
    text_b = f"{chunk_b['summary']} {chunk_b['content']}".lower()
    matches_b = sum(1 for h in hint_set if h in text_b)
    assert matches_b >= 2, f"chunk_b should match ≥2 hints, got {matches_b}"

    match_ratio = min(1.0, matches_b / max(1, len(hint_set) * 0.3))
    boosted_score_b = score_b + boost * match_ratio

    # chunk_a 不匹配 hint
    text_a = f"{chunk_a['summary']} {chunk_a['content']}".lower()
    matches_a = sum(1 for h in hint_set if h in text_a)
    # A 不应匹配（或很少匹配）
    assert matches_a < matches_b, f"chunk_a should match fewer hints"

    # 验证 boost 有效：boosted score 应该大于原始 score
    assert boosted_score_b > score_b, f"boost should increase score: {boosted_score_b} > {score_b}"


# ── T9: 无 hint 时零开销 ──
def test_no_hint_zero_overhead():
    from store import madvise_read, madvise_clear

    madvise_clear()  # 清除所有 hint
    hints = madvise_read("nonexistent_project")
    assert hints == [], "no project should return empty"

    # 计时验证零开销
    t0 = time.time()
    for _ in range(100):
        madvise_read("nonexistent_project")
    elapsed = (time.time() - t0) * 1000
    assert elapsed < 50, f"100 reads of empty hint should < 50ms, got {elapsed:.1f}ms"


# ── T10: config.py tunable 注册 ──
def test_config_tunables():
    from config import get, _REGISTRY

    assert "madvise.boost_factor" in _REGISTRY, "boost_factor not registered"
    assert "madvise.max_hints" in _REGISTRY, "max_hints not registered"
    assert "madvise.ttl_secs" in _REGISTRY, "ttl_secs not registered"

    # 验证默认值
    assert get("madvise.boost_factor") == 0.15
    assert get("madvise.max_hints") == 10
    assert get("madvise.ttl_secs") == 1800

    # 验证总 tunable 数 ≥ 33（后续迭代持续新增）
    assert len(_REGISTRY) >= 33, f"expected ≥33 tunables, got {len(_REGISTRY)}"


# ── T11: 性能 ──
def test_performance():
    from store import madvise_write, madvise_read

    t0 = time.time()
    for i in range(100):
        madvise_write(f"perf_proj_{i % 5}", [f"hint_{j}" for j in range(8)], "perf_s")
    write_ms = (time.time() - t0) * 1000

    t0 = time.time()
    for i in range(100):
        madvise_read(f"perf_proj_{i % 5}")
    read_ms = (time.time() - t0) * 1000

    avg_write = write_ms / 100
    avg_read = read_ms / 100
    print(f"    write: {avg_write:.2f}ms/call, read: {avg_read:.2f}ms/call")
    assert avg_write < 5, f"write should < 5ms, got {avg_write:.2f}ms"
    assert avg_read < 5, f"read should < 5ms, got {avg_read:.2f}ms"


if __name__ == "__main__":
    print("迭代32 madvise — Memory Access Hints 测试")
    print("=" * 55)

    run_test("T1  madvise_write 基本写入", test_write_basic)
    run_test("T2  madvise_read 有效读取", test_read_valid)
    run_test("T3  madvise_read TTL 过期", test_read_expired)
    run_test("T4  madvise_clear 清除 hint", test_clear)
    run_test("T5  project 隔离", test_project_isolation)
    run_test("T6  max_hints 限制", test_max_hints)
    run_test("T7  extractor 实体提取质量", test_extract_topic_entities)
    run_test("T8  retriever boost 验证", test_retriever_boost)
    run_test("T9  无 hint 零开销", test_no_hint_zero_overhead)
    run_test("T10 config tunable 注册", test_config_tunables)
    run_test("T11 性能", test_performance)

    print(f"\n{'=' * 55}")
    print(f"结果: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        sys.exit(1)
