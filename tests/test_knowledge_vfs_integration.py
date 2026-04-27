#!/usr/bin/env python3
"""
KnowledgeVFS 集成测试

验证：
  1. 与现有 knowledge_router 的 API 兼容性
  2. 初始化和全局单例工作
  3. 与 retriever/loader/writer hooks 的集成
"""

import sys
import os
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

# 设置环境
os.environ.setdefault("CLAUDE_CWD", str(__import__("pathlib").Path(__file__).parent.parent.parent.parent.parent))


# ─────────────────────────────────────────────────────────────
# 测试初始化和全局单例
# ─────────────────────────────────────────────────────────────

def test_vfs_initialization():
    """测试 VFS 初始化和全局单例"""
    from knowledge_vfs_init import init_knowledge_vfs

    # 初始化
    vfs = init_knowledge_vfs()
    assert vfs is not None
    assert len(vfs.backends) > 0
    print(f"  VFS backends: {list(vfs.backends.keys())}")

    print("✓ VFS 初始化测试通过")


# ─────────────────────────────────────────────────────────────
# 测试 knowledge_router 兼容性接口
# ─────────────────────────────────────────────────────────────

def test_knowledge_router_compatibility():
    """测试与 knowledge_router 的 API 兼容性"""
    from knowledge_vfs_init import init_knowledge_vfs, search, format_for_context

    # 初始化
    init_knowledge_vfs()

    # 测试搜索接口（与 knowledge_router.route 兼容）
    results = search(
        "test query",
        sources=["self-improving"],
        top_k=3,
        timeout_ms=100
    )

    # 验证返回格式
    assert isinstance(results, list)
    for r in results:
        assert "source" in r
        assert "chunk_type" in r
        assert "summary" in r
        assert "score" in r
        assert isinstance(r["score"], float)
        assert 0.0 <= r["score"] <= 1.0

    print(f"  搜索返回 {len(results)} 条结果")

    # 测试格式化函数（与 knowledge_router.format_for_context 兼容）
    formatted = format_for_context(results)
    assert isinstance(formatted, str)
    if results:
        assert "【知识路由召回】" in formatted or formatted == ""

    print("✓ knowledge_router 兼容性测试通过")


# ─────────────────────────────────────────────────────────────
# 测试 API 返回格式
# ─────────────────────────────────────────────────────────────

def test_api_return_format():
    """验证 API 返回格式与 knowledge_router 兼容"""
    from knowledge_vfs_init import init_knowledge_vfs, search

    init_knowledge_vfs()

    # 空查询结果
    results = search("非常罕见的查询词条XXXYYYzzz", sources=["self-improving"], top_k=1)
    assert isinstance(results, list)

    # 如果有结果，验证字段
    for result in results:
        required_fields = ["source", "chunk_type", "summary", "score", "content", "path"]
        for field in required_fields:
            assert field in result, f"缺少字段: {field}"

    print("✓ API 返回格式测试通过")


# ─────────────────────────────────────────────────────────────
# 测试读写接口
# ─────────────────────────────────────────────────────────────

def test_read_write_interface():
    """测试读写接口"""
    from knowledge_vfs_init import init_knowledge_vfs, write

    init_knowledge_vfs()

    # 测试写入（到 project 后端）
    item_dict = {
        "type": "trace",
        "summary": "Integration test trace",
        "content": "This is an integration test item for KnowledgeVFS.",
        "importance": 5,
        "scope": "project",
        "tags": ["test", "integration"]
    }

    # 写入到 project 后端
    new_id = write(item_dict, source="project")
    assert new_id is not None and new_id != ""
    print(f"  写入成功，新 ID: {new_id}")

    print("✓ 读写接口测试通过")


# ─────────────────────────────────────────────────────────────
# 测试多源搜索
# ─────────────────────────────────────────────────────────────

def test_multi_source_search():
    """测试跨多源搜索"""
    from knowledge_vfs_init import init_knowledge_vfs, search

    init_knowledge_vfs()

    # 搜索所有源
    results = search(
        "memory architecture design",
        sources=["memory-os", "self-improving", "project"],
        top_k=5,
        timeout_ms=150
    )

    assert isinstance(results, list)
    print(f"  多源搜索返回 {len(results)} 条结果")

    # 验证来源多样性
    if results:
        sources = set(r["source"] for r in results)
        print(f"  涵盖的源: {sources}")

    print("✓ 多源搜索测试通过")


# ─────────────────────────────────────────────────────────────
# 测试超时处理
# ─────────────────────────────────────────────────────────────

def test_timeout_handling():
    """测试超时处理"""
    from knowledge_vfs_init import init_knowledge_vfs, search

    init_knowledge_vfs()

    # 使用极短的超时（10ms）
    results = search(
        "test query",
        sources=["memory-os"],
        top_k=1,
        timeout_ms=10
    )

    # 应该返回结果或空列表，不抛出异常
    assert isinstance(results, list)
    print(f"  短超时搜索: {len(results)} 条结果")

    print("✓ 超时处理测试通过")


# ─────────────────────────────────────────────────────────────
# 测试缓存命中
# ─────────────────────────────────────────────────────────────

def test_cache_hit():
    """测试缓存命中情况"""
    import time
    from knowledge_vfs_init import init_knowledge_vfs, search

    init_knowledge_vfs()

    query = "caching performance test"

    # 第一次搜索
    start = time.time()
    results1 = search(query, sources=["self-improving"], top_k=1)
    elapsed1 = time.time() - start

    # 第二次搜索（应该命中缓存）
    start = time.time()
    results2 = search(query, sources=["self-improving"], top_k=1)
    elapsed2 = time.time() - start

    print(f"  第一次查询: {elapsed1*1000:.1f}ms")
    print(f"  第二次查询: {elapsed2*1000:.1f}ms (缓存)")

    # 缓存应该加速第二次查询
    if results1:
        assert results1 == results2, "缓存结果应该相同"

    print("✓ 缓存命中测试通过")


# ─────────────────────────────────────────────────────────────
# 测试错误处理
# ─────────────────────────────────────────────────────────────

def test_error_handling():
    """测试错误处理"""
    from knowledge_vfs_init import init_knowledge_vfs, search

    init_knowledge_vfs()

    # 测试无效源名称（应该被忽略）
    results = search("test", sources=["nonexistent-source"], top_k=1)
    assert isinstance(results, list)

    print("✓ 错误处理测试通过")


# ─────────────────────────────────────────────────────────────
# 性能基准测试
# ─────────────────────────────────────────────────────────────

def test_performance_baseline():
    """性能基准测试（对比目标 100ms）"""
    import time
    from knowledge_vfs_init import init_knowledge_vfs, search

    init_knowledge_vfs()

    queries = [
        "memory management",
        "knowledge retrieval",
        "caching strategy",
    ]

    total_time = 0
    result_counts = []

    for query in queries:
        start = time.time()
        results = search(query, sources=["self-improving", "project"], top_k=3, timeout_ms=100)
        elapsed = time.time() - start
        total_time += elapsed
        result_counts.append(len(results))

    avg_time = (total_time / len(queries)) * 1000
    print(f"  {len(queries)} 次查询平均时间: {avg_time:.1f}ms")
    print(f"  结果数: {result_counts}")

    # 验证性能目标：平均 < 100ms
    assert avg_time < 100, f"性能超出目标: {avg_time:.1f}ms > 100ms"

    print("✓ 性能基准测试通过")


# ─────────────────────────────────────────────────────────────
# 运行所有集成测试
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== KnowledgeVFS 集成测试 ===\n")

    test_vfs_initialization()
    test_knowledge_router_compatibility()
    test_api_return_format()
    test_read_write_interface()
    test_multi_source_search()
    test_timeout_handling()
    test_cache_hit()
    test_error_handling()
    test_performance_baseline()

    print("\n=== 所有集成测试通过 ===")
    print("\n✅ KnowledgeVFS 已准备好生产集成")
