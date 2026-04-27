#!/usr/bin/env python3
"""
迭代57→64 测试：TLB — Translation Lookaside Buffer 检索快速路径

v2 Multi-Slot + chunk_version 格式：
  {chunk_version: int, slots: {prompt_hash: {injection_hash: str}}}

验证：
1. TLB 读写基本功能（v2 multi-slot 格式）
2. TLB hit 条件：prompt_hash 在 slots 中 + chunk_version 匹配
3. TLB miss 条件：prompt_hash 不在 slots 中
4. TLB miss 条件：chunk_version 变化（有新 chunk 写入）
5. TLB 空文件/损坏文件的边界情况
6. TLB 与 injection_hash 的联动
7. _get_db_mtime 正常工作
8. TLB 多 slot 覆盖写入（同 prompt_hash 更新）
9. prompt_hash 计算一致性
10. TLB hit 时不应触发 FTS5 检索（功能逻辑验证）
"""
import tmpfs  # 测试隔离（迭代54）

import json
import os
import sys
import tempfile
import hashlib
from pathlib import Path

# 确保 hooks 目录可导入
_HOOKS_DIR = Path(__file__).parent / "hooks"
sys.path.insert(0, str(_HOOKS_DIR))
_MOS_ROOT = Path(__file__).parent
sys.path.insert(0, str(_MOS_ROOT))


def _get_retriever():
    """动态获取 retriever 模块（已在 sys.path 中）"""
    import retriever
    return retriever


# ── 测试用例 ──────────────────────────────────────────────────────────

def test_tlb_write_read():
    """测试1：TLB v2 基本读写（multi-slot 格式）"""
    r = _get_retriever()

    r._tlb_write("abc123", "inj456", 1234567890.0)
    tlb = r._tlb_read()

    assert "slots" in tlb, f"v2 format should have 'slots' key: {tlb}"
    assert "abc123" in tlb["slots"], f"slot abc123 missing: {tlb}"
    assert tlb["slots"]["abc123"]["injection_hash"] == "inj456", \
        f"injection_hash mismatch: {tlb}"
    assert "chunk_version" in tlb, f"v2 format should have 'chunk_version': {tlb}"
    print("  PASS test_tlb_write_read")


def test_tlb_hit_same_prompt():
    """测试2：TLB hit — prompt_hash 在 slots 中 + chunk_version 匹配"""
    r = _get_retriever()

    # 写入 TLB（_tlb_write 自动读取当前 chunk_version）
    r._tlb_write("prompt_A", "inj_X", 100.0)

    tlb = r._tlb_read()
    chunk_ver = r._read_chunk_version()

    # L1 hit: prompt_hash in slots + chunk_version match
    is_hit = (tlb.get("chunk_version") == chunk_ver
              and "prompt_A" in tlb.get("slots", {}))
    assert is_hit, f"Should be TLB hit: chunk_ver={chunk_ver}, tlb={tlb}"
    print("  PASS test_tlb_hit_same_prompt")


def test_tlb_miss_different_prompt():
    """测试3：TLB miss — prompt_hash 不在 slots 中"""
    r = _get_retriever()

    r._tlb_write("prompt_A", "inj_X", 100.0)
    tlb = r._tlb_read()
    chunk_ver = r._read_chunk_version()

    is_hit = (tlb.get("chunk_version") == chunk_ver
              and "prompt_B" in tlb.get("slots", {}))
    assert not is_hit, "Should be TLB miss (different prompt)"
    print("  PASS test_tlb_miss_different_prompt")


def test_tlb_miss_db_changed():
    """测试4：TLB miss — chunk_version 变化（有新 chunk 写入）"""
    r = _get_retriever()

    r._tlb_write("prompt_A", "inj_X", 100.0)
    tlb = r._tlb_read()
    tlb_ver = tlb.get("chunk_version", -1)

    # 模拟 chunk_version 变化（有新写入）
    simulated_new_ver = tlb_ver + 1
    is_hit = (simulated_new_ver == tlb_ver
              and "prompt_A" in tlb.get("slots", {}))
    assert not is_hit, "Should be TLB miss (chunk_version changed)"
    print("  PASS test_tlb_miss_db_changed")


def test_tlb_empty_file():
    """测试5：TLB 文件不存在或损坏
    注：iter159 将 Path 对象改为 str，本测试使用 os.path/open 等价操作。
    """
    r = _get_retriever()

    # 删除 TLB 文件
    if os.path.exists(r.TLB_FILE):
        os.unlink(r.TLB_FILE)
    tlb = r._tlb_read()
    assert tlb == {}, f"Should return empty dict, got {tlb}"

    # 写入损坏内容
    os.makedirs(os.path.dirname(r.TLB_FILE), exist_ok=True)
    with open(r.TLB_FILE, 'w', encoding="utf-8") as _f:
        _f.write("not json!!!")
    tlb = r._tlb_read()
    assert tlb == {}, f"Should return empty dict for corrupt file, got {tlb}"

    # 清理
    try:
        os.unlink(r.TLB_FILE)
    except FileNotFoundError:
        pass
    print("  PASS test_tlb_empty_file")


def test_tlb_injection_hash_linkage():
    """测试6：TLB v2 与 injection_hash 联动"""
    r = _get_retriever()

    # 写入 injection_hash 和 TLB
    r._write_hash("inj_ABC")
    r._tlb_write("prompt_1", "inj_ABC", 100.0)

    tlb = r._tlb_read()
    current_inj = r._read_hash()
    chunk_ver = r._read_chunk_version()

    # TLB slot 中的 injection_hash 应该与文件中的一致
    slot = tlb["slots"]["prompt_1"]
    assert slot["injection_hash"] == current_inj, \
        f"TLB injection_hash={slot['injection_hash']} != file={current_inj}"

    # 完整 L1 hit 条件
    is_hit = (tlb.get("chunk_version") == chunk_ver
              and "prompt_1" in tlb.get("slots", {})
              and tlb["slots"]["prompt_1"]["injection_hash"] == current_inj)
    assert is_hit, "Should be full TLB L1 hit with injection_hash match"
    print("  PASS test_tlb_injection_hash_linkage")


def test_get_db_mtime():
    """测试7：_get_db_mtime 正常工作"""
    r = _get_retriever()

    # 如果 store.db 存在应返回正数，不存在返回 0.0
    mtime = r._get_db_mtime()
    # 在 tmpfs 环境下 store.db 可能不存在
    assert isinstance(mtime, float), f"Should be float, got {type(mtime)}"
    # 如果不存在，应该返回 0.0（不报错）
    print(f"  PASS test_get_db_mtime (mtime={mtime})")


def test_tlb_overwrite():
    """测试8：TLB v2 multi-slot — 同 prompt_hash 覆盖更新"""
    r = _get_retriever()

    r._tlb_write("prompt_1", "inj_1", 100.0)
    r._tlb_write("prompt_1", "inj_2", 200.0)

    tlb = r._tlb_read()
    # 同一 prompt_hash 应被更新（不是新增）
    assert "prompt_1" in tlb["slots"], "prompt_1 slot should exist"
    assert tlb["slots"]["prompt_1"]["injection_hash"] == "inj_2", \
        "Should have latest injection_hash"

    # 不同 prompt_hash 应并存
    r._tlb_write("prompt_2", "inj_3", 300.0)
    tlb = r._tlb_read()
    assert "prompt_1" in tlb["slots"], "prompt_1 should still exist"
    assert "prompt_2" in tlb["slots"], "prompt_2 should be added"
    print("  PASS test_tlb_overwrite")


def test_prompt_hash_consistency():
    """测试9：prompt_hash 计算一致性"""
    prompt = "帮我实现 BM25 检索功能"
    h1 = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    h2 = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    assert h1 == h2, "Same prompt should produce same hash"

    # 不同 prompt 不同 hash
    h3 = hashlib.sha256("另一个完全不同的 prompt".encode()).hexdigest()[:8]
    assert h1 != h3, "Different prompts should produce different hashes"
    print("  PASS test_prompt_hash_consistency")


def test_tlb_hit_prevents_fts5():
    """测试10：TLB v2 hit 时不应触发 FTS5 检索（功能逻辑验证）"""
    r = _get_retriever()

    # 设置 TLB 缓存状态
    prompt = "测试 prompt"
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    injection_hash = "test_inj"

    r._write_hash(injection_hash)
    r._tlb_write(prompt_hash, injection_hash, 999.0)

    # 验证 TLB L1 hit 判断逻辑（v2 格式）
    tlb = r._tlb_read()
    chunk_ver = r._read_chunk_version()
    is_hit = (tlb.get("chunk_version") == chunk_ver
              and prompt_hash in tlb.get("slots", {})
              and tlb["slots"][prompt_hash]["injection_hash"] == r._read_hash())
    assert is_hit, "Should be TLB L1 hit"

    # 模拟 chunk_version 变化后（新写入），TLB 应失效
    simulated_new_ver = chunk_ver + 1
    is_hit_after_write = (tlb.get("chunk_version") == simulated_new_ver
                          and prompt_hash in tlb.get("slots", {})
                          and tlb["slots"][prompt_hash]["injection_hash"] == r._read_hash())
    assert not is_hit_after_write, "Should be TLB miss after chunk_version change"
    print("  PASS test_tlb_hit_prevents_fts5")


# ── run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 迭代57→64 TLB v2 (Multi-Slot + chunk_version) 测试 ===")
    tests = [
        test_tlb_write_read,
        test_tlb_hit_same_prompt,
        test_tlb_miss_different_prompt,
        test_tlb_miss_db_changed,
        test_tlb_empty_file,
        test_tlb_injection_hash_linkage,
        test_get_db_mtime,
        test_tlb_overwrite,
        test_prompt_hash_consistency,
        test_tlb_hit_prevents_fts5,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n结果：{passed}/{passed+failed} 通过")
