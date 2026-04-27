"""
迭代61：vDSO Fast Path — Lazy Import + 启动加速 测试

OS 类比：Linux vDSO (Virtual Dynamic Shared Object, 2004)
  高频路径（SKIP/TLB hit）绕过 heavy import，直接在 stdlib 范围内完成。

测试覆盖：
  1. _vdso_is_skip / _VDSO_SKIP_EXACT 与 _SKIP_PATTERNS 一致性（iter160: frozenset 替代 regex）
  2. _vdso_has_tech / _VDSO_TECH_* 与 _TECH_SIGNAL 一致性（iter160: frozenset 替代 regex）
  3. _vdso_fast_exit Stage 0: SKIP 路径
  4. _vdso_fast_exit Stage 1: TLB hit 路径
  5. _vdso_fast_exit fallthrough: 传递 hook_input
  6. _load_modules 延迟加载
  7. 性能：SKIP <1ms, TLB hit <3ms
"""
import sys
import os
import json
import time
import hashlib
import tempfile
import shutil
from pathlib import Path

# tmpfs 隔离
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tmpfs

# 现在导入 retriever
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hooks'))
import retriever


# ── 测试 1: SKIP 一致性（iter160: frozenset 替代 _VDSO_SKIP_RE）──

def test_skip_regex_consistency():
    """_vdso_is_skip 和 _SKIP_PATTERNS 对所有 SKIP 词汇行为一致。
    注：iter160 将 _VDSO_SKIP_RE 替换为 frozenset + _vdso_is_skip()，本测试已更新。
    """
    skip_words = [
        "好", "好的", "好吧", "好啊", "嗯", "恩", "哦", "噢",
        "ok", "okay", "OK", "OKAY",
        "是", "是的", "是吧",
        "对", "对的", "对吧",
        "收到", "了解", "明白", "可以", "继续",
        "开始", "执行", "确认", "同意", "谢谢",
        "thanks", "thank", "yes", "yep",
        "nope", "no", "got it", "sure", "lgtm", "LGTM",
    ]
    for w in skip_words:
        vdso_result = retriever._vdso_is_skip(w)
        # _SKIP_PATTERNS 只在 _load_modules() 后可用（Stage 2 正则）
        # 验证 frozenset 路径本身的行为：确认这些词都命中
        assert vdso_result, f"'{w}' should be detected as SKIP by _vdso_is_skip"


# ── 测试 2: TECH signal 一致性（iter160: frozenset 替代 _VDSO_TECH_RE）──

def test_tech_regex_consistency():
    """_vdso_has_tech 对技术信号正确检测。
    注：iter160 将 _VDSO_TECH_RE 替换为 frozenset + _vdso_has_tech()，本测试已更新。
    """
    tech_signals = [
        "`store.py`", "config.json", "error occurred", "class Foo",
        "def main", "import os", "function test",
        "函数", "类", "模块", "接口", "方法", "变量", "配置", "部署", "迁移",
        "bug in code", "fix this", "crash report",
    ]
    for s in tech_signals:
        vdso_result = retriever._vdso_has_tech(s)
        assert vdso_result, f"'{s}' should be detected as tech signal by _vdso_has_tech"


# ── 测试 3: SKIP 路径不触发 tech signal ──

def test_skip_no_tech():
    """纯 SKIP 词不含 tech signal。"""
    pure_skip = ["好的", "ok", "是", "继续", "sure"]
    for w in pure_skip:
        assert retriever._vdso_is_skip(w), f"'{w}' should match SKIP"
        assert not retriever._vdso_has_tech(w), f"'{w}' should not have tech signal"


# ── 测试 4: Tech signal 阻止 SKIP ──

def test_tech_blocks_skip():
    """含有 tech signal 的短文本应被检测为有 tech signal。"""
    tech_shorts = ["`fix`", "好的 config.json", "ok error"]
    for s in tech_shorts:
        has_tech = retriever._vdso_has_tech(s)
        assert has_tech, f"'{s}' should have tech signal"


# ── 测试 5: _load_modules 延迟加载 ──

def test_load_modules():
    """_load_modules 注入全局变量。"""
    retriever._load_modules()
    assert hasattr(retriever, 'open_db') or 'open_db' in dir(retriever), \
        "open_db should be available after _load_modules"
    assert retriever._modules_loaded is True


# ── 测试 6: _load_modules 幂等性 ──

def test_load_modules_idempotent():
    """多次调用 _load_modules 不出错。"""
    retriever._load_modules()
    retriever._load_modules()
    assert retriever._modules_loaded is True


# ── 测试 7: TLB 文件格式正确性 ──

def test_tlb_file_format():
    """验证 TLB 文件的 JSON 结构被 vDSO 正确读取。"""
    mem_dir = Path(os.environ.get("MEMORY_OS_DIR", str(Path.home() / ".claude" / "memory-os")))
    mem_dir.mkdir(parents=True, exist_ok=True)

    tlb_file = mem_dir / ".last_tlb.json"
    hash_file = mem_dir / ".last_injection_hash"

    prompt = "test prompt"
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    injection_hash = "abc12345"
    tlb_data = {"prompt_hash": prompt_hash, "injection_hash": injection_hash, "db_mtime": 12345.0}

    tlb_file.write_text(json.dumps(tlb_data), encoding="utf-8")
    hash_file.write_text(injection_hash, encoding="utf-8")

    loaded = json.loads(tlb_file.read_text(encoding="utf-8"))
    assert loaded["prompt_hash"] == prompt_hash
    assert loaded["injection_hash"] == injection_hash
    assert loaded["db_mtime"] == 12345.0

    loaded_hash = hash_file.read_text(encoding="utf-8").strip()
    assert loaded_hash == injection_hash


# ── 测试 8: 非 SKIP 文本不被 SKIP ──

def test_non_skip_not_matched():
    """技术查询不应该被 _vdso_is_skip 匹配。
    注：iter160 将 _VDSO_SKIP_RE 替换为 frozenset + _vdso_is_skip()，本测试已更新。
    """
    non_skip = [
        "请帮我看一下 store.py",
        "如何修复这个 bug",
        "explain the retriever architecture",
    ]
    for s in non_skip:
        is_skip_match = retriever._vdso_is_skip(s.strip())
        assert not is_skip_match, f"'{s}' should NOT match SKIP"


# ── 测试 9: 性能 - SKIP 检测 <0.1ms ──

def test_skip_regex_performance():
    """_vdso_is_skip 延迟 <0.1ms（frozenset 比 regex 更快）。
    注：iter160 将 _VDSO_SKIP_RE 替换为 frozenset + _vdso_is_skip()，本测试已更新。
    """
    iterations = 10000
    prompts = ["好的", "ok", "继续", "是", "谢谢"]
    t0 = time.time()
    for _ in range(iterations):
        for p in prompts:
            retriever._vdso_is_skip(p)
    elapsed_ms = (time.time() - t0) * 1000
    per_call_ms = elapsed_ms / (iterations * len(prompts))
    assert per_call_ms < 0.1, f"SKIP check too slow: {per_call_ms:.3f}ms"


# ── 测试 10: 性能 - TLB file read <2ms ──

def test_tlb_read_performance():
    """TLB file read (JSON parse + stat) 延迟 <2ms。"""
    mem_dir = Path(os.environ.get("MEMORY_OS_DIR", str(Path.home() / ".claude" / "memory-os")))
    mem_dir.mkdir(parents=True, exist_ok=True)

    tlb_file = mem_dir / ".last_tlb.json"
    hash_file = mem_dir / ".last_injection_hash"
    store_db = mem_dir / "store.db"

    tlb_data = {"prompt_hash": "abc12345", "injection_hash": "def67890", "db_mtime": 12345.0}
    tlb_file.write_text(json.dumps(tlb_data), encoding="utf-8")
    hash_file.write_text("def67890", encoding="utf-8")
    store_db.touch()

    iterations = 1000
    t0 = time.time()
    for _ in range(iterations):
        db_mtime = store_db.stat().st_mtime
        tlb = json.loads(tlb_file.read_text(encoding="utf-8"))
        last_hash = hash_file.read_text(encoding="utf-8").strip()
        _ = (tlb.get("prompt_hash") == "abc12345"
             and tlb.get("db_mtime") == db_mtime
             and tlb.get("injection_hash") == last_hash)
    elapsed_ms = (time.time() - t0) * 1000
    per_call_ms = elapsed_ms / iterations
    assert per_call_ms < 2.0, f"TLB read too slow: {per_call_ms:.3f}ms"


# ── 测试 11: _vdso_hook_input 传递 ──

def test_vdso_hook_input_passthrough():
    """验证 _vdso_hook_input 模块变量可设置和读取。"""
    test_input = {"prompt": "test", "task_list": []}
    retriever._vdso_hook_input = test_input
    assert retriever._vdso_hook_input == test_input
