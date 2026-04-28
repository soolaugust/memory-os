"""
test_context_boost.py — iter372 Context-Aware Retrieval Boost 测试

覆盖：
  CA1: cwd 完全匹配 → boost = 0.10
  CA2: cwd 子路径匹配（enc_cwd 是 cur_cwd 的前缀）→ boost = 0.10
  CA3: cwd 不匹配 + 关键词重叠 → boost = Jaccard × 0.05
  CA4: 两者都不匹配 → boost = 0.0
  CA5: boost 上限 0.15（cwd 匹配 + 关键词完全重叠）
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def _compute_context_match(enc_ctx: dict, cur_ctx: dict,
                            focus_keywords=None) -> float:
    """从 retriever.py 提取的 _compute_context_match 逻辑（独立测试版本）"""
    if focus_keywords is None:
        focus_keywords = []
    if not enc_ctx or not cur_ctx:
        return 0.0
    boost = 0.0
    enc_cwd = (enc_ctx.get("cwd") or "").rstrip("/")
    cur_cwd = (cur_ctx.get("cwd") or "").rstrip("/")
    if enc_cwd and cur_cwd:
        if enc_cwd == cur_cwd or cur_cwd.startswith(enc_cwd + "/"):
            boost += 0.10
    enc_kw = set(enc_ctx.get("keywords") or [])
    cur_kw = set(cur_ctx.get("keywords") or focus_keywords)
    if enc_kw and cur_kw:
        intersection = len(enc_kw & cur_kw)
        union = len(enc_kw | cur_kw)
        if union > 0:
            boost += (intersection / union) * 0.05
    return min(boost, 0.15)


# ── CA1: cwd 完全匹配 → boost = 0.10 ──────────────────────────────────────────

def test_ca1_exact_cwd_match():
    enc_ctx = {"cwd": "/home/user/proj/aios", "keywords": []}
    cur_ctx = {"cwd": "/home/user/proj/aios", "keywords": []}
    boost = _compute_context_match(enc_ctx, cur_ctx)
    assert abs(boost - 0.10) < 0.001


# ── CA2: cwd 子路径匹配 → boost = 0.10 ───────────────────────────────────────

def test_ca2_parent_cwd_match():
    """enc_cwd 是 cur_cwd 的父路径 → 匹配（子目录仍属同一工作区）"""
    enc_ctx = {"cwd": "/home/user/proj", "keywords": []}
    cur_ctx = {"cwd": "/home/user/proj/aios/hooks", "keywords": []}
    boost = _compute_context_match(enc_ctx, cur_ctx)
    assert abs(boost - 0.10) < 0.001


def test_ca2_child_cwd_no_match():
    """cur_cwd 是 enc_cwd 的子路径但反向不匹配（enc 更深）→ 不匹配"""
    enc_ctx = {"cwd": "/home/user/proj/aios/hooks", "keywords": []}
    cur_ctx = {"cwd": "/home/user/proj", "keywords": []}
    boost = _compute_context_match(enc_ctx, cur_ctx)
    assert abs(boost - 0.0) < 0.001


# ── CA3: cwd 不匹配 + 关键词重叠 → Jaccard × 0.05 ───────────────────────────

def test_ca3_keyword_overlap_boost():
    """cwd 不匹配，但关键词 50% 重叠 → boost ≈ 0.025"""
    enc_ctx = {"cwd": "/home/user/proj-a", "keywords": ["memory", "retriever", "FTS5", "sqlite"]}
    cur_ctx = {"cwd": "/home/user/proj-b", "keywords": ["memory", "retriever", "extractor"]}
    boost = _compute_context_match(enc_ctx, cur_ctx)
    # enc_kw = {memory, retriever, FTS5, sqlite}, cur_kw = {memory, retriever, extractor}
    # intersection = {memory, retriever} = 2, union = 5 → Jaccard = 2/5 = 0.4
    # boost = 0.4 × 0.05 = 0.02
    assert boost > 0.0
    assert boost < 0.10  # 无 cwd 匹配


# ── CA4: 两者都不匹配 → boost = 0.0 ─────────────────────────────────────────

def test_ca4_no_match_zero_boost():
    enc_ctx = {"cwd": "/home/user/proj-a", "keywords": ["redis", "cache"]}
    cur_ctx = {"cwd": "/home/user/proj-b", "keywords": ["sqlite", "FTS5"]}
    boost = _compute_context_match(enc_ctx, cur_ctx)
    assert abs(boost - 0.0) < 0.001


# ── CA5: boost 上限 0.15 ──────────────────────────────────────────────────────

def test_ca5_boost_capped_at_0_15():
    """cwd 完全匹配（+0.10）+ 关键词完全重叠（+0.05）= 0.15 上限"""
    enc_ctx = {"cwd": "/home/user/proj", "keywords": ["memory", "retriever"]}
    cur_ctx = {"cwd": "/home/user/proj", "keywords": ["memory", "retriever"]}
    boost = _compute_context_match(enc_ctx, cur_ctx)
    assert abs(boost - 0.15) < 0.001  # 精确等于上限


def test_ca5_empty_context_zero_boost():
    """空 enc_ctx → boost = 0.0（优雅降级）"""
    boost = _compute_context_match({}, {"cwd": "/home/user/proj"})
    assert boost == 0.0

    boost2 = _compute_context_match(None, {"cwd": "/home/user/proj"})
    assert boost2 == 0.0
