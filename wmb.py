"""
wmb.py — Working Memory Budget Manager
迭代316：工作记忆预算管理

认知科学基础：Cowan's Magic Number 4 (Cowan 2001)
  工作记忆同时能处理约4个chunk（4±1）。
  注入越多不等于越好——过多chunk会导致信息干扰，降低实际利用率。

OS 类比：Linux mm/readahead.c — 预读窗口管理，不是越大越好，
  太大的预读窗口会污染 page cache（缓存颠簸），
  最优窗口由访问模式动态调整。

分层映射：
  active     ↔  L1 cache（直接命中，立即使用）高relevance+高情境匹配
  background ↔  L2 cache（间接支撑，按需访问）间接相关，为active提供上下文
  dormant    ↔  L3/DRAM（标记但不加载，page fault 时再取）

依赖：仅 Python 标准库（无项目内部依赖，避免循环导入）
"""

from __future__ import annotations

__all__ = ["tier_chunks", "apply_wmb_budget", "wmb_stats"]


def tier_chunks(
    scored_chunks: list,
    top_k: int = 8,
    active_threshold: float = 0.65,
    background_threshold: float = 0.35,
    active_limit: int = 3,
    background_limit: int = 5,
) -> dict:
    """
    迭代316: Working Memory Budget — 将 scored_chunks 分层。

    输入：[(chunk_dict, score), ...] 任意顺序均可（内部会按score降序排列）
         也接受 [(score, chunk_dict), ...] 格式（自动检测）
    输出：{"active": [...], "background": [...], "dormant": [...]}
         每层中的元素均为 chunk_dict（不含分数）

    分层规则：
      - 取前 top_k 个候选（按分数降序）
      - scores 归一化到 [0,1]（相对归一化，最高分=1.0）
      - normalized_score >= active_threshold    → active（上限 active_limit=3 个）
      - normalized_score >= background_threshold → background（上限 background_limit=5 个）
      - 其余 → dormant（标记但不注入）

    Cowan 2001: 工作记忆槽位约4±1，active+background 总上限 = 8
    """
    if not scored_chunks:
        return {"active": [], "background": [], "dormant": []}

    # 自动检测输入格式：(chunk, score) 或 (score, chunk)
    # 约定：如果第一个元素的 [0] 是 dict，则格式为 (chunk, score)
    first = scored_chunks[0]
    if isinstance(first[0], dict):
        # 格式：(chunk_dict, score)
        pairs = [(float(s), c) for c, s in scored_chunks]
    else:
        # 格式：(score, chunk_dict)
        pairs = [(float(s), c) for s, c in scored_chunks]

    # 按分数降序排列，取前 top_k
    pairs.sort(key=lambda x: x[0], reverse=True)
    candidates = pairs[:top_k]

    if not candidates:
        return {"active": [], "background": [], "dormant": []}

    # 归一化到 [0, 1]（相对于当前候选集最高分）
    max_score = candidates[0][0]
    if max_score <= 0:
        # 全部分数非正：全部归 dormant
        return {
            "active": [],
            "background": [],
            "dormant": [c for _, c in candidates],
        }

    active: list = []
    background: list = []
    dormant: list = []

    for score, chunk in candidates:
        norm = score / max_score  # 归一化到 [0, 1]

        if norm >= active_threshold and len(active) < active_limit:
            active.append(chunk)
        elif norm >= background_threshold and len(background) < background_limit:
            # active 已满但分数仍高 → 溢出进 background
            background.append(chunk)
        else:
            dormant.append(chunk)

    return {"active": active, "background": background, "dormant": dormant}


def apply_wmb_budget(
    scored_chunks: list,
    top_k: int = 8,
    active_threshold: float = 0.65,
    background_threshold: float = 0.35,
    active_limit: int = 3,
    background_limit: int = 5,
) -> list:
    """
    迭代316: Working Memory Budget — 返回最终注入列表（active + background）。

    dormant 层不注入，等待 page fault 触发按需加载。

    输入：[(chunk_dict, score), ...] 或 [(score, chunk_dict), ...]
    输出：[chunk_dict, ...] 仅包含 active + background 层（保序：active在前）

    保证：
      - 永远不抛异常（内部 try/except 兜底）
      - 返回列表（最坏情况返回 []）
      - 总长度 <= active_limit + background_limit
    """
    try:
        tier = tier_chunks(
            scored_chunks,
            top_k=top_k,
            active_threshold=active_threshold,
            background_threshold=background_threshold,
            active_limit=active_limit,
            background_limit=background_limit,
        )
        return tier["active"] + tier["background"]
    except Exception:
        # 降级：返回空列表，由调用方按原逻辑处理
        return []


def wmb_stats(tier_result: dict) -> dict:
    """
    迭代316: Working Memory Budget — 返回分层统计信息。

    输入：tier_chunks() 的返回值
    输出：{
        "active_count":     int,   # active 层数量
        "background_count": int,   # background 层数量
        "dormant_count":    int,   # dormant 层数量（被抑制，不注入）
        "total":            int,   # 候选总量
        "injected":         int,   # 实际注入量（active + background）
        "dormant_suppressed": int, # 被抑制的 dormant 数量（同 dormant_count）
        "injected_ratio":   float, # 注入率 injected / total
    }
    """
    active_count = len(tier_result.get("active", []))
    background_count = len(tier_result.get("background", []))
    dormant_count = len(tier_result.get("dormant", []))
    total = active_count + background_count + dormant_count
    injected = active_count + background_count
    injected_ratio = injected / total if total > 0 else 0.0

    return {
        "active_count": active_count,
        "background_count": background_count,
        "dormant_count": dormant_count,
        "total": total,
        "injected": injected,
        "dormant_suppressed": dormant_count,
        "injected_ratio": round(injected_ratio, 3),
    }
