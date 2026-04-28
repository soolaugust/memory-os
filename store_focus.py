"""
store_focus.py — iter368 Attention Focus Stack（会话注意焦点栈）

OS 类比：CPU register file — 极少数"热"地址（~32 个寄存器）可在一个时钟周期内访问，
  相比 L1 cache（4 cycles）快 4×，相比 RAM（200 cycles）快 200×。
  寄存器分配器（register allocator）决定哪些变量"住"寄存器。

人的记忆类比：
  Cowan (2001) working memory focus of attention — 工作记忆中有一个"注意焦点"（focus of attention），
  能立即访问约 1-4 个概念，不需要额外提取代价。
  当前关注的主题在焦点中，相关知识激活阈值更低。

设计：
  - 维护 session 级别的"焦点关键词栈"（最多 MAX_FOCUS_ITEMS 个关键词）
  - 关键词来源：当前 prompt 中的实体词、新写入 chunk 的 summary 关键词
  - 存储在内存文件 .session_focus.json（跨请求持久化，session 切换时重置）
  - 检索时：焦点命中的 chunk 获得 FOCUS_BONUS (+0.12) 加分
  - LRU 语义：新关键词入栈时，超出容量的旧关键词弹出

表结构（持久化到 DB，用于多进程共享）：
  session_focus(session_id TEXT, keyword TEXT, updated_at TEXT, hit_count INTEGER)
  PRIMARY KEY (session_id, keyword)
"""

import re
from datetime import datetime, timezone

MAX_FOCUS_ITEMS = 12       # 焦点栈最大容量（类比 32 个寄存器，但 memory-os 关键词更"宽"）
FOCUS_BONUS = 0.12         # 焦点命中 bonus
FOCUS_DECAY_RATE = 0.85    # 每次未命中时衰减（LRU 语义）


# ── Schema ────────────────────────────────────────────────────────────────────

def ensure_focus_schema(conn):
    """创建 session_focus 表。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_focus (
            session_id  TEXT NOT NULL,
            keyword     TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            hit_count   INTEGER DEFAULT 1,
            PRIMARY KEY (session_id, keyword)
        )
    """)
    conn.commit()


# ── Focus Update ──────────────────────────────────────────────────────────────

def _extract_focus_keywords(text: str) -> list:
    """
    从文本中提取焦点关键词：
      1. 反引号包裹的标识符
      2. 全大写缩写（>= 2字符）
      3. 中文双字 bigram（频率最高的前 5 个）
      4. 英文技术词（≥ 5 字符的驼峰/下划线词）

    Returns: 去重后的关键词列表（最多 8 个）
    """
    keywords = []
    text_lower = text.lower()

    # 1. 反引号代码标识符
    for m in re.finditer(r'`([^`]{2,30})`', text):
        keywords.append(m.group(1).lower())

    # 2. 全大写缩写（排除常见非技术词）
    _EXCLUDE = {"I", "A", "OR", "AND", "THE", "IF", "OK", "IS", "IT",
                "LGTM", "TBD", "FYI", "ASAP", "BTW", "IMO"}
    for m in re.finditer(r'\b([A-Z][A-Z0-9_]{1,15})\b', text):
        w = m.group(1)
        if w not in _EXCLUDE:
            keywords.append(w.lower())

    # 3. 中文双字 bigram（≥ 3 次出现的高频词）
    cn_text = re.sub(r'[^\u4e00-\u9fff]', '', text)
    bigram_freq: dict = {}
    for i in range(len(cn_text) - 1):
        bg = cn_text[i:i+2]
        bigram_freq[bg] = bigram_freq.get(bg, 0) + 1
    # 取频率 >= 2 的 bigram（过滤低频噪音）
    for bg, cnt in sorted(bigram_freq.items(), key=lambda x: -x[1]):
        if cnt >= 2:
            keywords.append(bg)
        if len(keywords) >= 8:
            break

    # 4. 英文技术词（驼峰或下划线，≥ 5 字符）
    for m in re.finditer(r'\b([a-zA-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z_]{5,})\b', text):
        w = m.group(1).lower()
        if '_' in w or w[0].lower() != w[0].upper():
            keywords.append(w)

    # 去重，保持顺序，最多 8 个
    seen = set()
    result = []
    for kw in keywords:
        kw_clean = kw.strip().lower()
        if kw_clean and kw_clean not in seen and len(kw_clean) >= 2:
            seen.add(kw_clean)
            result.append(kw_clean)
            if len(result) >= 8:
                break
    return result


def update_focus(conn, session_id: str, text: str) -> list:
    """
    从 text 提取关键词，更新 session 焦点栈。
    LRU 语义：超出 MAX_FOCUS_ITEMS 时删除最旧的条目。
    Returns: 更新后的关键词列表
    """
    if not session_id or session_id == "unknown":
        return []

    keywords = _extract_focus_keywords(text)
    if not keywords:
        return get_focus(conn, session_id)

    now = datetime.now(timezone.utc).isoformat()

    for kw in keywords:
        conn.execute("""
            INSERT INTO session_focus (session_id, keyword, updated_at, hit_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(session_id, keyword) DO UPDATE SET
                updated_at = excluded.updated_at,
                hit_count  = hit_count + 1
        """, (session_id, kw, now))

    # LRU 淘汰：保留最近更新的 MAX_FOCUS_ITEMS 个
    rows = conn.execute(
        """SELECT keyword FROM session_focus WHERE session_id=?
           ORDER BY updated_at DESC, hit_count DESC
           LIMIT 1000""",
        (session_id,)
    ).fetchall()

    if len(rows) > MAX_FOCUS_ITEMS:
        to_delete = [r[0] for r in rows[MAX_FOCUS_ITEMS:]]
        for kw in to_delete:
            conn.execute(
                "DELETE FROM session_focus WHERE session_id=? AND keyword=?",
                (session_id, kw)
            )

    conn.commit()
    return [r[0] for r in rows[:MAX_FOCUS_ITEMS]]


def get_focus(conn, session_id: str) -> list:
    """获取 session 当前焦点关键词列表（按更新时间降序）。"""
    if not session_id or session_id == "unknown":
        return []
    rows = conn.execute(
        """SELECT keyword FROM session_focus WHERE session_id=?
           ORDER BY updated_at DESC, hit_count DESC
           LIMIT ?""",
        (session_id, MAX_FOCUS_ITEMS)
    ).fetchall()
    return [r[0] for r in rows]


def clear_focus(conn, session_id: str) -> None:
    """清除 session 焦点（session 切换时调用）。"""
    conn.execute("DELETE FROM session_focus WHERE session_id=?", (session_id,))
    conn.commit()


# ── Focus Score Bonus ─────────────────────────────────────────────────────────

def focus_score_bonus(focus_keywords: list, chunk_summary: str,
                      chunk_content: str = "") -> float:
    """
    计算 chunk 的焦点 bonus。
    焦点关键词命中越多 → bonus 越高（最多 FOCUS_BONUS）。

    OS 类比：register allocator live variable analysis —
      变量"活跃度"（use/def 计数）决定是否值得放入寄存器。
      焦点关键词在当前 chunk 中的命中数 ≈ 变量活跃度。

    Returns: float in [0, FOCUS_BONUS]
    """
    if not focus_keywords:
        return 0.0

    text = f"{chunk_summary} {chunk_content[:200]}".lower()
    matched = sum(1 for kw in focus_keywords if kw in text)
    if matched == 0:
        return 0.0

    # 对数缩放：1命中→0.06，2→0.09，3→0.11，4+→0.12
    import math
    ratio = min(1.0, math.log(1 + matched) / math.log(1 + len(focus_keywords)))
    return round(ratio * FOCUS_BONUS, 4)


# ── Focus Stats ───────────────────────────────────────────────────────────────

def focus_stats(conn, session_id: str) -> dict:
    """返回 session 焦点统计信息。"""
    rows = conn.execute(
        """SELECT keyword, hit_count, updated_at
           FROM session_focus WHERE session_id=?
           ORDER BY hit_count DESC, updated_at DESC""",
        (session_id,)
    ).fetchall()
    return {
        "session_id": session_id,
        "focus_count": len(rows),
        "keywords": [{"keyword": r[0], "hit_count": r[1], "updated_at": r[2]}
                     for r in rows],
    }
