#!/usr/bin/env python3
"""
derive_rules.py — 从 memory-os feedback/design_constraint chunk 自动派生 CLAUDE.md 门控规则

工作流：
1. 从 store.db 读取所有 feedback 和 design_constraint 类 chunk
2. 提取其中的"How to apply"字段（规则触发条件）
3. 在目标项目的 CLAUDE.md 中更新 [AUTO-DERIVED RULES] 区块

用法：
    python3 derive_rules.py --project <project_id> --claudemd <path>
    python3 derive_rules.py --all   # 更新所有已注册项目的 CLAUDE.md

可被 Stop hook 调用（chunk 写入后触发）。
"""
import sys
import os
import re
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from store import open_db
from utils import resolve_project_id

MARKER_START = "<!-- AUTO-DERIVED-RULES:START -->"
MARKER_END   = "<!-- AUTO-DERIVED-RULES:END -->"

# 从 chunk summary 提取 "How to apply:" 行
_HOW_RE = re.compile(
    r'\*\*How to apply:\*\*\s*(.+?)(?=\*\*Why:\*\*|\*\*How to apply:\*\*|$)',
    re.DOTALL | re.IGNORECASE
)
_WHY_RE = re.compile(
    r'\*\*Why:\*\*\s*(.+?)(?=\*\*How to apply:\*\*|\*\*Why:\*\*|$)',
    re.DOTALL | re.IGNORECASE
)


def fetch_actionable_chunks(conn: sqlite3.Connection, project: str) -> list[dict]:
    """
    取 feedback / design_constraint chunk，按 importance 降序。
    只取 importance >= 0.5 的（低质量 chunk 不应派生规则）。
    """
    cur = conn.execute("""
        SELECT id, chunk_type, summary, importance
        FROM chunks
        WHERE project = ?
          AND chunk_type IN ('feedback', 'design_constraint')
          AND importance >= 0.5
        ORDER BY importance DESC
        LIMIT 50
    """, (project,))
    rows = cur.fetchall()
    return [
        {"id": r[0], "chunk_type": r[1], "summary": r[2], "importance": r[3]}
        for r in rows
    ]


def extract_rule(summary: str) -> dict | None:
    """
    从 chunk summary 提取可执行规则。
    支持两种格式：
      1. 结构化：含 **How to apply:** 字段
      2. 非结构化：直接取 summary 第一句作为规则
    """
    how_m = _HOW_RE.search(summary)
    why_m = _WHY_RE.search(summary)

    if how_m:
        rule = how_m.group(1).strip().replace('\n', ' ')
        why  = why_m.group(1).strip().replace('\n', ' ') if why_m else ""
        return {"rule": rule, "why": why}

    # 非结构化：取第一句（去掉 markdown 标记）
    first = re.sub(r'\*\*.*?\*\*[：:]?\s*', '', summary.split('\n')[0]).strip()
    if len(first) > 20:
        return {"rule": first, "why": ""}
    return None


def build_rules_block(chunks: list[dict]) -> str:
    """生成 CLAUDE.md 中的自动派生规则区块。"""
    lines = [
        MARKER_START,
        "## 自动派生规则（由 memory-os feedback 生成，勿手动编辑）",
        f"<!-- 最后更新：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} -->",
        "",
    ]

    feedback_rules = []
    constraint_rules = []

    for c in chunks:
        r = extract_rule(c["summary"])
        if not r:
            continue
        entry = {
            "rule": r["rule"],
            "why": r["why"],
            "importance": c["importance"],
            "chunk_type": c["chunk_type"],
            "id": c["id"][:8],
        }
        if c["chunk_type"] == "design_constraint":
            constraint_rules.append(entry)
        else:
            feedback_rules.append(entry)

    if constraint_rules:
        lines.append("### 设计约束（违反 = 禁止继续）")
        lines.append("")
        for e in constraint_rules:
            lines.append(f"- **[{e['id']}]** {e['rule']}")
            if e["why"]:
                lines.append(f"  - Why: {e['why']}")
        lines.append("")

    if feedback_rules:
        lines.append("### 操作规则（来自历史 feedback）")
        lines.append("")
        for e in feedback_rules:
            lines.append(f"- **[{e['id']}]** {e['rule']}")
            if e["why"]:
                lines.append(f"  - Why: {e['why']}")
        lines.append("")

    lines.append(MARKER_END)
    return "\n".join(lines)


def update_claudemd(claudemd_path: Path, rules_block: str) -> bool:
    """
    在 CLAUDE.md 中替换 AUTO-DERIVED-RULES 区块。
    如果区块不存在则追加到文件末尾。
    返回 True 表示文件有变化。
    """
    if claudemd_path.exists():
        content = claudemd_path.read_text()
    else:
        content = ""

    if MARKER_START in content and MARKER_END in content:
        new_content = re.sub(
            re.escape(MARKER_START) + r'.*?' + re.escape(MARKER_END),
            rules_block,
            content,
            flags=re.DOTALL
        )
    else:
        new_content = content.rstrip() + "\n\n" + rules_block + "\n"

    if new_content == content:
        return False

    claudemd_path.write_text(new_content)
    return True


def run(project: str, claudemd_path: Path, db_path: Path | None = None) -> int:
    """执行派生，返回写入的规则数。"""
    conn = open_db(db_path) if db_path else open_db()
    chunks = fetch_actionable_chunks(conn, project)
    conn.close()

    if not chunks:
        return 0

    block = build_rules_block(chunks)
    changed = update_claudemd(claudemd_path, block)

    rule_count = block.count("- **[")
    if changed:
        print(f"[derive_rules] updated {claudemd_path} with {rule_count} rules "
              f"from {len(chunks)} chunks (project={project})")
    return rule_count


def main():
    parser = argparse.ArgumentParser(description="Derive CLAUDE.md rules from memory-os feedback chunks")
    parser.add_argument("--project", help="Project ID (auto-resolved from cwd if omitted)")
    parser.add_argument("--claudemd", help="Path to CLAUDE.md to update")
    parser.add_argument("--db", help="Path to store.db (default: ~/.claude/memory-os/store.db)")
    parser.add_argument("--dry-run", action="store_true", help="Print rules without writing")
    args = parser.parse_args()

    project = args.project or resolve_project_id(Path.cwd())
    if not project:
        print("[derive_rules] ERROR: cannot resolve project id", file=sys.stderr)
        sys.exit(1)

    # CLAUDE.md 路径：优先用参数，否则用 cwd
    if args.claudemd:
        claudemd = Path(args.claudemd)
    else:
        # 向上查找最近的 CLAUDE.md
        cwd = Path.cwd()
        claudemd = None
        for p in [cwd, *cwd.parents]:
            candidate = p / "CLAUDE.md"
            if candidate.exists():
                claudemd = candidate
                break
        if claudemd is None:
            claudemd = cwd / "CLAUDE.md"

    db_path = Path(args.db) if args.db else None

    if args.dry_run:
        conn = open_db(db_path) if db_path else open_db()
        chunks = fetch_actionable_chunks(conn, project)
        conn.close()
        block = build_rules_block(chunks)
        print(block)
        return

    count = run(project, claudemd, db_path)
    if count == 0:
        print(f"[derive_rules] no actionable chunks for project={project}")


if __name__ == "__main__":
    main()
