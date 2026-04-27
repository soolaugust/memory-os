#!/usr/bin/env python3
"""
store.db 噪声清洗工具
- 移除 bracket-prefix / self-ref / too-short / 旧 task_state chunk
- 执行前自动备份到 store.db.bak
"""
import sqlite3, re, shutil
from pathlib import Path

DB = Path.home() / ".claude/memory-os/store.db"
BAK = DB.parent / "store.db.bak"


def is_noise(ctype: str, summary: str) -> bool:
    s = (summary or "").strip()
    # bracket / list-marker 前缀（截断/前缀污染）
    if re.match(r'^[\[\]\-]', s): return True
    # 纯 markdown 符号
    if re.match(r'^[-=*`#]{2,}$', s): return True
    # 过短
    if len(s) < 10: return True
    # 自我引用 / 元数据泄漏
    noise_phrases = [
        "← project 字段", "(importance=", "Stop extractor，规则提取", "路径被重复写入",
    ]
    if any(x in s for x in noise_phrases): return True
    # 截断句（以助词/连词/标点开头，说明这是句子中间）
    if re.match(r'^[了的地得把被让向从以在]', s): return True
    # 占位符/无意义 decision
    placeholder = ["方案 X 是最优解", "extractor 升级", "KnowledgeRouter"]
    if s in placeholder: return True
    return False


def clean(dry_run: bool = False):
    shutil.copy2(DB, BAK)
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT id, chunk_type, summary FROM memory_chunks ORDER BY created_at"
    ).fetchall()

    noise_ids = []
    task_state_ids = [(r[0], r[2]) for r in rows if r[1] == "task_state"]

    for rid, ctype, summary in rows:
        if is_noise(ctype, summary):
            noise_ids.append(rid)

    if len(task_state_ids) > 1:
        for rid, _ in task_state_ids[:-1]:
            if rid not in noise_ids:
                noise_ids.append(rid)

    keep_ids = [r[0] for r in rows if r[0] not in set(noise_ids)]

    print(f"总计 {len(rows)} 条，移除 {len(noise_ids)} 条，保留 {len(keep_ids)} 条")
    if dry_run:
        print("[dry-run] 未实际写入")
        conn.close()
        return

    conn.execute("CREATE TEMP TABLE mc_keep AS SELECT * FROM memory_chunks WHERE 1=0")
    for kid in keep_ids:
        conn.execute("INSERT INTO mc_keep SELECT * FROM memory_chunks WHERE id=?", (kid,))
    conn.execute("DROP TABLE memory_chunks")
    conn.execute("ALTER TABLE mc_keep RENAME TO memory_chunks")
    conn.commit()

    remaining = conn.execute(
        "SELECT chunk_type, summary FROM memory_chunks ORDER BY created_at"
    ).fetchall()
    print("保留内容：")
    for ct, s in remaining:
        print(f"  [{ct}] {s[:70]}")
    conn.close()


if __name__ == "__main__":
    import sys
    dry = "--dry" in sys.argv
    clean(dry_run=dry)
