"""
tests/test_pin_decay.py — Pin Decay + Cap 测试（迭代356）

验证：
  1. pin_decay 解除超过 decay_days 的 soft pin
  2. pin_decay 不影响 hard pin
  3. pin_decay disabled 时不运行
  4. enforce_pin_cap 驱逐超限 soft pin
  5. enforce_pin_cap 不驱逐 hard pin
  6. enforce_pin_cap disabled 时不运行
  7. config 默认值合理
  8. pin rate < cap 时不驱逐
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import tmpfs  # noqa: F401 — tmpfs isolation
from store import open_db, ensure_schema, insert_chunk, pin_chunk, pin_decay, enforce_pin_cap, get_pinned_ids
from schema import MemoryChunk

_PROJECT = "test_pin_decay"


def _make_chunk(idx: int, importance: float = 0.8) -> dict:
    c = MemoryChunk(
        project=_PROJECT,
        source_session="sess",
        chunk_type="decision",
        content=f"content {idx}",
        summary=f"summary {idx}",
        importance=importance,
    )
    return c.to_dict()


def _setup_db():
    conn = open_db()
    ensure_schema(conn)
    return conn


# ─────────────────────────────────────────────────────────────
# 1. pin_decay — soft pin 自动解除
# ─────────────────────────────────────────────────────────────

class TestPinDecay:
    def test_decay_removes_stale_soft_pin(self):
        """超过 decay_days 天未访问的 soft pin 应被解除。"""
        conn = _setup_db()
        # 插入 chunk 并设置 last_accessed 为 60 天前
        cd = _make_chunk(1)
        conn.execute("""INSERT OR REPLACE INTO memory_chunks
            (id, project, source_session, chunk_type, content, summary, importance,
             retrievability, stability, last_accessed, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,datetime('now', '-60 days'),
                    datetime('now', '-60 days'),datetime('now', '-60 days'))""",
            (cd["id"], cd["project"], cd["source_session"], cd["chunk_type"],
             cd["content"], cd["summary"], cd["importance"],
             0.5, 1.0))
        conn.commit()
        pin_chunk(conn, cd["id"], _PROJECT, pin_type="soft")
        conn.commit()

        removed = pin_decay(conn, _PROJECT, decay_days=30)
        conn.commit()
        conn.close()

        assert removed == 1

    def test_decay_preserves_hard_pin(self):
        """hard pin 不受衰减影响。"""
        conn = _setup_db()
        cd = _make_chunk(2)
        conn.execute("""INSERT OR REPLACE INTO memory_chunks
            (id, project, source_session, chunk_type, content, summary, importance,
             retrievability, stability, last_accessed, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,datetime('now', '-60 days'),
                    datetime('now', '-60 days'),datetime('now', '-60 days'))""",
            (cd["id"], cd["project"], cd["source_session"], cd["chunk_type"],
             cd["content"], cd["summary"], cd["importance"],
             0.5, 1.0))
        conn.commit()
        pin_chunk(conn, cd["id"], _PROJECT, pin_type="hard")
        conn.commit()

        removed = pin_decay(conn, _PROJECT, decay_days=30)
        conn.commit()

        # hard pin 应保留
        still_pinned = get_pinned_ids(conn, _PROJECT, pin_type="hard")
        conn.close()
        assert cd["id"] in still_pinned
        assert removed == 0

    def test_decay_preserves_recent_soft_pin(self):
        """最近访问的 soft pin 不被衰减。"""
        conn = _setup_db()
        cd = _make_chunk(3)
        conn.execute("""INSERT OR REPLACE INTO memory_chunks
            (id, project, source_session, chunk_type, content, summary, importance,
             retrievability, stability, last_accessed, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,datetime('now', '-5 days'),
                    datetime('now', '-5 days'),datetime('now', '-5 days'))""",
            (cd["id"], cd["project"], cd["source_session"], cd["chunk_type"],
             cd["content"], cd["summary"], cd["importance"],
             0.5, 1.0))
        conn.commit()
        pin_chunk(conn, cd["id"], _PROJECT, pin_type="soft")
        conn.commit()

        removed = pin_decay(conn, _PROJECT, decay_days=30)
        conn.commit()
        conn.close()

        assert removed == 0  # 5 天 < 30 天，不应解除

    def test_decay_disabled_skips(self):
        """pin.decay_enabled=False 时 pin_decay 直接返回 0。"""
        conn = _setup_db()
        with patch("config.get", side_effect=lambda k: {
            "pin.decay_enabled": False,
            "pin.decay_days": 30,
        }.get(k, True)):
            removed = pin_decay(conn, _PROJECT, decay_days=30)
        conn.close()
        assert removed == 0


# ─────────────────────────────────────────────────────────────
# 2. enforce_pin_cap — pin 上限
# ─────────────────────────────────────────────────────────────

class TestEnforcePinCap:
    def test_cap_evicts_oldest_soft_pin(self):
        """超过 cap 时驱逐最旧 soft pin。"""
        conn = _setup_db()

        # 插入 10 个 chunk
        chunk_ids = []
        for i in range(10):
            cd = _make_chunk(100 + i)
            conn.execute("""INSERT OR REPLACE INTO memory_chunks
                (id, project, source_session, chunk_type, content, summary, importance,
                 retrievability, stability, last_accessed, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'),datetime('now'))""",
                (cd["id"], cd["project"], cd["source_session"], cd["chunk_type"],
                 cd["content"], cd["summary"], cd["importance"],
                 0.5, 1.0))
            chunk_ids.append(cd["id"])
        conn.commit()

        # pin 5 个（50% pin rate），cap = 15% → 最多 1-2 个
        for cid in chunk_ids[:5]:
            pin_chunk(conn, cid, _PROJECT, pin_type="soft")
        conn.commit()

        evicted = enforce_pin_cap(conn, _PROJECT, cap_pct=20)  # cap=20%=2个
        conn.commit()

        remaining_pins = conn.execute(
            "SELECT COUNT(*) FROM chunk_pins WHERE project=?", (_PROJECT,)
        ).fetchone()[0]
        conn.close()

        assert evicted >= 1  # 至少驱逐了一些
        assert remaining_pins <= 2  # 不超过 cap

    def test_cap_preserves_hard_pin(self):
        """cap 驱逐时不动 hard pin。"""
        conn = _setup_db()

        # 插入 5 个 chunk
        hard_id = None
        for i in range(5):
            cd = _make_chunk(200 + i)
            conn.execute("""INSERT OR REPLACE INTO memory_chunks
                (id, project, source_session, chunk_type, content, summary, importance,
                 retrievability, stability, last_accessed, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'),datetime('now'))""",
                (cd["id"], cd["project"], cd["source_session"], cd["chunk_type"],
                 cd["content"], cd["summary"], cd["importance"],
                 0.5, 1.0))
            if i == 0:
                hard_id = cd["id"]
        conn.commit()

        # 1 hard + 4 soft = 5 pins on 5 chunks = 100%，cap=20%=1个
        pin_chunk(conn, hard_id, _PROJECT, pin_type="hard")
        for i in range(1, 5):
            cid = f"chunk-{200 + i}-{_PROJECT}"  # dynamic chunk id
        # Use actual inserted chunk IDs
        rows = conn.execute(
            "SELECT id FROM memory_chunks WHERE project=? AND id != ?",
            (_PROJECT, hard_id)
        ).fetchall()
        for row in rows[:4]:
            pin_chunk(conn, row[0], _PROJECT, pin_type="soft")
        conn.commit()

        enforce_pin_cap(conn, _PROJECT, cap_pct=20)
        conn.commit()

        # hard pin 应保留
        hard_pinned = get_pinned_ids(conn, _PROJECT, pin_type="hard")
        conn.close()
        assert hard_id in hard_pinned

    def test_cap_no_eviction_when_within_limit(self):
        """pin rate < cap 时不驱逐。"""
        conn = _setup_db()

        # 插入 20 个 chunk，只 pin 2 个（10%）
        ids = []
        for i in range(20):
            cd = _make_chunk(300 + i)
            conn.execute("""INSERT OR REPLACE INTO memory_chunks
                (id, project, source_session, chunk_type, content, summary, importance,
                 retrievability, stability, last_accessed, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'),datetime('now'))""",
                (cd["id"], cd["project"], cd["source_session"], cd["chunk_type"],
                 cd["content"], cd["summary"], cd["importance"],
                 0.5, 1.0))
            ids.append(cd["id"])
        conn.commit()

        pin_chunk(conn, ids[0], _PROJECT, pin_type="soft")
        pin_chunk(conn, ids[1], _PROJECT, pin_type="soft")
        conn.commit()

        evicted = enforce_pin_cap(conn, _PROJECT, cap_pct=20)  # cap=20%=4个，当前2个<4个
        conn.commit()
        conn.close()

        assert evicted == 0

    def test_cap_disabled_skips(self):
        """pin.cap_apply_on_pin=False 时不驱逐。"""
        conn = _setup_db()
        with patch("config.get", side_effect=lambda k: {
            "pin.cap_apply_on_pin": False,
            "pin.cap_pct": 15,
        }.get(k, True)):
            evicted = enforce_pin_cap(conn, _PROJECT, cap_pct=15)
        conn.close()
        assert evicted == 0


# ─────────────────────────────────────────────────────────────
# 3. config 默认值
# ─────────────────────────────────────────────────────────────

class TestPinDecapConfig:
    def test_config_keys_exist(self):
        from config import get as sysctl
        assert isinstance(sysctl("pin.decay_enabled"), bool)
        assert isinstance(sysctl("pin.decay_days"), int)
        assert isinstance(sysctl("pin.cap_pct"), int)
        assert isinstance(sysctl("pin.cap_apply_on_pin"), bool)

    def test_defaults(self):
        from config import get as sysctl
        assert sysctl("pin.decay_enabled") is True
        assert sysctl("pin.decay_days") == 30
        assert sysctl("pin.cap_pct") == 15
        assert sysctl("pin.cap_apply_on_pin") is True
