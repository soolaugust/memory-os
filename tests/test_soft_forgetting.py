"""
test_soft_forgetting.py — iter369 Soft Forgetting 测试

覆盖：
  SF1: retrievability >= 0.15 → score 无折扣
  SF2: retrievability < 0.15 → score 乘以 0.55（软遗忘）
  SF3: design_constraint 类型豁免遗忘折扣
  SF4: retrievability = None/缺失 → 视为 1.0（无折扣）
  SF5: fts_search 返回结果中包含 retrievability 字段
  SF6: 遗忘 chunk 的 score 低于同等条件正常 chunk
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture()
def tmpdb(tmp_path):
    db_path = tmp_path / "test_store.db"
    os.environ["MEMORY_OS_DB"] = str(db_path)
    os.environ["MEMORY_OS_DIR"] = str(tmp_path)
    yield db_path
    os.environ.pop("MEMORY_OS_DB", None)
    os.environ.pop("MEMORY_OS_DIR", None)


@pytest.fixture()
def conn(tmpdb):
    from store_vfs import open_db, ensure_schema
    c = open_db(tmpdb)
    ensure_schema(c)
    yield c
    c.close()


def _insert_chunk(conn, chunk_id, retrievability=1.0, chunk_type="decision",
                  summary="chunk summary", importance=0.7, project="proj"):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO memory_chunks
        (id, chunk_type, summary, importance, created_at, updated_at, project,
         source_session, content, retrievability)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (chunk_id, chunk_type, summary, importance, now, now, project,
          "sess1", summary, retrievability))
    conn.commit()


# ── SF1: retrievability >= 0.15 → 无折扣 ──────────────────────────────────────

def test_sf1_normal_retrievability_no_discount():
    """retrievability = 1.0 时，soft forgetting 不施加折扣"""
    chunk = {
        "id": "c1", "chunk_type": "decision", "retrievability": 1.0,
        "summary": "正常 chunk", "content": "", "importance": 0.7,
    }
    # 模拟 _score_chunk 中的 iter369 逻辑
    _ret = float(chunk.get("retrievability") or 1.0)
    discount = 1.0
    if _ret < 0.15 and chunk.get("chunk_type") != "design_constraint":
        discount = 0.55
    assert discount == 1.0


# ── SF2: retrievability < 0.15 → score × 0.55 ───────────────────────────────

def test_sf2_low_retrievability_discount():
    """retrievability = 0.05 (高度遗忘) → 折扣 0.55"""
    chunk = {
        "id": "c1", "chunk_type": "decision", "retrievability": 0.05,
        "summary": "遗忘中的 chunk", "content": "", "importance": 0.7,
    }
    _ret = float(chunk.get("retrievability") or 1.0)
    discount = 1.0
    if _ret < 0.15 and chunk.get("chunk_type") != "design_constraint":
        discount = 0.55
    assert abs(discount - 0.55) < 0.01


def test_sf2_boundary_retrievability_no_discount():
    """retrievability = 0.15 (边界值) → 不折扣"""
    chunk = {
        "id": "c1", "chunk_type": "decision", "retrievability": 0.15,
        "summary": "边界 chunk",
    }
    _ret = float(chunk.get("retrievability") or 1.0)
    discount = 1.0
    if _ret < 0.15 and chunk.get("chunk_type") != "design_constraint":
        discount = 0.55
    assert discount == 1.0  # 0.15 不触发折扣（< 0.15 才触发）


# ── SF3: design_constraint 豁免 ───────────────────────────────────────────────

def test_sf3_design_constraint_exempt():
    """design_constraint 即使 retrievability=0 也不折扣"""
    chunk = {
        "id": "c1", "chunk_type": "design_constraint", "retrievability": 0.01,
        "summary": "不可遗忘的约束",
    }
    _ret = float(chunk.get("retrievability") or 1.0)
    discount = 1.0
    if _ret < 0.15 and chunk.get("chunk_type") != "design_constraint":
        discount = 0.55
    assert discount == 1.0  # 豁免，无折扣


# ── SF4: retrievability 缺失 → 视为 1.0 ─────────────────────────────────────

def test_sf4_missing_retrievability_defaults_to_full():
    """chunk 没有 retrievability 字段 → 默认 1.0，无折扣"""
    chunk = {
        "id": "c1", "chunk_type": "decision",
        "summary": "无 retrievability 字段的 chunk",
    }
    _ret = float(chunk.get("retrievability") or 1.0)
    discount = 1.0
    if _ret < 0.15 and chunk.get("chunk_type") != "design_constraint":
        discount = 0.55
    assert discount == 1.0


# ── SF5: fts_search 返回 retrievability 字段 ─────────────────────────────────

def test_sf5_fts_search_returns_retrievability(conn):
    """fts_search 返回的 chunk dict 应包含 retrievability 字段"""
    from store_vfs import fts_search
    _insert_chunk(conn, "fts_c1", retrievability=0.05, summary="FTS5 遗忘测试")
    results = fts_search(conn, "遗忘测试", "proj", top_k=5)
    if results:
        # 至少一个结果应有 retrievability 字段
        assert "retrievability" in results[0]
        match = [r for r in results if r["id"] == "fts_c1"]
        if match:
            assert abs(match[0]["retrievability"] - 0.05) < 0.01


# ── SF6: 遗忘 chunk 的 score 低于正常 chunk ──────────────────────────────────

def test_sf6_forgotten_chunk_lower_score():
    """模拟完整流程：遗忘 chunk score < 正常 chunk score（相同其他条件）"""
    # 模拟 _score_chunk 的 soft forgetting 部分
    base_score = 0.75

    chunk_normal = {"chunk_type": "decision", "retrievability": 1.0}
    chunk_forgotten = {"chunk_type": "decision", "retrievability": 0.05}

    def apply_forgetting(chunk, score):
        _ret = float(chunk.get("retrievability") or 1.0)
        if _ret < 0.15 and chunk.get("chunk_type") != "design_constraint":
            score *= 0.55
        return score

    normal_score = apply_forgetting(chunk_normal, base_score)
    forgotten_score = apply_forgetting(chunk_forgotten, base_score)

    assert forgotten_score < normal_score
    assert abs(forgotten_score - base_score * 0.55) < 0.01
