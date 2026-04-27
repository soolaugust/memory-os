"""
test_iter89_checkpoint_content.py — iter89: Checkpoint Content Retention 修复验证

修复项：
  1. chunk_snapshots 现在同时保存 content（之前只有 summary）
  2. max_hit_ids: 10 → 50（支持大工作集）

目标：
  - content retention: 13.9% → ≥80%
  - summary retention: 49% → ≥90%
  - 多轮（3轮）恢复率保持稳定（不衰减到 0%）
"""
import json
import sqlite3
import sys
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from store_core import ensure_schema
from store_criu import checkpoint_dump, checkpoint_restore
from store_swap import swap_out


def _make_conn():
    tmpdir = tempfile.mkdtemp()
    db_path = Path(tmpdir) / "test.db"
    conn = sqlite3.connect(str(db_path))
    ensure_schema(conn)
    return conn, tmpdir


def _cleanup(conn, tmpdir):
    conn.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


def _insert_chunks(conn, project, count=20):
    ids = []
    for i in range(count):
        cid = f"chunk_{project}_{i}"
        conn.execute("""
            INSERT INTO memory_chunks
            (id, created_at, updated_at, project, source_session, chunk_type,
             content, summary, tags, importance, retrievability, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cid,
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
            project,
            "sess_test",
            "decision",
            f"Full content block {i}: " + "x" * 200,
            f"Summary for chunk {i}",
            "[]", 0.7, 0.8,
            datetime.now(timezone.utc).isoformat(),
        ))
        ids.append(cid)
    conn.commit()
    return ids


class TestCheckpointContentRetention:
    """验证 content 字段被正确保存到 chunk_snapshots"""

    def test_snapshot_includes_content(self):
        """dump 后 snapshot 包含 content 字段"""
        conn, tmpdir = _make_conn()
        try:
            project = "test_content"
            ids = _insert_chunks(conn, project, 5)
            result = checkpoint_dump(conn, project, "sess1", hit_chunk_ids=ids)
            conn.commit()

            assert result["saved_ids"] == 5, f"Expected 5 saved, got {result['saved_ids']}"

            # 直接查 snapshot 内容
            row = conn.execute(
                "SELECT chunk_snapshots FROM checkpoints WHERE id = ?",
                (result["checkpoint_id"],)
            ).fetchone()
            snapshots = json.loads(row[0])

            for snap in snapshots:
                assert "content" in snap, f"Missing 'content' in snapshot: {snap.keys()}"
                assert len(snap["content"]) > 0, f"Empty content in snapshot for {snap['id']}"
                assert "summary" in snap, "Missing 'summary' in snapshot"
        finally:
            _cleanup(conn, tmpdir)

    def test_restore_returns_content_from_snapshot(self):
        """swap out 后，restore 从 snapshot 恢复带 content 的 chunks"""
        conn, tmpdir = _make_conn()
        try:
            project = "test_restore_content"
            ids = _insert_chunks(conn, project, 10)

            # dump
            dump_result = checkpoint_dump(conn, project, "sess1", hit_chunk_ids=ids)
            conn.commit()
            assert dump_result["saved_ids"] == 10

            # swap out 全部（模拟 kswapd 淘汰）
            swap_result = swap_out(conn, ids)
            conn.commit()
            assert swap_result["swapped_count"] > 0

            # restore — 应从 snapshot 恢复
            restored = checkpoint_restore(conn, project)
            assert restored is not None, "checkpoint_restore returned None"

            snapshot_chunks = [c for c in restored["chunks"] if c.get("_from_snapshot")]
            assert len(snapshot_chunks) > 0, "No snapshot chunks returned after swap_out"

            for sc in snapshot_chunks:
                assert "content" in sc, f"Snapshot chunk missing content: {sc.keys()}"
                assert len(sc["content"]) > 100, f"Content too short: {len(sc['content'])}"

        finally:
            _cleanup(conn, tmpdir)


class TestContentRetentionRate:
    """验证 content retention 率达到 ≥80%"""

    def test_content_retention_after_swap(self):
        """swap out 后 content retention ≥ 80%"""
        conn, tmpdir = _make_conn()
        try:
            project = "test_retention_rate"
            ids = _insert_chunks(conn, project, 20)

            # 计算原始 content chars
            rows = conn.execute(
                "SELECT content FROM memory_chunks WHERE project = ?", (project,)
            ).fetchall()
            original_content_chars = sum(len(r[0] or "") for r in rows)

            # dump
            checkpoint_dump(conn, project, "sess1", hit_chunk_ids=ids)
            conn.commit()

            # swap out all
            swap_out(conn, ids)
            conn.commit()

            # restore
            restored = checkpoint_restore(conn, project)
            assert restored is not None

            # 计算恢复的 content chars
            recovered_content_chars = sum(
                len(c.get("content", "")) for c in restored["chunks"]
            )

            retention_pct = recovered_content_chars / original_content_chars * 100
            assert retention_pct >= 80.0, (
                f"Content retention {retention_pct:.1f}% < 80% target\n"
                f"Original: {original_content_chars}, Recovered: {recovered_content_chars}"
            )

        finally:
            _cleanup(conn, tmpdir)

    def test_summary_retention_after_swap(self):
        """swap out 后 summary retention ≥ 90%"""
        conn, tmpdir = _make_conn()
        try:
            project = "test_summary_retention"
            ids = _insert_chunks(conn, project, 20)

            rows = conn.execute(
                "SELECT summary FROM memory_chunks WHERE project = ?", (project,)
            ).fetchall()
            original_summary_chars = sum(len(r[0] or "") for r in rows)

            checkpoint_dump(conn, project, "sess1", hit_chunk_ids=ids)
            conn.commit()
            swap_out(conn, ids)
            conn.commit()

            restored = checkpoint_restore(conn, project)
            assert restored is not None

            recovered_summary_chars = sum(
                len(c.get("summary", "")) for c in restored["chunks"]
            )
            retention_pct = recovered_summary_chars / original_summary_chars * 100
            assert retention_pct >= 90.0, (
                f"Summary retention {retention_pct:.1f}% < 90% target"
            )
        finally:
            _cleanup(conn, tmpdir)


class TestMultiRoundStability:
    """验证多轮 compaction 后恢复率稳定（不衰减到 0%）"""

    def test_three_round_retention_stable(self):
        """3 轮 checkpoint/restore 后，每轮 content retention 保持稳定"""
        conn, tmpdir = _make_conn()
        try:
            project = "test_multi_round"
            ids = _insert_chunks(conn, project, 30)

            rows = conn.execute(
                "SELECT content FROM memory_chunks WHERE project = ?", (project,)
            ).fetchall()
            original_chars = sum(len(r[0] or "") for r in rows)

            retention_rates = []

            for round_num in range(1, 4):
                # dump 当前工作集
                checkpoint_dump(conn, project, f"sess_{round_num}", hit_chunk_ids=ids)
                conn.commit()

                # restore
                restored = checkpoint_restore(conn, project)
                assert restored is not None, f"Round {round_num}: restore returned None"

                recovered_chars = sum(len(c.get("content", "")) for c in restored["chunks"])
                rate = recovered_chars / original_chars * 100
                retention_rates.append(rate)

                # swap out 一半（模拟内存压力）
                if round_num < 3:
                    swap_out(conn, ids[:len(ids)//2])
                    conn.commit()

            # 每轮都应 ≥ 40%（允许 swap 后部分丢失）
            for i, rate in enumerate(retention_rates):
                assert rate >= 40.0, (
                    f"Round {i+1} content retention {rate:.1f}% < 40% minimum\n"
                    f"All rates: {[f'{r:.1f}%' for r in retention_rates]}"
                )

            # 第 2、3 轮不应该完全归零（旧 bug：multi-round → 0%）
            assert retention_rates[1] > 0 or retention_rates[2] > 0, (
                "Rounds 2+3 both returned 0% — multi-round regression detected"
            )

        finally:
            _cleanup(conn, tmpdir)


class TestMaxHitIds:
    """验证 max_hit_ids=50 能覆盖大工作集"""

    def test_large_workset_preserved(self):
        """50 个 chunk 都能被保存（之前 max=10 会截断）"""
        conn, tmpdir = _make_conn()
        try:
            project = "test_large_ws"
            ids = _insert_chunks(conn, project, 50)

            result = checkpoint_dump(conn, project, "sess1", hit_chunk_ids=ids)
            conn.commit()

            # iter89 后 max_hit_ids=50，所有 50 个都应被保存
            assert result["saved_ids"] == 50, (
                f"Expected 50 saved_ids, got {result['saved_ids']} "
                f"(max_hit_ids may still be 10)"
            )
        finally:
            _cleanup(conn, tmpdir)
