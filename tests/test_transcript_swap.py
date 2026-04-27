#!/usr/bin/env python3
"""
迭代56 测试：Transcript-Aware Swap — PreCompact/PostCompact 对话记录提取与恢复

验证：
1. _extract_transcript_summary 正确从 JSONL 提取 user/assistant 文本
2. _summarize_text 正确去除代码块和长 JSON
3. 对话摘要字符上限控制
4. 空/不存在 transcript 文件的边界情况
5. PostCompact 恢复包含对话摘要
6. 端到端：swap out → swap in 对话摘要完整保留
"""
import tmpfs  # 测试隔离（迭代54）

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# 确保 hooks 目录可导入
_HOOKS_DIR = Path(__file__).parent / "hooks"
sys.path.insert(0, str(_HOOKS_DIR))

_MOS_ROOT = Path(__file__).parent
sys.path.insert(0, str(_MOS_ROOT))


def _import_save_module():
    """动态导入 save-task-state.py（文件名含连字符，不能直接 import）"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "save_task_state",
        str(Path.home() / ".claude" / "hooks" / "save-task-state.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_resume_module():
    """动态导入 resume-task-state.py"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "resume_task_state",
        str(Path.home() / ".claude" / "hooks" / "resume-task-state.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _create_mock_jsonl(tmpdir: str, messages: list) -> str:
    """创建模拟 transcript JSONL 文件"""
    path = os.path.join(tmpdir, "test_transcript.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "system", "message": {"role": "system", "content": "You are helpful."}}) + "\n")
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        f.write(json.dumps({"type": "last-prompt", "lastPrompt": "", "sessionId": "test"}) + "\n")
    return path


# ── 测试用例 ──────────────────────────────────────────────────────────

def test_extract_basic_conversation():
    """测试1：基本 user/assistant 对话提取"""
    save_mod = _import_save_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        messages = [
            {"type": "user", "message": {"role": "user", "content": "请帮我实现 BM25 检索功能"}},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "好的，我来实现 BM25 检索。首先分析当前代码结构。"}
            ]}},
            {"type": "user", "message": {"role": "user", "content": "效果很好，延迟如何？"}},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "实测延迟 3ms，远低于 50ms 约束，验证通过。"}
            ]}},
        ]
        jsonl_path = _create_mock_jsonl(tmpdir, messages)

        result = save_mod._extract_transcript_summary(jsonl_path)

        assert len(result) == 4, f"Expected 4 messages, got {len(result)}"
        assert result[0]["role"] == "user"
        assert "BM25" in result[0]["summary"]
        assert result[1]["role"] == "assistant"
        assert "BM25" in result[1]["summary"]
        assert result[2]["role"] == "user"
        assert "延迟" in result[2]["summary"]
        assert result[3]["role"] == "assistant"
        assert "3ms" in result[3]["summary"]
        print("  PASS test_extract_basic_conversation")


def test_summarize_removes_code_blocks():
    """测试2：_summarize_text 去除代码块"""
    save_mod = _import_save_module()

    text = "我实现了以下功能：\n```python\ndef bm25_search(query):\n    pass\n```\n验证通过，延迟 3ms。"
    result = save_mod._summarize_text(text, max_chars=500)

    assert "```" not in result, "Code block should be removed"
    assert "[代码块]" in result, "Should have code block placeholder"
    assert "3ms" in result, "Should keep conclusion text"
    print("  PASS test_summarize_removes_code_blocks")


def test_summarize_truncation():
    """测试3：长文本首尾策略截断"""
    save_mod = _import_save_module()

    head = "这是开头内容很重要" * 20
    tail = "这是结尾总结也很重要" * 20
    middle = "这是中间不太重要的内容" * 50
    text = f"{head}\n{middle}\n{tail}"

    result = save_mod._summarize_text(text, max_chars=200)

    assert len(result) <= 210, f"Too long: {len(result)}"
    assert "开头" in result, "Should keep head"
    assert "结尾" in result, "Should keep tail"
    assert " ... " in result, "Should have ellipsis"
    print("  PASS test_summarize_truncation")


def test_total_chars_limit():
    """测试4：对话摘要总字符上限控制"""
    save_mod = _import_save_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        messages = []
        for i in range(20):
            messages.append({"type": "user", "message": {
                "role": "user",
                "content": f"这是第 {i+1} 轮对话的用户消息，包含一些技术细节 x" * 5
            }})
            messages.append({"type": "assistant", "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": f"这是第 {i+1} 轮的 AI 回复，分析结果 y" * 5}]
            }})
        jsonl_path = _create_mock_jsonl(tmpdir, messages)

        result = save_mod._extract_transcript_summary(jsonl_path)

        total_chars = sum(len(m["summary"]) for m in result)
        assert total_chars <= save_mod.TRANSCRIPT_MAX_TOTAL_CHARS + 50, \
            f"Total chars {total_chars} exceeds limit {save_mod.TRANSCRIPT_MAX_TOTAL_CHARS}"
        assert len(result) <= save_mod.TRANSCRIPT_MAX_TURNS * 2, \
            f"Too many turns: {len(result)}"
        print(f"  PASS test_total_chars_limit (turns={len(result)}, chars={total_chars})")


def test_empty_transcript():
    """测试5：空/不存在 transcript 的边界情况"""
    save_mod = _import_save_module()

    result = save_mod._extract_transcript_summary("/nonexistent/path.jsonl")
    assert result == [], "Should return empty for nonexistent file"

    result = save_mod._extract_transcript_summary("")
    assert result == [], "Should return empty for empty path"

    result = save_mod._extract_transcript_summary(None)
    assert result == [], "Should return empty for None"

    with tempfile.TemporaryDirectory() as tmpdir:
        empty_path = os.path.join(tmpdir, "empty.jsonl")
        Path(empty_path).write_text("")
        result = save_mod._extract_transcript_summary(empty_path)
        assert result == [], "Should return empty for empty file"

    print("  PASS test_empty_transcript")


def test_tool_use_only_messages_skipped():
    """测试6：只有 tool_use 没有 text 的 assistant 消息被跳过"""
    save_mod = _import_save_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        messages = [
            {"type": "user", "message": {"role": "user", "content": "读取 store.py"}},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tool1", "name": "Read", "input": {"file_path": "store.py"}}
            ]}},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "store.py 有 4000 行代码，主要函数包括..."}
            ]}},
        ]
        jsonl_path = _create_mock_jsonl(tmpdir, messages)

        result = save_mod._extract_transcript_summary(jsonl_path)

        assert len(result) == 2, f"Expected 2 (user + text assistant), got {len(result)}: {result}"
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert "4000" in result[1]["summary"]
        print("  PASS test_tool_use_only_messages_skipped")


def test_postcompact_restore_includes_conversation():
    """测试7：PostCompact 恢复包含对话摘要"""
    resume_mod = _import_resume_module()
    from datetime import datetime, timezone

    state = {
        "swap_out_at": datetime.now(timezone.utc).isoformat(),
        "session_id": "test-session",
        "conversation_summary": [
            {"role": "user", "summary": "帮我优化 BM25 检索延迟"},
            {"role": "assistant", "summary": "分析发现 FTS5 路径延迟已降至 3ms"},
        ],
        "task_progress": [
            {"status": "in_progress", "content": "优化检索延迟"},
        ],
        "decisions": [
            {"summary": "BM25 延迟 3ms，验证通过", "importance": 0.8},
        ],
        "topics": ["BM25优化"],
        "madvise_hints": ["BM25", "FTS5", "延迟"],
        "reasoning_progress": [],
    }

    result = resume_mod._format_restore_context(state)

    assert "最近对话" in result, "Should have conversation section"
    assert "BM25" in result
    assert "用户:" in result, "Should have user prefix"
    assert "AI:" in result, "Should have AI prefix"
    assert "任务进度" in result, "Should have task progress"
    assert "已有结论" in result, "Should have decisions"
    print("  PASS test_postcompact_restore_includes_conversation")


def test_end_to_end_swap_cycle():
    """测试8：端到端 swap out → swap in 循环验证"""
    save_mod = _import_save_module()
    resume_mod = _import_resume_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        messages = [
            {"type": "user", "message": {"role": "user", "content": "实现 CRIU checkpoint 功能"}},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "checkpoint_dump 和 checkpoint_restore 已实现，33 测试通过。"},
            ]}},
            {"type": "user", "message": {"role": "user", "content": "下一步做什么？"}},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "建议实现 Context Pressure Governor，根据上下文压力动态调节注入量。"},
            ]}},
        ]
        jsonl_path = _create_mock_jsonl(tmpdir, messages)

        stdin_data = {
            "session_id": "e2e-test",
            "transcript_path": jsonl_path,
            "cwd": "/test",
        }
        swap_state = save_mod._collect_swap_state(stdin_data)

        assert len(swap_state["conversation_summary"]) == 4, \
            f"Expected 4 transcript turns, got {len(swap_state['conversation_summary'])}"
        assert "CRIU" in swap_state["conversation_summary"][0]["summary"]
        assert "Governor" in swap_state["conversation_summary"][3]["summary"]

        restore_text = resume_mod._format_restore_context(swap_state)

        assert "最近对话" in restore_text
        assert "CRIU" in restore_text or "checkpoint" in restore_text
        assert "Governor" in restore_text or "压力" in restore_text
        print("  PASS test_end_to_end_swap_cycle")


def test_large_jsonl_tail_read():
    """测试9：大文件尾部读取（验证 seek 偏移正确）"""
    save_mod = _import_save_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "large.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(2000):
                f.write(json.dumps({
                    "type": "attachment",
                    "attachment": {"content": "x" * 200},
                    "uuid": f"att-{i}",
                }) + "\n")
            f.write(json.dumps({
                "type": "user",
                "message": {"role": "user", "content": "这是关键用户消息"}
            }) + "\n")
            f.write(json.dumps({
                "type": "assistant",
                "message": {"role": "assistant", "content": [
                    {"type": "text", "text": "这是关键 AI 回复"}
                ]}
            }) + "\n")

        result = save_mod._extract_transcript_summary(path)

        assert len(result) >= 2, f"Should find at least 2 messages from tail, got {len(result)}"
        assert any("关键用户" in m["summary"] for m in result), "Should find key user message"
        assert any("关键 AI" in m["summary"] for m in result), "Should find key AI response"
        print("  PASS test_large_jsonl_tail_read")


# ── run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 迭代56 Transcript-Aware Swap 测试 ===")
    tests = [
        test_extract_basic_conversation,
        test_summarize_removes_code_blocks,
        test_summarize_truncation,
        test_total_chars_limit,
        test_empty_transcript,
        test_tool_use_only_messages_skipped,
        test_postcompact_restore_includes_conversation,
        test_end_to_end_swap_cycle,
        test_large_jsonl_tail_read,
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
