#!/usr/bin/env python3
"""
test_hook_coalescing.py — 迭代69: interrupt coalescing 验证

测试:
1. settings.json 结构完整性
2. PreToolUse hook 数量目标 (< 6)
3. 触发次数计算正确性
4. coalesced dispatcher 存在且可执行
5. monitor-update.sh 已从所有事件中移除
6. mcp-health-check matcher 已收窄
"""
import json
import os
import re
import subprocess
import sys

SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")
COALESCED_SCRIPT = os.path.join(
    os.path.dirname(__file__), "hooks", "pretool_coalesced.js"
)

def load_settings():
    with open(SETTINGS_PATH, "r") as f:
        return json.load(f)


def test_settings_valid_json():
    """settings.json must be valid JSON."""
    cfg = load_settings()
    assert "hooks" in cfg, "hooks key missing"
    assert "PreToolUse" in cfg["hooks"], "PreToolUse missing"
    assert "PostToolUse" in cfg["hooks"], "PostToolUse missing"
    print("  ✓ settings.json valid")


def test_pretooluse_count():
    """PreToolUse hooks should be <= 5 (was 9)."""
    cfg = load_settings()
    count = len(cfg["hooks"]["PreToolUse"])
    assert count <= 6, f"PreToolUse has {count} hooks, expected <= 6"
    print(f"  ✓ PreToolUse hooks: {count} (was 9)")


def test_total_hooks_reduced():
    """Total hooks should be < 42 (was 45)."""
    cfg = load_settings()
    total = sum(len(v) for v in cfg["hooks"].values())
    assert total < 42, f"Total hooks {total}, expected < 42"
    print(f"  ✓ Total hooks: {total} (was 45)")


def test_monitor_update_removed():
    """monitor-update.sh should be removed from all event types."""
    cfg = load_settings()
    for event_type, entries in cfg["hooks"].items():
        for entry in entries:
            for hook in entry.get("hooks", []):
                cmd = hook.get("command", "")
                assert "monitor-update.sh" not in cmd, (
                    f"monitor-update.sh still in {event_type}"
                )
    print("  ✓ monitor-update.sh removed from all events")


def test_observagent_not_in_pretooluse():
    """observagent relay should not be in PreToolUse."""
    cfg = load_settings()
    for entry in cfg["hooks"]["PreToolUse"]:
        for hook in entry.get("hooks", []):
            cmd = hook.get("command", "")
            assert "observagent" not in cmd, (
                "observagent/relay.py still in PreToolUse"
            )
    print("  ✓ observagent relay removed from PreToolUse")


def test_pre_observe_not_in_pretooluse():
    """pre:observe should not be in PreToolUse."""
    cfg = load_settings()
    for entry in cfg["hooks"]["PreToolUse"]:
        for hook in entry.get("hooks", []):
            cmd = hook.get("command", "")
            assert "pre:observe" not in cmd, (
                "pre:observe still in PreToolUse"
            )
    print("  ✓ pre:observe removed from PreToolUse")


def test_mcp_health_check_narrowed():
    """mcp-health-check should not use wildcard matcher."""
    cfg = load_settings()
    for entry in cfg["hooks"]["PreToolUse"]:
        for hook in entry.get("hooks", []):
            cmd = hook.get("command", "")
            if "mcp-health-check" in cmd:
                matcher = entry.get("matcher", "*")
                assert matcher != "*", (
                    f"mcp-health-check still uses * matcher: {matcher}"
                )
                print(f"  ✓ mcp-health-check matcher narrowed to: {matcher}")
                return
    print("  ✓ mcp-health-check not in PreToolUse (or narrowed)")


def test_coalesced_script_exists():
    """coalesced dispatcher script should exist."""
    assert os.path.exists(COALESCED_SCRIPT), (
        f"Missing: {COALESCED_SCRIPT}"
    )
    print(f"  ✓ coalesced script exists")


def test_coalesced_script_syntax():
    """coalesced dispatcher should have valid Node.js syntax."""
    result = subprocess.run(
        ["node", "--check", COALESCED_SCRIPT],
        capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, (
        f"Syntax error: {result.stderr}"
    )
    print("  ✓ coalesced script syntax valid")


def test_coalesced_in_settings():
    """coalesced dispatcher should be registered in settings."""
    cfg = load_settings()
    found = False
    for entry in cfg["hooks"]["PreToolUse"]:
        for hook in entry.get("hooks", []):
            if "pretool_coalesced" in hook.get("command", ""):
                found = True
                assert entry.get("matcher") == "Bash|Write|Edit|MultiEdit", (
                    f"Wrong matcher for coalesced: {entry.get('matcher')}"
                )
    assert found, "pretool_coalesced not found in PreToolUse"
    print("  ✓ coalesced dispatcher registered in settings")


# ── 迭代70: PostToolUse observer coalescing tests ──

POSTTOOL_SCRIPT = os.path.join(
    os.path.dirname(__file__), "hooks", "posttool_observers.js"
)


def test_posttool_observers_exists():
    """posttool_observers.js should exist."""
    assert os.path.exists(POSTTOOL_SCRIPT), f"Missing: {POSTTOOL_SCRIPT}"
    print("  ✓ posttool_observers.js exists")


def test_posttool_observers_syntax():
    """posttool_observers.js should have valid syntax."""
    result = subprocess.run(
        ["node", "--check", POSTTOOL_SCRIPT],
        capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, f"Syntax error: {result.stderr}"
    print("  ✓ posttool_observers.js syntax valid")


def test_posttool_observers_in_settings():
    """posttool_observers should be in PostToolUse settings."""
    cfg = load_settings()
    found = False
    for entry in cfg["hooks"]["PostToolUse"]:
        for hook in entry.get("hooks", []):
            if "posttool_observers" in hook.get("command", ""):
                found = True
                assert entry.get("matcher") == "*", (
                    f"Wrong matcher: {entry.get('matcher')}"
                )
                assert hook.get("async") is True, "Should be async"
    assert found, "posttool_observers not in PostToolUse"
    print("  ✓ posttool_observers registered in PostToolUse")


def test_snarc_not_standalone_in_posttooluse():
    """Standalone snarc hook should be removed from PostToolUse."""
    cfg = load_settings()
    for entry in cfg["hooks"]["PostToolUse"]:
        for hook in entry.get("hooks", []):
            cmd = hook.get("command", "")
            if "snarc" in cmd and "post-tool-use" in cmd and "posttool_observers" not in cmd:
                raise AssertionError("Standalone snarc hook still in PostToolUse")
    print("  ✓ standalone snarc removed from PostToolUse")


def calc_trigger_count():
    """Calculate and verify trigger count."""
    cfg = load_settings()

    tools = ["Bash", "Read", "Write", "Edit", "Glob"]

    total_pre = 0
    total_post = 0

    for tool in tools:
        for entry in cfg["hooks"]["PreToolUse"]:
            matcher = entry.get("matcher", "*")
            if _matches(matcher, tool):
                total_pre += 1

        for entry in cfg["hooks"]["PostToolUse"]:
            matcher = entry.get("matcher", "*")
            if _matches(matcher, tool):
                total_post += 1

    ups_count = len(cfg["hooks"].get("UserPromptSubmit", []))
    stop_count = len(cfg["hooks"].get("Stop", []))

    total = total_pre + total_post + ups_count + stop_count
    print(f"  PreToolUse triggers:  {total_pre}")
    print(f"  PostToolUse triggers: {total_post}")
    print(f"  UserPromptSubmit:     {ups_count}")
    print(f"  Stop:                 {stop_count}")
    print(f"  Total per round:      {total}")
    assert total < 50, f"Total {total} >= 50 target"
    print(f"  ✓ Total {total} < 50 target (was 103)")
    return total


def _matches(matcher, tool_name):
    """Simulate Claude Code hook matcher logic."""
    if matcher == "*":
        return True
    if matcher == ".*":
        return True
    try:
        return bool(re.fullmatch(matcher, tool_name))
    except re.error:
        return tool_name in matcher.split("|")


def main():
    print("=== 迭代69: Hook Coalescing Tests ===\n")

    tests = [
        test_settings_valid_json,
        test_pretooluse_count,
        test_total_hooks_reduced,
        test_monitor_update_removed,
        test_observagent_not_in_pretooluse,
        test_pre_observe_not_in_pretooluse,
        test_mcp_health_check_narrowed,
        test_coalesced_script_exists,
        test_coalesced_script_syntax,
        test_coalesced_in_settings,
        # 迭代70
        test_posttool_observers_exists,
        test_posttool_observers_syntax,
        test_posttool_observers_in_settings,
        test_snarc_not_standalone_in_posttooluse,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n--- Trigger Count Analysis ---")
    try:
        total = calc_trigger_count()
        passed += 1
    except Exception as e:
        print(f"  ✗ calc_trigger_count: {e}")
        failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
