#!/usr/bin/env bash
# Memory-OS 虚拟内存迭代脚本 — 聚焦上下文溢出问题
# 启动: nohup bash iterate-vm.sh >> /tmp/memory-os-vm.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FEISHU_DOC="${FEISHU_DOC:?'请设置环境变量 FEISHU_DOC=<飞书文档ID>'}"
PROJECT_MD="${PROJECT_MD:-"$HOME/self-improving/projects/memory-os.md"}"
WORKSPACE="${WORKSPACE:-"$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"}"
MAX_ROUNDS=30
FAIL_COUNT=0
MAX_FAILS=3

echo "========================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Memory-OS 虚拟内存迭代启动"
echo "========================================"

ROUND=0
while [ $ROUND -lt $MAX_ROUNDS ]; do
    ROUND=$((ROUND + 1))
    echo ""
    echo "========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 迭代轮次 #${ROUND} 开始"
    echo "========================================"

    if claude -p --dangerously-skip-permissions --model ppio/pa/claude-opus-4-6 \
        "你是 memory-os 虚拟内存迭代 agent。工作目录:${WORKSPACE}。所有决策默认是，不停顿。

## 核心使命
解决 AI 上下文窗口的根本痛点：上下文溢出导致 compaction 信息丢失，让用户感觉上下文无限。
本质是实现 OS 虚拟内存：swap out（compaction 前保存）→ swap in（compaction 后恢复）→ 对用户透明。

## 关键技术事实
- settings.json 已有 PreCompact 和 PostCompact hook 事件
- PreCompact → ~/.claude/hooks/save-task-state.py（文件不存在，需要创建）
- PostCompact → ~/.claude/hooks/resume-task-state.py（文件不存在，需要创建）
- PreCompact stdin 格式未知，第一步需要写探针确认
- memory-os 代码在 ${SCRIPT_DIR}/，store.py 有完整的 CRUD API
- 已有 CRIU checkpoint 机制（迭代49），可以复用
- 飞书文档 token: ${FEISHU_DOC}
- 项目状态: ${PROJECT_MD}，当前到迭代 52

## 迭代策略
每轮选择以下方向中最高价值的一个：

### 阶段 1：评测基础设施（必须最先做）
- 定义上下文连续性评分：compaction 前后关键信息保留率
- 创建 test_virtual_memory.py 评测套件
- 基线测量：当前无 swap 时的信息丢失率

### 阶段 2：PreCompact Swap Out
- 写探针到 ~/.claude/hooks/save-task-state.py 确认 stdin 格式
- 实现真正的 swap out：从 stdin 提取对话状态，保存到 store.db
- 保存内容：当前话题、关键结论、推理链进度、涉及文件、下一步计划

### 阶段 3：PostCompact Swap In
- ~/.claude/hooks/resume-task-state.py 从 store.db 恢复关键上下文
- 通过 additionalContext 注入回上下文窗口
- 目标：compaction 对用户透明

### 阶段 4：Context Pressure Governor
- 估算上下文压力（基于对话轮数/注入量）
- 压力低时多注入历史知识，压力高时精简
- 动态平衡信息密度和剩余空间

### 阶段 5：持续探索 OS 特性
- 写时复制 CoW、NUMA 感知、透明大页等
- 每个特性都要有评测证明收益

## 执行步骤
1. 读 ${PROJECT_MD} 了解已完成的迭代和上下文
2. 读 ${SCRIPT_DIR}/hooks/*.py 和 store.py 了解代码现状
3. 选择当前阶段最高价值方向
4. 实现并验证（必须有测试）
5. feishu docx append ${FEISHU_DOC} 追加结果
6. 更新 ${PROJECT_MD}
7. 打印摘要

开始。" 2>&1 | tail -80; then
        FAIL_COUNT=0
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 轮次 #${ROUND} 成功"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 轮次 #${ROUND} 失败 (${FAIL_COUNT}/${MAX_FAILS})"
        if [ $FAIL_COUNT -ge $MAX_FAILS ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 连续失败 ${MAX_FAILS} 次，停止"
            break
        fi
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 休息 60 秒..."
    sleep 60
done

echo ""
echo "========================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 迭代结束，共 ${ROUND} 轮"
echo "========================================"
