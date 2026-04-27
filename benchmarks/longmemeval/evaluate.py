#!/usr/bin/env python3
"""
evaluate.py — LongMemEval QA 评估器（Claude-as-Judge）

对标官方 evaluate_qa.py 的评估协议：
  - 二元判定：yes/no
  - 按 question_type 分别使用不同 prompt
  - temporal-reasoning 允许 off-by-one
  - knowledge-update 只要包含最新答案即算对
  - abstention 只要正确拒绝回答即算对

用法：
  python evaluate.py results/lme_s_full_XXXX.jsonl data/longmemeval_s.json
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import anthropic


JUDGE_PROMPT_TEMPLATE = """You are evaluating whether an AI assistant's answer correctly addresses a question about past conversations.

Question type: {question_type}
Question: {question}
Reference answer: {reference}
Assistant's answer: {hypothesis}

{type_specific_instruction}

Based on the above, does the assistant's answer correctly address the question?
Answer with ONLY "yes" or "no"."""

TYPE_INSTRUCTIONS = {
    "single-session-user": "The answer should contain the key information from the reference. Minor wording differences are acceptable.",
    "single-session-assistant": "The answer should contain the key information from the reference. Minor wording differences are acceptable.",
    "single-session-preference": "The answer should capture the user's preference as described in the reference. It does not need to be an exact match.",
    "temporal-reasoning": "The answer should be temporally correct. An off-by-one error in day counts is acceptable. The key temporal relationship or date must be correct.",
    "knowledge-update": "The answer should contain the UPDATED/LATEST information. If the answer mentions both old and new information but correctly identifies the latest, it is still correct.",
    "multi-session": "The answer should synthesize information correctly from multiple sessions. The key facts must be present.",
}


def evaluate_one(client, question, reference, hypothesis, question_type, model):
    """评估单个回答"""
    # 确保 reference 和 hypothesis 是字符串
    reference = str(reference)
    hypothesis = str(hypothesis)

    # abstention 问题
    is_abstention = False
    if "don't have" in hypothesis.lower() or "not available" in hypothesis.lower() or "cannot answer" in hypothesis.lower() or "no information" in hypothesis.lower():
        is_abstention = True

    # 对于 abstention 类型（question_id 以 _abs 结尾），正确拒绝 = correct
    # 但我们这里没有 question_id，所以靠 reference 中的 "unanswerable" 标记
    if "unanswerable" in reference.lower() or "not mentioned" in reference.lower():
        if is_abstention:
            return "yes"
        else:
            return "no"

    # 对于非 abstention 问题，如果模型拒绝回答 → wrong
    if is_abstention:
        return "no"

    type_instruction = TYPE_INSTRUCTIONS.get(question_type, TYPE_INSTRUCTIONS["single-session-user"])

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question_type=question_type,
        question=question,
        reference=reference,
        hypothesis=hypothesis,
        type_specific_instruction=type_instruction,
    )

    response = client.messages.create(
        model=model,
        max_tokens=10,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    result = response.content[0].text.strip().lower()
    return "yes" if result.startswith("yes") else "no"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="JSONL file with predictions")
    parser.add_argument("dataset", help="JSON file with ground truth")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                       help="Judge model")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # 加载预测
    predictions = {}
    with open(args.predictions) as f:
        for line in f:
            d = json.loads(line)
            predictions[d["question_id"]] = d

    # 加载数据集
    with open(args.dataset) as f:
        dataset = json.load(f)

    if args.limit:
        dataset = dataset[:args.limit]

    print(f"Evaluating {len(dataset)} questions with {args.model}")
    print(f"Predictions: {len(predictions)}")
    print()

    client = anthropic.Anthropic()

    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_evaluated = 0
    results = []

    start = time.time()

    for i, q in enumerate(dataset):
        qid = q["question_id"]
        qtype = q["question_type"]

        if qid not in predictions:
            continue

        pred = predictions[qid]
        hypothesis = pred.get("hypothesis", "")

        if not hypothesis:
            continue

        try:
            verdict = evaluate_one(
                client, q["question"], q["answer"], hypothesis, qtype, args.model
            )
        except Exception as e:
            print(f"  ERROR {qid}: {e}")
            verdict = "no"

        is_correct = verdict == "yes"
        total_correct += int(is_correct)
        total_evaluated += 1
        type_stats[qtype]["correct"] += int(is_correct)
        type_stats[qtype]["total"] += 1

        results.append({
            "question_id": qid,
            "question_type": qtype,
            "verdict": verdict,
            "hypothesis": hypothesis[:200],
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            acc = total_correct / total_evaluated if total_evaluated else 0
            print(f"  [{i+1}/{len(dataset)}] acc={acc:.1%} elapsed={elapsed:.0f}s")

    elapsed = time.time() - start

    overall_acc = total_correct / total_evaluated if total_evaluated else 0

    print(f"\n{'='*60}")
    print(f"RESULTS ({total_evaluated} questions, {elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"\n  Overall Accuracy: {overall_acc:.1%} ({total_correct}/{total_evaluated})\n")

    print(f"  By Question Type:")
    for qtype in sorted(type_stats.keys()):
        s = type_stats[qtype]
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"    {qtype:30s}: {acc:.1%} ({s['correct']}/{s['total']})")

    # 保存
    out_path = Path(args.predictions).with_suffix(".eval.json")
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "judge_model": args.model,
        "overall_accuracy": round(overall_acc, 4),
        "total_correct": total_correct,
        "total_evaluated": total_evaluated,
        "by_type": {
            qtype: {
                "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0,
                "correct": s["correct"],
                "total": s["total"],
            }
            for qtype, s in sorted(type_stats.items())
        },
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
