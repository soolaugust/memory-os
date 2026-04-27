#!/usr/bin/env python3
"""
LongMemEval Benchmark Adapter for AIOS Memory-OS

流程：
  1. Ingest: 将 haystack_sessions 写入临时 store.db（每个 session → 1 个 MemoryChunk）
  2. Retrieve: 用 fts_search 检索 top-k 相关 chunk
  3. Generate: 用 LLM 基于检索结果回答问题
  4. Output: JSONL 格式供官方 evaluate_qa.py 评分

用法：
  # 先跑 oracle 版（只有证据 session，测纯生成能力上界）
  python run_benchmark.py --dataset oracle --limit 50

  # 跑 S 版（~48 sessions/question，测检索 + 生成）
  python run_benchmark.py --dataset s --limit 50

  # 跑 retrieval-only（不调 LLM，只测检索召回率）
  python run_benchmark.py --dataset s --limit 500 --retrieval-only
"""

import sys
import json
import os
import sqlite3
import argparse
import time
import uuid
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# 添加 memory-os 根路径
_BENCH_DIR = Path(__file__).resolve().parent
_MOS_ROOT = _BENCH_DIR.parent.parent
sys.path.insert(0, str(_MOS_ROOT))


from store_vfs import open_db, ensure_schema, fts_search, insert_chunk, _cjk_tokenize


def create_isolated_db(tmpdir):
    """创建隔离的临时数据库（直接传 path，不改全局 env）"""
    db_path = tmpdir / "bench.db"
    conn = open_db(db_path)
    ensure_schema(conn)
    return conn


def ingest_sessions(conn, sessions, dates, session_ids, project_id):
    """
    将 LongMemEval 的 haystack_sessions 写入 store.db。

    每个 session → 1 个 MemoryChunk:
      - summary: session 的前两个 turn 的关键内容（截断到 200 字符）
      - content: 完整对话文本（截断到 2000 字符）
      - chunk_type: "conversation"
      - importance: 0.7（统一）
    """
    # insert_chunk imported at module level from store_vfs

    now = datetime.now(timezone.utc).isoformat()

    for i, (session, date_str, sid) in enumerate(zip(sessions, dates, session_ids)):
        # 拼接对话文本（纯净，不注入日期——日期存在 created_at metadata 中）
        turns = []
        for msg in session:
            role = msg["role"]
            content = msg["content"]
            turns.append(f"{role}: {content}")
        full_text = "\n".join(turns)

        # 构造 summary：user 的第一句话（纯净，不含日期，避免污染 BM25）
        user_msgs = [m["content"] for m in session if m["role"] == "user"]
        summary = user_msgs[0][:200] if user_msgs else full_text[:200]

        chunk = {
            "id": sid,
            "created_at": date_str,
            "updated_at": now,
            "project": project_id,
            "source_session": f"lme_session_{i}",
            "chunk_type": "conversation",
            "content": full_text[:24000],
            "summary": summary,
            "tags": json.dumps(["longmemeval"]),
            "importance": 0.7,
            "retrievability": 0.5,
            "last_accessed": now,
            "feishu_url": None,
            "access_count": 0,
            "oom_adj": 0,
            "lru_gen": 0,
        }
        insert_chunk(conn, chunk)

    conn.commit()


def retrieve_for_question(conn, question, project_id, top_k=10):
    """用 fts_search 检索相关 session chunk"""
    results = fts_search(conn, question, project_id, top_k=top_k)
    return results


def format_context(retrieved_chunks, max_chars=80000, include_dates=False):
    """将检索到的 chunk 格式化为 LLM 上下文。

    include_dates=True 时按时间排序并加日期 header（temporal 用）。
    include_dates=False 时保持 BM25 相关性排序（其他类型用，避免 lost-in-the-middle）。
    """
    if include_dates:
        chunks = sorted(retrieved_chunks, key=lambda c: c.get("created_at", ""))
    else:
        chunks = retrieved_chunks  # 保持 fts_search 返回的 BM25 相关性排序

    context_parts = []
    total = 0
    for chunk in chunks:
        text = chunk.get("content", chunk.get("summary", ""))
        if include_dates:
            date = chunk.get("created_at", "")
            if date:
                text = f"[Conversation date: {date}]\n{text}"
        if total + len(text) > max_chars:
            text = text[:max_chars - total]
        context_parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n---\n".join(context_parts)


def generate_answer_llm(question, context, question_date, model="claude-haiku", question_type=""):
    """调用 LLM 生成答案（支持 Anthropic Claude 和 OpenAI）"""

    # 按 question_type 使用完全不同的 prompt
    if question_type == "temporal-reasoning":
        prompt = f"""You are answering a question about past conversations. Use the provided conversation history to answer accurately and concisely.

Current date: {question_date}

Each conversation segment starts with [Conversation date: ...] showing when it took place. Use these timestamps to:
- Calculate time differences (days, weeks, months) between events
- Determine chronological order of events
- Answer "how long ago" questions by comparing conversation dates to the current date
- When someone says "yesterday", "last week", "today", interpret it relative to THAT conversation's date

Relevant conversation history:
{context}

Question: {question}

Show your date calculation briefly, then give the answer. Only say "I don't have enough information" if the events are truly not mentioned at all.

Answer:"""
    else:
        # 非 temporal 类型：用 v1 的原始简洁 prompt（已验证 44.4% 最佳）
        prompt = f"""You are answering a question about past conversations. Use the provided conversation history to answer accurately and concisely.

Current date: {question_date}

Relevant conversation history:
{context}

Question: {question}

Answer concisely based only on the conversation history above. If the information is not available, say "I don't have enough information to answer this question."
"""

    if model.startswith("claude") or model.startswith("ppio"):
        import anthropic
        client = anthropic.Anthropic()
        # 映射 model 简称
        model_id = {
            "claude-haiku": "claude-haiku-4-5-20251001",
            "claude-sonnet": os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-6"),
        }.get(model, model)
        response = client.messages.create(
            model=model_id,
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    else:
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()


def compute_retrieval_metrics(retrieved_ids, answer_session_ids):
    """计算 session 级别的检索指标"""
    if not answer_session_ids:
        return {"recall": None, "precision": None, "hit": False}

    answer_set = set(answer_session_ids)
    retrieved_set = set(retrieved_ids)
    hits = answer_set & retrieved_set

    recall = len(hits) / len(answer_set) if answer_set else 0
    precision = len(hits) / len(retrieved_set) if retrieved_set else 0

    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "hit": len(hits) > 0,
        "hits": len(hits),
        "expected": len(answer_set),
        "retrieved": len(retrieved_set),
    }


def run_benchmark(args):
    """主入口"""
    # 加载数据集
    if args.dataset == "oracle":
        data_path = _BENCH_DIR / "data" / "longmemeval_oracle.json"
    else:
        data_path = _BENCH_DIR / "data" / "longmemeval_s.json"

    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Download from HuggingFace first.")
        sys.exit(1)

    print(f"Loading {data_path.name}...")
    with open(data_path) as f:
        dataset = json.load(f)

    if args.type_filter:
        dataset = [q for q in dataset if q["question_type"] == args.type_filter]
    if args.limit:
        dataset = dataset[:args.limit]

    print(f"Questions: {len(dataset)}")
    print(f"Mode: {'retrieval-only' if args.retrieval_only else 'full (retrieve + generate)'}")
    print(f"Dataset: {args.dataset}")
    if not args.retrieval_only:
        print(f"LLM: {args.model}")
    print()

    # 结果容器
    outputs = []  # JSONL 输出
    retrieval_stats = defaultdict(list)
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0, "recall": []})
    errors = []

    start_time = time.time()

    for qi, q in enumerate(dataset):
        qid = q["question_id"]
        qtype = q["question_type"]
        question = q["question"]
        answer = q["answer"]
        sessions = q["haystack_sessions"]
        dates = q["haystack_dates"]
        session_ids = q["haystack_session_ids"]
        answer_sids = q.get("answer_session_ids", [])
        question_date = q.get("question_date", "")

        project_id = f"lme_{qid}"

        # 创建隔离的临时 DB
        tmpdir = Path(tempfile.mkdtemp(prefix="lme_"))
        try:
            conn = create_isolated_db(tmpdir)

            # 1. Ingest
            ingest_sessions(conn, sessions, dates, session_ids, project_id)

            # 2. Retrieve
            retrieved = retrieve_for_question(conn, question, project_id, top_k=args.top_k)
            retrieved_ids = [r["id"] for r in retrieved]

            # 3. 检索指标
            r_metrics = compute_retrieval_metrics(retrieved_ids, answer_sids)
            if r_metrics["recall"] is not None:
                retrieval_stats["recall"].append(r_metrics["recall"])
                retrieval_stats["hit"].append(1 if r_metrics["hit"] else 0)
                type_stats[qtype]["recall"].append(r_metrics["recall"])
            type_stats[qtype]["total"] += 1

            # 4. Generate (如果不是 retrieval-only)
            hypothesis = ""
            if not args.retrieval_only:
                context = format_context(retrieved, max_chars=args.max_context,
                                        include_dates=(qtype == "temporal-reasoning"))
                try:
                    hypothesis = generate_answer_llm(question, context, question_date, args.model, qtype)
                except Exception as e:
                    hypothesis = f"Error: {e}"
                    errors.append({"qid": qid, "error": str(e)})

            outputs.append({
                "question_id": qid,
                "hypothesis": hypothesis,
                "retrieval_recall": r_metrics["recall"],
                "retrieval_ids": retrieved_ids[:5],
            })

            conn.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # 进度
        if (qi + 1) % 10 == 0 or qi == len(dataset) - 1:
            elapsed = time.time() - start_time
            avg_recall = sum(retrieval_stats["recall"]) / len(retrieval_stats["recall"]) if retrieval_stats["recall"] else 0
            print(f"  [{qi+1}/{len(dataset)}] elapsed={elapsed:.0f}s avg_recall={avg_recall:.3f}")

    elapsed = time.time() - start_time

    # 输出结果
    print(f"\n{'='*60}")
    print(f"RESULTS ({args.dataset}, {len(dataset)} questions, {elapsed:.0f}s)")
    print(f"{'='*60}\n")

    # 检索指标
    avg_recall = sum(retrieval_stats["recall"]) / len(retrieval_stats["recall"]) if retrieval_stats["recall"] else 0
    hit_rate = sum(retrieval_stats["hit"]) / len(retrieval_stats["hit"]) if retrieval_stats["hit"] else 0

    print(f"Retrieval:")
    print(f"  Session Recall@{args.top_k}: {avg_recall:.3f}")
    print(f"  Hit Rate:                {hit_rate:.3f}")
    print()

    print(f"By Question Type:")
    for qtype in sorted(type_stats.keys()):
        data = type_stats[qtype]
        avg_r = sum(data["recall"]) / len(data["recall"]) if data["recall"] else 0
        print(f"  {qtype:30s}: recall={avg_r:.3f} (N={data['total']})")
    print()

    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  {e['qid']}: {e['error'][:80]}")

    # 保存 JSONL（供官方评估）
    out_dir = _BENCH_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "retrieval" if args.retrieval_only else "full"
    out_path = out_dir / f"lme_{args.dataset}_{mode}_{timestamp}.jsonl"

    with open(out_path, "w") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"Output: {out_path}")

    # 保存汇总
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "mode": mode,
        "questions": len(dataset),
        "top_k": args.top_k,
        "elapsed_seconds": round(elapsed, 1),
        "retrieval": {
            "avg_session_recall": round(avg_recall, 3),
            "hit_rate": round(hit_rate, 3),
        },
        "by_type": {
            qtype: {
                "recall": round(sum(d["recall"]) / len(d["recall"]), 3) if d["recall"] else 0,
                "count": d["total"],
            }
            for qtype, d in sorted(type_stats.items())
        },
        "errors": len(errors),
    }

    summary_path = out_dir / f"lme_{args.dataset}_{mode}_{timestamp}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Benchmark for AIOS Memory-OS")
    parser.add_argument("--dataset", choices=["oracle", "s"], default="oracle",
                       help="oracle=证据session only, s=~48 sessions/question")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit to first N questions (for pilot testing)")
    parser.add_argument("--type-filter", default=None,
                       help="Filter by question_type (e.g. temporal-reasoning)")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of chunks to retrieve")
    parser.add_argument("--retrieval-only", action="store_true",
                       help="Only test retrieval, skip LLM generation")
    parser.add_argument("--model", default="claude-haiku",
                       help="LLM model for answer generation (claude-haiku, claude-sonnet, gpt-4o-mini, etc.)")
    parser.add_argument("--max-context", type=int, default=80000,
                       help="Max chars for retrieved context")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
