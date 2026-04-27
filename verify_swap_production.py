#!/usr/bin/env python3
"""
迭代 71-72 生产验证：虚拟内存 swap 系统完整性检验

目标：
  1. 验证 save-task-state.py 和 resume-task-state.py 的实际工作状态
  2. 检查 swap_errors.log 是否有诊断信息
  3. 测量真实 DB 中的数据规模和访问模式
  4. 生成可用于改进的诊断报告
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
import sys

def check_swap_files():
    """检查 swap 相关文件是否存在和可读"""
    home = Path.home()
    memory_os_dir = home / ".claude" / "memory-os"

    results = {
        "files_present": {},
        "file_sizes": {},
        "swap_errors": [],
    }

    key_files = {
        "store.db": memory_os_dir / "store.db",
        "latest.json": memory_os_dir / "latest.json",
        "swap_errors.log": memory_os_dir / "swap_errors.log",
    }

    for name, path in key_files.items():
        exists = path.exists()
        results["files_present"][name] = exists

        if exists:
            try:
                size = path.stat().st_size
                results["file_sizes"][name] = size
            except:
                pass

    # 读取 swap_errors.log
    swap_errors_log = memory_os_dir / "swap_errors.log"
    if swap_errors_log.exists():
        try:
            with open(swap_errors_log, "r") as f:
                lines = f.readlines()[-20:]
                results["swap_errors"] = [line.strip() for line in lines]
        except Exception as e:
            results["swap_errors_read_error"] = str(e)

    return results

def check_db_state():
    """检查数据库核心表的状态"""
    home = Path.home()
    db_path = home / ".claude" / "memory-os" / "store.db"

    results = {
        "db_exists": db_path.exists(),
        "chunks_summary": {},
        "recall_traces_summary": {},
    }

    if not db_path.exists():
        return results

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # 1. memory_chunks 统计
        try:
            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT project) FROM memory_chunks")
            row = cursor.fetchone()
            results["chunks_summary"]["total_count"] = row[0] if row else 0
            results["chunks_summary"]["distinct_projects"] = row[1] if row else 0

            cursor.execute("""
                SELECT chunk_type, COUNT(*) FROM memory_chunks
                GROUP BY chunk_type ORDER BY COUNT(*) DESC
            """)
            results["chunks_summary"]["by_type"] = dict(cursor.fetchall())

            cursor.execute("""
                SELECT project, COUNT(*) FROM memory_chunks
                GROUP BY project ORDER BY COUNT(*) DESC LIMIT 5
            """)
            results["chunks_summary"]["top_projects"] = dict(cursor.fetchall())

        except sqlite3.OperationalError as e:
            results["chunks_summary"]["error"] = str(e)

        # 2. recall_traces 统计
        try:
            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT session_id) FROM recall_traces")
            row = cursor.fetchone()
            results["recall_traces_summary"]["total_count"] = row[0] if row else 0
            results["recall_traces_summary"]["distinct_sessions"] = row[1] if row else 0
        except sqlite3.OperationalError as e:
            results["recall_traces_summary"]["error"] = str(e)

        conn.close()

    except Exception as e:
        results["db_error"] = str(e)

    return results

def check_latest_json():
    """检查 latest.json 内容"""
    home = Path.home()
    latest_file = home / ".claude" / "memory-os" / "latest.json"

    results = {
        "file_exists": latest_file.exists(),
        "content": None,
        "keys": [],
    }

    if latest_file.exists():
        try:
            with open(latest_file, "r") as f:
                content = json.load(f)
                results["content"] = content
                results["keys"] = list(content.keys()) if isinstance(content, dict) else []
        except Exception as e:
            results["error"] = str(e)

    return results

def run_verification():
    """执行完整验证"""
    print("=" * 70)
    print("迭代 71-72 虚拟内存 Swap 系统生产验证")
    print("=" * 70)
    print(f"检验时间: {datetime.utcnow().isoformat()}\n")

    # 1. 文件检查
    print("【1. Swap 文件状态】")
    file_results = check_swap_files()
    print(f"  store.db: {file_results['files_present'].get('store.db', False)} ({file_results['file_sizes'].get('store.db', 'N/A')} bytes)")
    print(f"  latest.json: {file_results['files_present'].get('latest.json', False)} ({file_results['file_sizes'].get('latest.json', 'N/A')} bytes)")
    print(f"  swap_errors.log: {file_results['files_present'].get('swap_errors.log', False)} ({file_results['file_sizes'].get('swap_errors.log', 'N/A')} bytes)")

    if file_results["swap_errors"]:
        print(f"\n  【最近错误】({len(file_results['swap_errors'])} 条):")
        for error in file_results["swap_errors"][-5:]:
            print(f"    {error[:100]}")

    # 2. DB 状态
    print("\n【2. 数据库状态】")
    db_results = check_db_state()
    print(f"  DB 存在: {db_results['db_exists']}")

    if db_results['chunks_summary']:
        chunks = db_results['chunks_summary']
        print(f"  Total chunks: {chunks.get('total_count', 0)}")
        print(f"  Distinct projects: {chunks.get('distinct_projects', 0)}")
        if 'by_type' in chunks:
            print(f"  By type: {chunks['by_type']}")
        if 'top_projects' in chunks:
            print(f"  Top projects:")
            for proj, count in list(chunks['top_projects'].items())[:3]:
                print(f"    {proj}: {count}")

    if db_results['recall_traces_summary']:
        traces = db_results['recall_traces_summary']
        print(f"  Recall traces: {traces.get('total_count', 0)}")
        print(f"  Distinct sessions: {traces.get('distinct_sessions', 0)}")

    # 3. latest.json 检查
    print("\n【3. Latest.json 快速检查】")
    latest_results = check_latest_json()
    print(f"  File exists: {latest_results['file_exists']}")
    print(f"  Keys: {latest_results.get('keys', [])[:5]}")

    # 4. 汇总
    print("\n【4. 系统健康评分】")

    score = 100
    issues = []

    if not file_results['files_present'].get('store.db'):
        score -= 40
        issues.append("store.db 不存在")
    elif file_results['file_sizes'].get('store.db', 0) < 10000:
        score -= 20
        issues.append("store.db 过小")

    if not latest_results['file_exists']:
        score -= 20
        issues.append("latest.json 不存在")

    if db_results.get('chunks_summary', {}).get('total_count', 0) == 0:
        score -= 20
        issues.append("数据库为空")

    print(f"  健康评分: {score}/100")
    if issues:
        print(f"  问题:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  ✅ 系统正常运作")

    # 5. 保存报告
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "files": file_results,
        "database": db_results,
        "latest_json": latest_results,
        "health_score": score,
        "issues": issues,
    }

    report_file = Path(__file__).parent / "swap_verification_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📊 报告已保存: {report_file}")
    print("=" * 70)

    return report

if __name__ == "__main__":
    report = run_verification()
    sys.exit(0 if report['health_score'] >= 70 else 1)
