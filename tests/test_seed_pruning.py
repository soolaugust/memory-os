#!/usr/bin/env python3
"""
迭代79 测试：_is_quality_chunk() V8 Seed Pruning + Snapshot Filter
拦截纯状态快照和模糊方向声明
OS 类比：do_exit() → exit_mmap() — 进程退出时释放不再需要的页面
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import tmpfs  # noqa: F401 — 测试隔离

from hooks.extractor import _is_quality_chunk


if __name__ == "__main__":
    passed = failed = 0

    def check(name, actual, expected):
        global passed, failed
        if actual == expected:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}: expected {expected}, got {actual}")


    print("=== V4: 纯状态快照 ===")
    # 应被拦截（return False）— 这些是 point-in-time 数据，不是可复用决策
    check("data_scale_snapshot",
          _is_quality_chunk("数据规模：28 chunks（经过多轮 GC/hygiene 清洗后的精炼数据），88 条 recall traces"),
          False)
    check("current_status",
          _is_quality_chunk("当前状态：34 chunks, 零访问率 35.3%"),
          False)
    check("system_status",
          _is_quality_chunk("系统状态：PSI SOME, retriever 正常运行"),
          False)
    check("chunk_count_status",
          _is_quality_chunk("chunk数：27个，类型分布健康，decision 占比 55%"),
          False)
    check("stats_cn",
          _is_quality_chunk("统计：零访问率从 56% 降至 44%，清理了 10 条"),
          False)
    check("current_status_en",
          _is_quality_chunk("现状：extractor 写入正常，retriever 延迟稳定"),
          False)

    # 不应被拦截（含决策动词的状态描述 → 保留）
    check("scale_with_decision",
          _is_quality_chunk("数据规模：选择 cgroup 配额限制 200 chunks/project"),
          True)
    check("status_with_recommendation",
          _is_quality_chunk("当前状态：推荐将 kswapd 水位提高到 85%，因为写入频率增加"),
          True)

    print("\n=== V5: 模糊方向声明（无技术锚点）===")
    # 应被拦截 — 战略口号/方向性讨论，无具体技术锚点
    check("vague_direction_1",
          _is_quality_chunk("纵向深化 — 真正解决 AI 使用的体感痛点"),
          False)
    check("vague_direction_2",
          _is_quality_chunk("横向扩展 — 从 Memory 到完整 AIOS"),
          False)
    check("vague_direction_3",
          _is_quality_chunk("精简重构 — Less is More"),
          False)
    check("vague_direction_4",
          _is_quality_chunk("持续优化 — 追求极致性能"),
          False)
    check("vague_direction_5",
          _is_quality_chunk("架构升级 — 从单体到微服务"),
          False)
    check("vague_direction_6",
          _is_quality_chunk("深度学习 — 未来的方向"),
          False)

    # 不应被拦截 — 含具体技术锚点（数字/文件/代码标识）
    check("direction_with_metric",
          _is_quality_chunk("冷启动恢复 — 解决 /clear 后 5ms 延迟问题"),
          True)
    check("direction_with_percentage",
          _is_quality_chunk("零访问率 56.8%→44.4% — 删除 10 条重复后的改善"),
          True)
    check("direction_with_file",
          _is_quality_chunk("模块重构 — 将 retriever.py 的 BM25 逻辑提取到 bm25.py"),
          True)
    check("direction_with_code_ref",
          _is_quality_chunk("性能优化 — `kswapd_scan()` 从 O(N) 降到 O(log N)"),
          True)
    check("direction_with_arrow_num",
          _is_quality_chunk("Hook 合并 — 触发次数 103→34次/轮"),
          True)

    print("\n=== 边界测试 ===")
    # 正常决策不受影响
    check("normal_decision_1",
          _is_quality_chunk("选择 FTS5 替代 Python BM25 全表扫描"),
          True)
    check("normal_decision_2",
          _is_quality_chunk("采用 WAL 模式提升并发读写性能"),
          True)
    check("normal_decision_3",
          _is_quality_chunk("连接池在 hook subprocess 中无意义，改用 per-request scope"),
          True)
    check("quantitative_conclusion",
          _is_quality_chunk("召回延迟 1.35ms → 0.71ms（-47%）"),
          True)
    check("extractor_action",
          _is_quality_chunk("extractor.py 新增 _is_fragment() 碎片检测"),
          True)
    # 短 em-dash 也应被检测
    check("en_dash_vague",
          _is_quality_chunk("全面升级 – 提升用户体验"),
          False)
    # 无 dash 的正常句子不受影响
    check("no_dash_normal",
          _is_quality_chunk("选择 BM25 因为短文档效果更好，精度提升 15%"),
          True)

    print(f"\n{'='*50}")
    print(f"结果：{passed}/{passed+failed} 通过")
    if failed:
        print(f"FAILED ({failed} failures)")
        sys.exit(1)
    else:
        print("ALL PASSED")
