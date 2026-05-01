#!/usr/bin/env node
/**
 * filesize_guard.js — PreToolUse 大文件 + 图片文件拦截器
 *
 * L1 防御层：在 Read 调用前检查目标文件大小。
 * 文件 >100KB 且无 limit 参数时 block，强制分段读取。
 * 图片文件（png/jpg/jpeg/gif/webp/bmp/pdf/pptx）一律 block —
 *   二进制文件 Read 结果会以 base64 内联进 messages[]，
 *   严重膨胀对话历史，直接触发 32MB API 限制。
 *
 * 根因：大文件/图片 Read 结果 inline 进 messages[]，累积后超过 API 限制。
 *   - 文本大文件：21轮迭代累积超过 20MB
 *   - 图片/PDF/PPT：单张截图 30-130KB × 多页 × 多轮 → 超过 32MB
 * OS 类比：ulimit -f (file size limit) — 防止单个进程写入过大文件导致磁盘耗尽。
 *
 * 阈值设计（基于实测）：
 *   - store_core.py  = 74KB  → 超过 WARN，需指导
 *   - store_mm.py    = 119KB → 超过 BLOCK，必须分段
 *   - 截图 png/jpg   任意大小 → 直接 BLOCK（无意义 Read 二进制）
 *   - 典型 Python 模块 < 20KB → 不受影响
 *
 * 迭代 B7：会话级 context 累计增量追踪 (session_context_guard)
 *   OS 类比：cgroup memory.limit_in_bytes — 进程组级内存上限，
 *   超过阈值时触发 SIGKILL 或 throttle，而非只限制单个分配。
 *
 *   新增功能：
 *   - 追踪本 session 通过 Read 已注入的估计字节数
 *   - 当累计注入 > SESSION_WARN_MB 时：warn + 建议 Grep/LSP
 *   - 当累计注入 > SESSION_BLOCK_MB 时：block 所有 >20KB 的 Read（除非有 limit）
 *   - 状态文件：~/.claude/memory-os/thrashing_state.json（与 thrashing_detector 共享）
 *
 * 匹配工具：Read
 * 决策：block (图片/二进制 | >100KB 且无 limit | session超限) | warn (>20KB) | allow
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');

const MAX_FILE_BYTES = 100 * 1024; // 100KB — 超过此值必须分段读
const WARN_FILE_BYTES = 20 * 1024; // 20KB  — 超过此值提示使用 limit

// 会话级 context 增量阈值（与 thrashing_detector.py 的阈值对齐）
const SESSION_WARN_MB = 5;   // > 5MB → warn，但不 block
const SESSION_BLOCK_MB = 15; // > 15MB → block 中等文件（>20KB），防止 thrashing 恶化

// 状态文件路径（与 thrashing_detector.py 共享）
const STATE_FILE = path.join(os.homedir(), '.claude', 'memory-os', 'thrashing_state.json');

function loadState() {
  try {
    if (fs.existsSync(STATE_FILE)) {
      return JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
    }
  } catch (_) {}
  return { session_bytes: 0, last_warn_ts: 0, window_bytes_history: [] };
}

function getSessionMB() {
  const state = loadState();
  return (state.session_bytes || 0) / 1024 / 1024;
}

// 二进制/媒体文件扩展名 — Read 结果会 base64 膨胀，直接 block
const BINARY_EXTS = new Set([
  '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.ico',
  '.pdf', '.pptx', '.ppt', '.docx', '.doc', '.xlsx', '.xls',
  '.zip', '.tar', '.gz', '.mp4', '.mp3', '.mov', '.avi',
]);

async function main() {
  let raw = '';
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  raw = Buffer.concat(chunks).toString('utf8').substring(0, 128 * 1024);

  // Quick tool name extraction
  const toolMatch = raw.match(/"tool_name"\s*:\s*"([^"]+)"/);
  const toolName = toolMatch ? toolMatch[1] : '';

  // Only intercept Read tool
  if (toolName !== 'Read') {
    process.exit(0);
  }

  // Extract file_path and limit from tool_input
  let input;
  try {
    const parsed = JSON.parse(raw);
    input = parsed.tool_input || {};
  } catch {
    process.exit(0);
  }

  const filePath = input.file_path || '';
  const hasLimit = input.limit !== undefined && input.limit !== null;

  if (!filePath) {
    process.exit(0);
  }

  // Block 图片/二进制文件（无论大小，Read 无意义且会膨胀 context）
  const ext = path.extname(filePath).toLowerCase();
  if (BINARY_EXTS.has(ext)) {
    const result = {
      decision: 'block',
      reason: `[filesize_guard] ${path.basename(filePath)} 是二进制/媒体文件（${ext}），` +
              `Read 会将其 base64 内联进对话历史，严重膨胀 context 触发 32MB API 限制。` +
              `图片请用 browser_take_screenshot 后直接分析，PDF/PPT 请用 Bash 工具（pdfinfo/ls -lh）验证，` +
              `不要用 Read 读取二进制文件。`
    };
    process.stdout.write(JSON.stringify(result));
    process.exit(2);
  }

  // Check file size
  let stat;
  try {
    stat = fs.statSync(filePath);
  } catch {
    // File doesn't exist or not accessible — let Read handle the error
    process.exit(0);
  }

  if (!stat.isFile()) {
    process.exit(0);
  }

  const fileSize = stat.size;
  const fileSizeKB = (fileSize / 1024).toFixed(0);

  // Block: file >100KB AND no limit parameter set
  if (fileSize > MAX_FILE_BYTES && !hasLimit) {
    const result = {
      decision: 'block',
      reason: `[filesize_guard] ${path.basename(filePath)} 体积 ${fileSizeKB}KB，超过 ${MAX_FILE_BYTES/1024}KB 限制。` +
              `整体 Read 会将 ${fileSizeKB}KB 注入对话历史，累积后导致 API 限制触发。` +
              `请使用分段读取：Read(limit=100) 查看前段，Read(offset=100, limit=100) 读后段；` +
              `或用 Grep/LSP 工具直接定位目标符号，避免整体加载。`
    };
    process.stdout.write(JSON.stringify(result));
    process.exit(2);
  }

  // === 会话级 context 增量检查（session_context_guard）===
  // OS 类比：cgroup memory.limit_in_bytes — 进程组级累计上限
  const sessionMB = getSessionMB();

  if (sessionMB >= SESSION_BLOCK_MB && fileSize > WARN_FILE_BYTES && !hasLimit) {
    // session 已累计 >15MB，中等文件也需要分段
    const result = {
      decision: 'block',
      reason: `[filesize_guard:session_guard] ⚠ 本 session 已累计注入约 ${sessionMB.toFixed(1)}MB context` +
              `（阈值 ${SESSION_BLOCK_MB}MB）。Autocompact thrashing 风险极高。` +
              `${path.basename(filePath)} ${fileSizeKB}KB 被拦截。` +
              `请改用：Grep 搜索关键字、LSP goToDefinition、或 mcp__memory-os__memory_lookup。` +
              `如必须读取，请用 Read(limit=50) 只读所需行段。`
    };
    process.stdout.write(JSON.stringify(result));
    process.exit(2);
  }

  // Warn: file >20KB without limit，或 session 接近阈值
  if (fileSize > WARN_FILE_BYTES && !hasLimit) {
    const sessionNote = sessionMB >= SESSION_WARN_MB
      ? ` [session_guard: 已累计 ${sessionMB.toFixed(1)}MB，接近 thrashing 阈值]`
      : '';
    process.stderr.write(
      `[filesize_guard] ⚠ ${path.basename(filePath)} ${fileSizeKB}KB > 20KB，` +
      `建议用 limit 参数分段读取以节省上下文预算。${sessionNote}\n`
    );
  } else if (sessionMB >= SESSION_WARN_MB && fileSize > 5 * 1024) {
    // session 已超 warn 阈值，即使文件本身 >5KB 也附加提醒
    process.stderr.write(
      `[filesize_guard:session_guard] ⚠ session 已累计 ${sessionMB.toFixed(1)}MB context，` +
      `建议优先用 Grep/memory_lookup 替代 Read。\n`
    );
  }

  process.exit(0);
}

main().catch(err => {
  process.stderr.write(`[filesize_guard] error: ${err.message}\n`);
  process.exit(0); // Never block on hook failure
});
