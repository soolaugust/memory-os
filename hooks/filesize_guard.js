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
 * 匹配工具：Read
 * 决策：block (图片/二进制 | >100KB 且无 limit) | warn (>20KB) | allow
 */

'use strict';

const fs = require('fs');
const path = require('path');

const MAX_FILE_BYTES = 100 * 1024; // 100KB — 超过此值必须分段读
const WARN_FILE_BYTES = 20 * 1024; // 20KB  — 超过此值提示使用 limit

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

  // Warn: file >20KB without limit
  if (fileSize > WARN_FILE_BYTES && !hasLimit) {
    process.stderr.write(
      `[filesize_guard] ⚠ ${path.basename(filePath)} ${fileSizeKB}KB > 20KB，` +
      `建议用 limit 参数分段读取以节省上下文预算。\n`
    );
  }

  process.exit(0);
}

main().catch(err => {
  process.stderr.write(`[filesize_guard] error: ${err.message}\n`);
  process.exit(0); // Never block on hook failure
});
