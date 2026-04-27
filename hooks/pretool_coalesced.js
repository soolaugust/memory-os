#!/usr/bin/env node
/**
 * memory-os pretool_coalesced.js — Interrupt Coalescing Dispatcher
 *
 * 迭代69: 将 PreToolUse 的多个独立 hook 合并为一个 dispatcher。
 * OS 类比: interrupt coalescing — 将 N 次中断合并为 1 次批处理。
 *
 * 合并的 hooks:
 *   1. insaits-security (原: Bash|Write|Edit|MultiEdit, async)
 *   2. governance-capture (原: Bash|Write|Edit|MultiEdit, async)
 *
 * 效果: 对 Bash/Write/Edit/MultiEdit 工具，从 2 次 node 进程启动
 *        降为 1 次，共享 stdin 解析和进程开销。
 *
 * 对非匹配工具（Read/Glob/Grep/...），O(1) 快速退出。
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// 只对这些工具执行安全检查
const MUTATING_TOOLS = new Set(['Bash', 'Write', 'Edit', 'MultiEdit']);

const MAX_STDIN = 512 * 1024; // 512KB, enough for any hook input

async function main() {
  // Fast path: read tool name from stdin without full parse
  let raw = '';

  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  raw = Buffer.concat(chunks).toString('utf8').substring(0, MAX_STDIN);

  // Quick tool name extraction without full JSON parse
  const toolMatch = raw.match(/"tool_name"\s*:\s*"([^"]+)"/);
  const toolName = toolMatch ? toolMatch[1] : '';

  // Fast exit for non-mutating tools (Read, Glob, Grep, WebFetch, etc.)
  if (!MUTATING_TOOLS.has(toolName)) {
    process.exit(0);
  }

  // ECC plugin root resolution
  const pluginRoot = process.env.CLAUDE_PLUGIN_ROOT || '';
  const eccRoot = pluginRoot && fs.existsSync(path.join(pluginRoot, 'scripts', 'hooks', 'run-with-flags.js'))
    ? pluginRoot
    : '';

  if (!eccRoot) {
    // No ECC available, skip
    process.exit(0);
  }

  const runnerScript = path.join(eccRoot, 'scripts', 'hooks', 'run-with-flags.js');

  // Dispatch both hooks in parallel (fire-and-forget since both are async)
  const hooks = [
    {
      id: 'pre:insaits-security',
      script: 'scripts/hooks/insaits-security-wrapper.js',
      profiles: 'standard,strict',
      timeout: 15000,
    },
    {
      id: 'pre:governance-capture',
      script: 'scripts/hooks/governance-capture.js',
      profiles: 'standard,strict',
      timeout: 10000,
    },
  ];

  const promises = hooks.map(hook => {
    return new Promise((resolve) => {
      const child = spawn(process.execPath, [runnerScript, hook.id, hook.script, hook.profiles], {
        env: process.env,
        cwd: process.cwd(),
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: hook.timeout,
      });

      child.stdin.write(raw);
      child.stdin.end();

      // Collect stderr for debugging
      let stderr = '';
      child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });

      child.on('close', (code) => {
        if (code !== 0 && stderr) {
          process.stderr.write(`[coalesced:${hook.id}] ${stderr}`);
        }
        resolve(code);
      });

      child.on('error', (err) => {
        process.stderr.write(`[coalesced:${hook.id}] spawn error: ${err.message}\n`);
        resolve(1);
      });
    });
  });

  // Wait for all hooks to complete (they were async anyway)
  await Promise.all(promises);
  process.exit(0);
}

main().catch(err => {
  process.stderr.write(`[pretool_coalesced] fatal: ${err.message}\n`);
  process.exit(0); // Don't block tool execution on hook failure
});
