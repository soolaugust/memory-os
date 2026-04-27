#!/usr/bin/env node
/**
 * memory-os posttool_observers.js — PostToolUse Observer Coalescing
 *
 * 迭代70: 将 PostToolUse 的两个通配符 async 观察型 hook 合并为一个 dispatcher。
 * OS 类比: interrupt coalescing — 合并同类中断。
 *
 * 合并的 hooks:
 *   1. snarc post-tool-use.js (原: *, async, timeout 5s)
 *   2. continuous-learning observe.sh (原: *, async, timeout 10s)
 *
 * 效果: 从 2 个进程启动降为 1 个，共享 stdin 读取。
 * 两者都是 fire-and-forget 观察型，不影响工具执行结果。
 *
 * L2 防御: screenshot_gc — 截图完成后自动删除本地文件，防止图片累积
 *   进对话历史导致 32MB API 限制触发。
 *   根因: browser_take_screenshot 保存文件到磁盘，后续 Read 或其他工具
 *   将图片内容 base64 内联进 messages[]，多轮迭代后消息体超限。
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const MAX_STDIN = 512 * 1024;

// 截图 GC: 工具调用后自动删除截图文件，防止 context 膨胀
// 匹配 browser_take_screenshot 保存的文件
function screenshotGC(toolName, toolInput) {
  if (toolName !== 'browser_take_screenshot' && toolName !== 'mcp__plugin_everything-claude-code_playwright__browser_take_screenshot') {
    return;
  }
  try {
    const input = typeof toolInput === 'string' ? JSON.parse(toolInput) : (toolInput || {});
    const filename = input.filename;
    if (!filename) return;
    // 解析为绝对路径（相对路径基于 cwd）
    const filePath = path.isAbsolute(filename) ? filename : path.join(process.cwd(), filename);
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      process.stderr.write(`[screenshot_gc] deleted ${filePath} — prevent context bloat\n`);
    }
  } catch (e) {
    process.stderr.write(`[screenshot_gc] error: ${e.message}\n`);
  }
}

async function main() {
  // Read stdin once, share with all observers
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString('utf8').substring(0, MAX_STDIN);

  // L2 防御: 截图 GC（同步，优先执行）
  try {
    const data = JSON.parse(raw);
    const toolName = data.tool_name || data.toolName || '';
    const toolInput = data.tool_input || {};
    screenshotGC(toolName, toolInput);
  } catch (_) {}

  // Observer 1: snarc post-tool-use
  const snarcScript = path.join(
    process.env.HOME || require('os').homedir(),
    '.local', 'share', 'snarc', 'dist', 'hooks', 'handlers', 'post-tool-use.js'
  );

  // Observer 2: continuous-learning observe (ECC)
  const eccRoot = process.env.CLAUDE_PLUGIN_ROOT || '';
  const observeRunner = eccRoot
    ? path.join(eccRoot, 'scripts', 'hooks', 'run-with-flags-shell.sh')
    : '';

  const observers = [];

  // Dispatch snarc if available
  if (fs.existsSync(snarcScript)) {
    observers.push(dispatchNode(snarcScript, raw, 5000));
  }

  // Dispatch continuous-learning observe if ECC available
  if (observeRunner && fs.existsSync(observeRunner)) {
    observers.push(dispatchShell(
      observeRunner,
      ['post:observe', 'skills/continuous-learning-v2/hooks/observe.sh', 'standard,strict'],
      raw,
      10000
    ));
  }

  // Wait for all (fire-and-forget, but we still wait to avoid zombie processes)
  if (observers.length > 0) {
    await Promise.all(observers);
  }

  process.exit(0);
}

function dispatchNode(script, stdin, timeoutMs) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [script], {
      env: process.env,
      cwd: process.cwd(),
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: timeoutMs,
    });

    child.stdin.write(stdin);
    child.stdin.end();

    child.on('close', (code) => resolve(code));
    child.on('error', (err) => {
      process.stderr.write(`[posttool_observers:snarc] ${err.message}\n`);
      resolve(1);
    });
  });
}

function dispatchShell(script, args, stdin, timeoutMs) {
  return new Promise((resolve) => {
    const child = spawn('bash', [script, ...args], {
      env: process.env,
      cwd: process.cwd(),
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: timeoutMs,
    });

    child.stdin.write(stdin);
    child.stdin.end();

    child.on('close', (code) => resolve(code));
    child.on('error', (err) => {
      process.stderr.write(`[posttool_observers:observe] ${err.message}\n`);
      resolve(1);
    });
  });
}

main().catch(err => {
  process.stderr.write(`[posttool_observers] fatal: ${err.message}\n`);
  process.exit(0); // Never block tool execution
});
