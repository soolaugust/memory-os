#!/usr/bin/env node
/**
 * memory-os stop_coalesced.js — Stop Hook Coalescing Dispatcher
 *
 * 迭代71: 将 Stop 事件的 4 个独立 ECC Node 进程合并为 1 个 dispatcher。
 * OS 类比: Linux NAPI interrupt coalescing (2001)
 *   网卡中断到达后，NAPI 切换为轮询模式批量处理 N 个包，
 *   避免每个包一次中断上下文切换。
 *   等价地，4 个 Stop hook 各自启动 Node 进程（~200 行 root resolution 代码重复），
 *   现在合并为 1 次 stdin 读取 + 1 次 root resolution + 4 路并行 dispatch。
 *
 * 合并的 hooks:
 *   1. ECC session-end.js — 会话记录（原: matcher *, async, timeout 10s）
 *   2. ECC evaluate-session.js — 会话评估（原: matcher *, async, timeout 10s）
 *   3. ECC cost-tracker.js — 成本追踪（原: matcher *, async, timeout 10s）
 *   4. ECC desktop-notify.js — 桌面通知（原: matcher *, async, timeout 10s）
 *
 * 保留独立的 hooks（不合并）:
 *   - notify.sh — bash 脚本，async
 *   - snarc session-end.js — 不同 runtime，async
 *   - extractor.py — memory-os sync hook，不可 async 合并
 *
 * 效果: Stop hooks 7 → 4, 节省 3 个 Node 进程启动（~200ms 冷启动 × 3）
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const MAX_STDIN = 1024 * 1024; // 1MB, Stop hooks may have large transcript paths

async function main() {
  // ── 读 stdin 一次，共享给所有 hooks ──
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString('utf8').substring(0, MAX_STDIN);

  // ── ECC plugin root resolution（一次性，不重复 4 次）──
  const rel = path.join('scripts', 'hooks', 'run-with-flags.js');
  const hasRoot = (candidate) => {
    if (!candidate || typeof candidate !== 'string') return false;
    const resolved = path.resolve(candidate.trim());
    return resolved.length > 0 && fs.existsSync(path.join(resolved, rel));
  };

  let eccRoot = '';
  const envRoot = (process.env.CLAUDE_PLUGIN_ROOT || '').trim();
  if (hasRoot(envRoot)) {
    eccRoot = path.resolve(envRoot);
  } else {
    const home = require('os').homedir();
    const claudeDir = path.join(home, '.claude');
    const candidates = [
      claudeDir,
      path.join(claudeDir, 'plugins', 'everything-claude-code'),
      path.join(claudeDir, 'plugins', 'everything-claude-code@everything-claude-code'),
      path.join(claudeDir, 'plugins', 'marketplace', 'everything-claude-code'),
    ];
    for (const c of candidates) {
      if (hasRoot(c)) { eccRoot = c; break; }
    }
    // Cache directory scan fallback
    if (!eccRoot) {
      try {
        const cacheBase = path.join(claudeDir, 'plugins', 'cache', 'everything-claude-code');
        for (const org of fs.readdirSync(cacheBase, { withFileTypes: true })) {
          if (!org.isDirectory()) continue;
          for (const ver of fs.readdirSync(path.join(cacheBase, org.name), { withFileTypes: true })) {
            if (!ver.isDirectory()) continue;
            const candidate = path.join(cacheBase, org.name, ver.name);
            if (hasRoot(candidate)) { eccRoot = candidate; break; }
          }
          if (eccRoot) break;
        }
      } catch { /* no cache dir */ }
    }
  }

  if (!eccRoot) {
    // No ECC available — nothing to dispatch
    process.stderr.write('[stop_coalesced] WARNING: could not resolve ECC plugin root; skipping\n');
    process.stdout.write(raw); // passthrough for cost-tracker compatibility
    process.exit(0);
  }

  const runnerScript = path.join(eccRoot, rel);

  // ── 4 路并行 dispatch ──
  const hooks = [
    {
      id: 'stop:session-end',
      script: 'scripts/hooks/session-end.js',
      profiles: 'minimal,standard,strict',
      timeout: 30000,
    },
    {
      id: 'stop:evaluate-session',
      script: 'scripts/hooks/evaluate-session.js',
      profiles: 'minimal,standard,strict',
      timeout: 30000,
    },
    {
      id: 'stop:cost-tracker',
      script: 'scripts/hooks/cost-tracker.js',
      profiles: 'minimal,standard,strict',
      timeout: 30000,
      passthrough: true, // cost-tracker writes to stdout
    },
    {
      id: 'stop:desktop-notify',
      script: 'scripts/hooks/desktop-notify.js',
      profiles: 'standard,strict',
      timeout: 30000,
    },
  ];

  const promises = hooks.map(hook => dispatch(runnerScript, hook, raw));

  const results = await Promise.all(promises);

  // If cost-tracker produced stdout, write it (for passthrough compatibility)
  for (let i = 0; i < hooks.length; i++) {
    if (hooks[i].passthrough && results[i].stdout) {
      process.stdout.write(results[i].stdout);
    }
  }

  process.exit(0);
}

function dispatch(runnerScript, hook, stdin) {
  return new Promise((resolve) => {
    const child = spawn(
      process.execPath,
      [runnerScript, hook.id, hook.script, hook.profiles],
      {
        env: process.env,
        cwd: process.cwd(),
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: hook.timeout,
      }
    );

    child.stdin.write(stdin);
    child.stdin.end();

    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });

    child.on('close', (code) => {
      if (code !== 0 && stderr) {
        process.stderr.write(`[stop_coalesced:${hook.id}] ${stderr}\n`);
      }
      resolve({ code, stdout, stderr });
    });

    child.on('error', (err) => {
      process.stderr.write(`[stop_coalesced:${hook.id}] spawn error: ${err.message}\n`);
      resolve({ code: 1, stdout: '', stderr: err.message });
    });
  });
}

main().catch(err => {
  process.stderr.write(`[stop_coalesced] fatal: ${err.message}\n`);
  process.exit(0); // Never block on hook failure
});
