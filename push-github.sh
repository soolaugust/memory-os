#!/usr/bin/env bash
# push-github.sh — 一键同步到 GitHub（自动切换身份）
# 配置：在 ~/.gitconfig 或本仓库 .git/config 中设置
#   [push-github]
#       name = your-github-username
#       email = your@gmail.com
set -e

# 从 git config 读取 GitHub 身份（未配置则报错提示）
GITHUB_NAME=$(git config push-github.name 2>/dev/null || echo "")
GITHUB_EMAIL=$(git config push-github.email 2>/dev/null || echo "")

if [[ -z "$GITHUB_NAME" || -z "$GITHUB_EMAIL" ]]; then
  echo "❌ 请先配置 GitHub 身份："
  echo "   git config push-github.name  'your-username'"
  echo "   git config push-github.email 'your@gmail.com'"
  exit 1
fi

# 从 git config 读取本地（origin）身份
ORIGIN_NAME=$(git config user.name)
ORIGIN_EMAIL=$(git config user.email)

BRANCH=$(git rev-parse --abbrev-ref HEAD)
LOCAL_HEAD=$(git rev-parse HEAD)

echo "📦 当前分支: $BRANCH ($LOCAL_HEAD)"

# 1. 重写本地分支为 GitHub 身份（临时）
echo "🔄 切换为 GitHub 身份 ($GITHUB_NAME)..."
git filter-branch -f --env-filter "
export GIT_AUTHOR_NAME=\"$GITHUB_NAME\"
export GIT_AUTHOR_EMAIL=\"$GITHUB_EMAIL\"
export GIT_COMMITTER_NAME=\"$GITHUB_NAME\"
export GIT_COMMITTER_EMAIL=\"$GITHUB_EMAIL\"
" -- refs/heads/$BRANCH > /dev/null 2>&1

# 2. 推送到 GitHub
echo "🚀 推送到 GitHub..."
git push --force github $BRANCH

# 3. 恢复原始身份
echo "🔄 恢复本地身份 ($ORIGIN_NAME)..."
git filter-branch -f --env-filter "
export GIT_AUTHOR_NAME=\"$ORIGIN_NAME\"
export GIT_AUTHOR_EMAIL=\"$ORIGIN_EMAIL\"
export GIT_COMMITTER_NAME=\"$ORIGIN_NAME\"
export GIT_COMMITTER_EMAIL=\"$ORIGIN_EMAIL\"
" -- refs/heads/$BRANCH > /dev/null 2>&1

echo "✅ 完成！GitHub 已同步，本地身份已恢复"
