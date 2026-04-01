#!/bin/bash
# ============================================================
# git_push_all.sh
# Force-pushes everything in /home/epl05/EMProject to
# github.com/vectorsim/embed_sim_project  (main branch)
#
# Run: bash git_push_all.sh
# ============================================================

set -e  # stop on any error

REPO_DIR="/home/epl05/EMProject"
BRANCH="main"
REMOTE="origin"
REMOTE_URL="https://github.com/vectorsim/embed_sim_project.git"

echo ""
echo "══════════════════════════════════════════"
echo "  EmbedSim — Git Force Push Script"
echo "══════════════════════════════════════════"

cd "$REPO_DIR"
echo "📂 Working in: $(pwd)"

# ── Ensure remote is set correctly ──────────────────────────
echo ""
echo "▶ Checking remote..."
if git remote get-url "$REMOTE" &>/dev/null; then
    CURRENT_URL=$(git remote get-url "$REMOTE")
    echo "  Remote '$REMOTE' → $CURRENT_URL"
    if [ "$CURRENT_URL" != "$REMOTE_URL" ]; then
        echo "  ⚠ URL mismatch — updating remote..."
        git remote set-url "$REMOTE" "$REMOTE_URL"
        echo "  ✓ Remote updated to $REMOTE_URL"
    fi
else
    echo "  Remote '$REMOTE' not found — adding..."
    git remote add "$REMOTE" "$REMOTE_URL"
    echo "  ✓ Remote added."
fi

# ── Stage everything ─────────────────────────────────────────
echo ""
echo "▶ Staging all files..."
git add -A
echo "  ✓ All files staged."

# ── Commit (only if there is something new) ──────────────────
echo ""
echo "▶ Committing..."
if git diff --cached --quiet; then
    echo "  ℹ Nothing new to commit — working tree clean."
else
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    git commit -m "chore: full project sync $TIMESTAMP"
    echo "  ✓ Committed."
fi

# ── Force push — local wins, remote is overwritten ───────────
echo ""
echo "▶ Force-pushing to $REMOTE/$BRANCH ..."
git push "$REMOTE" "$BRANCH" --force
echo "  ✓ Push complete."

echo ""
echo "══════════════════════════════════════════"
echo "  Done. Check: https://github.com/vectorsim/embed_sim_project"
echo "══════════════════════════════════════════"
echo ""
