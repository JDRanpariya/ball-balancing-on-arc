#!/usr/bin/env bash
# Sync the arXiv preprint source into the Overleaf git clone and optionally push.
#
# Usage:
#   bash scripts/sync_overleaf.sh            # copy + commit in the clone
#   bash scripts/sync_overleaf.sh --push     # also push to Overleaf (needs OVERLEAF_GIT_TOKEN)
#
# OVERLEAF_GIT_TOKEN lives in ~/.zshrc. Override the clone location with OVERLEAF_DIR.
# NOTE: supplementary/ is intentionally NOT auto-synced: the repo uses a bib symlink
# that Overleaf cannot follow (see the Overleaf clone's OVERLEAF_NOTES.md). Sync it by hand.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"            # paper/ram
OVL="${OVERLEAF_DIR:-$HOME/Desktop/ram-overleaf}"
PROJ="68ef6899746b319e964305c3"
PUSH="${1:-}"

[ -d "$OVL/.git" ] || { echo "No Overleaf clone at $OVL"; exit 1; }

cp "$HERE/main.tex" "$HERE/bibliography.bib" "$OVL/"
cp "$HERE/sections/"*.tex "$OVL/sections/"
cp "$HERE/figures/"* "$OVL/figures/" 2>/dev/null || true

cd "$OVL"
git add -A
if git diff --cached --quiet; then
  echo "No file changes to sync."
else
  git commit -m "Sync main paper from public main"
fi

if [ "$PUSH" = "--push" ]; then
  # Token lives in ~/.zshrc; pull it in when the shell did not export it.
  if [ -z "${OVERLEAF_GIT_TOKEN:-}" ]; then
    OVERLEAF_GIT_TOKEN="$(grep -oE 'OVERLEAF_GIT_TOKEN=[^ ]+' "$HOME/.zshrc" 2>/dev/null \
      | head -1 | cut -d= -f2- | sed 's/[\"'\'']//g')"
  fi
  : "${OVERLEAF_GIT_TOKEN:?Set OVERLEAF_GIT_TOKEN (it is in ~/.zshrc)}"
  URL="https://git:${OVERLEAF_GIT_TOKEN}@git.overleaf.com/${PROJ}"
  # Overleaf's web editor makes its own commits; merge them before pushing.
  git pull --no-edit "$URL" main 2>&1 | sed -E "s/${OVERLEAF_GIT_TOKEN}/<TOKEN>/g" \
    || { echo "Overleaf pull failed (conflict?). Resolve in $OVL, then re-run."; exit 1; }
  # Push whenever the local branch is ahead, even if this run copied no new files
  # (a previous run may have committed but failed to push). Overleaf's branch is main.
  git push "$URL" HEAD:main 2>&1 | sed -E "s/${OVERLEAF_GIT_TOKEN}/<TOKEN>/g"
fi
echo "Done."
